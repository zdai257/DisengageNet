"""
network_builder_3d.py — depth-aware extension of the GazeLLE baseline.

Architecturally identical to ``GazeMoE`` (same DINOv2 backbone, same
multi-scale-style transformer trunk, same heatmap and in/out heads) plus a
*parallel* depth head that produces a (64, 64) relative-depth map per person.
Both heads share every parameter up to and including the transformer trunk —
the depth supervision therefore acts as a geometry-aware auxiliary task that
sharpens the spatial features used by the heatmap head.

Only the depth value at (and immediately around) the gaze target is meaningful
for the 3-D gaze task: the corresponding training script weights the depth
loss by the GT spatial heatmap so that distant pixels contribute negligibly.

The model exposes the same ``get_gazelle_state_dict`` / ``load_gazelle_state_dict``
helpers as ``GazeMoE`` so existing 2-D pretraining checkpoints (heatmap only)
can be loaded directly — the new ``depth_head.*`` keys will simply be reported
as missing and trained from scratch.
"""

import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block

import network.utils as utils
from network.backbone import DinoV2Backbone
from network.network_builder import positionalencoding2d


class GazeLLE3D(nn.Module):
    def __init__(self, backbone, inout=True, dim=256, num_layers=3,
                 in_size=(448, 448), out_size=(64, 64)):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout

        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        self.register_buffer(
            "pos_embed",
            positionalencoding2d(self.dim, self.featmap_h, self.featmap_w)
            .squeeze(dim=0).squeeze(dim=0),
        )
        self.transformer = nn.Sequential(*[
            Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
            for _ in range(num_layers)
        ])

        # ------------------------------------------------------------------
        # Heatmap head — identical to GazeLLE.
        # ------------------------------------------------------------------
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # NEW: Depth head — shares the transformer trunk with heatmap_head,
        # then a slightly deeper decoder so it can learn local geometry
        # without competing channel-wise with the heatmap.
        # Sigmoid because the GT depth is min-max normalised to [0, 1] per
        # image (matches DepthAnythingV2 relative-depth convention).
        # ------------------------------------------------------------------
        self.depth_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.head_token = nn.Embedding(1, self.dim)
        if self.inout:
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
            self.inout_token = nn.Embedding(1, self.dim)

    # ----------------------------------------------------------------------
    def forward(self, input):
        """
        input["images"]: [B, 3, H, W]
        input["bboxes"]: list[B] of list[N_i] of (xmin, ymin, xmax, ymax) in [0, 1]

        Returns:
            {
                "heatmap": list[B] of [N_i, 64, 64] in [0, 1],
                "inout"  : list[B] of [N_i]        in [0, 1] (None if inout=False),
                "depth"  : list[B] of [N_i, 64, 64] in [0, 1] (relative depth),
            }
        """
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]

        x = self.backbone.forward(input["images"])
        x = self.linear(x)
        x = x + self.pos_embed
        x = utils.repeat_tensors(x, num_ppl_per_img)

        head_maps = torch.cat(
            self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
        head_map_emb = (head_maps.unsqueeze(dim=1)
                        * self.head_token.weight.unsqueeze(-1).unsqueeze(-1))
        x = x + head_map_emb

        x = x.flatten(start_dim=2).permute(0, 2, 1)  # b c h w -> b (h w) c

        if self.inout:
            x = torch.cat([
                self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1),
                x,
            ], dim=1)

        x = self.transformer(x)

        if self.inout:
            inout_tokens = x[:, 0, :]
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :]
        else:
            inout_preds = None

        # Restore (b, c, h, w) layout for the convolutional decoders.
        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]) \
             .permute(0, 3, 1, 2)

        # ----- heatmap branch -----
        h = self.heatmap_head(x).squeeze(dim=1)
        h = torchvision.transforms.functional.resize(h, self.out_size)
        heatmap_preds = utils.split_tensors(h, num_ppl_per_img)

        # ----- depth branch -----
        d = self.depth_head(x).squeeze(dim=1)
        d = torchvision.transforms.functional.resize(d, self.out_size)
        depth_preds = utils.split_tensors(d, num_ppl_per_img)

        return {
            "heatmap": heatmap_preds,
            "inout":   inout_preds,
            "depth":   depth_preds,
        }

    # ----------------------------------------------------------------------
    def get_input_head_maps(self, bboxes):
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None:
                    img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    width, height = self.featmap_w, self.featmap_h
                    xmin = round(xmin * width)
                    ymin = round(ymin * height)
                    xmax = round(xmax * width)
                    ymax = round(ymax * height)
                    head_map = torch.zeros((height, width))
                    head_map[ymin:ymax, xmin:xmax] = 1
                    img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps

    # ----------------------------------------------------------------------
    # Checkpoint helpers — fully compatible with the GazeLLE 2-D format so
    # that existing pretraining checkpoints (heatmap-only) load cleanly and
    # the depth head simply trains from scratch.
    # ----------------------------------------------------------------------
    def get_gazelle_state_dict(self, include_backbone=False):
        if include_backbone:
            return self.state_dict()
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith("backbone")}

    def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
        current_state_dict = self.state_dict()
        keys1 = current_state_dict.keys()
        keys2 = ckpt_state_dict.keys()

        if not include_backbone:
            keys1 = set(k for k in keys1 if not k.startswith("backbone"))
            keys2 = set(k for k in keys2 if not k.startswith("backbone"))
        else:
            keys1 = set(keys1)
            keys2 = set(keys2)

        if len(keys2 - keys1) > 0:
            print("WARNING unused keys in provided state dict: ", keys2 - keys1)
        if len(keys1 - keys2) > 0:
            print("WARNING provided state dict missing keys (will be trained "
                  "from scratch — expected for the new depth_head.*): ",
                  keys1 - keys2)

        for k in list(keys1 & keys2):
            current_state_dict[k] = ckpt_state_dict[k]

        self.load_state_dict(current_state_dict, strict=False)


# --------------------------------------------------------------------------
# Factory — mirrors get_gazelle_model() so it is configuration-driven.
# --------------------------------------------------------------------------

def get_gazelle3d_model(configuration):
    factory = {
        "gazelle3d_dinov2_vitb14":          gazelle3d_dinov2_vitb14,
        "gazelle3d_dinov2_vitl14":          gazelle3d_dinov2_vitl14,
        "gazelle3d_dinov2_vitb14_inout":    gazelle3d_dinov2_vitb14_inout,
        "gazelle3d_dinov2_vitl14_inout":    gazelle3d_dinov2_vitl14_inout,
    }
    name = configuration["model"]["name"]
    assert name in factory, f"Unknown 3-D model name: {name}"
    return factory[name]()


def gazelle3d_dinov2_vitb14():
    backbone = DinoV2Backbone("dinov2_vitb14")
    transform = backbone.get_transform((448, 448))
    return GazeLLE3D(backbone, inout=False), transform


def gazelle3d_dinov2_vitl14():
    backbone = DinoV2Backbone("dinov2_vitl14")
    transform = backbone.get_transform((448, 448))
    return GazeLLE3D(backbone, inout=False), transform


def gazelle3d_dinov2_vitb14_inout():
    backbone = DinoV2Backbone("dinov2_vitb14")
    transform = backbone.get_transform((448, 448))
    return GazeLLE3D(backbone, inout=True), transform


def gazelle3d_dinov2_vitl14_inout():
    backbone = DinoV2Backbone("dinov2_vitl14")
    transform = backbone.get_transform((448, 448))
    return GazeLLE3D(backbone, inout=True), transform
