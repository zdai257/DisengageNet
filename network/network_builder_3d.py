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
import torch.nn.functional as F
import torchvision
from timm.models.vision_transformer import Block

import network.utils as utils
from network.backbone import DinoV2Backbone
from network.network_builder import positionalencoding2d


class GazeLLE3D(nn.Module):
    """Depth-aware extension of GazeLLE — head-anchored relative depth.

    The model has two parallel cross-attention depth queries — one for the
    gaze target and one for the head reference — both decoded by a *shared*
    scalar MLP.  Supervision is applied to the **log-ratio**
    ``log(z_gaze) − log(z_head)``, which is invariant to the per-image
    min-max normalisation used to produce DepthAnythingV2 pseudo-labels.
    This eliminates the scale ambiguity that plagued the earlier dense /
    single-query designs (see paper notes; option A and the original dense
    head were retired because they hypothesised an absolute scale that the
    monocular signal cannot identify).

    Outputs:
        heatmap     : list[B] of [N_i, 64, 64] in [0, 1]  — spatial probability
        inout       : list[B] of [N_i]        in [0, 1] (None if inout=False)
        depth_gaze  : list[B] of [N_i]        in [0, 1] (sigmoid scalar)
        depth_head  : list[B] of [N_i]        in [0, 1] (sigmoid scalar)

    Optional architectural mode:

    Option B — ``use_refine=True``
        Adds ``num_refine_layers`` transformer blocks that consume the
        original trunk features PLUS a depth-conditioning map derived from
        the predicted log-ratio (``log z_gaze − log z_head``) and emit a
        refined heatmap.  This couples depth ↔ heatmap so improving the
        anchored depth also sharpens 2-D localisation (Option B+C in the
        design notes).
    """

    def __init__(self, backbone, inout=True, dim=256, num_layers=3,
                 in_size=(448, 448), out_size=(64, 64),
                 use_refine=False,
                 num_refine_layers=1):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout
        self.use_refine        = use_refine
        self.num_refine_layers = int(num_refine_layers) if use_refine else 0

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
        # Anchored depth: two parallel query tokens (gaze target + head
        # reference), shared scalar MLP.  The two scalars are individually
        # unidentifiable; only their log-ratio is meaningful and supervised.
        # ------------------------------------------------------------------
        self.depth_token_gaze = nn.Embedding(1, self.dim)
        self.depth_token_head = nn.Embedding(1, self.dim)
        self.depth_scalar_head = nn.Sequential(
            nn.Linear(self.dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # Optional depth-conditioned refinement (Option B, use_refine=True).
        # Conditioning signal is the predicted log-ratio broadcast across
        # the spatial feature grid — itself scale-invariant.
        # ------------------------------------------------------------------
        if self.use_refine:
            self.refine_blocks = nn.Sequential(*[
                Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
                for _ in range(self.num_refine_layers)
            ])
            self.depth_to_feat = nn.Conv2d(1, self.dim, kernel_size=1)
            self.refined_heatmap_head = nn.Sequential(
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                nn.Conv2d(dim, 1, kernel_size=1, bias=False),
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
                "heatmap"   : list[B] of [N_i, 64, 64] in [0, 1],
                "inout"     : list[B] of [N_i]        in [0, 1] (None if inout=False),
                "depth_gaze": list[B] of [N_i]        in [0, 1] — head-anchored,
                "depth_head": list[B] of [N_i]        in [0, 1] — head-anchored,
            }

        Only the *log-ratio* ``log(depth_gaze) − log(depth_head)`` is a
        meaningful per-sample quantity; the two scalars are individually
        unidentifiable under per-image min-max normalisation.
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

        x_seq = x.flatten(start_dim=2).permute(0, 2, 1)  # b c h w -> b (h w) c

        # ----- Prepend special tokens in fixed order ---------------------
        # Order: [inout?, depth_token_gaze, depth_token_head, ...spatial]
        prefix_tokens = []
        if self.inout:
            prefix_tokens.append(
                self.inout_token.weight.unsqueeze(0).repeat(x_seq.shape[0], 1, 1))
        prefix_tokens.append(
            self.depth_token_gaze.weight.unsqueeze(0).repeat(x_seq.shape[0], 1, 1))
        prefix_tokens.append(
            self.depth_token_head.weight.unsqueeze(0).repeat(x_seq.shape[0], 1, 1))
        x_seq = torch.cat(prefix_tokens + [x_seq], dim=1)

        x_seq = self.transformer(x_seq)

        # ----- Peel off special tokens in matching order -----------------
        offset = 0
        if self.inout:
            inout_tok = x_seq[:, offset, :]
            inout_preds = self.inout_head(inout_tok).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            offset += 1
        else:
            inout_preds = None

        z_gaze = self.depth_scalar_head(x_seq[:, offset,     :]).squeeze(dim=-1)
        z_head = self.depth_scalar_head(x_seq[:, offset + 1, :]).squeeze(dim=-1)
        offset += 2

        x_seq = x_seq[:, offset:, :]

        # Restore (b, c, h, w) layout for the convolutional decoder / refine.
        x_feat = x_seq.reshape(
            x_seq.shape[0], self.featmap_h, self.featmap_w, x_seq.shape[2]
        ).permute(0, 3, 1, 2)

        # ----- Heatmap head (pre-refinement / default output) ------------
        h_init = self.heatmap_head(x_feat).squeeze(dim=1)
        h_out  = torchvision.transforms.functional.resize(h_init, self.out_size)

        # ----- Optional depth-conditioned heatmap refinement -------------
        # Conditioning signal: predicted scale-invariant log-ratio,
        # broadcast across the spatial grid as a single-channel feature.
        if self.use_refine:
            log_ratio = (torch.log(z_gaze.clamp(min=1e-4))
                         - torch.log(z_head.clamp(min=1e-4)))           # [B']
            d_cond = log_ratio.view(-1, 1, 1, 1).expand(
                -1, 1, self.featmap_h, self.featmap_w)
            depth_feat = self.depth_to_feat(d_cond)                       # [B', dim, h, w]
            x_refined  = x_feat + depth_feat
            x_refined  = x_refined.flatten(start_dim=2).permute(0, 2, 1)
            x_refined  = self.refine_blocks(x_refined)
            x_refined  = x_refined.reshape(
                x_refined.shape[0], self.featmap_h, self.featmap_w, self.dim
            ).permute(0, 3, 1, 2)
            h_final = self.refined_heatmap_head(x_refined).squeeze(dim=1)
            h_out   = torchvision.transforms.functional.resize(h_final, self.out_size)

        heatmap_preds   = utils.split_tensors(h_out,  num_ppl_per_img)
        depth_gaze_pred = utils.split_tensors(z_gaze, num_ppl_per_img)
        depth_head_pred = utils.split_tensors(z_head, num_ppl_per_img)

        return {
            "heatmap":    heatmap_preds,
            "inout":      inout_preds,
            "depth_gaze": depth_gaze_pred,
            "depth_head": depth_head_pred,
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
    """Factory dispatch.

    Optional architectural flags read from ``configuration["model"]``:
        use_refine        (bool, default False) — Option B (Option B+C with
                                                   the anchored depth always on)
        num_refine_layers (int,  default 1)

    Anchored depth (Option C) is *always* enabled in GazeLLE3D — there is no
    longer a scale-dependent depth mode to opt into.
    """
    factory = {
        "gazelle3d_dinov2_vitb14":          gazelle3d_dinov2_vitb14,
        "gazelle3d_dinov2_vitl14":          gazelle3d_dinov2_vitl14,
        "gazelle3d_dinov2_vitb14_inout":    gazelle3d_dinov2_vitb14_inout,
        "gazelle3d_dinov2_vitl14_inout":    gazelle3d_dinov2_vitl14_inout,
    }
    name = configuration["model"]["name"]
    assert name in factory, f"Unknown 3-D model name: {name}"
    cfg = configuration["model"]
    return factory[name](
        use_refine        = bool(cfg.get("use_refine",        False)),
        num_refine_layers = int(cfg.get("num_refine_layers",  1)),
    )


def _build(backbone_name, inout, **flags):
    backbone = DinoV2Backbone(backbone_name)
    transform = backbone.get_transform((448, 448))
    return GazeLLE3D(backbone, inout=inout, **flags), transform


def gazelle3d_dinov2_vitb14(**flags):
    return _build("dinov2_vitb14", inout=False, **flags)


def gazelle3d_dinov2_vitl14(**flags):
    return _build("dinov2_vitl14", inout=False, **flags)


def gazelle3d_dinov2_vitb14_inout(**flags):
    return _build("dinov2_vitb14", inout=True, **flags)


def gazelle3d_dinov2_vitl14_inout(**flags):
    return _build("dinov2_vitl14", inout=True, **flags)
