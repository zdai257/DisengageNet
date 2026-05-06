"""
network_builder_gt3d.py — depth-aware 3-D gaze-target model (``GT3D``).

Renamed from the earlier ``GazeLLE3D`` to mark this as a new line of work
rather than an incremental tweak of GazeLLE.  The architecture is built
around three ideas:

1.  A DINOv2 trunk + 3-block transformer that produces per-person spatial
    features, identical in spirit to GazeLLE.
2.  Two parallel cross-attention depth queries (gaze target, head reference)
    decoded by a *shared* scalar MLP — the head-anchored log-ratio
    ``log(z_gaze) − log(z_head)`` is the only supervisable, scale-invariant
    quantity we expose.
3.  An optional depth → heatmap refinement loop (``use_refine=True``) that
    feeds the predicted depth back into the spatial features so the 2-D
    peak can shift in response to the predicted depth.  Three conditioning
    variants are available (``refine_cond``):

       * ``"scalar"`` — broadcast the log-ratio scalar (1-channel)
       * ``"pair"``   — broadcast ``[log z_gaze, log z_head]``  (2-channel)
       * ``"dense"``  — predict a low-rank dense depth feature from the
                        trunk and project it back into the trunk dim

Module layout is named so that a 4-bucket optimiser dispatch (substrings
``inout`` / ``depth`` / ``block`` / ``fuse``) routes every *trainable*
parameter to one of the four LR groups — the base ``lr`` arm is never
reached:

    fuse_proj.*, fuse_head_token.*               → fuse_lr   (shallow)
    trunk_blocks.*, refine_blocks.*              → block_lr  (transformer)
    inout_token.*, inout_head.*                  → inout_lr  (output head)
    depth_token_*, depth_scalar_head.*,
    depth_to_feat.*, dense_depth_decoder.*,
    depth_heatmap_head.*,
    depth_refined_heatmap_head.*                 → depth_lr  (output head)

The heatmap output heads are tagged ``depth_*`` because in this model the
heatmap is depth-conditioned (use_refine=True) or co-trained with the
depth task through the shared trunk (use_refine=False); the ``depth_lr``
group is therefore the natural home for them.

``pos_embed`` is registered as a buffer (not a parameter) and is excluded
from ``named_parameters()`` — it does not need an LR.

Outputs:
    heatmap     : list[B] of [N_i, 64, 64] in [0, 1]
    inout       : list[B] of [N_i]        in [0, 1] (None if inout=False)
    depth_gaze  : list[B] of [N_i]        in [0, 1] (sigmoid scalar)
    depth_head  : list[B] of [N_i]        in [0, 1] (sigmoid scalar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from timm.models.vision_transformer import Block

import network.utils as utils
from network.backbone import DinoV2Backbone
from network.network_builder import positionalencoding2d


# --------------------------------------------------------------------------
# GT3D — depth-aware gaze-target model.
# --------------------------------------------------------------------------

class GT3D(nn.Module):
    """Depth-aware 3-D gaze-target estimator (head-anchored).

    Args:
        backbone          : a ``DinoV2Backbone`` instance.
        inout             : if True, attach the in-frame / out-of-frame head.
        dim               : trunk hidden dim.
        num_layers        : number of trunk transformer blocks.
        in_size           : input image (H, W).
        out_size          : output heatmap (H, W).
        use_refine        : enable the depth → heatmap refinement loop.
        refine_cond       : conditioning signal when ``use_refine`` is on:
                            ``"scalar"`` (1-channel log-ratio),
                            ``"pair"``   (2-channel ``[log z_g, log z_h]``),
                            ``"dense"``  (low-rank dense depth feature).
        num_refine_layers : number of refinement transformer blocks.
        dense_depth_dim   : channels of the dense depth feature
                            (only used when ``refine_cond == "dense"``).
    """

    SUPPORTED_REFINE_COND = ("scalar", "pair", "dense")

    def __init__(self, backbone, inout=True, dim=256, num_layers=3,
                 in_size=(448, 448), out_size=(64, 64),
                 use_refine=False,
                 refine_cond="scalar",
                 num_refine_layers=1,
                 dense_depth_dim=32):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout
        self.use_refine        = bool(use_refine)
        self.num_refine_layers = int(num_refine_layers) if self.use_refine else 0
        self.dense_depth_dim   = int(dense_depth_dim)
        if self.use_refine:
            assert refine_cond in self.SUPPORTED_REFINE_COND, (
                f"refine_cond must be one of {self.SUPPORTED_REFINE_COND}, "
                f"got {refine_cond!r}")
        self.refine_cond = refine_cond

        # ------------------------------------------------------------------
        # Backbone-projection ("fuse") stage.
        # ------------------------------------------------------------------
        self.fuse_proj = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        self.register_buffer(
            "pos_embed",
            positionalencoding2d(self.dim, self.featmap_h, self.featmap_w)
            .squeeze(dim=0).squeeze(dim=0),
        )
        self.fuse_head_token = nn.Embedding(1, self.dim)

        # ------------------------------------------------------------------
        # Trunk transformer blocks.
        # ------------------------------------------------------------------
        self.trunk_blocks = nn.Sequential(*[
            Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
            for _ in range(num_layers)
        ])

        # ------------------------------------------------------------------
        # Anchored depth: two parallel query tokens (gaze target + head
        # reference), shared scalar MLP.  Only their log-ratio is supervised.
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
        # Heatmap path: lazy build — exactly one of the two heads exists.
        #   use_refine = False → ``depth_heatmap_head`` only.
        #   use_refine = True  → refine stack + ``depth_refined_heatmap_head``.
        # The ``depth_`` prefix puts both heads in the depth_lr bucket.
        # ------------------------------------------------------------------
        if not self.use_refine:
            self.depth_heatmap_head = nn.Sequential(
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                nn.Conv2d(dim, 1, kernel_size=1, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.refine_blocks = nn.Sequential(*[
                Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
                for _ in range(self.num_refine_layers)
            ])
            self.depth_refined_heatmap_head = nn.Sequential(
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                nn.Conv2d(dim, 1, kernel_size=1, bias=False),
                nn.Sigmoid(),
            )
            in_ch = self._cond_channels(self.refine_cond, self.dense_depth_dim)
            self.depth_to_feat = nn.Conv2d(in_ch, self.dim, kernel_size=1)
            if self.refine_cond == "dense":
                self.dense_depth_decoder = nn.Sequential(
                    nn.ConvTranspose2d(self.dim, self.dim,
                                       kernel_size=2, stride=2),
                    nn.GELU(),
                    nn.Conv2d(self.dim, self.dense_depth_dim, kernel_size=1),
                )

        # ------------------------------------------------------------------
        # In-frame / out-of-frame head (auxiliary).
        # ------------------------------------------------------------------
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
    @staticmethod
    def _cond_channels(refine_cond, dense_depth_dim):
        if refine_cond == "scalar":
            return 1
        if refine_cond == "pair":
            return 2
        if refine_cond == "dense":
            return int(dense_depth_dim)
        raise ValueError(f"Unknown refine_cond: {refine_cond!r}")

    # ----------------------------------------------------------------------
    def _build_cond(self, x_feat, z_gaze, z_head):
        """Build the (B', C, h, w) conditioning tensor for the refine path."""
        h, w = self.featmap_h, self.featmap_w
        eps = 1e-4
        if self.refine_cond == "scalar":
            log_ratio = (torch.log(z_gaze.clamp(min=eps))
                         - torch.log(z_head.clamp(min=eps)))         # [B']
            cond = log_ratio.view(-1, 1, 1, 1).expand(-1, 1, h, w)
        elif self.refine_cond == "pair":
            log_zg = torch.log(z_gaze.clamp(min=eps)).view(-1, 1, 1, 1)
            log_zh = torch.log(z_head.clamp(min=eps)).view(-1, 1, 1, 1)
            cond = torch.cat([log_zg, log_zh], dim=1).expand(-1, 2, h, w)
        else:                                                          # "dense"
            dense = self.dense_depth_decoder(x_feat)                   # [B', K, 2h, 2w]
            cond = F.interpolate(dense, size=(h, w),
                                 mode="bilinear", align_corners=False)
        return cond

    # ----------------------------------------------------------------------
    def forward(self, input):
        """
        input["images"]: [B, 3, H, W]
        input["bboxes"]: list[B] of list[N_i] of (xmin, ymin, xmax, ymax) in [0, 1]
        """
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]

        x = self.backbone.forward(input["images"])
        x = self.fuse_proj(x)
        x = x + self.pos_embed
        x = utils.repeat_tensors(x, num_ppl_per_img)

        head_maps = torch.cat(
            self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
        head_map_emb = (head_maps.unsqueeze(dim=1)
                        * self.fuse_head_token.weight.unsqueeze(-1).unsqueeze(-1))
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

        x_seq = self.trunk_blocks(x_seq)

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

        # ----- Heatmap path ----------------------------------------------
        if not self.use_refine:
            h_out = self.depth_heatmap_head(x_feat).squeeze(dim=1)
        else:
            cond       = self._build_cond(x_feat, z_gaze, z_head)         # [B', C, h, w]
            depth_feat = self.depth_to_feat(cond)                          # [B', dim, h, w]
            x_refined  = x_feat + depth_feat
            x_refined  = x_refined.flatten(start_dim=2).permute(0, 2, 1)
            x_refined  = self.refine_blocks(x_refined)
            x_refined  = x_refined.reshape(
                x_refined.shape[0], self.featmap_h, self.featmap_w, self.dim
            ).permute(0, 3, 1, 2)
            h_out = self.depth_refined_heatmap_head(x_refined).squeeze(dim=1)

        h_out = torchvision.transforms.functional.resize(h_out, self.out_size)

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
    # Checkpoint helpers — backbone-stripped state dicts.
    # ----------------------------------------------------------------------
    def get_gt3d_state_dict(self, include_backbone=False):
        if include_backbone:
            return self.state_dict()
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith("backbone")}

    def load_gt3d_state_dict(self, ckpt_state_dict, include_backbone=False):
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
                  "from scratch): ", keys1 - keys2)

        for k in list(keys1 & keys2):
            current_state_dict[k] = ckpt_state_dict[k]

        self.load_state_dict(current_state_dict, strict=False)


# --------------------------------------------------------------------------
# Factory — mirrors get_gazelle_model() so it is configuration-driven.
# --------------------------------------------------------------------------

def get_gt3d_model(configuration):
    """Factory dispatch.

    Optional architectural flags read from ``configuration["model"]``:
        use_refine        (bool, default False)
        refine_cond       (str,  default "scalar"; one of scalar|pair|dense)
        num_refine_layers (int,  default 1)
        dense_depth_dim   (int,  default 32; only used when refine_cond=="dense")
    """
    factory = {
        "gt3d_dinov2_vitb14":          gt3d_dinov2_vitb14,
        "gt3d_dinov2_vitl14":          gt3d_dinov2_vitl14,
        "gt3d_dinov2_vitb14_inout":    gt3d_dinov2_vitb14_inout,
        "gt3d_dinov2_vitl14_inout":    gt3d_dinov2_vitl14_inout,
    }
    name = configuration["model"]["name"]
    assert name in factory, f"Unknown GT3D model name: {name}"
    cfg = configuration["model"]
    return factory[name](
        use_refine        = bool(cfg.get("use_refine",        False)),
        refine_cond       = str (cfg.get("refine_cond",       "scalar")),
        num_refine_layers = int (cfg.get("num_refine_layers", 1)),
        dense_depth_dim   = int (cfg.get("dense_depth_dim",   32)),
    )


def _build(backbone_name, inout, **flags):
    backbone = DinoV2Backbone(backbone_name)
    transform = backbone.get_transform((448, 448))
    return GT3D(backbone, inout=inout, **flags), transform


def gt3d_dinov2_vitb14(**flags):
    return _build("dinov2_vitb14", inout=False, **flags)


def gt3d_dinov2_vitl14(**flags):
    return _build("dinov2_vitl14", inout=False, **flags)


def gt3d_dinov2_vitb14_inout(**flags):
    return _build("dinov2_vitb14", inout=True, **flags)


def gt3d_dinov2_vitl14_inout(**flags):
    return _build("dinov2_vitl14", inout=True, **flags)
