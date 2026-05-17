"""
network/network_builder_gt3d.py — depth-aware 3-D gaze-target model (``GT3D``).

Rewritten as a *thin* depth-aware extension on top of the GazeMoE 2-D path
in network/network_builder_update2.py.  The 2-D heatmap branch is
bit-identical to GazeMoE (DINOv2 backbone → MultiScaleFusionLite (or
1×1 fuse_proj) → positional encoding + head-map token → trunk transformer
(vanilla / shared / MoE) → ConvTranspose2d heatmap head).

This guarantees the 2-D AUC/L2 metrics cannot regress from auxiliary
depth losses by architecture alone — any change is solely attributable
to the auxiliary supervision.

Extensions on top of GazeMoE:

1.  Two **anchored depth query tokens** (``depth_token_gaze`` and
    ``depth_token_head``) prepended to the trunk sequence, decoded by a
    *shared* scalar MLP (``depth_scalar_head``).  Only their head-anchored
    log-ratio ``log z_gaze − log z_head`` is supervisable under the
    per-image min-max normalisation used by DepthAnythingV2 pseudo-labels.

2.  A small **dense depth decoder** (one ConvTranspose + one 1×1 + sigmoid)
    over the *per-person* trunk features.  Acts as an encoder regulariser
    when trained against DA2 pseudo-labels via :func:`dense_si_log_loss`
    (defined in train_depth_goo.py).  It is cheap (~ one extra ConvT) and
    produces a 64×64 dense depth map that can be used at inference too
    (e.g. for the 3-D metrics) without ever loading a .npy depth file at
    test time.

3.  A runtime curriculum helper :meth:`set_detach_depth_grads`.  When
    enabled, the inputs to *both* depth heads are detached from the trunk,
    so the depth losses only update the depth-head parameters — used
    during the first few epochs of training to let the heatmap head
    converge before the depth losses are allowed to perturb the trunk.

Parameter naming respects the 4-bucket LR dispatch already used by
train_depth_gazefollow.py and train_gazefollow-depthaware.py:

    fuse_proj.*, fuse_head_token.*, ms_fusion.*       → fuse_lr   (shallow)
    trunk_blocks.*                                    → block_lr  (transformer)
    inout_token.*, inout_head.*                       → inout_lr  (output head)
    depth_token_*, depth_scalar_head.*,
    dense_depth_decoder.*                             → depth_lr  (output head)

``pos_embed`` is a buffer (not a parameter) and is excluded from
``named_parameters()`` — it does not need an LR.

Forward outputs:
    heatmap     : list[B] of [N_i, 64, 64] in [0, 1]
    inout       : list[B] of [N_i]         in [0, 1]   or None
    depth_gaze  : list[B] of [N_i]         in [0, 1]   or None
    depth_head  : list[B] of [N_i]         in [0, 1]   or None
    depth_dense : list[B] of [N_i, 64, 64] in [0, 1]   or None
"""

import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block

import network.utils as utils
from network.backbone import DinoV2Backbone
from network.network_builder_update2 import (
    DinoV2BackboneMultiScale,
    MultiScaleFusionLite,
    MoEBlock,
    SharedTransformer,
    positionalencoding2d,
)


# --------------------------------------------------------------------------
# GT3D — depth-aware gaze-target model (GazeMoE-derived).
# --------------------------------------------------------------------------

class GT3D(nn.Module):
    """Depth-aware 3-D gaze-target estimator built on top of GazeMoE.

    Args:
        backbone           : a ``DinoV2BackboneMultiScale`` instance.
        inout              : if True, attach the in-frame / out-of-frame head.
        dim                : trunk hidden dim.
        mlp_ratio          : transformer MLP expansion ratio.
        num_layers         : number of trunk transformer blocks.
        num_experts        : number of routed experts (only used if moe_type='moe').
        num_shared_experts : number of shared experts in the MoE FFN.
        top_k              : top-K expert selection.
        dropout            : trunk drop-path / FFN dropout.
        moe_type           : "vanilla" | "shared" | "moe".
        is_msf             : if 0 → legacy 1×1 fuse_proj; else MS-fusion with
                             ``num_scales=is_msf`` (matches GazeMoE convention).
        in_size            : input image (H, W).
        out_size           : output heatmap (H, W).
        use_anchored_depth : enable the (z_gaze, z_head) query-token pair.
        use_dense_depth    : enable the dense depth decoder.
    """

    def __init__(
        self,
        backbone,
        *,
        inout=True,
        dim=256,
        mlp_ratio=4,
        num_layers=3,
        num_experts=4,
        num_shared_experts=1,
        top_k=2,
        dropout=0.1,
        moe_type="vanilla",
        is_msf=1,
        in_size=(448, 448),
        out_size=(64, 64),
        use_anchored_depth=True,
        use_dense_depth=True,
    ):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = bool(inout)
        self.is_msf = bool(is_msf)
        self.use_anchored_depth = bool(use_anchored_depth)
        self.use_dense_depth    = bool(use_dense_depth)
        self._detach_depth_grads = False

        # ------------------------------------------------------------------
        # Fuse stage: legacy single 1×1 (``fuse_proj``) OR multi-scale fusion.
        # Naming is chosen so the LR-bucket dispatch ("fuse" / "fusion")
        # routes these params to the ``fuse_lr`` group.
        # ------------------------------------------------------------------
        if not self.is_msf:
            self.fuse_proj = nn.Conv2d(
                backbone.get_dimension(), self.dim, kernel_size=1
            )
        else:
            multi_scale_channels = backbone.get_multi_scale_channels()
            self.ms_fusion = MultiScaleFusionLite(
                in_channels_list=multi_scale_channels,
                out_channels=self.dim,
                target_size=(self.featmap_h, self.featmap_w),
            )

        # ------------------------------------------------------------------
        # Positional encoding (buffer — not learnable, not in the LR dispatch).
        # ------------------------------------------------------------------
        self.register_buffer(
            "pos_embed",
            positionalencoding2d(self.dim, self.featmap_h, self.featmap_w)
            .squeeze(dim=0).squeeze(dim=0),
        )

        # Head-map token: name contains "fuse" → routed to fuse_lr.
        self.fuse_head_token = nn.Embedding(1, self.dim)

        # ------------------------------------------------------------------
        # Trunk transformer blocks (identical to GazeMoE).
        # ------------------------------------------------------------------
        if moe_type == "vanilla":
            self.trunk_blocks = nn.Sequential(*[
                Block(dim=self.dim, num_heads=8,
                      mlp_ratio=self.mlp_ratio, drop_path=dropout)
                for _ in range(num_layers)
            ])
        elif moe_type == "shared":
            vanilla = Block(dim=self.dim, num_heads=8,
                            mlp_ratio=self.mlp_ratio, drop_path=dropout)
            self.trunk_blocks = SharedTransformer(vanilla, num_layers)
        else:  # MoE
            self.trunk_blocks = nn.Sequential(*[
                MoEBlock(
                    dim=self.dim, num_heads=8,
                    mlp_ratio=self.mlp_ratio, drop_path=dropout,
                    num_experts=num_experts,
                    num_shared_experts=num_shared_experts,
                    top_k=top_k,
                )
                for _ in range(num_layers)
            ])

        # ------------------------------------------------------------------
        # 2-D heatmap head — bit-identical to GazeMoE.
        # Named ``heatmap_head`` (no "depth" substring) so it lives in the
        # *default* (``pre_lr``) LR bucket, exactly like GazeMoE.
        # ------------------------------------------------------------------
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # In-frame / out-of-frame auxiliary head (optional).
        # ------------------------------------------------------------------
        if self.inout:
            self.inout_token = nn.Embedding(1, self.dim)
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 1), nn.Sigmoid(),
            )

        # ------------------------------------------------------------------
        # Anchored depth: two parallel query tokens + shared scalar MLP.
        # Only their head-anchored log-ratio is supervised (see
        # anchored_si_log_loss / anchored_log_huber_loss).
        # ------------------------------------------------------------------
        if self.use_anchored_depth:
            self.depth_token_gaze  = nn.Embedding(1, self.dim)
            self.depth_token_head  = nn.Embedding(1, self.dim)
            self.depth_scalar_head = nn.Sequential(
                nn.Linear(self.dim, 128), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(128, 1), nn.Sigmoid(),
            )

        # ------------------------------------------------------------------
        # Dense depth decoder (auxiliary encoder regulariser).  Tiny: one
        # ConvT (32→64) + one 1×1 + sigmoid.  Operates on per-person trunk
        # features; train against DA2 pseudo-labels via dense_si_log_loss.
        # ------------------------------------------------------------------
        if self.use_dense_depth:
            self.dense_depth_decoder = nn.Sequential(
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                nn.GELU(),
                nn.Conv2d(dim, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

    # ----------------------------------------------------------------------
    def set_detach_depth_grads(self, enabled: bool):
        """Curriculum helper.

        When ``True``, the inputs to both depth heads (the anchored depth-
        token outputs and the per-person feature map fed to the dense
        decoder) are detached from the trunk.  Gradients from the depth
        losses then update *only* the depth-head parameters; the trunk,
        fusion, head-map token, and heatmap head remain unaffected.

        Used during the first few epochs of training to let the heatmap
        head converge before depth gradients are allowed to perturb the
        shared trunk.
        """
        self._detach_depth_grads = bool(enabled)

    # ----------------------------------------------------------------------
    def forward(self, input):
        """
        input["images"]: [B, 3, H, W]
        input["bboxes"]: list[B] of list[N_i] of (xmin, ymin, xmax, ymax) in [0, 1]
        """
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]

        # ---- Backbone + fuse stage -----------------------------------
        feats = self.backbone.forward(input["images"])
        if not self.is_msf:
            x = self.fuse_proj(feats)
        else:
            x = self.ms_fusion(feats)
        x = x + self.pos_embed
        x = utils.repeat_tensors(x, num_ppl_per_img)

        # ---- Inject per-person head map ------------------------------
        head_maps = torch.cat(
            self.get_input_head_maps(input["bboxes"]), dim=0
        ).to(x.device)
        head_map_emb = (head_maps.unsqueeze(dim=1)
                        * self.fuse_head_token.weight.unsqueeze(-1).unsqueeze(-1))
        x = x + head_map_emb

        # ---- Flatten + prepend special tokens in fixed order ---------
        # Order:  [inout?, depth_token_gaze?, depth_token_head?, ...spatial]
        x_seq = x.flatten(start_dim=2).permute(0, 2, 1)  # [B', HW, dim]
        B_ = x_seq.shape[0]

        prefix_tokens = []
        if self.inout:
            prefix_tokens.append(
                self.inout_token.weight.unsqueeze(0).repeat(B_, 1, 1))
        if self.use_anchored_depth:
            prefix_tokens.append(
                self.depth_token_gaze.weight.unsqueeze(0).repeat(B_, 1, 1))
            prefix_tokens.append(
                self.depth_token_head.weight.unsqueeze(0).repeat(B_, 1, 1))
        if prefix_tokens:
            x_seq = torch.cat(prefix_tokens + [x_seq], dim=1)

        # ---- Trunk transformer ---------------------------------------
        x_seq = self.trunk_blocks(x_seq)

        # ---- Peel off special tokens in the same fixed order ---------
        offset = 0
        inout_preds = None
        if self.inout:
            inout_tok   = x_seq[:, offset, :]
            inout_preds = self.inout_head(inout_tok).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            offset += 1

        depth_gaze_pred = None
        depth_head_pred = None
        if self.use_anchored_depth:
            zg_tok = x_seq[:, offset,     :]
            zh_tok = x_seq[:, offset + 1, :]
            if self._detach_depth_grads:
                # Stop gradient from L_ratio / L_ray3d into the trunk.
                # depth_scalar_head's own params still receive gradients.
                zg_tok = zg_tok.detach()
                zh_tok = zh_tok.detach()
            z_gaze = self.depth_scalar_head(zg_tok).squeeze(dim=-1)
            z_head = self.depth_scalar_head(zh_tok).squeeze(dim=-1)
            depth_gaze_pred = utils.split_tensors(z_gaze, num_ppl_per_img)
            depth_head_pred = utils.split_tensors(z_head, num_ppl_per_img)
            offset += 2

        # Strip prefix tokens, restore spatial layout for conv decoders.
        x_seq = x_seq[:, offset:, :]
        x_feat = x_seq.reshape(
            B_, self.featmap_h, self.featmap_w, self.dim
        ).permute(0, 3, 1, 2)

        # ---- Heatmap head (never detached — primary task) ------------
        h_out = self.heatmap_head(x_feat).squeeze(dim=1)
        h_out = torchvision.transforms.functional.resize(h_out, self.out_size)
        heatmap_preds = utils.split_tensors(h_out, num_ppl_per_img)

        # ---- Dense depth decoder (auxiliary) -------------------------
        depth_dense_pred = None
        if self.use_dense_depth:
            x_for_dense = (x_feat.detach()
                           if self._detach_depth_grads else x_feat)
            d_out = self.dense_depth_decoder(x_for_dense).squeeze(dim=1)
            d_out = torchvision.transforms.functional.resize(d_out, self.out_size)
            depth_dense_pred = utils.split_tensors(d_out, num_ppl_per_img)

        return {
            "heatmap":     heatmap_preds,
            "inout":       inout_preds,
            "depth_gaze":  depth_gaze_pred,
            "depth_head":  depth_head_pred,
            "depth_dense": depth_dense_pred,
        }

    # ----------------------------------------------------------------------
    def get_input_head_maps(self, bboxes):
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None:
                    img_head_maps.append(
                        torch.zeros(self.featmap_h, self.featmap_w))
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
    # ``get_gazelle_state_dict`` / ``load_gazelle_state_dict`` are kept as
    # compatibility shims so train_gazefollow-depthaware.py's existing
    # checkpointing calls keep working with no extra glue.
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

    # GazeMoE-named aliases for backward compatibility with existing trainers.
    def get_gazelle_state_dict(self, include_backbone=False):
        return self.get_gt3d_state_dict(include_backbone)

    def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
        self.load_gt3d_state_dict(ckpt_state_dict, include_backbone)


# --------------------------------------------------------------------------
# Factory — configuration-driven, mirrors get_gazemoe_model().
# --------------------------------------------------------------------------

def get_gt3d_model(configuration):
    """Factory dispatch driven by ``configuration['model']['name']``.

    Accepted names (all build a GT3D; ``_inout`` variants enable the
    in-frame / out-of-frame auxiliary head):

        gt3d_dinov2_vitb14            gt3d_dinov2_vitl14
        gt3d_dinov2_vitb14_inout      gt3d_dinov2_vitl14_inout

    A backward-compat alias is also accepted so the existing GazeMoE
    configuration name continues to work:

        gazemoe_dinov2_vitl14_inout   →  gt3d_dinov2_vitl14_inout

    Optional architectural flags read from ``configuration['model']``:
        use_anchored_depth (bool, default True)
        use_dense_depth    (bool, default True)
        moe_type           (str,  default "vanilla")   — "vanilla|shared|moe"
        is_msf             (int,  default 1)           — 0 → 1×1 fuse_proj; ≥1 → MSF
        mlp_ratio          (int,  default 4)
        num_experts        (int,  default 4)           — used only when moe_type=='moe'
        num_shared_experts (int,  default 1)
        top_k              (int,  default 2)
        decoder.hidden_size, decoder.depth, decoder.dropout — same as GazeMoE.
    """
    factory = {
        "gt3d_dinov2_vitb14":             _build("dinov2_vitb14", inout=False),
        "gt3d_dinov2_vitl14":             _build("dinov2_vitl14", inout=False),
        "gt3d_dinov2_vitb14_inout":       _build("dinov2_vitb14", inout=True),
        "gt3d_dinov2_vitl14_inout":       _build("dinov2_vitl14", inout=True),
        # Backward-compat alias for the existing config name.
        "gazemoe_dinov2_vitl14_inout":    _build("dinov2_vitl14", inout=True),
    }
    name = configuration["model"]["name"]
    assert name in factory, (
        f"Unknown GT3D model name: {name!r}.  "
        f"Accepted: {sorted(factory.keys())}"
    )
    cfg = configuration["model"]
    decoder_cfg = cfg.get("decoder", {})

    return factory[name](
        d_model            = int(decoder_cfg.get("hidden_size", 256)),
        mlp_ratio          = int(cfg.get("mlp_ratio",   4)),
        num_layers         = int(decoder_cfg.get("depth", 3)),
        num_experts        = int(cfg.get("num_experts", 4)),
        num_shared_experts = int(cfg.get("num_shared_experts", 1)),
        top_k              = int(cfg.get("top_k",       2)),
        dropout            = float(decoder_cfg.get("dropout", 0.1)),
        moe_type           = str (cfg.get("moe_type",   "vanilla")),
        is_msf             = int (cfg.get("is_msf",     1)),
        use_anchored_depth = bool(cfg.get("use_anchored_depth", True)),
        use_dense_depth    = bool(cfg.get("use_dense_depth",    True)),
    )


# --------------------------------------------------------------------------
# Backbone-bound builders.  Each returns a *callable* that consumes the rest
# of the configuration-derived kwargs (matches the GazeMoE factory pattern).
# --------------------------------------------------------------------------

def _build(backbone_name, *, inout):
    def builder(d_model, mlp_ratio, num_layers, num_experts,
                num_shared_experts, top_k, dropout, moe_type,
                is_msf, use_anchored_depth, use_dense_depth):
        # Always use the multi-scale wrapper so num_scales is configurable;
        # is_msf == 0 still falls back to the legacy 1×1 ``fuse_proj`` path
        # inside GT3D.  We pass at least 1 scale to the wrapper since it
        # rejects num_scales < 1.
        backbone = DinoV2BackboneMultiScale(
            backbone_name, num_scales=max(int(is_msf), 1)
        )
        transform = backbone.get_transform((448, 448))
        model = GT3D(
            backbone,
            inout=inout,
            dim=d_model,
            mlp_ratio=mlp_ratio,
            num_layers=num_layers,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            dropout=dropout,
            moe_type=moe_type,
            is_msf=is_msf,
            use_anchored_depth=use_anchored_depth,
            use_dense_depth=use_dense_depth,
        )
        return model, transform
    return builder
