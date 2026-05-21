"""
network/network_builder_sharingan.py
====================================

Builder + thin uniform-interface wrapper for

    Tafasca, Gupta, Odobez.
    "Sharingan: A Transformer Architecture for Multi-Person Gaze Following."
    CVPR 2024.   https://github.com/idiap/sharingan

Wiring
------
The upstream repo is expected at ``../sharingan`` (containing the ``src/``
package).  The released checkpoints (Lightning state dicts) sit at
``./sharingan/checkpoints/{gazefollow,videoattentiontarget,childplay}.pt``
and were trained with the model kwargs of ``src/conf/config_*.yaml``:

    patch_size=16, token_dim=768, image_size=224, heatmap_size=64,
    gaze_feature_dim=512,
    encoder_depth=12, encoder_num_heads=12, encoder_num_global_tokens=0,
    encoder_mlp_ratio=4.0, encoder_use_qkv_bias=True,
    encoder_drop_rate=0.0, encoder_attn_drop_rate=0.0,
    encoder_drop_path_rate=0.0,
    decoder_feature_dim=128, decoder_hooks=[2, 5, 8, 11],
    decoder_hidden_dims=[48, 96, 192, 384], decoder_use_bn=True

Forward contract (upstream)
    Input:   ``{"image": [B, 3, 224, 224],
                 "heads": [B, N, 3, 224, 224],
                 "head_bboxes": [B, N, 4]   (normalised xyxy)}``
    Returns: ``(gaze_vec, gaze_hm, inout)``
      * ``gaze_hm  : [B, N, 64, 64]`` — raw (un-sigmoided) gaze heatmap
      * ``inout    : [B, N, 1]``      — raw in/out logit
      * Target person is **last** along N (per the training step).

For our per-head eval flow we feed N=1 and read index 0.  Released ckpts
are Lightning snapshots with keys prefixed ``model.`` (we strip that).
The upstream pulls in training-only deps (pytorch_lightning, wandb,
transformers, termcolor); we stub those at import time so the wrapper
runs in a lean inference env.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T


# ---------------------------------------------------------------------------
# 1.  Constants — match upstream config_{gf,vat,cp}.yaml
# ---------------------------------------------------------------------------

SHARINGAN_KWARGS: Dict[str, Any] = {
    "patch_size": 16,
    "token_dim": 768,
    "image_size": 224,
    "heatmap_size": 64,
    "gaze_feature_dim": 512,
    "encoder_depth": 12,
    "encoder_num_heads": 12,
    "encoder_num_global_tokens": 0,
    "encoder_mlp_ratio": 4.0,
    "encoder_use_qkv_bias": True,
    "encoder_drop_rate": 0.0,
    "encoder_attn_drop_rate": 0.0,
    "encoder_drop_path_rate": 0.0,
    "decoder_feature_dim": 128,
    "decoder_hooks": [2, 5, 8, 11],
    "decoder_hidden_dims": [48, 96, 192, 384],
    "decoder_use_bn": True,
}

# Image / head normalisation stats (from src/datasets/{gazefollow,vat,cp}.py).
IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD  = [0.28674, 0.27776, 0.27995]


# ---------------------------------------------------------------------------
# 2.  Training-only stubs (Lightning, wandb, transformers, termcolor) so
#     importing the upstream module does NOT require those packages.
# ---------------------------------------------------------------------------

def _install_sharingan_import_stubs() -> None:
    if "pytorch_lightning" not in sys.modules:
        pl = types.ModuleType("pytorch_lightning")
        pl.LightningModule = type("LightningModule", (nn.Module,), {})
        sys.modules["pytorch_lightning"] = pl
    if "wandb" not in sys.modules:
        wb = types.ModuleType("wandb")
        wb.log = lambda *a, **k: None
        wb.define_metric = lambda *a, **k: None
        wb.Image = lambda *a, **k: None
        wb.Histogram = lambda *a, **k: None
        sys.modules["wandb"] = wb
    if "transformers" not in sys.modules:
        tr = types.ModuleType("transformers")
        tr.get_cosine_schedule_with_warmup = lambda *a, **k: None
        sys.modules["transformers"] = tr
    if "termcolor" not in sys.modules:
        tc = types.ModuleType("termcolor")
        tc.colored = lambda s, *a, **k: s
        sys.modules["termcolor"] = tc


# ---------------------------------------------------------------------------
# 3.  Wrapper — exposes the ``forward_eval(images, samples, transform,
#                                          device)`` contract.
# ---------------------------------------------------------------------------

class SharinganWrapper(nn.Module):
    """Adapter around the upstream Sharingan model.

    The upstream model consumes:
        * a 224×224 scene image,
        * one or more 224×224 head crops + their normalised bboxes,
    and returns ``(gaze_vec, gaze_hm, inout)`` with the target person at
    the last index along the N axis.  We pass a single head per sample
    (N=1) and read index 0.
    """

    def __init__(self, upstream_model: nn.Module,
                  expand_k: float = 0.1, image_size: int = 224):
        super().__init__()
        self.model = upstream_model
        self.expand_k = expand_k
        self.image_size = image_size

        # Head pre-processing: resize → tensor → Sharingan normalise.
        self._head_tf = T.Compose([
            T.Resize((image_size, image_size), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ])

    # --- bbox helpers (ported from sharingan/src/utils/common.py) ---------
    @staticmethod
    def _expand_bbox(bbox_pix: torch.Tensor, img_w: int, img_h: int,
                      k: float = 0.1) -> torch.Tensor:
        x1, y1, x2, y2 = bbox_pix
        bw, bh = x2 - x1, y2 - y1
        ex, ey = k * bw, k * bh
        return torch.tensor([
            max(0.0,            float(x1 - ex / 2.0)),
            max(0.0,            float(y1 - ey / 2.0)),
            min(float(img_w),   float(x2 + ex / 2.0)),
            min(float(img_h),   float(y2 + ey / 2.0)),
        ])

    @staticmethod
    def _square_bbox(bbox_pix: torch.Tensor, img_w: int, img_h: int
                      ) -> torch.Tensor:
        x1, y1, x2, y2 = bbox_pix
        bw, bh = x2 - x1, y2 - y1
        side = float(max(bw, bh))
        cx, cy = float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)
        nx1 = cx - side / 2.0
        ny1 = cy - side / 2.0
        nx2 = cx + side / 2.0
        ny2 = cy + side / 2.0
        # Recenter inside the image frame if the square pokes out.
        if nx1 < 0:
            nx2 -= nx1; nx1 = 0.0
        if ny1 < 0:
            ny2 -= ny1; ny1 = 0.0
        if nx2 > img_w:
            nx1 -= (nx2 - img_w); nx2 = float(img_w)
        if ny2 > img_h:
            ny1 -= (ny2 - img_h); ny2 = float(img_h)
        return torch.tensor([max(0.0, nx1), max(0.0, ny1),
                              min(float(img_w), nx2),
                              min(float(img_h), ny2)])

    # --------------------------------------------------------------------
    @torch.no_grad()
    def forward_eval(self, images: List[Image.Image],
                      samples: List[Dict[str, Any]],
                      transform: Callable, device: torch.device
                      ) -> Dict[str, torch.Tensor]:
        # Scene tensor — the ``transform`` from the factory already does
        # 224×224 + Sharingan-normalised mean/std.
        scene_t = torch.stack([transform(im) for im in images]).to(device)

        head_imgs:    List[torch.Tensor] = []
        head_bboxes:  List[torch.Tensor] = []
        for im, s in zip(images, samples):
            w, h = im.size
            b = s["bbox_norm"]
            bp = torch.tensor([b[0] * w, b[1] * h, b[2] * w, b[3] * h],
                              dtype=torch.float32)
            bp = self._expand_bbox(bp, w, h, k=self.expand_k)
            bp = self._square_bbox(bp, w, h)
            x1, y1, x2, y2 = bp.tolist()
            crop = im.crop((int(round(x1)), int(round(y1)),
                            int(round(x2)), int(round(y2)))) \
                   if (x2 > x1 + 1 and y2 > y1 + 1) else im
            head_imgs.append(self._head_tf(crop))
            # Normalise the (squared, expanded) bbox to [0, 1] for the model.
            head_bboxes.append(torch.tensor([
                x1 / w, y1 / h, x2 / w, y2 / h], dtype=torch.float32
            ).clamp_(0.0, 1.0))

        # [B, N=1, 3, 224, 224] and [B, N=1, 4].
        heads_t  = torch.stack(head_imgs).unsqueeze(1).to(device)
        bboxes_t = torch.stack(head_bboxes).unsqueeze(1).to(device)

        batch = {"image": scene_t, "heads": heads_t, "head_bboxes": bboxes_t}
        gaze_vec, gaze_hm, inout = self.model(batch)
        # gaze_hm: [B, N=1, 64, 64];  inout: [B, N=1, 1]  (raw logit).

        hm = gaze_hm[:, -1].detach().float()           # target = last person
        # Heatmap is raw (no sigmoid) but our 2-D metrics are rank-/argmax-
        # based so the unbounded scale is fine.  Keep it as is.
        inout_logit = inout[:, -1, 0].detach().float()
        inout_t = torch.sigmoid(inout_logit).cpu()
        return {"heatmap": hm.cpu(), "inout": inout_t}


# ---------------------------------------------------------------------------
# 4.  Factory
# ---------------------------------------------------------------------------

def _load_sharingan_state_dict(model: nn.Module, ckpt_path: str) -> None:
    """Load the released Lightning ckpt into the raw ``Sharingan`` module.

    The released files are ``{"pytorch-lightning_version": ...,
    "state_dict": {"model.<...>": tensor, ...}}``; strip the ``model.``
    prefix to align with the raw class.
    """
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    cleaned = {}
    for k, v in sd.items():
        nk = k[len("model."):] if k.startswith("model.") else k
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[sharingan] Loaded {ckpt_path}  "
          f"missing={len(missing)}  unexpected={len(unexpected)}")
    if missing and len(missing) <= 8:
        print(f"[sharingan]   missing: {missing}")
    if unexpected and len(unexpected) <= 8:
        print(f"[sharingan]   unexpected: {unexpected}")


def get_sharingan_model(
    upstream_root: str = "../sharingan",
    ckpt_path: Optional[str] = None,
):
    """Return ``(SharinganWrapper, transform)`` wired to the official
    upstream release at ``upstream_root`` (e.g. ``../sharingan``) using
    the released-checkpoint constructor kwargs.

    The ``transform`` returned is the **scene-image** transform; the head
    crop pipeline lives inside ``SharinganWrapper`` (it needs the original
    PIL image to do bbox expansion + squaring before cropping).
    """
    if not os.path.isdir(upstream_root):
        raise ImportError(
            f"Sharingan upstream code not found at '{upstream_root}'.  "
            "Clone https://github.com/idiap/sharingan as a sibling of this "
            "repo (or pass `upstream_root=<path-to-clone>`).")

    _install_sharingan_import_stubs()
    sys.path.insert(0, upstream_root)
    from src.modeling.sharingan import Sharingan  # type: ignore

    model = Sharingan(**SHARINGAN_KWARGS)
    print(f"[sharingan] Built Sharingan with config_gf/vat/cp kwargs  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    if ckpt_path and os.path.isfile(ckpt_path):
        _load_sharingan_state_dict(model, ckpt_path)
    elif ckpt_path:
        print(f"[sharingan] WARNING ckpt_path '{ckpt_path}' not found — "
              f"running with random-init weights.")

    scene_transform = T.Compose([
        T.Resize((SHARINGAN_KWARGS["image_size"],
                  SHARINGAN_KWARGS["image_size"]), antialias=True),
        T.ToTensor(),
        T.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])
    return SharinganWrapper(model,
                             image_size=SHARINGAN_KWARGS["image_size"]), \
           scene_transform
