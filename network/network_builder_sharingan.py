"""
network/network_builder_sharingan.py
====================================

Builder + thin uniform-interface wrapper for

    Tafasca, Gupta, Odobez.
    "Sharingan: A Transformer Architecture for Multi-Person Gaze Following."
    CVPR 2024.   https://github.com/idiap/sharingan

The Sharingan architecture is a multi-person Transformer that wraps a
MultiMAE-pretrained ViT scene encoder with a per-head token-injection
pipeline and a Gaze360-style auxiliary gaze-direction encoder.  Faithfully
reimplementing the full model from scratch (encoder + gaze tokens +
contrastive aux loss + dataset-specific configurations) is non-trivial
and out of scope for an in-house quick-reproduction file, so this builder
follows a **vendoring pattern**:

    1. Clone the upstream repo as a sibling checkout::

           cd ..
           git clone https://github.com/idiap/sharingan.git
           cd DisengageNet
           ln -s ../sharingan/src vendor/sharingan_src    # optional convenience

    2. Place the pretrained weights as instructed by the upstream README
       under ``vendor/sharingan_weights/`` (or pass the paths via the
       factory).

    3. The :func:`get_sharingan_model` factory below imports the upstream
       ``LightningModule`` (or the raw ``nn.Module`` underneath it) and
       wraps it with a ``SharinganWrapper`` that exposes the
       ``forward_eval`` interface required by ``test_models.py``.

If the upstream code is not present, the factory raises a clear
ImportError with the instructions above — preferred over silently
producing wrong numbers from a partial reimplementation.

Public API:
    get_sharingan_model(
        upstream_src_root: str = "../sharingan/src",
        ckpt_path: str | None  = None,
        config_name: str       = "config_gf",
    ) -> (model, transform)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T


class SharinganWrapper(nn.Module):
    """Adapter that exposes the upstream Sharingan model under the same
    ``forward_eval`` contract used by ``test_models.py``.

    The Sharingan forward signature differs from the in-house and
    Chong-style models: it consumes multi-person head crops + head
    bounding boxes plus the scene image, and returns one heatmap per head.
    For our single-head-per-sample eval flow we feed a singleton head
    list per image; the wrapper unpacks the first head's prediction.
    """

    def __init__(self, upstream_model: nn.Module,
                 head_input_size: int = 224,
                 scene_input_size: int = 384):
        super().__init__()
        self.model = upstream_model
        self.head_input_size  = head_input_size
        self.scene_input_size = scene_input_size

    # --------------------------------------------------------------------
    @torch.no_grad()
    def forward_eval(self, images: List[Image.Image],
                      samples: List[Dict[str, Any]],
                      transform: Callable, device: torch.device
                      ) -> Dict[str, torch.Tensor]:
        # Build scene tensor.
        scene_t = torch.stack([transform(im) for im in images]).to(device)

        # Per-image head crops resized to head_input_size.
        head_tf = T.Compose([
            T.Resize((self.head_input_size, self.head_input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        head_imgs:  List[torch.Tensor] = []
        head_boxes: List[torch.Tensor] = []
        for im, s in zip(images, samples):
            w, h = im.size
            b = s["bbox_norm"]
            x1, y1 = int(round(b[0] * w)), int(round(b[1] * h))
            x2, y2 = int(round(b[2] * w)), int(round(b[3] * h))
            x1, x2 = sorted((max(0, x1), min(w, x2)))
            y1, y2 = sorted((max(0, y1), min(h, y2)))
            crop = im.crop((x1, y1, x2, y2)) if (x2 > x1 and y2 > y1) else im
            head_imgs.append(head_tf(crop))
            head_boxes.append(torch.tensor(b, dtype=torch.float32))
        # Singleton head per image: [B, N=1, ...]
        heads_t = torch.stack(head_imgs).unsqueeze(1).to(device)         # [B, 1, 3, Hh, Wh]
        boxes_t = torch.stack(head_boxes).unsqueeze(1).to(device)         # [B, 1, 4]

        # The upstream model's exact forward signature depends on the
        # vendored release; we expect a dict / tuple return.  Try the
        # most common shapes in turn.
        out = None
        for kwargs in (
            {"scene": scene_t, "heads": heads_t, "head_boxes": boxes_t},
            {"image": scene_t, "head_imgs": heads_t, "head_bboxes": boxes_t},
            {"x": scene_t,     "heads_x": heads_t, "boxes": boxes_t},
        ):
            try:
                out = self.model(**kwargs)
                break
            except TypeError:
                continue
        if out is None:
            raise RuntimeError(
                "Could not call the upstream Sharingan model with any of the "
                "expected keyword arrangements.  Please adapt the SharinganWrapper "
                "to match your vendored Sharingan release.")

        if isinstance(out, dict):
            hm   = out.get("heatmap", out.get("heatmaps"))
            inout = out.get("inout",  out.get("watching", None))
        else:                                  # (heatmaps, inout) tuple
            hm, inout = out

        if hm is None:
            raise RuntimeError("Sharingan output did not include a heatmap.")
        if hm.dim() == 5:                      # [B, N, 1, H, W]
            hm = hm[:, 0, 0]
        elif hm.dim() == 4:                    # [B, N, H, W]
            hm = hm[:, 0]
        if hm.shape[-1] != 64 or hm.shape[-2] != 64:
            hm = F.interpolate(hm.unsqueeze(1), size=(64, 64),
                                mode="bilinear", align_corners=False).squeeze(1)
        hm = torch.sigmoid(hm) if (hm.min() < 0 or hm.max() > 1) else hm

        if inout is None:
            inout_t = torch.full((hm.shape[0],), float("nan"))
        else:
            if inout.dim() >= 2:
                inout = inout[:, 0]
            inout_t = torch.sigmoid(inout).view(-1).cpu().float()
        return {"heatmap": hm.cpu(), "inout": inout_t}


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def get_sharingan_model(
    upstream_src_root: str = "../sharingan/src",
    ckpt_path: str | None = None,
    config_name: str = "config_gf",
):
    """Return ``(SharinganWrapper, transform)``.

    The upstream Sharingan repo is expected to be present at
    ``upstream_src_root``.  We import its model module, then optionally
    load a checkpoint at ``ckpt_path``.
    """
    if not os.path.isdir(upstream_src_root):
        raise ImportError(
            "Sharingan upstream code not found at "
            f"'{upstream_src_root}'.  Clone https://github.com/idiap/sharingan "
            "as a sibling of this repo (or pass `upstream_src_root=...`) "
            "and download the pretrained weights as the README instructs."
        )
    sys.path.insert(0, upstream_src_root)

    try:
        # Upstream layout: ``src/model.py`` defines ``Sharingan(nn.Module)``,
        # ``src/litmodel.py`` wraps it in a LightningModule.  We prefer the
        # raw nn.Module to keep this builder framework-free.
        from model import Sharingan  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Could not import the Sharingan model class from "
            f"'{upstream_src_root}/model.py'.  The upstream module names "
            f"may have changed; see {upstream_src_root}/README.md."
        ) from e

    model = Sharingan()
    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        # Strip a Lightning prefix if present.
        if any(k.startswith("model.") for k in sd):
            sd = {k.replace("model.", "", 1): v for k, v in sd.items()
                   if k.startswith("model.")}
        model.load_state_dict(sd, strict=False)

    # Sharingan expects ImageNet-normalised 384×384 scene images by default.
    transform = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return SharinganWrapper(model), transform
