"""
network/network_builder_gatector.py
===================================

Builder + thin uniform-interface wrapper for

    Wang, Hu, Li, Chen, Zhang.
    "GaTector: A Unified Framework for Gaze Object Prediction."
    CVPR 2022.
    https://github.com/CodeMonsterPHD/GaTector-A-Unified-Framework-for-Gaze-Object-Prediction

GaTector is a YOLOv4-based unified gaze + object framework with a
specific-general-specific (SGS) feature extractor and a defocus layer
for object detection.  The upstream code combines a (vendored)
YOLOv4 detector, gaze-heatmap head, and energy-aggregation loss across
multiple modules under ``lib/`` and is non-trivial to reproduce from
scratch without the YOLOv4 anchors / pretrained weights that the
upstream README ships separately.

This file therefore follows a **vendoring pattern** (analogous to
``network_builder_sharingan.py``): the upstream code is expected to be
available as a sibling checkout, and we wrap it in a uniform interface
for ``test_models.py``.

Vendoring quick-start
---------------------
.. code-block:: bash

    cd ..
    git clone https://github.com/CodeMonsterPHD/GaTector-A-Unified-Framework-for-Gaze-Object-Prediction.git GaTector
    # follow the upstream README to download pretrained weights into
    # GaTector/data/anchors and any model checkpoints.

The factory ``get_gatector_model`` imports
``GaTector/lib/gatector_model.py`` (or whichever module the upstream
release exposes) and loads the checkpoint passed as ``ckpt_path``.

Public API:
    get_gatector_model(
        upstream_root: str = "../GaTector",
        ckpt_path: str | None = None,
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


class GaTectorWrapper(nn.Module):
    """Adapter that exposes the upstream GaTector model under the same
    ``forward_eval`` contract as the other reproduction wrappers.

    We do *not* report GaTector's object-detection metrics here — only the
    gaze-target metrics (AUC, AvgL2, L2-3D, Angle-3D, DOA, optionally AP),
    which are the headline numbers compared across the gaze literature.
    """

    def __init__(self, upstream_model: nn.Module, input_size: int = 224):
        super().__init__()
        self.model = upstream_model
        self.input_size = input_size

    # --------------------------------------------------------------------
    @torch.no_grad()
    def forward_eval(self, images: List[Image.Image],
                      samples: List[Dict[str, Any]],
                      transform: Callable, device: torch.device
                      ) -> Dict[str, torch.Tensor]:
        scene_t = torch.stack([transform(im) for im in images]).to(device)

        head_imgs:  List[torch.Tensor] = []
        head_poses: List[torch.Tensor] = []
        for im, s in zip(images, samples):
            w, h = im.size
            b = s["bbox_norm"]
            x1, y1 = int(round(b[0] * w)), int(round(b[1] * h))
            x2, y2 = int(round(b[2] * w)), int(round(b[3] * h))
            x1, x2 = sorted((max(0, x1), min(w, x2)))
            y1, y2 = sorted((max(0, y1), min(h, y2)))
            crop = im.crop((x1, y1, x2, y2)) if (x2 > x1 and y2 > y1) else im
            head_imgs.append(transform(crop))

            pos = torch.zeros(1, self.input_size, self.input_size)
            ux1 = int(round(b[0] * self.input_size))
            uy1 = int(round(b[1] * self.input_size))
            ux2 = int(round(b[2] * self.input_size))
            uy2 = int(round(b[3] * self.input_size))
            ux1, ux2 = sorted((max(0, ux1), min(self.input_size, ux2)))
            uy1, uy2 = sorted((max(0, uy1), min(self.input_size, uy2)))
            if ux2 > ux1 and uy2 > uy1:
                pos[:, uy1:uy2, ux1:ux2] = 1.0
            head_poses.append(pos)

        heads_t = torch.stack(head_imgs).to(device)
        poses_t = torch.stack(head_poses).to(device)

        # The upstream GaTector model returns multiple outputs (object
        # detection bboxes, gaze heatmap, energy aggregation, ...).  We
        # try a few common return shapes and extract the gaze heatmap.
        out = None
        for kwargs in (
            {"scene": scene_t, "face": heads_t, "head": poses_t},
            {"x": scene_t, "head_x": heads_t, "head_pos": poses_t},
            {"image": scene_t, "head_image": heads_t, "head_mask": poses_t},
        ):
            try:
                out = self.model(**kwargs)
                break
            except TypeError:
                continue
        if out is None:
            raise RuntimeError(
                "Could not call the upstream GaTector model with any expected "
                "kwargs; please adapt GaTectorWrapper to match your vendored "
                "release.")

        if isinstance(out, dict):
            hm    = out.get("heatmap", out.get("gaze_heatmap"))
            inout = out.get("inout", out.get("watch_outside", None))
        elif isinstance(out, (tuple, list)):
            hm    = out[0]
            inout = out[1] if len(out) > 1 else None
        else:
            hm, inout = out, None

        if hm is None:
            raise RuntimeError("GaTector output did not include a gaze heatmap.")
        if hm.dim() == 4:                       # [B, 1, H, W]
            hm = hm.squeeze(1)
        if hm.shape[-1] != 64 or hm.shape[-2] != 64:
            hm = F.interpolate(hm.unsqueeze(1), size=(64, 64),
                                mode="bilinear", align_corners=False).squeeze(1)
        hm = torch.sigmoid(hm) if (hm.min() < 0 or hm.max() > 1) else hm

        if inout is None:
            inout_t = torch.full((hm.shape[0],), float("nan"))
        else:
            inout_t = torch.sigmoid(inout).view(-1).cpu().float()
        return {"heatmap": hm.cpu(), "inout": inout_t}


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def get_gatector_model(
    upstream_root: str = "../GaTector",
    ckpt_path: str | None = None,
):
    """Return ``(GaTectorWrapper, transform)``.

    Vendor the upstream repo at ``upstream_root`` and pass any pretrained
    checkpoint via ``ckpt_path``.
    """
    if not os.path.isdir(upstream_root):
        raise ImportError(
            "GaTector upstream code not found at "
            f"'{upstream_root}'.  Clone the repo from "
            "https://github.com/CodeMonsterPHD/GaTector-A-Unified-Framework-for-Gaze-Object-Prediction "
            "as a sibling of this repo (or pass `upstream_root=...`)."
        )
    sys.path.insert(0, upstream_root)
    sys.path.insert(0, os.path.join(upstream_root, "lib"))

    try:
        # Upstream module name varies by release; try the canonical paths.
        from gatector_model import GaTector  # type: ignore
    except ImportError:
        try:
            from lib.gatector_model import GaTector  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Could not import the GaTector model class from "
                f"'{upstream_root}'.  Check the upstream README for the "
                "correct module path; you may need to expose a thin "
                "`gatector_model.py` that re-exports the model class."
            ) from e

    model = GaTector()
    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd, strict=False)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return GaTectorWrapper(model), transform
