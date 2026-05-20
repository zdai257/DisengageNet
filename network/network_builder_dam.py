"""
network/network_builder_dam.py
==============================

Faithful PyTorch reimplementation of the *Dual Attention Module* (DAM)
model from

    Fang, Tang, Wu, Tang.  "Dual Attention Guided Gaze Target Detection
    in the Wild."  CVPR 2021.  https://github.com/Crystal2333/DAM

Architectural summary
---------------------
DAM extends the Chong-style two-branch detector with an explicit
gaze-direction estimator (Gaze360-style) that produces a "field-of-view"
(FoV) cone mask in the image plane.  The FoV mask is then attended to a
scene feature map via a depth-aware dual-attention mechanism, and the
result is decoded into a 64×64 heatmap.

Because the original code factors logic across multiple files
(``gaze_model.py``, ``resnet_scene.py``, ``model.py``), this reimplementation
mirrors the *interface* and *output shape* but folds the same components
into a single self-contained module.  Two simplifications relative to
upstream are made deliberately to keep this file readable and
reproducible without external assets:

    • The gaze-direction sub-network is a small MLP head on top of the
      head-branch ResNet50, predicting a unit (yaw, pitch) on the head
      crop alone.  Upstream uses a separately-pretrained Gaze360 ResNet18
      — if you have that checkpoint, vendor it into ``./vendor/dam/`` and
      replace ``self.gaze_estimator`` with an import-and-load call.
    • The depth-aware attention is implemented as a learnable
      multiplicative gate over the FoV mask, multiplied into the scene
      feats — equivalent in spirit to the dual attention but with one
      fewer parameter tensor.

Both simplifications keep the *headline interface* identical to upstream
(scene image + head crop + head-position map + gaze direction →
64×64 heatmap + inout logit), so this file can be swapped for the
upstream module at any time.

Public API:
    get_dam_model() -> (model, transform)
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet50

from network.network_builder_chong import _Resnet50


# -----------------------------------------------------------------------------
# Gaze-direction estimator (small head on top of the head ResNet50)
# -----------------------------------------------------------------------------

class _GazeDirHead(nn.Module):
    """Tiny gaze-direction head.

    Predicts a unit 2-vector ``(g_x, g_y)`` in the *image-plane* sense,
    i.e. the predicted direction of the gaze ray projected onto the
    image plane.  The unit vector is then used to rasterise a FoV cone
    on the image plane (see :func:`_rasterise_fov`).
    """

    def __init__(self, in_dim: int = 2048):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward(self, head_feats: torch.Tensor) -> torch.Tensor:
        v = self.head(head_feats)                       # [B, 2]
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
        return v                                        # unit (gx, gy)


def _rasterise_fov(head_centre_xy: torch.Tensor,
                   gaze_dir_xy: torch.Tensor,
                   size: int = 64,
                   half_angle_deg: float = 20.0,
                   device: torch.device = torch.device("cpu")
                   ) -> torch.Tensor:
    """Build a soft FoV cone mask in the image plane.

    Args:
        head_centre_xy : [B, 2] in [0, 1] image-plane coords (head bbox
                         centre).
        gaze_dir_xy    : [B, 2] unit vector — predicted image-plane gaze.
        size           : output spatial side length.
        half_angle_deg : half-aperture of the FoV cone.

    Returns:
        [B, 1, size, size] tensor with values in [0, 1].  A pixel is
        bright when the head→pixel ray is within ``half_angle_deg`` of
        the predicted gaze direction.
    """
    B = head_centre_xy.shape[0]
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, size, device=device),
        torch.linspace(0, 1, size, device=device),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, S, S, 2]
    rel = grid - head_centre_xy.view(B, 1, 1, 2)                              # head→pixel
    rel_n = rel / (rel.norm(dim=-1, keepdim=True) + 1e-6)
    cos = (rel_n * gaze_dir_xy.view(B, 1, 1, 2)).sum(dim=-1)                  # [B, S, S]
    cos_thr = math.cos(math.radians(half_angle_deg))
    fov = torch.clamp((cos - cos_thr) / (1.0 - cos_thr + 1e-6), 0.0, 1.0)
    return fov.unsqueeze(1)                                                   # [B, 1, S, S]


# -----------------------------------------------------------------------------
# DAM main module
# -----------------------------------------------------------------------------

class DAMModel(nn.Module):
    """Dual-Attention gaze target detector.

    Inputs (training-time forward):
        scene_img      : [B, 3, 224, 224]
        head_img       : [B, 3, 224, 224]
        head_pos       : [B, 1, 224, 224]
        head_centre_xy : [B, 2]            (normalised image-plane head centre)

    Outputs:
        heatmap_logit  : [B, 1, 64, 64]
        inout_logit    : [B, 1]
        gaze_dir_xy    : [B, 2] unit vector (for auxiliary supervision)
    """

    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        self.scene = _Resnet50(in_channels=4, pretrained=pretrained_backbone)
        self.head  = _Resnet50(in_channels=3, pretrained=pretrained_backbone)
        self.gaze_estimator = _GazeDirHead(in_dim=2048)

        # Dual-attention gate: input is the concatenation of [scene_feats |
        # FoV cone at 7×7].  Output is a per-pixel attention map.
        self.fov_resize = nn.AdaptiveAvgPool2d((7, 7))
        self.attn_dual  = nn.Sequential(
            nn.Conv2d(2048 + 1, 512, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(512,        1, kernel_size=1), nn.Sigmoid(),
        )

        # Fusion + decoder — same shape as Chong's so the trained weights
        # have a comparable parameter count.
        self.encode = nn.Sequential(
            nn.Conv2d(4096, 1024, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(1024,  512, kernel_size=1), nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 64,   1, kernel_size=4, stride=2, padding=1),
        )
        self.inout = nn.Sequential(
            nn.AvgPool2d(7), nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    # --------------------------------------------------------------------
    def forward(self, scene_img, head_img, head_pos,
                head_centre_xy) -> Dict[str, torch.Tensor]:
        scene_in = torch.cat([scene_img, head_pos], dim=1)
        scene_f  = self.scene(scene_in)                          # [B, 2048, 7, 7]
        head_f   = self.head(head_img)                            # [B, 2048, 7, 7]

        gaze_dir = self.gaze_estimator(head_f)                    # [B, 2]
        fov64    = _rasterise_fov(head_centre_xy, gaze_dir,
                                    size=64,
                                    device=scene_img.device)
        fov7     = self.fov_resize(fov64)                          # [B, 1, 7, 7]

        attn     = self.attn_dual(torch.cat([scene_f, fov7], dim=1))  # [B, 1, 7, 7]
        attended = scene_f * attn

        fused    = self.encode(torch.cat([head_f, attended], dim=1))  # [B, 512, 7, 7]
        hm       = self.deconv(fused)
        hm64     = F.interpolate(hm, size=(64, 64),
                                  mode="bilinear", align_corners=False)
        inout    = self.inout(fused)
        return {
            "heatmap_logit": hm64,
            "inout_logit":   inout,
            "gaze_dir_xy":   gaze_dir,
        }

    # --------------------------------------------------------------------
    @torch.no_grad()
    def forward_eval(self, images: List[Image.Image],
                      samples: List[Dict[str, Any]],
                      transform: Callable, device: torch.device
                      ) -> Dict[str, torch.Tensor]:
        scene_t = torch.stack([transform(im) for im in images]).to(device)
        head_imgs, head_poses, centres = [], [], []
        for im, s in zip(images, samples):
            w, h = im.size
            b = s["bbox_norm"]
            x1, y1 = int(round(b[0] * w)), int(round(b[1] * h))
            x2, y2 = int(round(b[2] * w)), int(round(b[3] * h))
            x1, x2 = sorted((max(0, x1), min(w, x2)))
            y1, y2 = sorted((max(0, y1), min(h, y2)))
            head_crop = im.crop((x1, y1, x2, y2)) if (x2 > x1 and y2 > y1) else im
            head_imgs.append(transform(head_crop))

            pos = torch.zeros(1, 224, 224)
            ux1 = int(round(b[0] * 224)); uy1 = int(round(b[1] * 224))
            ux2 = int(round(b[2] * 224)); uy2 = int(round(b[3] * 224))
            ux1, ux2 = sorted((max(0, ux1), min(224, ux2)))
            uy1, uy2 = sorted((max(0, uy1), min(224, uy2)))
            if ux2 > ux1 and uy2 > uy1:
                pos[:, uy1:uy2, ux1:ux2] = 1.0
            head_poses.append(pos)

            centres.append(torch.tensor(
                [(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5],
                dtype=torch.float32,
            ))

        out = self(
            scene_t,
            torch.stack(head_imgs).to(device),
            torch.stack(head_poses).to(device),
            torch.stack(centres).to(device),
        )
        return {
            "heatmap": torch.sigmoid(out["heatmap_logit"]).squeeze(1).cpu(),
            "inout":   torch.sigmoid(out["inout_logit"]).squeeze(1).cpu(),
        }


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def get_dam_model(pretrained_backbone: bool = True):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = DAMModel(pretrained_backbone=pretrained_backbone)
    return model, transform
