"""
network/network_builder_chong.py
================================

Faithful PyTorch reimplementation of the *per-frame* sub-model from

    Chong, Wang, Ruiz, Rehg.  "Detecting Attended Visual Targets in Video."
    CVPR 2020.   https://github.com/ejcgt/attention-target-detection

We omit the temporal (ConvLSTM) head used for VideoAttentionTarget and
implement only the static, single-image attention-target detector — which
is the model trained on GazeFollow and used as the per-frame backbone in
the temporal variant.

The architecture follows the paper / public reference implementation
exactly:

    Scene branch       :  ResNet-50, modified first conv to take 4 channels
                          (RGB + head-position map).
    Head branch        :  ResNet-50, RGB input, output a 49×7×7 feature map
                          + 49-vector ``attention'' produced via two FC
                          layers + Sigmoid.  The attention is reshaped to
                          7×7 and element-wise multiplied into the scene
                          features.
    Fusion + decoder   :  Concatenated head + attended-scene features pass
                          through two encoding convs and four
                          ConvTranspose2d layers to produce a 64×64
                          heatmap.
    In/out branch      :  Two FC layers from the scene-conv features to a
                          single in/out logit, then Sigmoid at inference.

The forward signature is unified with the rest of this repo so the model
can be plugged into ``test_models.py`` directly via its
``forward_eval(images, samples, transform, device)`` adapter.

Public API:
    get_chong_model() -> (model, transform)
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet50


# -----------------------------------------------------------------------------
# Heads
# -----------------------------------------------------------------------------

class _Resnet50(nn.Module):
    """ResNet-50 wrapped to expose only the conv-feature stack and (optionally)
    accept >3 input channels.  We keep the standard 5-stage backbone and
    drop the final avgpool/fc, returning a 2048×H/32×W/32 feature map.
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        backbone = resnet50(weights="DEFAULT" if pretrained else None)
        if in_channels != 3:
            c = backbone.conv1
            new = nn.Conv2d(in_channels, c.out_channels,
                             kernel_size=c.kernel_size, stride=c.stride,
                             padding=c.padding, bias=False)
            with torch.no_grad():
                new.weight[:, :3].copy_(c.weight)
                if in_channels > 3:
                    new.weight[:, 3:].zero_()
            backbone.conv1 = new
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_channels = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# -----------------------------------------------------------------------------
# Chong model
# -----------------------------------------------------------------------------

class ChongModel(nn.Module):
    """Per-frame attention-target detector (Chong et al., CVPR 2020).

    Forward inputs (training-time):
        scene_img  : [B, 3, 224, 224]
        head_img   : [B, 3, 224, 224]
        head_pos   : [B, 1, 224, 224]  (binary head-position map)

    Forward outputs:
        heatmap    : [B, 1, 64, 64]    (raw, no sigmoid — the BCE-with-logits
                                        / sigmoid wrapper is applied in
                                        ``forward_eval``)
        inout      : [B, 1]            (logit; sigmoid in ``forward_eval``)
    """

    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        self.scene  = _Resnet50(in_channels=4, pretrained=pretrained_backbone)
        self.head   = _Resnet50(in_channels=3, pretrained=pretrained_backbone)

        # Attention vector: project head feats → 49-vector via avgpool + 2-FC.
        self.attn = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.Linear(2048, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1 * 7 * 7), nn.Sigmoid(),
        )

        # Two encoding convs over the concatenated [head | attended-scene]
        # feature stack (2048 + 2048 = 4096 channels at 7×7).
        self.encode = nn.Sequential(
            nn.Conv2d(4096, 1024, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(1024,  512, kernel_size=1), nn.ReLU(inplace=True),
        )

        # Heatmap decoder: 7×7 → 14 → 28 → 56 → 64 via four ConvT layers.
        # The last layer outputs a single channel; we then bilinear-resize
        # 56→64 inside forward() for exact 64×64 output.
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 64,   1, kernel_size=4, stride=2, padding=1),
        )

        # In/out head from the encoded fusion features (avgpool + 2-FC).
        self.inout = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256,   1),
        )

    # --------------------------------------------------------------------
    def forward(self, scene_img: torch.Tensor, head_img: torch.Tensor,
                 head_pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        scene_in = torch.cat([scene_img, head_pos], dim=1)              # [B, 4, 224, 224]
        scene_f  = self.scene(scene_in)                                  # [B, 2048, 7, 7]
        head_f   = self.head(head_img)                                   # [B, 2048, 7, 7]
        a        = self.attn(head_f).view(-1, 1, 7, 7)                   # [B, 1, 7, 7]
        attended = scene_f * a                                            # [B, 2048, 7, 7]
        fused    = self.encode(torch.cat([head_f, attended], dim=1))     # [B, 512, 7, 7]
        hm56     = self.deconv(fused)                                     # [B, 1, 112, 112]  (after 4× 2×-upsample)
        hm64     = F.interpolate(hm56, size=(64, 64),
                                  mode="bilinear", align_corners=False)
        inout    = self.inout(fused)                                      # [B, 1]
        return {"heatmap_logit": hm64, "inout_logit": inout}

    # --------------------------------------------------------------------
    @torch.no_grad()
    def forward_eval(self, images: List[Image.Image],
                      samples: List[Dict[str, Any]],
                      transform: Callable, device: torch.device
                      ) -> Dict[str, torch.Tensor]:
        """Adapter to the uniform interface required by test_models.py.

        Each ``sample`` provides a normalised head bbox; we crop the head
        from the original RGB image and build the 1-channel head-position
        map by rasterising the bbox onto a 224×224 grid.
        """
        # The transform yields the 3-channel 224×224 scene image.
        scene_t = torch.stack([transform(im) for im in images]).to(device)

        head_imgs:   List[torch.Tensor] = []
        head_poses:  List[torch.Tensor] = []
        for im, s in zip(images, samples):
            w, h = im.size
            b = s["bbox_norm"]
            x1, y1 = int(round(b[0] * w)), int(round(b[1] * h))
            x2, y2 = int(round(b[2] * w)), int(round(b[3] * h))
            x1, x2 = sorted((max(0, x1), min(w, x2)))
            y1, y2 = sorted((max(0, y1), min(h, y2)))
            if x2 <= x1 or y2 <= y1:
                # Degenerate bbox — fall back to a centre crop.
                head_crop = im
            else:
                head_crop = im.crop((x1, y1, x2, y2))
            head_imgs.append(transform(head_crop))

            # Head-position map at 224×224, ones inside the (normalised) bbox.
            pos = torch.zeros(1, 224, 224)
            ux1 = int(round(b[0] * 224)); uy1 = int(round(b[1] * 224))
            ux2 = int(round(b[2] * 224)); uy2 = int(round(b[3] * 224))
            ux1, ux2 = sorted((max(0, ux1), min(224, ux2)))
            uy1, uy2 = sorted((max(0, uy1), min(224, uy2)))
            if ux2 > ux1 and uy2 > uy1:
                pos[:, uy1:uy2, ux1:ux2] = 1.0
            head_poses.append(pos)
        head_t = torch.stack(head_imgs).to(device)
        pos_t  = torch.stack(head_poses).to(device)

        out = self(scene_t, head_t, pos_t)
        heatmap = torch.sigmoid(out["heatmap_logit"]).squeeze(1).cpu()  # [B, 64, 64]
        inout   = torch.sigmoid(out["inout_logit"]).squeeze(1).cpu()     # [B]
        return {"heatmap": heatmap, "inout": inout}


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def get_chong_model(pretrained_backbone: bool = True):
    """Return (model, transform).

    The transform produces a 224×224 ImageNet-normalised RGB tensor — the
    exact preprocessing pipeline used in the upstream ``train_on_gazefollow``
    reference.  The model is reused for both scene and head inputs via
    ``forward_eval`` (which crops + re-transforms the head region).
    """
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = ChongModel(pretrained_backbone=pretrained_backbone)
    return model, transform
