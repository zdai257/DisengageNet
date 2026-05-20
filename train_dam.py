#!/usr/bin/env python
"""
train_dam.py
============

Minimal CLI-driven trainer for the Dual-Attention model (Fang et al.).
Same structure as ``train_chong.py``; only the model construction and
``train_step`` differ.

Example::

    python train_dam.py \\
        --data_root ./gazefollow_extended \\
        --out_dir results/repro/dam \\
        --batch_size 32 --epochs 15 --lr 1e-4
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List

import torch
import torch.nn.functional as F
from PIL import Image

from network.network_builder_dam import get_dam_model
from network.train_helpers_repro import TrainConfig, run_training


def _build_inputs(images: List[Image.Image], samples: List[Dict[str, Any]],
                   transform: Callable, device: torch.device):
    scene_t = torch.stack([transform(im) for im in images]).to(device)
    head_imgs, head_poses, centres = [], [], []
    for im, s in zip(images, samples):
        w, h = im.size
        b = s["bbox_norm"]
        x1, y1 = int(round(b[0] * w)), int(round(b[1] * h))
        x2, y2 = int(round(b[2] * w)), int(round(b[3] * h))
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        crop = im.crop((x1, y1, x2, y2)) if (x2 > x1 and y2 > y1) else im
        head_imgs.append(transform(crop))
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
    return (scene_t,
            torch.stack(head_imgs).to(device),
            torch.stack(head_poses).to(device),
            torch.stack(centres).to(device))


def train_step(model, images, samples, transform, device,
                gt_heatmap, gt_inout, cfg):
    scene_t, head_t, pos_t, centres = _build_inputs(images, samples, transform, device)
    out = model(scene_t, head_t, pos_t, centres)
    l_hm = F.binary_cross_entropy_with_logits(
        out["heatmap_logit"].squeeze(1), gt_heatmap)
    l_io = F.binary_cross_entropy_with_logits(
        out["inout_logit"].squeeze(1), gt_inout)
    loss = l_hm + cfg.inout_weight * l_io
    return {"loss": loss, "l_hm": l_hm, "l_io": l_io}


def eval_step(model, images, samples, transform, device):
    return model.forward_eval(images=images, samples=samples,
                                transform=transform, device=device)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data_root",    type=str, required=True)
    ap.add_argument("--out_dir",      type=str, default="results/repro/dam")
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--epochs",       type=int,   default=15)
    ap.add_argument("--lr",           type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_workers",  type=int,   default=3)
    ap.add_argument("--device",       type=str,   default="cuda:0")
    ap.add_argument("--inout_weight", type=float, default=1.0)
    args = ap.parse_args()

    model, transform = get_dam_model()
    cfg = TrainConfig(
        data_root=args.data_root, out_dir=args.out_dir,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, num_workers=args.num_workers,
        device=args.device, inout_weight=args.inout_weight,
    )
    run_training(model, transform, cfg, train_step, eval_step)


if __name__ == "__main__":
    main()
