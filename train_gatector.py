#!/usr/bin/env python
"""
train_gatector.py
=================

Thin reproduction trainer for GaTector (Wang et al., CVPR 2022).

Like ``train_sharingan.py``, this script does NOT reproduce the full
upstream training recipe (YOLOv4 detection + gaze + energy aggregation
+ defocus layer + GOO-specific anchors).  It is a simple BCE-heatmap
finetune of a *vendored* GaTector backbone so you can compare gaze
metrics fairly under the same evaluation pipeline.

For headline GaTector numbers (including the object-detection /
GOP / mDAP scores), use the upstream training recipe.

Example::

    python train_gatector.py \\
        --data_root ../GOOSynthV3 \\
        --upstream_root ../GaTector \\
        --ckpt_path ../GaTector/weights/goosynth_pretrained.pt \\
        --out_dir results/repro/gatector \\
        --batch_size 32 --epochs 10 --lr 1e-4
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List

import torch
import torch.nn.functional as F
from PIL import Image

from network.network_builder_gatector import get_gatector_model
from network.train_helpers_repro import TrainConfig, run_training


def train_step(model, images, samples, transform, device,
                gt_heatmap, gt_inout, cfg):
    out = model.forward_eval(images=images, samples=samples,
                              transform=transform, device=device)
    hm_pred = out["heatmap"].to(device)
    l_hm = F.binary_cross_entropy(hm_pred.clamp(1e-6, 1 - 1e-6), gt_heatmap)
    inout = out.get("inout")
    if inout is not None and torch.isfinite(inout).all():
        l_io = F.binary_cross_entropy(inout.clamp(1e-6, 1 - 1e-6).to(device), gt_inout)
        loss = l_hm + cfg.inout_weight * l_io
    else:
        l_io = torch.zeros((), device=device)
        loss = l_hm
    return {"loss": loss, "l_hm": l_hm, "l_io": l_io}


def eval_step(model, images, samples, transform, device):
    return model.forward_eval(images=images, samples=samples,
                                transform=transform, device=device)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data_root",     type=str, required=True)
    ap.add_argument("--upstream_root", type=str, default="../GaTector")
    ap.add_argument("--ckpt_path",     type=str, default=None)
    ap.add_argument("--out_dir",      type=str, default="results/repro/gatector")
    ap.add_argument("--batch_size",   type=int,   default=32)
    ap.add_argument("--epochs",       type=int,   default=10)
    ap.add_argument("--lr",           type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_workers",  type=int,   default=3)
    ap.add_argument("--device",       type=str,   default="cuda:0")
    ap.add_argument("--inout_weight", type=float, default=1.0)
    args = ap.parse_args()

    model, transform = get_gatector_model(
        upstream_root=args.upstream_root,
        ckpt_path=args.ckpt_path,
    )
    cfg = TrainConfig(
        data_root=args.data_root, out_dir=args.out_dir,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, num_workers=args.num_workers,
        device=args.device, inout_weight=args.inout_weight,
    )
    run_training(model, transform, cfg, train_step, eval_step)


if __name__ == "__main__":
    main()
