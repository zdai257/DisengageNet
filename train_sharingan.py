#!/usr/bin/env python
"""
train_sharingan.py
==================

Thin reproduction trainer for Sharingan (Tafasca et al., CVPR 2024).

Sharingan's original training recipe relies on a fairly involved
PyTorch Lightning + Hydra setup (multi-stage backbone unfreezing,
MultiMAE-pretrained ViT scene encoder, contrastive auxiliary loss).
This script does NOT reimplement that recipe — it provides a thin
single-loss BCE-heatmap finetune that you can use to *adapt* a
pre-existing Sharingan checkpoint to your data layout for fair
comparison.

For the headline Sharingan numbers, vendor the upstream repo
(see ``network/network_builder_sharingan.py``) and use its own
``main.py``.  This script is for quick reproduction-finetuning only.

Example::

    python train_sharingan.py \\
        --data_root ./gazefollow_extended \\
        --upstream_src_root ../sharingan/src \\
        --ckpt_path ../sharingan/checkpoints/gazefollow.pt \\
        --out_dir results/repro/sharingan \\
        --batch_size 16 --epochs 5 --lr 5e-5
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List

import torch
import torch.nn.functional as F
from PIL import Image

from network.network_builder_sharingan import get_sharingan_model
from network.train_helpers_repro import TrainConfig, run_training


def train_step(model, images, samples, transform, device,
                gt_heatmap, gt_inout, cfg):
    # We re-use the wrapper's ``forward_eval`` to produce a heatmap, then
    # compute BCE against the GT heatmap.  This is a single-loss adaptation
    # path; for the full upstream loss, vendor the upstream litmodel.py.
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
    ap.add_argument("--data_root",          type=str, required=True)
    ap.add_argument("--upstream_src_root",  type=str, default="../sharingan/src")
    ap.add_argument("--ckpt_path",          type=str, default=None,
                    help="Optional Sharingan checkpoint to warm-start from.")
    ap.add_argument("--out_dir",      type=str, default="results/repro/sharingan")
    ap.add_argument("--batch_size",   type=int,   default=16)
    ap.add_argument("--epochs",       type=int,   default=5)
    ap.add_argument("--lr",           type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_workers",  type=int,   default=3)
    ap.add_argument("--device",       type=str,   default="cuda:0")
    ap.add_argument("--inout_weight", type=float, default=1.0)
    args = ap.parse_args()

    model, transform = get_sharingan_model(
        upstream_src_root=args.upstream_src_root,
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
