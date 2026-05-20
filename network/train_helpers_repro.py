"""
network/train_helpers_repro.py
==============================

Shared utilities for the reproduction-model trainers (train_chong.py,
train_dam.py, train_sharingan.py, train_gatector.py).  Deliberately
*not* coupled to ``configuration.yaml`` — every option is passed in
explicitly from the per-model trainer script's CLI.

Provides:
    • ``GazeFollowReproDataset``  — minimal GazeFollow-style loader that
      flattens the preprocessed JSON into one sample per head.
    • ``make_gaussian_heatmap``   — sigma=3 64×64 Gaussian target.
    • ``run_training``            — generic training+eval loop that drives
      any model with a user-supplied ``train_step`` and ``eval_step``
      callback (model-specific).
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eval import gazefollow_auc, gazefollow_l2


# ============================================================================
# Dataset (one head per sample) — common to all reproduction models.
# ============================================================================

class GazeFollowReproDataset(Dataset):
    """Reads ``<data_root>/{train,test}_preprocessed.json`` and yields, per
    sample, a dict that the model-specific ``prepare_batch_fn`` will turn
    into the actual model inputs.
    """

    def __init__(self, data_root: str, split: str,
                 in_frame_only_train: bool = True):
        self.data_root = data_root
        self.split = split
        self.is_train = (split == "train")
        json_path = os.path.join(data_root, f"{split}_preprocessed.json")
        with open(json_path, "r") as f:
            self.frames = json.load(f)
        self.idxs: List[Tuple[int, int]] = []
        for i, frm in enumerate(self.frames):
            for j, head in enumerate(frm.get("heads", [])):
                if (not in_frame_only_train) or (not self.is_train) \
                        or head.get("inout", 1) == 1:
                    self.idxs.append((i, j))

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, idx):
        fi, hi = self.idxs[idx]
        frm = self.frames[fi]
        head = copy.deepcopy(frm["heads"][hi])
        img = Image.open(os.path.join(self.data_root, frm["path"])).convert("RGB")
        sample = {
            "path":       frm["path"],
            "bbox_norm":  list(head["bbox_norm"]),
            "gazex_norm": list(head.get("gazex_norm", [])),
            "gazey_norm": list(head.get("gazey_norm", [])),
            "inout":      int(head.get("inout", 1)),
            "width":      int(img.width),
            "height":     int(img.height),
        }
        return img, sample


def _collate(batch):
    imgs, samples = zip(*batch)
    return list(imgs), list(samples)


# ============================================================================
# Heatmap target — sigma=3 isotropic 2-D Gaussian at the GT gaze pixel.
# ============================================================================

def make_gaussian_heatmap(gx_norm: float, gy_norm: float,
                          H: int = 64, W: int = 64, sigma: float = 3.0
                          ) -> torch.Tensor:
    if gx_norm < 0 or gy_norm < 0:
        return torch.zeros(H, W)
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    cx = gx_norm * W
    cy = gy_norm * H
    g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return g


# ============================================================================
# Generic training loop — drives any of the four reproduction models.
# ============================================================================

@dataclass
class TrainConfig:
    data_root:    str
    out_dir:      str
    batch_size:   int = 32
    epochs:       int = 15
    lr:           float = 1e-4
    weight_decay: float = 0.0
    num_workers:  int = 3
    device:       str = "cuda:0"
    log_every:    int = 50
    inout_weight: float = 1.0          # 0 to disable in/out auxiliary loss
    gt_heatmap_sigma: float = 3.0


def run_training(
    model: nn.Module,
    transform: Callable,
    cfg: TrainConfig,
    train_step: Callable,
    eval_step: Callable,
    optimizer_factory: Optional[Callable[[nn.Module], torch.optim.Optimizer]] = None,
) -> None:
    """Run the standard BCE-heatmap + (optional) BCE-inout training loop.

    Args:
        model              : nn.Module — the reproduction model.
        transform          : RGB transform yielding the scene tensor.
        cfg                : :class:`TrainConfig` with hyperparameters.
        train_step         : Callable
            ``train_step(model, images, samples, transform, device,
                          gt_heatmap, gt_inout, cfg) -> dict``
            must return ``{'loss': scalar tensor, 'l_hm': ..., 'l_io': ...}``.
        eval_step          : Callable with signature
            ``eval_step(model, images, samples, transform, device)
              -> {'heatmap': [B,64,64] CPU tensor,
                  'inout':   [B] CPU tensor or None}``
        optimizer_factory  : Optional callable; defaults to Adam over all
            trainable parameters with cfg.lr / cfg.weight_decay.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)
    model = model.to(device)

    train_ds = GazeFollowReproDataset(cfg.data_root, "train")
    test_ds  = GazeFollowReproDataset(cfg.data_root, "test",
                                       in_frame_only_train=False)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                           collate_fn=_collate, num_workers=cfg.num_workers,
                           pin_memory=False)
    test_dl  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                           collate_fn=_collate, num_workers=cfg.num_workers,
                           pin_memory=False)
    print(f"train={len(train_ds)}  test={len(test_ds)}  device={device}")

    if optimizer_factory is None:
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = optimizer_factory(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-7)

    best_avg_l2 = float("inf")
    best_epoch = -1
    for epoch in range(cfg.epochs):
        # ---- TRAIN ----
        model.train()
        t0 = time.time()
        running = {"loss": 0.0, "l_hm": 0.0, "l_io": 0.0, "n": 0}
        for it, (images, samples) in enumerate(
                tqdm(train_dl, desc=f"train ep{epoch + 1}/{cfg.epochs}")):
            gt_hm = torch.stack([
                make_gaussian_heatmap(s["gazex_norm"][0] if s["gazex_norm"] else -1.0,
                                       s["gazey_norm"][0] if s["gazey_norm"] else -1.0,
                                       sigma=cfg.gt_heatmap_sigma)
                for s in samples
            ]).to(device)
            gt_io = torch.tensor([s["inout"] for s in samples],
                                  dtype=torch.float32, device=device)

            optimizer.zero_grad()
            stats = train_step(model, images, samples, transform, device,
                                gt_hm, gt_io, cfg)
            stats["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()

            for k in ("loss", "l_hm", "l_io"):
                if k in stats:
                    running[k] += float(stats[k].item() if hasattr(stats[k], "item") else stats[k])
            running["n"] += 1

        scheduler.step()
        elapsed = time.time() - t0
        avg = {k: running[k] / max(running["n"], 1) for k in ("loss", "l_hm", "l_io")}
        print(f"[ep {epoch:02d}] train loss={avg['loss']:.4f}  "
              f"hm={avg['l_hm']:.4f}  io={avg['l_io']:.4f}  "
              f"({elapsed:.0f}s)")

        # ---- SAVE ----
        ck_path = os.path.join(cfg.out_dir, f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), ck_path)

        # ---- EVAL ----
        model.eval()
        aucs, avg_l2s = [], []
        inout_preds, inout_gts = [], []
        with torch.no_grad():
            for images, samples in tqdm(test_dl, desc=f"eval ep{epoch + 1}",
                                         total=len(test_dl)):
                out = eval_step(model, images, samples, transform, device)
                hm = out["heatmap"]      # [B, 64, 64]
                io = out.get("inout")
                for i, s in enumerate(samples):
                    if s["inout"] == 1 and s["gazex_norm"] and s["gazey_norm"]:
                        auc = gazefollow_auc(hm[i], s["gazex_norm"], s["gazey_norm"],
                                              s["height"], s["width"])
                        avg_l2, _ = gazefollow_l2(hm[i], s["gazex_norm"], s["gazey_norm"])
                        aucs.append(auc); avg_l2s.append(avg_l2)
                    if io is not None and s.get("inout") is not None:
                        inout_preds.append(float(io[i].item()))
                        inout_gts.append(int(s["inout"]))
        AUC   = float(np.mean(aucs))    if aucs    else float("nan")
        L2    = float(np.mean(avg_l2s)) if avg_l2s else float("nan")
        AP    = (float(average_precision_score(inout_gts, inout_preds))
                  if inout_preds and len(set(inout_gts)) > 1 else None)
        ap_s  = f"  AP={AP:.4f}" if AP is not None else ""
        print(f"[ep {epoch:02d}] eval AUC={AUC:.4f}  AvgL2={L2:.4f}{ap_s}")
        if not math.isnan(L2) and L2 < best_avg_l2:
            best_avg_l2 = L2; best_epoch = epoch
    print(f"Done.  best AvgL2={best_avg_l2:.4f} @ epoch {best_epoch}")
