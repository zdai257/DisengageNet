"""
train_gazefollow-depthaware.py — depth-aware GazeFollow pretraining.

Replaces the original simple ``get_depthaware_heatmap``-only pipeline with
a multi-task GT3D pipeline that exploits DepthAnythingV2 pseudo-labels at
*training time only* (model inference does not need any .npy depth file).

Loss
----
    L = L_hm                          (BCE on depth-aware Gaussian heatmap)
      + w_d_dense   · L_d_dense       (SI-log on dense DA2 pseudo-label)
      + w_ratio     · L_ratio         (anchored log-ratio of z_gaze, z_head)
      + w_ray3d     · L_ray3d         (3-D head→target ray-tracing loss)
      + w_consist   · L_consist       (optional; OFF unless w_consist > 0)
      + w_inout     · L_inout         (optional inout head)

The 2-D heatmap target itself is still produced by
``utils.get_depthaware_heatmap`` (per-user request — its double-inversion
bug is now fixed in network/utils.py).  All depth-related auxiliary
supervision is *additional* to that target.

Curriculum (0-based epoch indices)
----------------------------------
    epoch  0 .. warmup-1                : L_hm only
                                          detach_depth_grads = True
    epoch  warmup .. ray_start-1        : L_hm + 0.5·L_d_dense + 0.5·L_ratio
                                          detach_depth_grads = True for the
                                          first ``detach_depth_grads_to_trunk_for_epochs``
                                          epochs total
    epoch  ray_start .. N-(anneal+1)    : all losses, weights linearly ramped
                                          from *_start → *_end
                                          detach_depth_grads = False
    epoch  N-anneal .. N-1              : all losses pinned at *_end weights
                                          (annealing tail)

Evaluation
----------
    Primary 2-D metrics  : AUC, AvgL2, MinL2 (existing gazefollow_* helpers).
    Secondary 3-D metrics: RatioMAE, RatioRMSE, δ1, L2-3D, Angle-3D
                           (computed from the model's depth scalar outputs
                            against DA2 GT sampled at the mean annotator
                            location and the head bbox centre).
"""

import copy
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR
from tqdm import tqdm
import wandb
import yaml

from eval import gazefollow_auc, gazefollow_l2
from network.network_builder_gt3d import get_gt3d_model
import network.utils as utils
from network.utils import get_depthaware_heatmap

# Centralised loss / metric helpers — train_depth_goo.py is the depth-aware
# loss hub.  train_depth_gazefollow.py uses the same pattern.
from train_depth_goo import (
    LOSS_SCALAR,
    FocalLoss,
    # Anchored / SI-log losses on (z_gaze, z_head).
    anchored_log_huber_loss,
    anchored_si_log_loss,
    # 3-D metrics.
    anchored_delta1,
    anchored_l2_3d,
    gaze3d_angle_relative,
    ratio_mae,
    ratio_rmse,
    # New helpers added in this iteration.
    anchored_consistency_loss,
    dense_si_log_loss,
    gaze_ray_3d_loss,
)


# --------------------------------------------------------------------------
# Joint augmentations (RGB + depth must move together).
# --------------------------------------------------------------------------

def _joint_random_crop(image, depth, bbox, gazex, gazey, inout):
    """utils.random_crop, but cropping the depth map identically."""
    width, height = image.size
    bxmin, bymin, bxmax, bymax = bbox
    cxmin = min(bxmin, min(gazex)) if inout else bxmin
    cymin = min(bymin, min(gazey)) if inout else bymin
    cxmax = max(bxmax, max(gazex)) if inout else bxmax
    cymax = max(bymax, max(gazey)) if inout else bymax
    try:
        xmin = random.randint(0, int(cxmin))
        ymin = random.randint(0, int(cymin))
        xmax = random.randint(int(cxmax), width)
        ymax = random.randint(int(cymax), height)
    except ValueError:
        return image, depth, bbox, gazex, gazey
    image = TF.crop(image, ymin, xmin, ymax - ymin, xmax - xmin)
    depth = TF.crop(depth, ymin, xmin, ymax - ymin, xmax - xmin)
    bbox  = [bxmin - xmin, bymin - ymin, bxmax - xmin, bymax - ymin]
    gazex = [x - xmin for x in gazex]
    gazey = [y - ymin for y in gazey]
    return image, depth, bbox, gazex, gazey


def _joint_horiz_flip(image, depth, bbox, gazex, gazey, inout):
    width, _ = image.size
    image = TF.hflip(image)
    depth = TF.hflip(depth)
    xmin, ymin, xmax, ymax = bbox
    bbox = [width - xmax, ymin, width - xmin, ymax]
    if inout:
        gazex = [width - x for x in gazex]
    return image, depth, bbox, gazex, gazey


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class GazeDepthDataset(torch.utils.data.Dataset):
    """GazeFollow with depth-aware heatmap target + dense / anchored depth GT.

    Train items
    -----------
        image_t        : [3, H, W]
        bbox_norm      : [4]   in [0, 1]
        gazex_norm     : list  in [0, 1]
        gazey_norm     : list  in [0, 1]
        inout          : 0-D tensor
        height, width  : int (original image size, post-aug)
        heatmap_da     : [64, 64] Gaussian × depth weighting
                         (utils.get_depthaware_heatmap, after the
                          double-inversion fix in network/utils.py)
        dense_depth_t  : [64, 64] per-image min-max-normalised DA2 map in [0,1]
        gt_z_gaze      : float  in [0, 1]   (DA2 value at gaze pixel)
        gt_z_head      : float  in [0, 1]   (DA2 value at head bbox centre)

    Test items
    ----------
        Same as train minus the heatmap (eval uses GT annotations directly).
        Depth is still loaded at test time *only to compute 3-D metrics*;
        the model itself never consumes a depth tensor at inference.
    """

    def __init__(self, dataset_name, path, split, transform,
                 in_frame_only=True, sample_rate=1, aug_groups=None,
                 depth_dir='depth'):
        self.dataset_name = dataset_name
        self.path = path
        self.depth_dir = depth_dir
        self.split = split
        self.is_train = (split == "train")
        self.aug = self.is_train
        self.transform = transform
        self.in_frame_only = in_frame_only
        self.sample_rate = sample_rate
        self.aug_groups = aug_groups if aug_groups is not None else []

        if dataset_name == "gazefollow":
            with open(os.path.join(self.path,
                                   f"{split}_preprocessed.json"), "r") as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Invalid dataset for this trainer: {dataset_name!r}")

        self.data_idxs = []
        for i, frame in enumerate(self.data):
            for j, head in enumerate(frame['heads']):
                if not self.in_frame_only or head['inout'] == 1:
                    self.data_idxs.append((i, j))

    # ----------------------------------------------------------------------
    def _depth_path(self, image_rel_path):
        """Map ``train/0001/0001.jpg`` → ``depth/train/0001/0001.npy``."""
        rel = image_rel_path.replace("images" + os.sep,
                                     self.depth_dir + os.sep, 1)
        if rel == image_rel_path:                     # no "images/" segment
            rel = os.path.join(self.depth_dir, image_rel_path)
        rel = os.path.splitext(rel)[0] + ".npy"
        return os.path.join(self.path, rel)

    def _load_depth_npy(self, image_rel_path):
        return np.load(self._depth_path(image_rel_path)).astype(np.float32)

    def _load_depth_pil(self, image_rel_path):
        return Image.fromarray(self._load_depth_npy(image_rel_path), mode="F")

    @staticmethod
    def _sample_anchored_from_64(d_norm_64, gazex_norm, gazey_norm, bbox_norm):
        """Sample (gt_z_gaze, gt_z_head) from a [64,64] min-max-normalised array."""
        u_g = int(np.clip(round(gazex_norm * 63), 0, 63))
        v_g = int(np.clip(round(gazey_norm * 63), 0, 63))
        gt_z_gaze = float(d_norm_64[v_g, u_g])

        hcx = (bbox_norm[0] + bbox_norm[2]) * 0.5
        hcy = (bbox_norm[1] + bbox_norm[3]) * 0.5
        u_h = int(np.clip(round(hcx * 63), 0, 63))
        v_h = int(np.clip(round(hcy * 63), 0, 63))
        gt_z_head = float(d_norm_64[v_h, u_h])
        return gt_z_gaze, gt_z_head

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        img_idx, head_idx = self.data_idxs[idx]
        img_data = self.data[img_idx]
        head_data = copy.deepcopy(img_data['heads'][head_idx])
        bbox_norm  = head_data['bbox_norm']
        gazex_norm = list(head_data['gazex_norm'])
        gazey_norm = list(head_data['gazey_norm'])
        inout = head_data['inout']

        img_path = os.path.join(self.path, img_data['path'])
        img = Image.open(img_path).convert("RGB")
        depth_pil = self._load_depth_pil(img_data['path'])   # for joint aug
        width, height = img.size

        if self.aug:
            bbox  = list(head_data['bbox'])
            gazex = list(head_data['gazex'])
            gazey = list(head_data['gazey'])

            if 'crop' in self.aug_groups and np.random.sample() <= 0.5:
                img, depth_pil, bbox, gazex, gazey = _joint_random_crop(
                    img, depth_pil, bbox, gazex, gazey, inout)
            if 'crop' in self.aug_groups and np.random.sample() <= 0.5:
                img, depth_pil, bbox, gazex, gazey = _joint_horiz_flip(
                    img, depth_pil, bbox, gazex, gazey, inout)
            if 'crop' in self.aug_groups and np.random.sample() <= 0.5:
                bbox = utils.random_bbox_jitter(img, bbox)

            width, height = img.size
            bbox_norm  = [bbox[0] / width,  bbox[1] / height,
                          bbox[2] / width,  bbox[3] / height]
            gazex_norm = [x / float(width)  for x in gazex]
            gazey_norm = [y / float(height) for y in gazey]

            if 'photometric' in self.aug_groups and np.random.sample() <= 0.5:
                photometric = T.Compose([
                    T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2,
                                                 saturation=0.2, hue=0.1)], p=0.5),
                    T.RandomGrayscale(p=0.2),
                    T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                    T.RandomAutocontrast(p=0.1),
                ])
                img = photometric(img)

        img_t = self.transform(img)

        # ---- Depth → 64×64 per-image min-max-normalised in [0, 1] -----
        d_arr_64 = np.asarray(depth_pil.resize((64, 64), Image.BILINEAR),
                              dtype=np.float32)
        d_min, d_max = float(d_arr_64.min()), float(d_arr_64.max())
        d_norm_64 = (d_arr_64 - d_min) / ((d_max - d_min) + 1e-8)

        # Anchored GT scalars: train uses the single annotation; test uses
        # the mean annotator location (one stable reference per image).
        if self.is_train:
            gx_for_depth = gazex_norm[0]
            gy_for_depth = gazey_norm[0]
        else:
            gx_for_depth = float(np.mean(gazex_norm))
            gy_for_depth = float(np.mean(gazey_norm))
        gt_z_gaze, gt_z_head = self._sample_anchored_from_64(
            d_norm_64, gx_for_depth, gy_for_depth, bbox_norm)

        if self.is_train:
            # Keep the depth-aware heatmap target as the user wants.
            # get_depthaware_heatmap accepts arbitrary-resolution depth and
            # resizes internally; pass the post-augmentation depth so the
            # target is consistent with the augmented gaze coordinates.
            depth_full = np.asarray(depth_pil, dtype=np.float32)
            heatmap = get_depthaware_heatmap(
                depth_full, gazex_norm[0], gazey_norm[0], 64, 64)
            dense_depth_t = torch.from_numpy(d_norm_64).float()  # [64, 64]
            return (img_t, bbox_norm, gazex_norm, gazey_norm,
                    torch.tensor(inout), height, width,
                    heatmap, dense_depth_t,
                    float(gt_z_gaze), float(gt_z_head))
        return (img_t, bbox_norm, gazex_norm, gazey_norm,
                torch.tensor(inout), height, width,
                float(gt_z_gaze), float(gt_z_head))

    def __len__(self):
        return len(self.data_idxs)


def collate_fn(batch):
    transposed = list(zip(*batch))
    return tuple(
        torch.stack(items) if isinstance(items[0], torch.Tensor) else list(items)
        for items in transposed
    )


# --------------------------------------------------------------------------
# Curriculum helper — compute per-epoch loss weights and the
# detach_depth_grads toggle.
# --------------------------------------------------------------------------

def _curriculum_weights(epoch, *, total_epochs, warmup_epochs, ray_start_epoch,
                        annealing_last_epochs, detach_until_epoch,
                        w_d_dense_start, w_d_dense_end,
                        w_ratio_start,   w_ratio_end,
                        w_ray3d_start,   w_ray3d_end):
    """Return (w_d_dense, w_ratio, w_ray3d, detach_grads, phase_name)."""
    in_warmup    = epoch < warmup_epochs
    in_depth_wu  = (warmup_epochs <= epoch < ray_start_epoch)
    annealing_start = total_epochs - annealing_last_epochs
    in_annealing = epoch >= annealing_start
    in_full      = (ray_start_epoch <= epoch < annealing_start)

    if in_warmup:
        wd, wr, wr3 = 0.0, 0.0, 0.0
        phase = "warmup"
    elif in_depth_wu:
        wd  = 0.5 * w_d_dense_start
        wr  = 0.5 * w_ratio_start
        wr3 = 0.0
        phase = "depth_warmup"
    elif in_full:
        denom = max(annealing_start - ray_start_epoch - 1, 1)
        t = (epoch - ray_start_epoch) / float(denom)
        t = max(0.0, min(1.0, t))
        wd  = w_d_dense_start + (w_d_dense_end - w_d_dense_start) * t
        wr  = w_ratio_start   + (w_ratio_end   - w_ratio_start)   * t
        wr3 = w_ray3d_start   + (w_ray3d_end   - w_ray3d_start)   * t
        phase = "full"
    else:  # in_annealing
        wd, wr, wr3 = w_d_dense_end, w_ratio_end, w_ray3d_end
        phase = "annealing"

    detach = epoch < detach_until_epoch
    return wd, wr, wr3, detach, phase


# --------------------------------------------------------------------------
# Scheduler helper (warmup + cosine, identical to train_depth_gazefollow.py)
# --------------------------------------------------------------------------

def _make_warmup_cosine_lambda(total_ep, warmup_ep, eta, ref_lr):
    def lr_lambda(epoch):
        if epoch < warmup_ep:
            return max((epoch + 1) / float(warmup_ep), 1e-8)
        t = epoch - warmup_ep
        T = max(total_ep - warmup_ep - 1, 1)
        cos_f = 0.5 * (1.0 + math.cos(math.pi * min(t / T, 1.0)))
        em = eta / max(ref_lr, 1e-12)
        return em + (1.0 - em) * cos_f
    return lr_lambda


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    with open('configuration.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    cfg_m, cfg_t, cfg_d = config['model'], config['train'], config['data']

    wandb.init(
        project=cfg_m['name'],
        name="pretrain_gf_depth",
        config=config,
    )

    # ---- Checkpoint dir -------------------------------------------------
    checkpoint_dir = "_".join([
        cfg_m['name'],
        cfg_m.get('moe_type', 'vanilla'),
        str(cfg_m.get('is_msf', 1)),
        cfg_t['pre_optimizer'],
        "bs" + str(cfg_t['pre_batch_size']),
        cfg_m['pbce_loss'],
        str(cfg_t['pre_lr']),
        str(cfg_t['pre_fuse_lr']),
        str(cfg_t['pre_block_lr']),
        "depth",
    ])
    exp_dir = os.path.join(config['logging']['pre_dir'], checkpoint_dir)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Pretrained checkpoint saved at: {exp_dir}")

    # ---- Model ----------------------------------------------------------
    model, transform = get_gt3d_model(config)
    print(f"Model: {cfg_m['name']}  "
          f"(anchored_depth={getattr(model, 'use_anchored_depth', True)}, "
          f"dense_depth={getattr(model, 'use_dense_depth', True)})")

    # Freeze backbone (same convention as the legacy GazeFollow pretrain).
    for param in model.backbone.parameters():
        param.requires_grad = False
    print(f"Learnable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model.to(device)

    # ---- Datasets / loaders --------------------------------------------
    depth_dir = cfg_d.get('depth_dir', 'depth')
    train_dataset = GazeDepthDataset(
        'gazefollow', cfg_d['pre_train_path'], 'train', transform,
        aug_groups=cfg_d.get('augmentations', []), depth_dir=depth_dir)
    eval_dataset = GazeDepthDataset(
        'gazefollow', cfg_d['pre_test_path'], 'test', transform,
        depth_dir=depth_dir)

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg_t['pre_batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware'].get('pin_memory', False),
    )
    eval_dl = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=cfg_t['pre_batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware'].get('pin_memory', False),
    )

    # ---- Per-bucket LR dispatch ----------------------------------------
    pre_lr       = float(cfg_t['pre_lr'])
    pre_fuse_lr  = float(cfg_t.get('pre_fuse_lr',  pre_lr))
    pre_block_lr = float(cfg_t.get('pre_block_lr', pre_lr))
    pre_inout_lr = float(cfg_t.get('inout_lr',     pre_lr))
    pre_depth_lr = float(cfg_t.get('depth_lr',     pre_lr))

    param_dicts = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "inout" in n:
            lr = pre_inout_lr
        elif "depth" in n:
            lr = pre_depth_lr
        elif "block" in n:
            lr = pre_block_lr
        elif ("fuse" in n) or ("fusion" in n) or ("linear" in n):
            lr = pre_fuse_lr
        else:
            lr = pre_lr
        param_dicts.append({'params': p, 'lr': lr})

    opt_name = cfg_t['pre_optimizer']
    if opt_name == 'Adam':
        optimizer = Adam(param_dicts)
    elif opt_name == 'AdamW':
        optimizer = AdamW(param_dicts)
    else:
        raise TypeError(f"Optimizer not supported: {opt_name}")

    # ---- Scheduler ------------------------------------------------------
    pre_epochs = int(cfg_t['pre_epochs'])
    sched_cfg  = cfg_t.get('pre_lr_scheduler', cfg_t['lr_scheduler'])
    sched_type = sched_cfg.get('type', 'cosine')
    if sched_type == 'warmup_cosine':
        warmup_epochs_sched = int(sched_cfg.get('warmup_epochs', 2))
        eta_min = float(sched_cfg.get('min_lr', 1e-7))
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=_make_warmup_cosine_lambda(
                pre_epochs, warmup_epochs_sched, eta_min, pre_lr),
        )
        print(f"LR schedule: warmup_cosine  warmup_epochs={warmup_epochs_sched}  "
              f"total_epochs={pre_epochs}  min_lr={eta_min}")
    elif sched_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get('step_size', pre_epochs)),
            eta_min=float(sched_cfg.get('min_lr', 1e-7)),
        )
    else:
        scheduler = StepLR(
            optimizer,
            step_size=int(sched_cfg.get('step_size', pre_epochs)),
            gamma=float(sched_cfg.get('gamma', 0.1)),
        )

    # ---- Loss configuration --------------------------------------------
    # Heatmap loss: BCE / MSE selectable.
    if cfg_m['pbce_loss'] == "mse":
        SCALAR = 36
        heatmap_loss_fn = nn.MSELoss(reduction=cfg_m.get('reduction', 'mean'))
    elif cfg_m['pbce_loss'] == "bce":
        SCALAR = 1
        heatmap_loss_fn = nn.BCELoss()
    else:
        raise TypeError(
            f"Heatmap loss not supported: {cfg_m['pbce_loss']!r}  "
            f"(expected 'mse' or 'bce')")

    inout_loss_fn = (FocalLoss() if int(cfg_m.get('is_focal_loss', 0)) == 1
                     else nn.BCELoss())

    depth_loss_name = str(cfg_m.get('depth_loss', 'si_log')).lower()
    si_lambda       = float(cfg_m.get('lambda_si',     0.5))

    # ---- Curriculum / weight schedule ----------------------------------
    warmup_epochs          = int(cfg_m.get('curriculum_warmup_epochs',         1))
    ray_start_epoch        = int(cfg_m.get('ray_start_epoch',                  5))
    annealing_last_epochs  = int(cfg_m.get('curriculum_annealing_last_epochs', 2))
    detach_until_epoch     = int(cfg_m.get('detach_depth_grads_to_trunk_for_epochs', 2))

    w_d_dense_start = float(cfg_m.get('w_d_dense_start', 0.30))
    w_d_dense_end   = float(cfg_m.get('w_d_dense_end',   0.10))
    w_ratio_start   = float(cfg_m.get('w_ratio_start',   0.50))
    w_ratio_end     = float(cfg_m.get('w_ratio_end',     0.20))
    w_ray3d_start   = float(cfg_m.get('w_ray3d_start',   0.05))
    w_ray3d_end     = float(cfg_m.get('w_ray3d_end',     0.20))
    w_consist       = float(cfg_m.get('w_consist',       0.0))   # default OFF
    w_inout         = float(cfg_m.get('bce_weight',      0.0))   # legacy alias

    ray_gamma       = float(cfg_m.get('ray_gamma',       0.5))
    ray_w_cos       = float(cfg_m.get('ray_w_cos',       1.0))
    ray_w_l1        = float(cfg_m.get('ray_w_l1',        0.5))
    dense_alpha     = float(cfg_m.get('dense_alpha_gaussian', 0.5))

    grad_clip       = float(cfg_t.get('gradient_clipping', 0.0))

    print(f"Curriculum: warmup={warmup_epochs}  "
          f"ray_start={ray_start_epoch}  "
          f"annealing_tail={annealing_last_epochs}  "
          f"detach_until={detach_until_epoch}  "
          f"depth_loss={depth_loss_name}  "
          f"w_consist={w_consist} ({'ON' if w_consist > 0 else 'OFF'})")

    # ---- Training loop --------------------------------------------------
    best_avg_l2 = float("inf")
    best_epoch  = None

    for epoch in range(pre_epochs):
        # ---- Schedule: weights + detach flag --------------------------
        w_d_dense, w_ratio, w_ray3d, detach_flag, phase = _curriculum_weights(
            epoch,
            total_epochs=pre_epochs,
            warmup_epochs=warmup_epochs,
            ray_start_epoch=ray_start_epoch,
            annealing_last_epochs=annealing_last_epochs,
            detach_until_epoch=detach_until_epoch,
            w_d_dense_start=w_d_dense_start, w_d_dense_end=w_d_dense_end,
            w_ratio_start  =w_ratio_start,   w_ratio_end  =w_ratio_end,
            w_ray3d_start  =w_ray3d_start,   w_ray3d_end  =w_ray3d_end,
        )
        model.set_detach_depth_grads(detach_flag)
        print(f"[Epoch {epoch}] phase={phase}  "
              f"w_d_dense={w_d_dense:.3f}  w_ratio={w_ratio:.3f}  "
              f"w_ray3d={w_ray3d:.3f}  detach={detach_flag}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # ---- TRAIN EPOCH ---------------------------------------------
        model.train()
        sums = {'total': 0.0, 'hm': 0.0, 'd_dense': 0.0,
                'ratio': 0.0, 'ray3d': 0.0, 'consist': 0.0, 'io': 0.0}
        n_iters = len(train_dl)

        for cur_iter, batch in tqdm(enumerate(train_dl), total=n_iters,
                                     desc=f"Epoch {epoch + 1}/{pre_epochs}"):
            (imgs, bboxes, gazex, gazey, inout, heights, widths,
             heatmaps, dense_depth_gt, gt_zg, gt_zh) = batch

            optimizer.zero_grad()
            preds = model({
                "images": imgs.to(device),
                "bboxes": [[bbox] for bbox in bboxes],
            })

            pred_hm    = torch.stack(preds['heatmap']).squeeze(dim=1)        # [B, 64, 64]
            pred_dense = torch.stack(preds['depth_dense']).squeeze(dim=1)    # [B, 64, 64]
            pred_zg    = torch.cat(preds['depth_gaze'], 0)                   # [B]
            pred_zh    = torch.cat(preds['depth_head'], 0)                   # [B]

            gt_hm_d    = heatmaps.to(device)
            gt_dense_d = dense_depth_gt.to(device)
            gt_zg_d    = torch.as_tensor(gt_zg, dtype=torch.float32, device=device)
            gt_zh_d    = torch.as_tensor(gt_zh, dtype=torch.float32, device=device)

            # ---- 1. Heatmap loss (primary) ---------------------------
            l_hm = SCALAR * heatmap_loss_fn(pred_hm, gt_hm_d)

            # ---- 2. Dense depth distillation -------------------------
            if w_d_dense > 0:
                l_d_dense = dense_si_log_loss(
                    pred_dense, gt_dense_d,
                    gaussian_mask=gt_hm_d,
                    alpha=dense_alpha,
                    lambda_si=si_lambda,
                )
            else:
                l_d_dense = torch.zeros((), device=device)

            # ---- 3. Anchored log-ratio -------------------------------
            if w_ratio > 0:
                if depth_loss_name == 'si_log':
                    l_ratio = anchored_si_log_loss(
                        pred_zg, pred_zh, gt_zg_d, gt_zh_d,
                        lambda_si=si_lambda)
                else:
                    l_ratio = anchored_log_huber_loss(
                        pred_zg, pred_zh, gt_zg_d, gt_zh_d)
            else:
                l_ratio = torch.zeros((), device=device)

            # ---- 4. 3-D ray-tracing loss (enabled after warmup) ------
            if w_ray3d > 0:
                gt_gx   = torch.tensor([g[0] for g in gazex],
                                       dtype=torch.float32, device=device)
                gt_gy   = torch.tensor([g[0] for g in gazey],
                                       dtype=torch.float32, device=device)
                head_cx = torch.tensor([(b[0] + b[2]) / 2.0 for b in bboxes],
                                       dtype=torch.float32, device=device)
                head_cy = torch.tensor([(b[1] + b[3]) / 2.0 for b in bboxes],
                                       dtype=torch.float32, device=device)
                inout_mask = (inout.to(device) > 0)
                l_ray3d = gaze_ray_3d_loss(
                    pred_hm, pred_zg, pred_zh,
                    gt_gx, gt_gy, gt_zg_d, gt_zh_d,
                    head_cx, head_cy,
                    gamma=ray_gamma, w_cos=ray_w_cos, w_l1=ray_w_l1,
                    mask=inout_mask,
                )
            else:
                l_ray3d = torch.zeros((), device=device)

            # ---- 5. (Optional) consistency between dense & anchored --
            if w_consist > 0:
                head_cx_c = [(b[0] + b[2]) / 2.0 for b in bboxes]
                head_cy_c = [(b[1] + b[3]) / 2.0 for b in bboxes]
                gx_c      = [g[0] for g in gazex]
                gy_c      = [g[0] for g in gazey]
                l_consist = anchored_consistency_loss(
                    pred_dense, pred_zg, pred_zh,
                    gx_c, gy_c, head_cx_c, head_cy_c,
                )
            else:
                l_consist = torch.zeros((), device=device)

            # ---- 6. (Optional) inout BCE / focal --------------------
            if w_inout > 0 and preds['inout'] is not None:
                pred_io = torch.cat(preds['inout'], 0)
                gt_io   = inout.float().to(device)
                l_io    = inout_loss_fn(pred_io, gt_io)
            else:
                l_io = torch.zeros((), device=device)

            loss = (l_hm
                    + w_d_dense * l_d_dense
                    + w_ratio   * l_ratio
                    + w_ray3d   * l_ray3d
                    + w_consist * l_consist
                    + w_inout   * l_io)

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            sums['total']   += float(loss.item())
            sums['hm']      += float(l_hm.item())
            sums['d_dense'] += float(l_d_dense.item())
            sums['ratio']   += float(l_ratio.item())
            sums['ray3d']   += float(l_ray3d.item())
            sums['consist'] += float(l_consist.item())
            sums['io']      += float(l_io.item())

            if cur_iter % config['logging']['save_every'] == 0:
                wandb.log({
                    "train/loss":         float(loss.item()),
                    "train/heatmap_loss": float(l_hm.item()),
                    "train/d_dense_loss": float(l_d_dense.item()),
                    "train/ratio_loss":   float(l_ratio.item()),
                    "train/ray3d_loss":   float(l_ray3d.item()),
                    "train/consist_loss": float(l_consist.item()),
                    "train/inout_loss":   float(l_io.item()),
                    "train/w_d_dense":    w_d_dense,
                    "train/w_ratio":      w_ratio,
                    "train/w_ray3d":      w_ray3d,
                    "train/detach":       int(detach_flag),
                    "train/lr":           optimizer.param_groups[0]['lr'],
                })

        scheduler.step()
        ck_path = os.path.join(exp_dir, f'epoch_{epoch}.pt')
        torch.save(model.get_gazelle_state_dict(), ck_path)
        print(f"Saved checkpoint to {ck_path}")

        print(f"  Train means: total={sums['total']/n_iters:.4f}  "
              f"hm={sums['hm']/n_iters:.4f}  "
              f"d_dense={sums['d_dense']/n_iters:.4f}  "
              f"ratio={sums['ratio']/n_iters:.4f}  "
              f"ray3d={sums['ray3d']/n_iters:.4f}  "
              f"consist={sums['consist']/n_iters:.4f}  "
              f"io={sums['io']/n_iters:.4f}")

        # ---- EVAL (2D primary + 3D secondary) ------------------------
        model.eval()
        avg_l2s, min_l2s, aucs = [], [], []
        pred_zg_all, pred_zh_all = [], []
        gt_zg_all,   gt_zh_all   = [], []
        l2_3d_all, angle_3d_all  = [], []

        for batch in tqdm(eval_dl, total=len(eval_dl), desc=f"Eval {epoch}"):
            (imgs, bboxes, gazex, gazey, inout, heights, widths,
             gt_zg, gt_zh) = batch
            with torch.no_grad():
                preds = model({
                    "images": imgs.to(device),
                    "bboxes": [[bbox] for bbox in bboxes],
                })

            pred_hm = torch.stack(preds['heatmap']).squeeze(dim=1)
            pred_zg = torch.cat(preds['depth_gaze'], 0).detach().cpu()
            pred_zh = torch.cat(preds['depth_head'], 0).detach().cpu()
            gt_zg_t = torch.as_tensor(gt_zg, dtype=torch.float32)
            gt_zh_t = torch.as_tensor(gt_zh, dtype=torch.float32)

            pred_zg_all.append(pred_zg)
            pred_zh_all.append(pred_zh)
            gt_zg_all.append(gt_zg_t)
            gt_zh_all.append(gt_zh_t)

            for i in range(pred_hm.shape[0]):
                hm_cpu = pred_hm[i].detach().cpu()
                auc = gazefollow_auc(hm_cpu, gazex[i], gazey[i],
                                     heights[i], widths[i])
                avg_l2, min_l2 = gazefollow_l2(hm_cpu, gazex[i], gazey[i])
                aucs.append(auc)
                avg_l2s.append(avg_l2)
                min_l2s.append(min_l2)

                gx_mean = float(np.mean(gazex[i]))
                gy_mean = float(np.mean(gazey[i]))
                bbox = bboxes[i]
                head_cx = (float(bbox[0]) + float(bbox[2])) * 0.5
                head_cy = (float(bbox[1]) + float(bbox[3])) * 0.5
                l2_3d_all.append(anchored_l2_3d(
                    hm_cpu, pred_zg[i], pred_zh[i],
                    gx_mean, gy_mean,
                    gt_zg_t[i].item(), gt_zh_t[i].item(),
                ))
                angle_3d_all.append(gaze3d_angle_relative(
                    hm_cpu, pred_zg[i], pred_zh[i],
                    gx_mean, gy_mean,
                    gt_zg_t[i].item(), gt_zh_t[i].item(),
                    head_cx, head_cy,
                ))

        epoch_auc    = float(np.mean(aucs))
        epoch_avg_l2 = float(np.mean(avg_l2s))
        epoch_min_l2 = float(np.mean(min_l2s))

        pred_zg_cat = torch.cat(pred_zg_all)
        pred_zh_cat = torch.cat(pred_zh_all)
        gt_zg_cat   = torch.cat(gt_zg_all)
        gt_zh_cat   = torch.cat(gt_zh_all)
        R_MAE  = float(ratio_mae (pred_zg_cat, pred_zh_cat, gt_zg_cat, gt_zh_cat).item())
        R_RMSE = float(ratio_rmse(pred_zg_cat, pred_zh_cat, gt_zg_cat, gt_zh_cat).item())
        D1     = float(anchored_delta1(
            pred_zg_cat, pred_zh_cat, gt_zg_cat, gt_zh_cat).item())
        L2_3D  = float(np.mean(l2_3d_all))    if l2_3d_all    else 0.0
        A3D    = float(np.mean(angle_3d_all)) if angle_3d_all else 0.0

        wandb.log({
            "eval/auc":        epoch_auc,
            "eval/avg_l2":     epoch_avg_l2,
            "eval/min_l2":     epoch_min_l2,
            "eval/ratio_mae":  R_MAE,
            "eval/ratio_rmse": R_RMSE,
            "eval/delta1":     D1,
            "eval/l2_3d":      L2_3D,
            "eval/angle_3d":   A3D,
            "epoch":           epoch,
        })
        print(f"EVAL EPOCH {epoch}: AUC={epoch_auc:.4f}  "
              f"AvgL2={epoch_avg_l2:.4f}  MinL2={epoch_min_l2:.4f}  |  "
              f"RatioMAE={R_MAE:.4f}  RatioRMSE={R_RMSE:.4f}  "
              f"δ1={D1:.4f}  L2-3D={L2_3D:.4f}  Angle-3D={A3D:.2f}°")

        # User: prioritise AvgL2 (not MinL2) as the 2-D headline metric.
        if epoch_avg_l2 < best_avg_l2:
            best_avg_l2 = epoch_avg_l2
            best_epoch  = epoch

    print(f"Completed training. Best AvgL2 of {round(best_avg_l2, 4)} "
          f"obtained at epoch {best_epoch}.")
    wandb.finish()


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
