"""
train_goo-depthaware.py — depth-aware GOO-Synth finetuning
(simplified: GazeMoE + ONE composite auxiliary loss).

Mirrors the strategy used by ``train_gazefollow-depthaware.py`` but on the
GOO-Synth dataset.  The model architecture is *unchanged* from the SoTA
GazeMoE: no depth heads, no multi-task curriculum.  A single composite
auxiliary loss using DepthAnythingV2 pseudo-labels regulates the
heatmap into normalised (x, y, depth) ∈ [0, 1]³ coherence:

    L = SCALAR · BCE(pred_heatmap, depth_aware_heatmap_target)
      + w_aux_3d · gaze3d_aux_loss(...)

``gaze3d_aux_loss`` (defined in ``train_depth_goo.py``) is::

    L_aux = 1 − cos((pred_xy − head_c), (gt_xy − head_c))     ← 2-D angular
          + depth_weight · |d_pred − d_gt|                     ← depth-coherence

The depth-aware heatmap target (``utils.get_depthaware_heatmap``) is preserved
exactly as in the GazeFollow recipe.

Inference contract
------------------
Plain GazeMoE — at inference it consumes only the image and head bbox.
No .npy depth file is needed at deploy time.

Dataset
-------
GOO-Synth:
    <goo_path>/goosynth_train_preprocess.json   (train)
    <goo_path>/goosynth_test_preprocess.json    (test)
    <goo_path>/<depth_dir>/<image_id>.npy       (DepthAnythingV2 cached maps)

GOO frames carry exactly one head with single-annotator gaze coordinates
in pixel space, so we use the VAT-style rectangular AUC + single-point
L2 as the 2-D headline metrics (matching the rest of the GOO literature
and the existing ``train_depth_goo.py`` convention).
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

from eval import vat_auc, vat_l2
from network.network_builder_update2 import get_gazemoe_model
import network.utils as utils
from network.utils import get_depthaware_heatmap

from train_depth_goo import gaze3d_aux_loss, depth_order_accuracy_proxy


# --------------------------------------------------------------------------
# Joint augmentations (RGB + depth must move together) — depth map is a
# training-only signal, so we keep it aligned through crop/flip/jitter.
# --------------------------------------------------------------------------

def _joint_random_crop(image, depth, bbox, gazex, gazey, inout):
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

class GOODepthDataset(torch.utils.data.Dataset):
    """GOO-Synth with depth-aware heatmap target + per-image min-max DA2
    depth map (training-only auxiliary signal).

    The JSON schema (``goosynth_{split}_preprocess.json``) carries one head
    per frame in pixel-space (``bbox``, ``gazex``, ``gazey``); we normalise
    on the fly after augmentation.

    Train items::

        image_t        : [3, H, W]
        bbox_norm      : [4]   in [0, 1]
        gazex_norm     : list  in [0, 1]   (single-annotator → length-1)
        gazey_norm     : list  in [0, 1]
        inout          : 0-D tensor
        height, width  : int (post-aug image size)
        heatmap_da     : [64, 64] depth-aware Gaussian target
        depth_64       : [64, 64] per-image min-max-normalised DA2 in [0, 1]

    Test items: same minus the heatmap target; depth is still emitted so
    the eval pipeline can compute the 3-D-proxy metrics.
    """

    def __init__(self, path, split, transform,
                 in_frame_only=True, aug_groups=None, depth_dir='depth'):
        self.path = path
        self.depth_dir = depth_dir
        self.split = split
        self.is_train = (split == "train")
        self.aug = self.is_train
        self.transform = transform
        self.in_frame_only = in_frame_only
        self.aug_groups = aug_groups if aug_groups is not None else []

        json_path = os.path.join(self.path,
                                  f"goosynth_{split}_preprocess.json")
        with open(json_path, "r") as f:
            self.frames = json.load(f)

        # GOO-Synth has exactly one head per frame (head_idx = 0).
        self.data_idxs = []
        for i, frame in enumerate(self.frames):
            head = frame['heads'][0]
            if not self.in_frame_only or head.get('inout', 1) == 1:
                self.data_idxs.append(i)

    # ----------------------------------------------------------------------
    def _depth_path(self, image_rel_path):
        rel = image_rel_path.replace("images" + os.sep,
                                     self.depth_dir + os.sep, 1)
        if rel == image_rel_path:
            rel = os.path.join(self.depth_dir, image_rel_path)
        rel = os.path.splitext(rel)[0] + ".npy"
        return os.path.join(self.path, rel)

    def _load_depth_pil(self, image_rel_path):
        depth = np.load(self._depth_path(image_rel_path)).astype(np.float32)
        return Image.fromarray(depth, mode="F")

    # ----------------------------------------------------------------------
    def __getitem__(self, idx):
        frame = self.frames[self.data_idxs[idx]]
        head  = copy.deepcopy(frame['heads'][0])
        inout = int(head.get('inout', 1))

        img_path = os.path.join(self.path, frame['path'])
        img = Image.open(img_path).convert("RGB")
        depth_pil = self._load_depth_pil(frame['path'])

        # GOO carries pixel-space bbox/gazex/gazey under the same keys as
        # GazeFollow's pre-augmentation values.
        bbox  = list(head['bbox'])
        gazex = list(head['gazex'])
        gazey = list(head['gazey'])

        if self.aug:
            if 'crop' in self.aug_groups and np.random.sample() <= 0.5:
                img, depth_pil, bbox, gazex, gazey = _joint_random_crop(
                    img, depth_pil, bbox, gazex, gazey, inout)
            if 'crop' in self.aug_groups and np.random.sample() <= 0.5:
                img, depth_pil, bbox, gazex, gazey = _joint_horiz_flip(
                    img, depth_pil, bbox, gazex, gazey, inout)
            if 'crop' in self.aug_groups and np.random.sample() <= 0.5:
                bbox = utils.random_bbox_jitter(img, bbox)

            if 'photometric' in self.aug_groups and np.random.sample() <= 0.5:
                photometric = T.Compose([
                    T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2,
                                                 saturation=0.2, hue=0.1)], p=0.5),
                    T.RandomGrayscale(p=0.2),
                    T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                    T.RandomAutocontrast(p=0.1),
                ])
                img = photometric(img)

        width, height = img.size
        bbox_norm  = [bbox[0] / width,  bbox[1] / height,
                      bbox[2] / width,  bbox[3] / height]
        gazex_norm = [x / float(width)  for x in gazex]
        gazey_norm = [y / float(height) for y in gazey]

        img_t = self.transform(img)

        # ---- Depth → 64×64 per-image min-max-normalised in [0, 1] -----
        d_arr_64 = np.asarray(depth_pil.resize((64, 64), Image.BILINEAR),
                              dtype=np.float32)
        d_min, d_max = float(d_arr_64.min()), float(d_arr_64.max())
        d_norm_64 = (d_arr_64 - d_min) / ((d_max - d_min) + 1e-8)
        depth_64_t = torch.from_numpy(d_norm_64).float()         # [64, 64]

        if self.is_train:
            # Depth-aware Gaussian target — same recipe as the GazeFollow
            # trainer; uses post-augmentation depth + gaze coordinates.
            depth_full = np.asarray(depth_pil, dtype=np.float32)
            heatmap = get_depthaware_heatmap(
                depth_full, gazex_norm[0], gazey_norm[0], 64, 64)
            return (img_t, bbox_norm, gazex_norm, gazey_norm,
                    torch.tensor(inout), height, width,
                    heatmap, depth_64_t)
        return (img_t, bbox_norm, gazex_norm, gazey_norm,
                torch.tensor(inout), height, width,
                depth_64_t)

    def __len__(self):
        return len(self.data_idxs)


def collate_fn(batch):
    transposed = list(zip(*batch))
    return tuple(
        torch.stack(items) if isinstance(items[0], torch.Tensor) else list(items)
        for items in transposed
    )


# --------------------------------------------------------------------------
# 3-D-proxy eval metrics (no depth prediction needed — use GT-sampled depth).
# --------------------------------------------------------------------------

def _sample_depth_at(depth_64, x_norm, y_norm):
    """Nearest-pixel sample of a [H, W] depth map at normalised (x, y).
    Uses ``floor(c · W)`` so that ``argmax_col / W`` round-trips back to
    ``argmax_col`` — consistent with the soft-argmax normalisation.
    """
    H, W = depth_64.shape
    x = int(np.clip(int(float(x_norm) * W), 0, W - 1))
    y = int(np.clip(int(float(y_norm) * H), 0, H - 1))
    return float(depth_64[y, x])


def l2_3d_proxy(heatmap, gazex, gazey, depth_64):
    argmax = int(heatmap.flatten().argmax().item())
    pred_y, pred_x = np.unravel_index(argmax, heatmap.shape)
    pred_x = pred_x / float(heatmap.shape[1])
    pred_y = pred_y / float(heatmap.shape[0])
    gt_x = float(np.mean(gazex))
    gt_y = float(np.mean(gazey))
    d_pred = _sample_depth_at(depth_64, pred_x, pred_y)
    d_gt   = _sample_depth_at(depth_64, gt_x,   gt_y)
    return float(np.sqrt(
        (pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2 + (d_pred - d_gt) ** 2
    ))


def angle_3d_proxy(heatmap, gazex, gazey, head_cx, head_cy, depth_64):
    argmax = int(heatmap.flatten().argmax().item())
    pred_y, pred_x = np.unravel_index(argmax, heatmap.shape)
    pred_x = pred_x / float(heatmap.shape[1])
    pred_y = pred_y / float(heatmap.shape[0])
    gt_x = float(np.mean(gazex))
    gt_y = float(np.mean(gazey))
    d_head = _sample_depth_at(depth_64, head_cx, head_cy)
    d_pred = _sample_depth_at(depth_64, pred_x,  pred_y)
    d_gt   = _sample_depth_at(depth_64, gt_x,    gt_y)
    pv = np.array([pred_x - head_cx, pred_y - head_cy, d_pred - d_head],
                  dtype=np.float64)
    gv = np.array([gt_x   - head_cx, gt_y   - head_cy, d_gt   - d_head],
                  dtype=np.float64)
    pn = float(np.linalg.norm(pv)) + 1e-8
    gn = float(np.linalg.norm(gv)) + 1e-8
    cos = float(np.dot(pv, gv) / (pn * gn))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


# --------------------------------------------------------------------------
# Scheduler helper (warmup + cosine).
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
        name="goo_depth_aux",
        config=config,
    )

    checkpoint_dir = "_".join([
        "GOO",
        cfg_m['name'],
        cfg_m.get('moe_type', 'vanilla'),
        str(cfg_m.get('is_msf', 1)),
        cfg_t['pre_optimizer'],
        "bs" + str(cfg_t['pre_batch_size']),
        cfg_m['pbce_loss'],
        str(cfg_t['pre_lr']),
        "aux" + str(cfg_m.get('w_aux_3d', 0.1)),
    ])
    exp_dir = os.path.join(config['logging']['pre_dir'], checkpoint_dir)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Checkpoint dir: {exp_dir}")

    # ---- Model — plain GazeMoE (no depth heads, no curriculum) ---------
    model, transform = get_gazemoe_model(config)
    print(f"Model: {cfg_m['name']} (plain GazeMoE; depth used only as aux loss)")

    # Freeze backbone (same as the GazeFollow recipe).
    for param in model.backbone.parameters():
        param.requires_grad = False
    print(f"Learnable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- Warm-start from a SoTA GazeMoE checkpoint ----------------------
    pretrained_path = cfg_m.get('pretrained_path', '')
    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"Warm-starting from {pretrained_path}")
        sd = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        model.load_gazelle_state_dict(sd, include_backbone=False)
    elif pretrained_path:
        print(f"WARN pretrained_path set but file not found: {pretrained_path!r}  "
              f"— training from scratch.")

    model.to(device)

    # ---- Datasets / loaders --------------------------------------------
    depth_dir = cfg_d.get('depth_dir', 'depth')
    goo_path  = cfg_d.get('goo_path',  '../GOOSynthV3')

    train_dataset = GOODepthDataset(
        goo_path, 'train', transform,
        aug_groups=cfg_d.get('augmentations', []), depth_dir=depth_dir)
    eval_dataset = GOODepthDataset(
        goo_path, 'test', transform,
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

    # ---- Per-bucket LR dispatch (GazeMoE convention) -------------------
    pre_lr       = float(cfg_t['pre_lr'])
    pre_fuse_lr  = float(cfg_t.get('pre_fuse_lr',  pre_lr))
    pre_block_lr = float(cfg_t.get('pre_block_lr', pre_lr))

    param_dicts = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ('ms_fusion' in n) or ('fuse' in n) or ('fusion' in n) or ('linear' in n):
            lr = pre_fuse_lr
        elif ('transformer' in n) or ('block' in n):
            lr = pre_block_lr
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

    # ---- Single auxiliary-loss schedule --------------------------------
    w_aux_3d_target  = float(cfg_m.get('w_aux_3d',             0.1))
    aux_warmup_eps   = int  (cfg_m.get('aux_3d_warmup_epochs', 1))
    aux_depth_weight = float(cfg_m.get('aux_3d_depth_weight',  1.0))
    aux_temperature  = cfg_m.get('aux_3d_temperature', None)
    if aux_temperature is not None:
        aux_temperature = float(aux_temperature)

    grad_clip = float(cfg_t.get('gradient_clipping', 0.0))

    print(f"Auxiliary loss: gaze3d_aux  w_aux_3d={w_aux_3d_target}  "
          f"warmup={aux_warmup_eps} epoch(s)  "
          f"depth_weight={aux_depth_weight}  "
          f"temperature={aux_temperature}")

    # ---- Training loop --------------------------------------------------
    best_avg_l2 = float("inf")
    best_epoch  = None

    for epoch in range(pre_epochs):
        # Linear warmup of w_aux_3d from 0 → w_aux_3d_target.
        if aux_warmup_eps <= 0 or epoch >= aux_warmup_eps:
            w_aux_cur = w_aux_3d_target
        else:
            w_aux_cur = w_aux_3d_target * (epoch + 1) / float(aux_warmup_eps + 1)

        print(f"[Epoch {epoch}] w_aux_3d={w_aux_cur:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # ---- TRAIN EPOCH ---------------------------------------------
        model.train()
        sums = {'total': 0.0, 'hm': 0.0, 'aux': 0.0}
        n_iters = len(train_dl)

        for cur_iter, batch in tqdm(enumerate(train_dl), total=n_iters,
                                     desc=f"Epoch {epoch + 1}/{pre_epochs}"):
            (imgs, bboxes, gazex, gazey, inout, heights, widths,
             heatmaps, depth_64) = batch

            optimizer.zero_grad()
            preds = model({
                "images": imgs.to(device),
                "bboxes": [[bbox] for bbox in bboxes],
            })
            pred_hm = torch.stack(preds['heatmap']).squeeze(dim=1)   # [B, 64, 64]

            # ---- 1. Heatmap BCE (primary, dominant) ----
            l_hm = SCALAR * heatmap_loss_fn(pred_hm, heatmaps.to(device))

            # ---- 2. Composite auxiliary 3-D loss (single term) ----
            if w_aux_cur > 0:
                gt_gx   = torch.tensor([g[0] for g in gazex],
                                       dtype=torch.float32, device=device)
                gt_gy   = torch.tensor([g[0] for g in gazey],
                                       dtype=torch.float32, device=device)
                head_cx = torch.tensor([(b[0] + b[2]) / 2.0 for b in bboxes],
                                       dtype=torch.float32, device=device)
                head_cy = torch.tensor([(b[1] + b[3]) / 2.0 for b in bboxes],
                                       dtype=torch.float32, device=device)
                inout_mask = (inout.to(device) > 0)
                l_aux = gaze3d_aux_loss(
                    pred_hm,
                    gt_gx, gt_gy, head_cx, head_cy,
                    depth_norm=depth_64.to(device),
                    mask=inout_mask,
                    depth_weight=aux_depth_weight,
                    temperature=aux_temperature,
                )
            else:
                l_aux = torch.zeros((), device=device)

            loss = l_hm + w_aux_cur * l_aux
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            sums['total'] += float(loss.item())
            sums['hm']    += float(l_hm.item())
            sums['aux']   += float(l_aux.item())

            if cur_iter % config['logging']['save_every'] == 0:
                wandb.log({
                    "train/loss":         float(loss.item()),
                    "train/heatmap_loss": float(l_hm.item()),
                    "train/aux_loss":     float(l_aux.item()),
                    "train/w_aux_3d":     w_aux_cur,
                    "train/lr":           optimizer.param_groups[0]['lr'],
                })

        scheduler.step()
        ck_path = os.path.join(exp_dir, f'epoch_{epoch}.pt')
        torch.save(model.get_gazelle_state_dict(), ck_path)
        print(f"Saved checkpoint to {ck_path}")
        print(f"  Train means: total={sums['total']/n_iters:.4f}  "
              f"hm={sums['hm']/n_iters:.4f}  "
              f"aux={sums['aux']/n_iters:.4f}")

        # ---- EVAL ----
        # 2-D primary metrics (VAT-style AUC + L2 for single-annotator GOO)
        # plus 3-D-proxy secondary metrics from sampled GT DA2 depth.
        model.eval()
        aucs, l2s = [], []
        l2_3d_all, angle_3d_all = [], []
        doa_correct, doa_valid = 0, 0

        for batch in tqdm(eval_dl, total=len(eval_dl), desc=f"Eval {epoch}"):
            (imgs, bboxes, gazex, gazey, inout, heights, widths,
             depth_64) = batch
            with torch.no_grad():
                preds = model({
                    "images": imgs.to(device),
                    "bboxes": [[bbox] for bbox in bboxes],
                })
            pred_hm = torch.stack(preds['heatmap']).squeeze(dim=1)

            for i in range(pred_hm.shape[0]):
                if int(inout[i].item()) != 1:
                    continue
                hm_cpu = pred_hm[i].detach().cpu()
                gx, gy = gazex[i], gazey[i]
                if not gx or not gy:
                    continue
                aucs.append(vat_auc(hm_cpu, gx[0], gy[0]))
                l2s.append(vat_l2(hm_cpu, gx[0], gy[0]))

                # 3-D-proxy metrics (no depth prediction).
                d_np = depth_64[i].numpy()
                hm_np = hm_cpu.numpy()
                bbox = bboxes[i]
                hcx = (float(bbox[0]) + float(bbox[2])) * 0.5
                hcy = (float(bbox[1]) + float(bbox[3])) * 0.5

                l2_3d_all.append(l2_3d_proxy(hm_np, gx, gy, d_np))
                angle_3d_all.append(angle_3d_proxy(
                    hm_np, gx, gy, hcx, hcy, d_np))
                is_correct, is_valid = depth_order_accuracy_proxy(
                    hm_np, gx, gy, hcx, hcy, d_np)
                if is_valid:
                    doa_valid   += 1
                    doa_correct += int(bool(is_correct))

        epoch_auc    = float(np.mean(aucs))      if aucs      else 0.0
        epoch_avg_l2 = float(np.mean(l2s))       if l2s       else 0.0
        L2_3D = float(np.mean(l2_3d_all))    if l2_3d_all    else 0.0
        A3D   = float(np.mean(angle_3d_all)) if angle_3d_all else 0.0
        DOA   = (100.0 * doa_correct / doa_valid) if doa_valid > 0 else 0.0

        wandb.log({
            "eval/auc":         epoch_auc,
            "eval/avg_l2":      epoch_avg_l2,
            "eval/l2_3d":       L2_3D,
            "eval/angle_3d":    A3D,
            "eval/doa_percent": DOA,
            "eval/doa_n_valid": doa_valid,
            "epoch":            epoch,
        })
        print(f"EVAL EPOCH {epoch}: AUC={epoch_auc:.4f}  "
              f"AvgL2={epoch_avg_l2:.4f}  |  "
              f"L2-3D(proxy)={L2_3D:.4f}  Angle-3D(proxy)={A3D:.2f}°  "
              f"DOA={DOA:.2f}% (n_valid={doa_valid})")

        # Prioritise AvgL2 as the 2-D headline metric (per GazeFollow recipe).
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
