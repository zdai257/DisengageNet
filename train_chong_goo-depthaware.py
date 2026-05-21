"""
train_chong_goo-depthaware.py — depth-aware GOO-Synth pretraining for Chong et al.

Combines:
    • Chong per-frame model + dual-branch inputs from ``train_chong.py``
    • GOO-Synth depth-aware recipe from ``train_goo-depthaware.py``

Loss (in-frame heatmap + optional in/out + aux 3-D proxy)::

    L = BCEWithLogits(pred_hm, depth_aware_heatmap_target)
      + w_inout · BCEWithLogits(inout_pred, inout_gt)
      + w_aux_3d · gaze3d_aux_loss(...)

At inference the model still consumes only RGB + head bbox (224×224 ResNet
pipeline); depth is a training-only auxiliary signal from cached DA2 maps.

Dataset
-------
    <goo_path>/goosynth_train_preprocess.json
    <goo_path>/goosynth_test_preprocess.json
    <goo_path>/<depth_dir>/<image_id>.npy

Checkpoints are plain ``model.state_dict()`` files compatible with::

    python test_models.py --arch chong --checkpoint <path> ...
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR
from tqdm import tqdm
import wandb
import yaml

from eval import vat_auc, vat_l2
from network.network_builder_chong import get_chong_model
import network.utils as utils
from network.utils import get_depthaware_heatmap

from train_depth_goo import gaze3d_aux_loss, depth_order_accuracy_proxy


# --------------------------------------------------------------------------
# Chong input builder (from train_chong.py)
# --------------------------------------------------------------------------

def _build_inputs(images: List[Image.Image], samples: List[Dict[str, Any]],
                  transform, device: torch.device):
    """Build (scene, head, head-position-map) batches for ChongModel."""
    scene_t = torch.stack([transform(im) for im in images]).to(device)
    head_imgs, head_poses = [], []
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
    return (scene_t,
            torch.stack(head_imgs).to(device),
            torch.stack(head_poses).to(device))


def _samples_from_batch(bboxes, gazex, gazey, inout, heights, widths):
    return [
        {
            "bbox_norm":  list(b),
            "gazex_norm": list(gx),
            "gazey_norm": list(gy),
            "inout":      int(io),
            "width":      int(h),
            "height":     int(w),
        }
        for b, gx, gy, io, h, w in zip(bboxes, gazex, gazey, inout, heights, widths)
    ]


# --------------------------------------------------------------------------
# Joint augmentations (RGB + depth)
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
# Dataset — GOO-Synth with depth; returns PIL for Chong head-crop path
# --------------------------------------------------------------------------

class GOODepthChongDataset(torch.utils.data.Dataset):
    """GOO-Synth loader that keeps augmented RGB as PIL (Chong needs raw
    crops for the head branch).  Depth-aware heatmap + DA2 depth as in
    ``train_goo-depthaware.GOODepthDataset``."""

    def __init__(self, path, split, in_frame_only=True, aug_groups=None,
                 depth_dir="depth"):
        self.path = path
        self.depth_dir = depth_dir
        self.split = split
        self.is_train = (split == "train")
        self.aug = self.is_train
        self.in_frame_only = in_frame_only
        self.aug_groups = aug_groups if aug_groups is not None else []

        json_path = os.path.join(self.path, f"goosynth_{split}_preprocess.json")
        with open(json_path, "r") as f:
            self.frames = json.load(f)

        self.data_idxs = []
        for i, frame in enumerate(self.frames):
            head = frame["heads"][0]
            if not self.in_frame_only or head.get("inout", 1) == 1:
                self.data_idxs.append(i)

    def _depth_path(self, image_rel_path):
        rel = image_rel_path.replace("\\", "/")
        if rel.startswith("images/"):
            rel = rel[len("images/"):]
        rel = os.path.splitext(rel)[0] + ".npy"
        primary = os.path.join(self.path, self.depth_dir, rel)
        if os.path.isfile(primary):
            return primary
        alt = os.path.join(
            self.path, self.depth_dir,
            os.path.splitext(image_rel_path.replace("\\", "/"))[0] + ".npy")
        return alt if os.path.isfile(alt) else primary

    def _load_depth_pil(self, image_rel_path):
        depth_path = self._depth_path(image_rel_path)
        if not os.path.isfile(depth_path):
            raise FileNotFoundError(
                f"Depth map not found: {depth_path!r}  "
                f"(image={image_rel_path!r}; run preprocess_Depth.py)")
        depth = np.load(depth_path).astype(np.float32)
        return Image.fromarray(depth, mode="F")

    def __getitem__(self, idx):
        frame = self.frames[self.data_idxs[idx]]
        head = copy.deepcopy(frame["heads"][0])
        inout = int(head.get("inout", 1))

        img = Image.open(os.path.join(self.path, frame["path"])).convert("RGB")
        depth_pil = self._load_depth_pil(frame["path"])

        bbox = list(head["bbox"])
        gazex = list(head["gazex"])
        gazey = list(head["gazey"])

        if self.aug:
            if "crop" in self.aug_groups and np.random.sample() <= 0.5:
                img, depth_pil, bbox, gazex, gazey = _joint_random_crop(
                    img, depth_pil, bbox, gazex, gazey, inout)
            if "crop" in self.aug_groups and np.random.sample() <= 0.5:
                img, depth_pil, bbox, gazex, gazey = _joint_horiz_flip(
                    img, depth_pil, bbox, gazex, gazey, inout)
            if "crop" in self.aug_groups and np.random.sample() <= 0.5:
                bbox = utils.random_bbox_jitter(img, bbox)

            if "photometric" in self.aug_groups and np.random.sample() <= 0.5:
                photometric = T.Compose([
                    T.RandomApply([T.ColorJitter(
                        brightness=0.2, contrast=0.2,
                        saturation=0.2, hue=0.1)], p=0.5),
                    T.RandomGrayscale(p=0.2),
                    T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                    T.RandomAutocontrast(p=0.1),
                ])
                img = photometric(img)

        width, height = img.size
        bbox_norm = [bbox[0] / width, bbox[1] / height,
                     bbox[2] / width, bbox[3] / height]
        gazex_norm = [x / float(width) for x in gazex]
        gazey_norm = [y / float(height) for y in gazey]

        d_arr_64 = np.asarray(depth_pil.resize((64, 64), Image.BILINEAR),
                              dtype=np.float32)
        d_min, d_max = float(d_arr_64.min()), float(d_arr_64.max())
        d_norm_64 = (d_arr_64 - d_min) / ((d_max - d_min) + 1e-8)
        depth_64_t = torch.from_numpy(d_norm_64).float()

        if self.is_train:
            depth_full = np.asarray(depth_pil, dtype=np.float32)
            heatmap = get_depthaware_heatmap(
                depth_full, gazex_norm[0], gazey_norm[0], 64, 64)
            return (img, bbox_norm, gazex_norm, gazey_norm,
                    torch.tensor(inout), height, width,
                    heatmap, depth_64_t)
        return (img, bbox_norm, gazex_norm, gazey_norm,
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
# 3-D-proxy eval helpers (from train_goo-depthaware.py)
# --------------------------------------------------------------------------

def _sample_depth_at(depth_64, x_norm, y_norm):
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
    d_gt = _sample_depth_at(depth_64, gt_x, gt_y)
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
    d_pred = _sample_depth_at(depth_64, pred_x, pred_y)
    d_gt = _sample_depth_at(depth_64, gt_x, gt_y)
    pv = np.array([pred_x - head_cx, pred_y - head_cy, d_pred - d_head],
                  dtype=np.float64)
    gv = np.array([gt_x - head_cx, gt_y - head_cy, d_gt - d_head],
                  dtype=np.float64)
    pn = float(np.linalg.norm(pv)) + 1e-8
    gn = float(np.linalg.norm(gv)) + 1e-8
    cos = float(np.dot(pv, gv) / (pn * gn))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


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
    with open("configuration.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = config["hardware"]["device"] if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    cfg_m, cfg_t, cfg_d = config["model"], config["train"], config["data"]

    wandb.init(
        project="chong_goo_depthaware",
        name="chong_goo_depth_aux",
        config=config,
    )

    checkpoint_dir = "_".join([
        "Chong_GOO",
        "depth_aux",
        cfg_t["pre_optimizer"],
        "bs" + str(cfg_t.get("chong_batch_size", cfg_t["pre_batch_size"])),
        str(cfg_t["pre_lr"]),
        "aux" + str(cfg_m.get("w_aux_3d", 0.1)),
    ])
    exp_dir = os.path.join(
        config["logging"].get("chong_pre_dir",
                               config["logging"].get("pre_dir", "results")),
        checkpoint_dir)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Checkpoint dir: {exp_dir}")

    model, transform = get_chong_model(pretrained_backbone=True)
    print("Model: Chong et al. (CVPR 2020 per-frame) + depth-aware aux loss")
    print(f"Learnable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    pretrained_path = cfg_m.get("pretrained_path", "")
    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"Warm-starting from {pretrained_path}")
        sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[warn] partial load: missing={len(missing)}  "
                  f"unexpected={len(unexpected)}")
    elif pretrained_path:
        print(f"WARN pretrained_path set but file not found: {pretrained_path!r}  "
              f"— using ImageNet-init ResNet backbones only.")

    model.to(device)

    depth_dir = cfg_d.get("depth_dir", "depth")
    goo_path = cfg_d.get("goo_path", "../GOOSynthV3")
    batch_size = int(cfg_t.get("chong_batch_size", cfg_t["pre_batch_size"]))

    train_dataset = GOODepthChongDataset(
        goo_path, "train",
        aug_groups=cfg_d.get("augmentations", []), depth_dir=depth_dir)
    eval_dataset = GOODepthChongDataset(
        goo_path, "test", depth_dir=depth_dir)

    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"].get("pin_memory", False),
    )
    eval_dl = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"].get("pin_memory", False),
    )

    pre_lr = float(cfg_t["pre_lr"])
    opt_name = cfg_t["pre_optimizer"]
    wd = float(cfg_t.get("pre_weight_decay", cfg_t.get("weight_decay", 0.0)))
    if opt_name == "Adam":
        optimizer = Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=pre_lr, weight_decay=wd)
    elif opt_name == "AdamW":
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=pre_lr, weight_decay=wd)
    else:
        raise TypeError(f"Optimizer not supported: {opt_name}")

    pre_epochs = int(cfg_t["pre_epochs"])
    sched_cfg = cfg_t.get("pre_lr_scheduler", cfg_t["lr_scheduler"])
    sched_type = sched_cfg.get("type", "cosine")
    if sched_type == "warmup_cosine":
        warmup_epochs_sched = int(sched_cfg.get("warmup_epochs", 2))
        eta_min = float(sched_cfg.get("min_lr", 1e-7))
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=_make_warmup_cosine_lambda(
                pre_epochs, warmup_epochs_sched, eta_min, pre_lr),
        )
    elif sched_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get("step_size", pre_epochs)),
            eta_min=float(sched_cfg.get("min_lr", 1e-7)),
        )
    else:
        scheduler = StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", pre_epochs)),
            gamma=float(sched_cfg.get("gamma", 0.1)),
        )

    w_inout = float(cfg_m.get("bce_weight", 1.0))
    w_aux_3d_target = float(cfg_m.get("w_aux_3d", 0.1))
    aux_warmup_eps = int(cfg_m.get("aux_3d_warmup_epochs", 1))
    aux_depth_weight = float(cfg_m.get("aux_3d_depth_weight", 1.0))
    aux_temperature = cfg_m.get("aux_3d_temperature", None)
    if aux_temperature is not None:
        aux_temperature = float(aux_temperature)
    grad_clip = float(cfg_t.get("gradient_clipping", 5.0))

    print(f"Loss: depth-aware BCE + inout(w={w_inout}) + gaze3d_aux(w={w_aux_3d_target})")

    best_avg_l2 = float("inf")
    best_epoch = None

    for epoch in range(pre_epochs):
        if aux_warmup_eps <= 0 or epoch >= aux_warmup_eps:
            w_aux_cur = w_aux_3d_target
        else:
            w_aux_cur = w_aux_3d_target * (epoch + 1) / float(aux_warmup_eps + 1)

        print(f"[Epoch {epoch}] w_aux_3d={w_aux_cur:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        model.train()
        sums = {"total": 0.0, "hm": 0.0, "io": 0.0, "aux": 0.0}
        n_iters = len(train_dl)

        for cur_iter, batch in tqdm(enumerate(train_dl), total=n_iters,
                                     desc=f"Epoch {epoch + 1}/{pre_epochs}"):
            (imgs, bboxes, gazex, gazey, inout, heights, widths,
             heatmaps, depth_64) = batch

            samples = _samples_from_batch(
                bboxes, gazex, gazey, inout, heights, widths)
            scene_t, head_t, pos_t = _build_inputs(
                imgs, samples, transform, device)

            optimizer.zero_grad()
            out = model(scene_t, head_t, pos_t)
            pred_logit = out["heatmap_logit"].squeeze(1)
            gt_hm = heatmaps.to(device)
            gt_io = inout.to(device).float()

            l_hm = F.binary_cross_entropy_with_logits(pred_logit, gt_hm)
            l_io = F.binary_cross_entropy_with_logits(
                out["inout_logit"].squeeze(1), gt_io)
            pred_hm = torch.sigmoid(pred_logit)

            if w_aux_cur > 0:
                gt_gx = torch.tensor([g[0] for g in gazex],
                                     dtype=torch.float32, device=device)
                gt_gy = torch.tensor([g[0] for g in gazey],
                                     dtype=torch.float32, device=device)
                head_cx = torch.tensor([(b[0] + b[2]) / 2.0 for b in bboxes],
                                       dtype=torch.float32, device=device)
                head_cy = torch.tensor([(b[1] + b[3]) / 2.0 for b in bboxes],
                                       dtype=torch.float32, device=device)
                inout_mask = (gt_io > 0)
                l_aux = gaze3d_aux_loss(
                    pred_hm, gt_gx, gt_gy, head_cx, head_cy,
                    depth_norm=depth_64.to(device),
                    mask=inout_mask,
                    depth_weight=aux_depth_weight,
                    temperature=aux_temperature,
                )
            else:
                l_aux = torch.zeros((), device=device)

            loss = l_hm + w_inout * l_io + w_aux_cur * l_aux
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            sums["total"] += float(loss.item())
            sums["hm"] += float(l_hm.item())
            sums["io"] += float(l_io.item())
            sums["aux"] += float(l_aux.item())

            if cur_iter % config["logging"]["save_every"] == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/heatmap_loss": l_hm.item(),
                    "train/inout_loss": l_io.item(),
                    "train/aux_loss": l_aux.item(),
                    "train/w_aux_3d": w_aux_cur,
                    "train/lr": optimizer.param_groups[0]["lr"],
                })

        scheduler.step()
        ck_path = os.path.join(exp_dir, f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), ck_path)
        print(f"Saved checkpoint to {ck_path}")
        print(f"  Train means: total={sums['total']/n_iters:.4f}  "
              f"hm={sums['hm']/n_iters:.4f}  io={sums['io']/n_iters:.4f}  "
              f"aux={sums['aux']/n_iters:.4f}")

        model.eval()
        aucs, l2s = [], []
        l2_3d_all, angle_3d_all = [], []
        doa_correct, doa_valid = 0, 0

        for batch in tqdm(eval_dl, total=len(eval_dl), desc=f"Eval {epoch}"):
            (imgs, bboxes, gazex, gazey, inout, heights, widths,
             depth_64) = batch

            samples = _samples_from_batch(
                bboxes, gazex, gazey, inout, heights, widths)
            with torch.no_grad():
                out = model.forward_eval(
                    images=imgs, samples=samples,
                    transform=transform, device=torch.device(device))
            pred_hm = out["heatmap"]

            for i in range(pred_hm.shape[0]):
                if int(inout[i].item()) != 1:
                    continue
                hm_cpu = pred_hm[i]
                gx, gy = gazex[i], gazey[i]
                if not gx or not gy:
                    continue
                aucs.append(vat_auc(hm_cpu, gx[0], gy[0]))
                l2s.append(vat_l2(hm_cpu, gx[0], gy[0]))

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
                    doa_valid += 1
                    doa_correct += int(bool(is_correct))

        epoch_auc = float(np.mean(aucs)) if aucs else 0.0
        epoch_avg_l2 = float(np.mean(l2s)) if l2s else 0.0
        L2_3D = float(np.mean(l2_3d_all)) if l2_3d_all else 0.0
        A3D = float(np.mean(angle_3d_all)) if angle_3d_all else 0.0
        DOA = (100.0 * doa_correct / doa_valid) if doa_valid > 0 else 0.0

        wandb.log({
            "eval/auc": epoch_auc,
            "eval/avg_l2": epoch_avg_l2,
            "eval/l2_3d": L2_3D,
            "eval/angle_3d": A3D,
            "eval/doa_percent": DOA,
            "eval/doa_n_valid": doa_valid,
            "epoch": epoch,
        })
        print(f"EVAL EPOCH {epoch}: AUC={epoch_auc:.4f}  "
              f"AvgL2={epoch_avg_l2:.4f}  |  "
              f"L2-3D(proxy)={L2_3D:.4f}  Angle-3D(proxy)={A3D:.2f}°  "
              f"DOA={DOA:.2f}% (n_valid={doa_valid})")

        if epoch_avg_l2 < best_avg_l2:
            best_avg_l2 = epoch_avg_l2
            best_epoch = epoch

    print(f"Completed training. Best AvgL2 of {round(best_avg_l2, 4)} "
          f"obtained at epoch {best_epoch}.")
    wandb.finish()


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
