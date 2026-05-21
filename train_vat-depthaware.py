"""
train_vat-depthaware.py — depth-aware VAT finetuning (GazeMoE + aux 3D loss).

Finetunes a GazeFollow-pretrained GazeMoE on VideoAttentionTarget with the
same two depth-aware changes as train_gazefollow-depthaware.py:

    L = SCALAR · BCE(pred_hm, depth_aware_heatmap)   [in-frame only]
      + w_inout · BCE/Focal(inout_pred, inout_gt)
      + w_aux_3d · gaze3d_aux_loss(...)              [in-frame only, warmup]

Everything else follows train_videoattentiontarget.py (sample_rate, per-bucket
LR incl. inout_lr, VAT AUC/L2 eval, in/out AP).
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
from sklearn.metrics import average_precision_score
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
from train_videoattentiontarget import FocalLoss


# --------------------------------------------------------------------------
# Joint augmentations (RGB + depth must move together)
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


def _load_vat_frames(json_path, sample_rate):
    sequences = json.load(open(json_path, "r"))
    frames = []
    for seq in sequences:
        for j in range(0, len(seq["frames"]), sample_rate):
            frames.append(seq["frames"][j])
    return frames


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class VATDepthDataset(torch.utils.data.Dataset):
    """VAT with depth-aware heatmap target + per-image min-max DA2 depth map."""

    def __init__(self, path, split, transform, in_frame_only=False,
                 sample_rate=1, aug_groups=None, depth_dir="depth"):
        self.path = path
        self.depth_dir = depth_dir
        self.split = split
        self.is_train = (split == "train")
        self.aug = self.is_train
        self.transform = transform
        self.in_frame_only = in_frame_only
        self.aug_groups = aug_groups if aug_groups is not None else []

        json_path = os.path.join(path, f"{split}_preprocessed.json")
        self.data = _load_vat_frames(json_path, sample_rate)

        self.data_idxs = []
        for i, frame in enumerate(self.data):
            for j, head in enumerate(frame["heads"]):
                if not self.in_frame_only or head.get("inout", 1) == 1:
                    self.data_idxs.append((i, j))

    def _depth_path(self, image_rel_path):
        """Map image rel-path → cached DA2 .npy.

        ``preprocess_Depth`` writes::
            <data_root>/<depth_dir>/<relpath-within-image_dir>.npy
        i.e. the ``images/`` prefix from dataset JSON paths is *not*
        replicated under ``depth/``.
        """
        rel = image_rel_path.replace("\\", "/")
        if rel.startswith("images/"):
            rel = rel[len("images/"):]
        rel = os.path.splitext(rel)[0] + ".npy"
        primary = os.path.join(self.path, self.depth_dir, rel)
        if os.path.isfile(primary):
            return primary
        # Fallback: full mirror incl. images/ (older preprocess runs).
        alt = os.path.join(self.path, self.depth_dir,
                           os.path.splitext(image_rel_path.replace("\\", "/"))[0]
                           + ".npy")
        return alt if os.path.isfile(alt) else primary

    def _load_depth_pil(self, image_rel_path):
        path = self._depth_path(image_rel_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Depth map not found: {path!r}  "
                f"(image={image_rel_path!r}; run preprocess_Depth.py on this dataset)")
        return Image.fromarray(np.load(path).astype(np.float32), mode="F")

    def __getitem__(self, idx):
        img_idx, head_idx = self.data_idxs[idx]
        img_data = self.data[img_idx]
        head_data = copy.deepcopy(img_data["heads"][head_idx])
        inout = int(head_data.get("inout", 1))

        img_path = os.path.join(self.path, img_data["path"])
        img = Image.open(img_path).convert("RGB")
        depth_pil = self._load_depth_pil(img_data["path"])
        width, height = img.size

        bbox_norm  = list(head_data["bbox_norm"])
        gazex_norm = list(head_data["gazex_norm"])
        gazey_norm = list(head_data["gazey_norm"])

        if self.aug:
            bbox  = list(head_data["bbox"])
            gazex = list(head_data["gazex"])
            gazey = list(head_data["gazey"])

            if "crop" in self.aug_groups and np.random.sample() <= 0.5:
                img, depth_pil, bbox, gazex, gazey = _joint_random_crop(
                    img, depth_pil, bbox, gazex, gazey, inout)
            if "crop" in self.aug_groups and np.random.sample() <= 0.5:
                img, depth_pil, bbox, gazex, gazey = _joint_horiz_flip(
                    img, depth_pil, bbox, gazex, gazey, inout)
            if "crop" in self.aug_groups and np.random.sample() <= 0.5:
                bbox = utils.random_bbox_jitter(img, bbox)

            width, height = img.size
            bbox_norm  = [bbox[0] / width,  bbox[1] / height,
                          bbox[2] / width,  bbox[3] / height]
            gazex_norm = [x / float(width)  for x in gazex]
            gazey_norm = [y / float(height) for y in gazey]

            if "photometric" in self.aug_groups and np.random.sample() <= 0.5:
                photometric = T.Compose([
                    T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2,
                                                 saturation=0.2, hue=0.1)], p=0.5),
                    T.RandomGrayscale(p=0.2),
                    T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                    T.RandomAutocontrast(p=0.1),
                ])
                img = photometric(img)

        img_t = self.transform(img)

        d_arr_64 = np.asarray(depth_pil.resize((64, 64), Image.BILINEAR),
                              dtype=np.float32)
        d_min, d_max = float(d_arr_64.min()), float(d_arr_64.max())
        d_norm_64 = (d_arr_64 - d_min) / ((d_max - d_min) + 1e-8)
        depth_64_t = torch.from_numpy(d_norm_64).float()

        if self.is_train:
            if inout == 1:
                depth_full = np.asarray(depth_pil, dtype=np.float32)
                heatmap = get_depthaware_heatmap(
                    depth_full, gazex_norm[0], gazey_norm[0], 64, 64)
            else:
                heatmap = torch.zeros(64, 64)
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
# 3-D-proxy eval metrics
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
    gt_x, gt_y = float(gazex[0]), float(gazey[0])
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
    gt_x, gt_y = float(gazex[0]), float(gazey[0])
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

    wandb.init(project=cfg_m["name"], name="vat_depth_aux", config=config)

    checkpoint_dir = "_".join([
        "VAT",
        cfg_m["name"],
        cfg_m.get("moe_type", "vanilla"),
        str(cfg_m.get("is_msf", 1)),
        cfg_t["optimizer"],
        "bs" + str(cfg_t["batch_size"]),
        cfg_m["pbce_loss"],
        str(cfg_t["lr"]),
        "aux" + str(cfg_m.get("w_aux_3d", 0.1)),
    ])
    exp_dir = os.path.join(config["logging"]["log_dir"], checkpoint_dir)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Checkpoint dir: {exp_dir}")

    model, transform = get_gazemoe_model(config)
    print(f"Model: {cfg_m['name']} (plain GazeMoE; depth used only as aux loss)")

    for param in model.backbone.parameters():
        param.requires_grad = False
    print(f"Learnable parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    pretrained_path = cfg_m.get("pretrained_path", "")
    if pretrained_path and os.path.isfile(pretrained_path):
        print(f"Warm-starting from {pretrained_path}")
        sd = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model.load_gazelle_state_dict(sd, include_backbone=False)
    elif pretrained_path:
        print(f"WARN pretrained_path not found: {pretrained_path!r}")

    model.to(device)

    depth_dir = cfg_d.get("depth_dir", "depth")
    train_dataset = VATDepthDataset(
        cfg_d["train_path"], "train", transform,
        in_frame_only=False, sample_rate=6,
        aug_groups=cfg_d.get("augmentations", []), depth_dir=depth_dir)
    eval_dataset = VATDepthDataset(
        cfg_d["test_path"], "test", transform,
        in_frame_only=False, sample_rate=1, depth_dir=depth_dir)

    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg_t["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"].get("pin_memory", False))
    eval_dl = torch.utils.data.DataLoader(
        eval_dataset, batch_size=cfg_t["batch_size"], shuffle=False,
        collate_fn=collate_fn, num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"].get("pin_memory", False))

    lr       = float(cfg_t["lr"])
    fuse_lr  = float(cfg_t.get("fuse_lr",  lr))
    block_lr = float(cfg_t.get("block_lr", lr))
    inout_lr = float(cfg_t.get("inout_lr", lr))

    param_dicts = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ("ms_fusion" in n) or ("fuse" in n) or ("fusion" in n) or ("linear" in n):
            param_dicts.append({"params": p, "lr": fuse_lr})
        elif ("transformer" in n) or ("block" in n):
            param_dicts.append({"params": p, "lr": block_lr})
        elif "inout" in n:
            param_dicts.append({"params": p, "lr": inout_lr})
        else:
            param_dicts.append({"params": p, "lr": lr})

    opt_name = cfg_t["optimizer"]
    if opt_name == "Adam":
        optimizer = Adam(param_dicts)
    elif opt_name == "AdamW":
        optimizer = AdamW(param_dicts, weight_decay=float(cfg_t.get("weight_decay", 0.0)))
    else:
        raise TypeError(f"Optimizer not supported: {opt_name}")

    epochs = int(cfg_t["epochs"])
    sched_cfg = cfg_t.get("lr_scheduler", {})
    if sched_cfg.get("type") == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=int(sched_cfg.get("step_size", epochs)),
            eta_min=float(sched_cfg.get("min_lr", 1e-7)))
    else:
        scheduler = StepLR(
            optimizer, step_size=int(sched_cfg.get("step_size", epochs)),
            gamma=float(sched_cfg.get("gamma", 0.1)))

    if cfg_m["pbce_loss"] == "mse":
        SCALAR = 36
        heatmap_loss_fn = nn.MSELoss(reduction=cfg_m.get("reduction", "mean"))
    elif cfg_m["pbce_loss"] == "bce":
        SCALAR = 1
        heatmap_loss_fn = nn.BCELoss()
    else:
        raise TypeError(f"Heatmap loss not supported: {cfg_m['pbce_loss']!r}")

    if cfg_m.get("is_focal_loss", 0) == 1:
        inout_loss_fn = FocalLoss()
    else:
        inout_loss_fn = nn.BCELoss()

    w_aux_3d_target  = float(cfg_m.get("w_aux_3d", 0.1))
    aux_warmup_eps   = int  (cfg_m.get("aux_3d_warmup_epochs", 1))
    aux_depth_weight = float(cfg_m.get("aux_3d_depth_weight", 1.0))
    aux_temperature  = cfg_m.get("aux_3d_temperature", None)
    if aux_temperature is not None:
        aux_temperature = float(aux_temperature)
    grad_clip = float(cfg_t.get("gradient_clipping", 0.0))

    print(f"Auxiliary loss: gaze3d_aux  w_aux_3d={w_aux_3d_target}  "
          f"warmup={aux_warmup_eps} epoch(s)")

    best_l2 = float("inf")
    best_epoch = None

    for epoch in range(epochs):
        if aux_warmup_eps <= 0 or epoch >= aux_warmup_eps:
            w_aux_cur = w_aux_3d_target
        else:
            w_aux_cur = w_aux_3d_target * (epoch + 1) / float(aux_warmup_eps + 1)

        print(f"[Epoch {epoch}] w_aux_3d={w_aux_cur:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        model.train()
        sums = {"total": 0.0, "hm": 0.0, "inout": 0.0, "aux": 0.0}
        n_iters = len(train_dl)

        for cur_iter, batch in tqdm(enumerate(train_dl), total=n_iters,
                                     desc=f"Epoch {epoch + 1}/{epochs}"):
            (imgs, bboxes, gazex, gazey, inout, heights, widths,
             heatmaps, depth_64) = batch

            optimizer.zero_grad()
            preds = model({
                "images": imgs.to(device),
                "bboxes": [[bbox] for bbox in bboxes],
            })
            pred_hm = torch.stack(preds["heatmap"]).squeeze(dim=1)
            pred_inout = torch.stack(preds["inout"]).squeeze(dim=1)

            in_mask = inout.to(device).bool()
            heatmaps_dev = heatmaps.to(device)
            if in_mask.any():
                l_hm = SCALAR * heatmap_loss_fn(
                    pred_hm[in_mask], heatmaps_dev[in_mask])
            else:
                l_hm = torch.zeros((), device=device)

            l_inout = inout_loss_fn(pred_inout, inout.float().to(device))

            if w_aux_cur > 0 and in_mask.any():
                gt_gx = torch.tensor([g[0] for g in gazex],
                                     dtype=torch.float32, device=device)
                gt_gy = torch.tensor([g[0] for g in gazey],
                                     dtype=torch.float32, device=device)
                head_cx = torch.tensor([(b[0] + b[2]) / 2.0 for b in bboxes],
                                       dtype=torch.float32, device=device)
                head_cy = torch.tensor([(b[1] + b[3]) / 2.0 for b in bboxes],
                                       dtype=torch.float32, device=device)
                l_aux = gaze3d_aux_loss(
                    pred_hm, gt_gx, gt_gy, head_cx, head_cy,
                    depth_norm=depth_64.to(device),
                    mask=in_mask, depth_weight=aux_depth_weight,
                    temperature=aux_temperature,
                )
            else:
                l_aux = torch.zeros((), device=device)

            loss = l_hm + l_inout + w_aux_cur * l_aux
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            sums["total"] += float(loss.item())
            sums["hm"]    += float(l_hm.item())
            sums["inout"] += float(l_inout.item())
            sums["aux"]   += float(l_aux.item())

            if cur_iter % config["logging"]["save_every"] == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/heatmap_loss": l_hm.item(),
                    "train/inout_loss": l_inout.item(),
                    "train/aux_loss": l_aux.item(),
                    "train/w_aux_3d": w_aux_cur,
                })

        scheduler.step()
        ck_path = os.path.join(exp_dir, f"epoch_{epoch}.pt")
        torch.save(model.get_gazelle_state_dict(), ck_path)
        print(f"Saved checkpoint to {ck_path}")
        print(f"  Train means: total={sums['total']/n_iters:.4f}  "
              f"hm={sums['hm']/n_iters:.4f}  inout={sums['inout']/n_iters:.4f}  "
              f"aux={sums['aux']/n_iters:.4f}")

        # ---- EVAL ----
        model.eval()
        l2s, aucs = [], []
        l2_3d_all, angle_3d_all = [], []
        doa_correct, doa_valid = 0, 0
        all_inout_preds, all_inout_gts = [], []

        for batch in tqdm(eval_dl, total=len(eval_dl), desc=f"Eval {epoch}"):
            (imgs, bboxes, gazex, gazey, inout, heights, widths,
             depth_64) = batch
            with torch.no_grad():
                preds = model({
                    "images": imgs.to(device),
                    "bboxes": [[bbox] for bbox in bboxes],
                })
            pred_hm = torch.stack(preds["heatmap"]).squeeze(dim=1)
            pred_inout = torch.stack(preds["inout"]).squeeze(dim=1)

            for i in range(pred_hm.shape[0]):
                all_inout_preds.append(float(pred_inout[i].item()))
                all_inout_gts.append(int(inout[i].item()))

                if int(inout[i].item()) != 1:
                    continue
                hm_cpu = pred_hm[i].detach().cpu()
                aucs.append(vat_auc(hm_cpu, gazex[i][0], gazey[i][0]))
                l2s.append(vat_l2(hm_cpu, gazex[i][0], gazey[i][0]))

                d_np = depth_64[i].numpy()
                hm_np = hm_cpu.numpy()
                bbox = bboxes[i]
                hcx = (float(bbox[0]) + float(bbox[2])) * 0.5
                hcy = (float(bbox[1]) + float(bbox[3])) * 0.5
                l2_3d_all.append(l2_3d_proxy(hm_np, gazex[i], gazey[i], d_np))
                angle_3d_all.append(angle_3d_proxy(
                    hm_np, gazex[i], gazey[i], hcx, hcy, d_np))
                is_correct, is_valid = depth_order_accuracy_proxy(
                    hm_np, gazex[i], gazey[i], hcx, hcy, d_np)
                if is_valid:
                    doa_valid   += 1
                    doa_correct += int(bool(is_correct))

        epoch_auc = float(np.mean(aucs)) if aucs else 0.0
        epoch_l2  = float(np.mean(l2s))  if l2s  else 0.0
        L2_3D = float(np.mean(l2_3d_all))    if l2_3d_all    else 0.0
        A3D   = float(np.mean(angle_3d_all)) if angle_3d_all else 0.0
        DOA   = (100.0 * doa_correct / doa_valid) if doa_valid else 0.0
        epoch_ap = float(average_precision_score(all_inout_gts, all_inout_preds))

        wandb.log({
            "eval/auc": epoch_auc, "eval/l2": epoch_l2,
            "eval/l2_3d": L2_3D, "eval/angle_3d": A3D,
            "eval/doa_percent": DOA, "eval/inout_ap": epoch_ap,
            "epoch": epoch,
        })
        print(f"EVAL EPOCH {epoch}: AUC={epoch_auc:.4f}  L2={epoch_l2:.4f}  "
              f"AP={epoch_ap:.4f}  |  L2-3D(proxy)={L2_3D:.4f}  "
              f"Angle-3D(proxy)={A3D:.2f}°  DOA={DOA:.2f}%")

        if epoch_l2 < best_l2:
            best_l2 = epoch_l2
            best_epoch = epoch

    print(f"Completed. Best L2={round(best_l2, 4)} at epoch {best_epoch}.")
    wandb.finish()


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
