"""
train_depth_gazefollow.py — depth-aware gaze pretraining on GazeFollow.

Mirrors ``train_depth_goo.py`` (same anchored ``GT3D`` model, same
scale-invariant log-ratio losses, same anchored evaluation metrics) but
adapted for the GazeFollow corpus stored at ``./gazefollow_extended``:

    gazefollow_extended/
    ├── train/<scene>/<id>.jpg
    ├── test2/<scene>/<id>.jpg            (or test/, depending on extraction)
    ├── train_preprocessed.json
    ├── test_preprocessed.json
    └── depth/<same/relative/path>.npy    ← from preprocess_Depth.py

Differences from train_depth_goo.py:
    * JSON layout follows train_gazefollow.py (``data_idxs`` over (img, head))
    * Multi-annotator test annotations — eval uses GazeFollow-style metrics
      (``gazefollow_auc`` / ``gazefollow_l2``); anchored depth GT at test
      time is sampled at the *mean* annotator location.
    * Photometric augmentation pipeline borrowed from train_gazefollow.py.
    * Re-uses every loss / metric / augmentation helper from
      train_depth_goo.py via ordinary import.

Run:
    python preprocess_Depth.py --data_path ./gazefollow_extended --image_dir .
    python train_depth_gazefollow.py
"""

import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, StepLR
from tqdm import tqdm
import wandb
import yaml

from eval import gazefollow_auc, gazefollow_l2
from network.network_builder_gt3d import get_gt3d_model
import network.utils as utils
from network.utils import get_heatmap

# Re-use the depth-aware machinery defined in train_depth_goo.py — same
# anchored losses, metrics, joint augmentations, FocalLoss, constants.
from train_depth_goo import (
    DEPTH_EPS,
    DEPTH_HUBER_DELTA,
    LOSS_SCALAR,
    FocalLoss,
    sample_anchored_depth_gt,
    _bbox_jitter,
    _joint_horiz_flip,
    _joint_random_crop,
    anchored_delta1,
    anchored_l2_3d,
    anchored_log_huber_loss,
    anchored_si_log_loss,
    gaze3d_angle_relative,
    ratio_mae,
    ratio_rmse,
)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class GazeFollowDepth(torch.utils.data.Dataset):
    """GazeFollow with anchored depth supervision.

    Train: one gaze annotation per sample.
    Test : multiple annotator gaze points per image.  AUC and L2 are
           computed against all annotators (handled by ``gazefollow_*``);
           the anchored depth target is sampled at the *mean* annotator
           pixel — a stable single-point reference for the log-ratio.

    Preprocessed JSON schema follows ``train_gazefollow.py``::

        [{path, heads: [{bbox, bbox_norm, gazex, gazey,
                         gazex_norm, gazey_norm, inout}]}, ...]
    """

    def __init__(self, data_path, transform, split="train",
                 depth_dir="depth", in_frame_only=True, aug_groups=None):
        self.path          = data_path
        self.split         = split
        self.transform     = transform
        self.depth_dir     = depth_dir
        self.in_frame_only = in_frame_only
        self.is_train      = (split == "train")
        self.aug_groups    = aug_groups if aug_groups is not None else []

        with open(os.path.join(data_path, f"{split}_preprocessed.json"), "r") as f:
            self.data = json.load(f)

        self.data_idxs = []
        for i, frame in enumerate(self.data):
            for j, head in enumerate(frame["heads"]):
                if not in_frame_only or head["inout"] == 1:
                    self.data_idxs.append((i, j))

        # Photometric pipeline from train_gazefollow.py — RGB-only, never
        # touches the depth map (whose values would be corrupted by it).
        self._photo = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2,
                                         saturation=0.2, hue=0.1)], p=0.5),
            T.RandomGrayscale(p=0.2),
            T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
            T.RandomAutocontrast(p=0.1),
        ])

    def __len__(self):
        return len(self.data_idxs)

    # --------------------------------------------------------------------
    def _depth_path(self, image_rel_path):
        """Map ``train/0001/0001.jpg`` → ``depth/train/0001/0001.npy``."""
        rel = image_rel_path.replace("images" + os.sep,
                                     self.depth_dir + os.sep, 1)
        if rel == image_rel_path:                     # no "images/" segment
            rel = os.path.join(self.depth_dir, image_rel_path)
        rel = os.path.splitext(rel)[0] + ".npy"
        return os.path.join(self.path, rel)

    def _load_depth_pil(self, image_rel_path):
        depth = np.load(self._depth_path(image_rel_path)).astype(np.float32)
        return Image.fromarray(depth, mode="F")

    @staticmethod
    def _sample_anchored(depth_pil, gazex_norm, gazey_norm, bbox_norm):
        return sample_anchored_depth_gt(depth_pil, gazex_norm, gazey_norm, bbox_norm)

    # --------------------------------------------------------------------
    def __getitem__(self, idx):
        img_idx, head_idx = self.data_idxs[idx]
        img_data  = self.data[img_idx]
        head_data = copy.deepcopy(img_data["heads"][head_idx])

        bbox_norm  = head_data["bbox_norm"]
        gazex_norm = list(head_data["gazex_norm"])
        gazey_norm = list(head_data["gazey_norm"])
        inout      = head_data["inout"]

        img = Image.open(os.path.join(self.path, img_data["path"])).convert("RGB")
        depth = self._load_depth_pil(img_data["path"])
        width, height = img.size

        if self.is_train:
            # Spatial augmentations operate on absolute-pixel coords; the
            # joint helpers also crop / flip the depth map identically.
            bbox  = list(head_data["bbox"])
            gazex = list(head_data["gazex"])
            gazey = list(head_data["gazey"])

            if "crop" in self.aug_groups and np.random.sample() <= 0.5:
                img, depth, bbox, gazex, gazey = _joint_random_crop(
                    img, depth, bbox, gazex, gazey, inout)
            if "crop" in self.aug_groups and np.random.sample() <= 0.5:
                img, depth, bbox, gazex, gazey = _joint_horiz_flip(
                    img, depth, bbox, gazex, gazey, inout)
            if "crop" in self.aug_groups and np.random.sample() <= 0.5:
                bbox = _bbox_jitter(img, bbox)

            width, height = img.size
            bbox_norm  = [bbox[0] / width,  bbox[1] / height,
                          bbox[2] / width,  bbox[3] / height]
            gazex_norm = [x / float(width)  for x in gazex]
            gazey_norm = [y / float(height) for y in gazey]

            if "photometric" in self.aug_groups and np.random.sample() <= 0.5:
                img = self._photo(img)

        img_t = self.transform(img)

        # Anchored depth GT — train: single annotation; test: mean.
        if self.is_train:
            gx_for_depth = gazex_norm[0]
            gy_for_depth = gazey_norm[0]
        else:
            gx_for_depth = float(np.mean(gazex_norm))
            gy_for_depth = float(np.mean(gazey_norm))
        gt_z_gaze, gt_z_head = self._sample_anchored(
            depth, gx_for_depth, gy_for_depth, bbox_norm)

        if self.is_train:
            heatmap = get_heatmap(gazex_norm[0], gazey_norm[0], 64, 64)
            return (img_t, bbox_norm, gazex_norm, gazey_norm,
                    torch.tensor(inout), height, width,
                    heatmap, gt_z_gaze, gt_z_head)
        return (img_t, bbox_norm, gazex_norm, gazey_norm,
                torch.tensor(inout), height, width,
                gt_z_gaze, gt_z_head)


def collate_fn(batch):
    """Stack tensors, pass through Python objects (mirrors train_gazefollow)."""
    transposed = list(zip(*batch))
    return tuple(
        torch.stack(items) if isinstance(items[0], torch.Tensor) else list(items)
        for items in transposed
    )


# --------------------------------------------------------------------------
# Evaluation — GazeFollow-style heatmap metrics + scale-invariant depth
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    aucs, avg_l2s, min_l2s = [], [], []
    pred_zg_all, pred_zh_all = [], []
    gt_zg_all,   gt_zh_all   = [], []
    l2_3d_all, angle_3d_all  = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        (images, bboxes, gazex, gazey, inout, heights, widths,
         gt_zg, gt_zh) = batch
        preds = model({"images": images.to(device),
                       "bboxes": [[bbox] for bbox in bboxes]})

        # Each image has exactly 1 head, so [N_i=1, 64, 64] per image.
        pred_zg = torch.cat(preds["depth_gaze"], 0).detach().cpu()   # [B]
        pred_zh = torch.cat(preds["depth_head"], 0).detach().cpu()   # [B]

        pred_zg_all.append(pred_zg)
        pred_zh_all.append(pred_zh)
        gt_zg_t = torch.as_tensor(gt_zg, dtype=torch.float32)
        gt_zh_t = torch.as_tensor(gt_zh, dtype=torch.float32)
        gt_zg_all.append(gt_zg_t)
        gt_zh_all.append(gt_zh_t)

        for j in range(images.shape[0]):
            heatmap_j = preds["heatmap"][j][0].detach().cpu()        # [64, 64]
            auc       = gazefollow_auc(
                heatmap_j, gazex[j], gazey[j], heights[j], widths[j])
            avg_l2, min_l2 = gazefollow_l2(heatmap_j, gazex[j], gazey[j])
            aucs.append(auc)
            avg_l2s.append(avg_l2)
            min_l2s.append(min_l2)

            # Anchored L2-3D / Angle-3D both use the mean annotator location
            # (matches the depth GT sampled at __getitem__).
            gx_mean = float(np.mean(gazex[j]))
            gy_mean = float(np.mean(gazey[j]))
            l2_3d_all.append(anchored_l2_3d(
                heatmap_j, pred_zg[j], pred_zh[j],
                gx_mean, gy_mean,
                gt_zg_t[j].item(), gt_zh_t[j].item()))

            bbox = bboxes[j]
            head_cx = (float(bbox[0]) + float(bbox[2])) * 0.5
            head_cy = (float(bbox[1]) + float(bbox[3])) * 0.5
            angle_3d_all.append(gaze3d_angle_relative(
                heatmap_j, pred_zg[j], pred_zh[j],
                gx_mean, gy_mean,
                gt_zg_t[j].item(), gt_zh_t[j].item(),
                head_cx, head_cy,
            ))

    pred_zg_t = torch.cat(pred_zg_all)
    pred_zh_t = torch.cat(pred_zh_all)
    gt_zg_t   = torch.cat(gt_zg_all)
    gt_zh_t   = torch.cat(gt_zh_all)

    AUC     = float(np.mean(aucs))
    AvgL2   = float(np.mean(avg_l2s))
    MinL2   = float(np.mean(min_l2s))
    R_MAE   = float(ratio_mae (pred_zg_t, pred_zh_t, gt_zg_t, gt_zh_t).item())
    R_RMSE  = float(ratio_rmse(pred_zg_t, pred_zh_t, gt_zg_t, gt_zh_t).item())
    D1      = float(anchored_delta1(pred_zg_t, pred_zh_t, gt_zg_t, gt_zh_t).item())
    L2_3D   = float(np.mean(l2_3d_all))    if l2_3d_all    else 0.0
    A3D     = float(np.mean(angle_3d_all)) if angle_3d_all else 0.0

    print(f"  AUC={AUC:.4f}  AvgL2={AvgL2:.4f}  MinL2={MinL2:.4f}  |  "
          f"RatioMAE={R_MAE:.4f}  RatioRMSE={R_RMSE:.4f}  "
          f"δ1={D1:.4f}  L2-3D={L2_3D:.4f}  Angle-3D={A3D:.2f}°")
    return {
        "AUC":         AUC,
        "AvgL2":       AvgL2,
        "MinL2":       MinL2,
        "ratio_mae":   R_MAE,
        "ratio_rmse":  R_RMSE,
        "delta1":      D1,
        "l2_3d":       L2_3D,
        "angle_3d":    A3D,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    with open("configuration.yaml", "r") as f:
        config = yaml.safe_load(f)

    device       = config["hardware"]["device"] if torch.cuda.is_available() else "cpu"
    train_path   = config["data"].get("pre_train_path", "./gazefollow_extended")
    test_path    = config["data"].get("pre_test_path",  train_path)
    depth_dir    = config["data"].get("depth_dir", "depth")
    aug_groups   = config["data"].get("augmentations", ["crop", "photometric"])
    print(f"Device: {device}")
    print(f"Train: {train_path}   Test: {test_path}   Depth dir: {depth_dir}")

    wandb.init(
        project=config["model"]["name"],
        name="train_depth_gazefollow",
        config=config,
    )

    # ---- Loss weights ------------------------------------------------
    cfg_model = config["model"]
    w_heatmap = float(cfg_model.get("mse_weight",   1.0))
    w_inout   = float(cfg_model.get("bce_weight",   0.0))
    w_depth   = float(cfg_model.get("depth_weight", 0.5))
    print(f"Loss weights: heatmap={w_heatmap}  inout={w_inout}  depth={w_depth}")

    # ---- Model -------------------------------------------------------
    model, _ = get_gt3d_model(config)
    print(f"Model: {cfg_model['name']}")

    pretrained = cfg_model.get("pretrained_path", "")
    if pretrained and os.path.isfile(pretrained):
        print(f"Loading warm-start weights from {pretrained}")
        model.load_gt3d_state_dict(
            torch.load(pretrained, map_location=device, weights_only=True))

    for name, param in model.named_parameters():
        param.requires_grad = "backbone" not in name

    if not pretrained:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() > 1:
                torch.nn.init.xavier_normal_(param)
            else:
                torch.nn.init.zeros_(param)

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ---- Optimizer (uses pre_* hyperparameters per train_gazefollow) -
    # 4-bucket LR dispatch driven by canonical substrings in module names.
    # By design every trainable parameter lands in exactly one bucket; the
    # base ``pre_lr`` arm should never be reached (kept as a safety net only):
    #   "inout"  → inout_lr   (inout_token, inout_head)
    #   "depth"  → depth_lr   (depth_token_*, depth_scalar_head, depth_to_feat,
    #                          dense_depth_decoder, depth_heatmap_head,
    #                          depth_refined_heatmap_head)
    #   "block"  → block_lr   (trunk_blocks, refine_blocks)
    #   "fuse"   → fuse_lr    (fuse_proj, fuse_head_token)
    pre_lr       = float(config["train"].get("pre_lr",       config["train"]["lr"]))
    pre_fuse_lr  = float(config["train"].get("pre_fuse_lr",  pre_lr))
    pre_block_lr = float(config["train"].get("pre_block_lr", pre_lr))
    pre_inout_lr = float(config["train"].get("inout_lr",     pre_lr))
    pre_depth_lr = float(config["train"].get("depth_lr",     pre_lr))

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
        elif "fuse" in n:
            lr = pre_fuse_lr
        else:
            lr = pre_lr
        param_dicts.append({"params": p, "lr": lr})

    opt_name = config["train"].get("pre_optimizer", config["train"]["optimizer"])
    wd       = float(config["train"]["weight_decay"])
    if opt_name == "Adam":
        optimizer = Adam(param_dicts, weight_decay=wd)
    elif opt_name == "AdamW":
        optimizer = AdamW(param_dicts, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    # ---- Scheduler ---------------------------------------------------
    sched = config["train"].get("pre_lr_scheduler", config["train"]["lr_scheduler"])
    if sched["type"] == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=sched["step_size"], eta_min=float(sched["min_lr"]))
    elif sched["type"] == "warmup":
        ws = sched["step_size"]
        scheduler = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, e / ws))
    else:
        scheduler = StepLR(
            optimizer, step_size=sched["step_size"], gamma=sched["gamma"])

    # ---- Transforms (Resize before ToTensor, like train_depth_goo) ---
    res = config["data"]["input_resolution"]
    img_transform = T.Compose([
        T.Resize((res, res)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.Resize((res, res)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ---- Dataloaders -------------------------------------------------
    train_dataset = GazeFollowDepth(
        train_path, img_transform, split="train",
        depth_dir=depth_dir, in_frame_only=True, aug_groups=aug_groups)
    test_dataset  = GazeFollowDepth(
        test_path,  val_transform, split="test",
        depth_dir=depth_dir, in_frame_only=True)

    pre_bs = int(config["train"].get("pre_batch_size",
                                     config["train"]["batch_size"]))
    eval_bs = int(config["eval"]["batch_size"])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=pre_bs,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=eval_bs,
        collate_fn=collate_fn,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
    )

    # ---- Loss functions ----------------------------------------------
    inout_loss_fn   = (FocalLoss() if cfg_model.get("is_focal_loss", 0) == 1
                       else nn.BCELoss())
    heatmap_loss_fn = (nn.MSELoss() if cfg_model.get("pbce_loss") == "mse"
                       else nn.BCELoss())

    depth_loss_name = str(cfg_model.get("depth_loss", "log_huber")).lower()
    lambda_si       = float(cfg_model.get("lambda_si", 0.5))
    print(f"Depth loss: {depth_loss_name}  (lambda_si={lambda_si} if si_log)")

    # ---- Checkpoint dir ----------------------------------------------
    pre_dir = config["logging"].get("pre_dir", "results/pretrain_gf_depth/")
    ckpt_dir = os.path.join(
        pre_dir,
        "_".join([
            "depthgf",
            cfg_model["name"].replace("/", "_"),
            opt_name,
            "bs"  + str(pre_bs),
            "lr"  + str(pre_lr),
            "hw"  + str(w_heatmap),
            "iw"  + str(w_inout),
            "dw"  + str(w_depth),
            depth_loss_name,
        ]),
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints → {ckpt_dir}")

    # ---- Training loop -----------------------------------------------
    pre_epochs = int(config["train"].get("pre_epochs",
                                         config["train"]["epochs"]))
    save_every = int(config["logging"]["save_every"])
    log_every  = int(config["logging"].get("log_every", save_every))
    best_l2    = float("inf")
    best_path  = None

    for epoch in range(pre_epochs):
        model.train()
        sums = {"total": 0.0, "hm": 0.0, "io": 0.0, "depth": 0.0}

        for cur_iter, batch in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{pre_epochs}")):
            (images, bboxes, gazex, gazey, inout, _heights, _widths,
             gt_hm, gt_zg, gt_zh) = batch

            preds = model({"images": images.to(device),
                           "bboxes": [[bbox] for bbox in bboxes]})
            pred_hm = torch.cat(preds["heatmap"],    0)             # [B, 64, 64]
            pred_zg = torch.cat(preds["depth_gaze"], 0)             # [B]
            pred_zh = torch.cat(preds["depth_head"], 0)             # [B]
            pred_inouts = (torch.cat(preds["inout"], 0)
                           if preds["inout"] is not None else None)

            gt_hm_d = gt_hm.to(device)
            gt_zg_d = torch.as_tensor(gt_zg, dtype=torch.float32, device=device)
            gt_zh_d = torch.as_tensor(gt_zh, dtype=torch.float32, device=device)
            gt_io_d = torch.as_tensor(
                [int(io) for io in inout], dtype=torch.float32, device=device)

            l_hm = heatmap_loss_fn(pred_hm, gt_hm_d) * LOSS_SCALAR

            if pred_inouts is not None and w_inout > 0:
                l_io = inout_loss_fn(pred_inouts, gt_io_d)
            else:
                l_io = torch.zeros((), device=device)

            if depth_loss_name == "si_log":
                l_depth = anchored_si_log_loss(
                    pred_zg, pred_zh, gt_zg_d, gt_zh_d, lambda_si=lambda_si)
            else:
                l_depth = anchored_log_huber_loss(
                    pred_zg, pred_zh, gt_zg_d, gt_zh_d)

            total = w_heatmap * l_hm + w_inout * l_io + w_depth * l_depth

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            sums["total"] += total.item()
            sums["hm"]    += l_hm.item()
            sums["io"]    += float(l_io.item())
            sums["depth"] += l_depth.item()

            if cur_iter % log_every == 0:
                wandb.log({
                    "train/loss":         total.item(),
                    "train/heatmap_loss": l_hm.item(),
                    "train/inout_loss":   float(l_io.item()),
                    "train/depth_loss":   l_depth.item(),
                })

        scheduler.step()
        n = len(train_loader)
        print(f"Epoch [{epoch + 1}/{pre_epochs}]  "
              f"Total={sums['total']/n:.6f}  "
              f"HM={sums['hm']/n:.6f}  "
              f"IO={sums['io']/n:.6f}  "
              f"Depth={sums['depth']/n:.6f}")

        metrics = evaluate(model, test_loader, device)
        wandb.log({
            "eval/auc":        metrics["AUC"],
            "eval/avg_l2":     metrics["AvgL2"],
            "eval/min_l2":     metrics["MinL2"],
            "eval/ratio_mae":  metrics["ratio_mae"],
            "eval/ratio_rmse": metrics["ratio_rmse"],
            "eval/delta1":     metrics["delta1"],
            "eval/l2_3d":      metrics["l2_3d"],
            "eval/angle_3d":   metrics["angle_3d"],
            "epoch":           epoch,
        })

        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler":            scheduler.state_dict(),
                "metrics":              metrics,
            }, ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

        if metrics["MinL2"] < best_l2 and epoch >= save_every:
            best_l2   = metrics["MinL2"]
            best_path = os.path.join(ckpt_dir, f"best_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler":            scheduler.state_dict(),
                "metrics":              metrics,
            }, best_path)
            print(f"  Best model → epoch {epoch + 1}  MinL2={best_l2:.4f}")

    if best_path is not None:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        with torch.no_grad():
            metrics = evaluate(model, test_loader, device)
        final_path = os.path.join(
            ckpt_dir,
            f"Best_ep{ckpt['epoch']}_minL2{int(metrics['MinL2']*100)}"
            f"_auc{int(metrics['AUC']*100)}_d1{int(metrics['delta1']*100)}.pt",
        )
        torch.save({
            "model_state_dict": model.state_dict(),
            "metrics":          metrics,
        }, final_path)
        print(f"Final best model → {final_path}")

    wandb.finish()


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
