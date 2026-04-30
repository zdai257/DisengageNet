"""
train_depth_goo.py — depth-aware gaze pretraining on GOOSynthV3.

Extends the train_goo.py pipeline with an additional depth-prediction head
(GazeLLE3D) trained against DepthAnythingV2 pseudo-labels cached under
``<goo_path>/depth/`` by preprocess_Depth.py.

Loss = L_heatmap (BCE)
     + λ_io · L_inout (BCE / Focal)
     + λ_d  · L_depth (Gaussian-masked log-Huber)

The depth loss is *spatially weighted by the GT heatmap* (sigma=3 Gaussian
around the gaze target).  Pixels far from the target therefore contribute
~0 to the loss, so the depth head is forced to learn

    "what is the depth value at and immediately around the gaze target",

instead of the much easier (and uninformative for this task) global
pixel-by-pixel depth-map reconstruction.
"""

import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
from tqdm import tqdm
import yaml

from eval import eval_metrics, average_precision_score, vat_auc, vat_l2
from network.network_builder_3d import get_gazelle3d_model
import network.utils as utils
from network.utils import get_heatmap


# --------------------------------------------------------------------------
# Constants & defaults
# --------------------------------------------------------------------------

LOSS_SCALAR        = 1.0
DEPTH_EPS          = 1e-4    # numerical floor for log()
DEPTH_HUBER_DELTA  = 0.1     # transition between quadratic and linear regions


# --------------------------------------------------------------------------
# Joint augmentation helpers (RGB + depth must move together)
# --------------------------------------------------------------------------

def _joint_random_crop(image, depth, bbox, gazex, gazey, inout):
    """Mirror of utils.random_crop but cropping the depth map identically."""
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


def _bbox_jitter(image, bbox, jitter=0.2):
    width, height = image.size
    xmin, ymin, xmax, ymax = bbox
    xmin_j = (np.random.random_sample() * 2 * jitter - jitter) * (xmax - xmin)
    xmax_j = (np.random.random_sample() * 2 * jitter - jitter) * (xmax - xmin)
    ymin_j = (np.random.random_sample() * 2 * jitter - jitter) * (ymax - ymin)
    ymax_j = (np.random.random_sample() * 2 * jitter - jitter) * (ymax - ymin)
    return [
        max(0,      xmin + xmin_j),
        max(0,      ymin + ymin_j),
        min(width,  xmax + xmax_j),
        min(height, ymax + ymax_j),
    ]


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class GOOSynthDepth(torch.utils.data.Dataset):
    """
    Reads goosynth_{split}_preprocess.json and the matching DepthAnythingV2
    depth maps cached at ``<data_path>/depth/<split>/<image_id>.npy``.

    Anchored depth supervision: from each cached depth map we sample only
    *two* scalars after augmentation — the value at the gaze-target pixel
    (``gt_z_gaze``) and at the head bbox centre (``gt_z_head``).  Both are
    in the same per-image min-max-normalised [0, 1] space, so their
    log-ratio is a scale-invariant supervisable quantity.

    Each item (train):
        image     : [3, H, W] tensor (Resize→ToTensor→Normalize)
        bbox_norm : [[x1,y1,x2,y2]] in [0, 1]
        gazex/y   : [[x]], [[y]] in [0, 1]
        inout     : [int]
        heatmap   : [64, 64]   sigma=3 Gaussian GT
        gt_z_gaze : float in [0, 1]
        gt_z_head : float in [0, 1]

    Each item (test): same minus the heatmap (which is recomputed by the
    eval pipeline against multi-annotator GT when applicable).
    """

    def __init__(self, data_path, img_transform, split="train", depth_dir="depth"):
        json_path = os.path.join(data_path, f"goosynth_{split}_preprocess.json")
        self.frames    = json.load(open(json_path, "rb"))
        self.data_path = data_path
        self.depth_dir = depth_dir
        self.transform = img_transform
        self.split     = split
        self.is_train  = (split == "train")

    def __len__(self):
        return len(self.frames)

    def _depth_path(self, image_rel_path):
        """Map images/<split>/<id>.jpg → <depth_dir>/<split>/<id>.npy."""
        rel = image_rel_path.replace("images" + os.sep, self.depth_dir + os.sep, 1)
        if rel == image_rel_path:                      # no "images/" segment found
            rel = os.path.join(self.depth_dir, image_rel_path)
        rel = os.path.splitext(rel)[0] + ".npy"
        return os.path.join(self.data_path, rel)

    def _load_depth_pil(self, image_rel_path):
        """Load the cached .npy depth and wrap it as a single-channel float PIL."""
        path = self._depth_path(image_rel_path)
        depth = np.load(path).astype(np.float32)        # H × W (any size)
        return Image.fromarray(depth, mode="F")         # PIL float image

    @staticmethod
    def _sample_anchored(depth_pil, gazex_norm, gazey_norm, bbox_norm):
        """Sample two scalars from a per-image-normalised 64×64 depth map.

        Returns (gt_z_gaze, gt_z_head).  Both lie in [0, 1] under the same
        per-image min-max normalisation, so their log-ratio is invariant
        to per-image rescaling.
        """
        depth_64 = depth_pil.resize((64, 64), Image.BILINEAR)
        d_arr = np.asarray(depth_64, dtype=np.float32)
        d_min, d_max = float(d_arr.min()), float(d_arr.max())
        d_norm = (d_arr - d_min) / (d_max - d_min + 1e-8)

        u_g = int(np.clip(round(gazex_norm * 63), 0, 63))
        v_g = int(np.clip(round(gazey_norm * 63), 0, 63))
        gt_z_gaze = float(d_norm[v_g, u_g])

        hcx = (bbox_norm[0] + bbox_norm[2]) * 0.5
        hcy = (bbox_norm[1] + bbox_norm[3]) * 0.5
        u_h = int(np.clip(round(hcx * 63), 0, 63))
        v_h = int(np.clip(round(hcy * 63), 0, 63))
        gt_z_head = float(d_norm[v_h, u_h])
        return gt_z_gaze, gt_z_head

    def __getitem__(self, idx):
        frame = self.frames[idx]
        image = Image.open(
            os.path.join(self.data_path, frame["path"])).convert("RGB")
        depth = self._load_depth_pil(frame["path"])

        head  = frame["heads"][0]
        bbox  = list(head["bbox"])
        gazex = list(head["gazex"])
        gazey = list(head["gazey"])
        inout = head["inout"]

        if self.is_train:
            if np.random.sample() <= 0.5:
                image, depth, bbox, gazex, gazey = _joint_random_crop(
                    image, depth, bbox, gazex, gazey, inout)
            if np.random.sample() <= 0.5:
                image, depth, bbox, gazex, gazey = _joint_horiz_flip(
                    image, depth, bbox, gazex, gazey, inout)
            if np.random.sample() <= 0.5:
                bbox = _bbox_jitter(image, bbox)

        width, height = image.size
        bbox_norm  = [bbox[0] / width,  bbox[1] / height,
                      bbox[2] / width,  bbox[3] / height]
        gazex_norm = [x / float(width)  for x in gazex]
        gazey_norm = [y / float(height) for y in gazey]

        image_t = self.transform(image)

        # Anchored depth GT — same in train and eval, post-augmentation.
        gt_z_gaze, gt_z_head = self._sample_anchored(
            depth, gazex_norm[0], gazey_norm[0], bbox_norm)

        if not self.is_train:
            return (image_t, [bbox_norm], [gazex_norm], [gazey_norm], [inout],
                    gt_z_gaze, gt_z_head)

        gt_heatmap = get_heatmap(gazex_norm[0], gazey_norm[0], 64, 64)
        return (image_t, [bbox_norm], [gazex_norm], [gazey_norm], [inout],
                gt_heatmap, gt_z_gaze, gt_z_head)


def collate_train(batch):
    images, bboxes, gazex, gazey, inout, heatmaps, z_gaze, z_head = zip(*batch)
    return (torch.stack(images),
            list(bboxes), list(gazex), list(gazey), list(inout),
            torch.stack(heatmaps),
            torch.tensor(z_gaze, dtype=torch.float32),
            torch.tensor(z_head, dtype=torch.float32))


def collate(batch):
    images, bboxes, gazex, gazey, inout, z_gaze, z_head = zip(*batch)
    return (torch.stack(images),
            list(bboxes), list(gazex), list(gazey), list(inout),
            torch.tensor(z_gaze, dtype=torch.float32),
            torch.tensor(z_head, dtype=torch.float32))


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Class-balancing focal loss (alpha=0.02/0.98 default for GOO inout)."""
    def __init__(self, alpha=0.02 / 0.98, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt  = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# --------------------------------------------------------------------------
# Anchored depth losses — head-anchored log-ratio supervision (Fix 2).
# --------------------------------------------------------------------------
#
# Both losses operate on the scale-invariant quantity
#     log(z_gaze) − log(z_head)
# so the per-image min-max normalisation of the DepthAnythingV2 pseudo-label
# cancels exactly in both prediction and ground truth.  The model's two
# scalar outputs are individually unidentifiable; only their log-ratio is
# meaningful and supervised.
# --------------------------------------------------------------------------

def _safe_log_ratio(z_gaze, z_head, eps=DEPTH_EPS):
    """log(z_gaze) − log(z_head) with both terms clamped above ``eps``."""
    return (torch.log(z_gaze.clamp(min=eps))
            - torch.log(z_head.clamp(min=eps)))


def anchored_log_huber_loss(pred_z_gaze, pred_z_head, gt_z_gaze, gt_z_head,
                            delta=DEPTH_HUBER_DELTA, eps=DEPTH_EPS):
    """Log-Huber loss on the scale-invariant log-ratio.

    Inputs are all 1-D tensors of length N (per-sample scalars).  Returns
    a 0-D tensor (mean over the batch).
    """
    pred = _safe_log_ratio(pred_z_gaze, pred_z_head, eps)
    gt   = _safe_log_ratio(gt_z_gaze,   gt_z_head,   eps)
    diff   = pred - gt
    abs_d  = diff.abs()
    quad   = 0.5 * diff * diff
    linear = delta * (abs_d - 0.5 * delta)
    return torch.where(abs_d < delta, quad, linear).mean()


def anchored_si_log_loss(pred_z_gaze, pred_z_head, gt_z_gaze, gt_z_head,
                         lambda_si=0.5, eps=DEPTH_EPS):
    """Scale-invariant log loss (Eigen et al., 2014) on the log-ratio.

    L = mean(d²) − λ · mean(d)²,  where d = pred_log_ratio − gt_log_ratio.

    λ = 0.0 → plain log-MSE on the ratio (no extra scale subtraction)
    λ = 1.0 → fully scale-invariant — pure variance of d across the batch
    λ = 0.5 → Eigen's recommended balance.

    Notice that even for λ = 0 this loss is already scale-invariant, because
    the supervision target *is* the log-ratio.  The λ-term provides
    additional batch-level invariance that can absorb residual systematic
    shifts in the per-image normalisation of GT pseudo-labels (e.g. due to
    DA2 outliers near image borders).
    """
    pred = _safe_log_ratio(pred_z_gaze, pred_z_head, eps)
    gt   = _safe_log_ratio(gt_z_gaze,   gt_z_head,   eps)
    d    = pred - gt
    return (d * d).mean() - float(lambda_si) * d.mean() ** 2


# --------------------------------------------------------------------------
# Anchored evaluation metrics — every metric below is scale-invariant by
# construction (operates on the log-ratio rather than absolute depth).
# --------------------------------------------------------------------------

def anchored_log_ratio(z_gaze, z_head, eps=DEPTH_EPS):
    """Scale-free supervisable quantity: log(z_gaze) − log(z_head)."""
    return _safe_log_ratio(z_gaze, z_head, eps)


def ratio_mae(pred_z_gaze, pred_z_head, gt_z_gaze, gt_z_head, eps=DEPTH_EPS):
    """Mean absolute error of the log-ratio across the batch."""
    pred = _safe_log_ratio(pred_z_gaze, pred_z_head, eps)
    gt   = _safe_log_ratio(gt_z_gaze,   gt_z_head,   eps)
    return (pred - gt).abs().mean()


def ratio_rmse(pred_z_gaze, pred_z_head, gt_z_gaze, gt_z_head, eps=DEPTH_EPS):
    """RMSE of the log-ratio across the batch.

    Equivalent to RMSE-log of (z_gaze / z_head), which is the canonical
    headline depth metric in head-anchored coordinates.
    """
    pred = _safe_log_ratio(pred_z_gaze, pred_z_head, eps)
    gt   = _safe_log_ratio(gt_z_gaze,   gt_z_head,   eps)
    d    = pred - gt
    return torch.sqrt((d * d).mean())


def anchored_delta1(pred_z_gaze, pred_z_head, gt_z_gaze, gt_z_head,
                    threshold=1.25, eps=DEPTH_EPS):
    """δ₁ accuracy on the head-anchored ratio.

    Fraction of samples where the predicted ratio differs from the GT ratio
    by less than a factor of ``threshold`` (default 1.25 → ±25 %).
    """
    diff = _safe_log_ratio(pred_z_gaze, pred_z_head, eps) \
         - _safe_log_ratio(gt_z_gaze,   gt_z_head,   eps)
    correct = (torch.exp(diff.abs()) < float(threshold)).float()
    return correct.mean()


def anchored_l2_3d(pred_heatmap, pred_z_gaze, pred_z_head,
                   gt_gazex_norm, gt_gazey_norm, gt_z_gaze, gt_z_head,
                   eps=DEPTH_EPS):
    """Joint 2-D + log-ratio Euclidean distance, scale-invariant.

    Components:
        Δx = u*/(W-1) − gt_gazex_norm
        Δy = v*/(H-1) − gt_gazey_norm
        Δz = log(pred_z_gaze/pred_z_head) − log(gt_z_gaze/gt_z_head)

    Returns a Python float — Δz is in log-space (dimensionless), Δx/Δy in
    [0, 1].  A 0.1 log-ratio error corresponds to ≈10 % depth-ratio error.
    """
    h, w = pred_heatmap.shape
    flat_idx = torch.argmax(pred_heatmap.flatten()).item()
    v_star = flat_idx // w
    u_star = flat_idx % w
    pred_lr = (torch.log(pred_z_gaze.clamp(min=eps))
               - torch.log(pred_z_head.clamp(min=eps)))
    gt_lr   = (torch.log(torch.as_tensor(gt_z_gaze).clamp(min=eps))
               - torch.log(torch.as_tensor(gt_z_head).clamp(min=eps)))
    dx = u_star / max(w - 1, 1) - float(gt_gazex_norm)
    dy = v_star / max(h - 1, 1) - float(gt_gazey_norm)
    dz = float(pred_lr.item() - gt_lr.item())
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    """Compute heatmap metrics + scale-invariant anchored depth metrics.

    Heatmap metrics  : AUC, L2 (existing GazeLLE convention), AP for inout.
    Depth metrics    : Ratio-MAE / Ratio-RMSE in log-ratio space, δ₁ at 1.25,
                       and Anchored-L2-3D (joint 2-D peak + log-ratio).
    """
    model.eval()
    aucs, l2s, inout_preds, inout_gts = [], [], [], []
    pred_zg_all, pred_zh_all = [], []
    gt_zg_all,   gt_zh_all   = [], []
    l2_3d_all                = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        images, bboxes, gazex, gazey, inout, gt_zg, gt_zh = batch
        preds = model({"images": images.to(device), "bboxes": bboxes})

        pred_zg = torch.cat(preds["depth_gaze"], 0).detach().cpu()
        pred_zh = torch.cat(preds["depth_head"], 0).detach().cpu()

        pred_zg_all.append(pred_zg)
        pred_zh_all.append(pred_zh)
        gt_zg_all.append(gt_zg.float())
        gt_zh_all.append(gt_zh.float())

        sample_idx = 0
        for i in range(images.shape[0]):
            for j in range(len(bboxes[i])):
                if inout[i][j] == 1:
                    aucs.append(vat_auc(preds["heatmap"][i][j],
                                        gazex[i][j][0], gazey[i][j][0]))
                    l2s.append(vat_l2(preds["heatmap"][i][j],
                                      gazex[i][j][0], gazey[i][j][0]))
                    l2_3d_all.append(anchored_l2_3d(
                        preds["heatmap"][i][j].detach().cpu(),
                        pred_zg[sample_idx], pred_zh[sample_idx],
                        gazex[i][j][0], gazey[i][j][0],
                        gt_zg[sample_idx].item(), gt_zh[sample_idx].item(),
                    ))
                inout_preds.append(preds["inout"][i][j].item()
                                   if preds["inout"] is not None else 1.0)
                inout_gts.append(inout[i][j])
                sample_idx += 1

    pred_zg_t = torch.cat(pred_zg_all)
    pred_zh_t = torch.cat(pred_zh_all)
    gt_zg_t   = torch.cat(gt_zg_all)
    gt_zh_t   = torch.cat(gt_zh_all)

    AUC = float(np.mean(aucs)) if aucs else 0.0
    L2  = float(np.mean(l2s))  if l2s  else 0.0
    AP  = average_precision_score(inout_gts, inout_preds)
    R_MAE  = float(ratio_mae (pred_zg_t, pred_zh_t, gt_zg_t, gt_zh_t).item())
    R_RMSE = float(ratio_rmse(pred_zg_t, pred_zh_t, gt_zg_t, gt_zh_t).item())
    D1     = float(anchored_delta1(pred_zg_t, pred_zh_t, gt_zg_t, gt_zh_t).item())
    L2_3D  = float(np.mean(l2_3d_all)) if l2_3d_all else 0.0

    print(f"  AUC={AUC:.4f}  L2={L2:.4f}  AP={AP:.4f}  |  "
          f"RatioMAE={R_MAE:.4f}  RatioRMSE={R_RMSE:.4f}  "
          f"δ1={D1:.4f}  L2-3D={L2_3D:.4f}")
    return {
        "AUC":         AUC,
        "L2":          L2,
        "AP":          AP,
        "ratio_mae":   R_MAE,
        "ratio_rmse":  R_RMSE,
        "delta1":      D1,
        "l2_3d":       L2_3D,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    with open("configuration.yaml", "r") as f:
        config = yaml.safe_load(f)

    device    = config["hardware"]["device"] if torch.cuda.is_available() else "cpu"
    data_path = config["data"]["goo_path"]
    print(f"Device: {device}  |  Data: {data_path}")

    # ---- Loss weights (configurable, sane defaults) ------------------
    cfg_model    = config["model"]
    w_heatmap    = float(cfg_model.get("mse_weight",   1.0))
    w_inout      = float(cfg_model.get("bce_weight",   0.0))
    w_depth      = float(cfg_model.get("depth_weight", 0.5))
    print(f"Loss weights: heatmap={w_heatmap}  inout={w_inout}  depth={w_depth}")

    # ---- Model -------------------------------------------------------
    # Drop-in 3-D variant.  Override config['model']['name'] e.g. to
    # "gazelle3d_dinov2_vitl14_inout" to pick the depth-aware backbone.
    model, _ = get_gazelle3d_model(config)
    print(f"Model: {cfg_model['name']}")

    # Optionally warm-start the heatmap branch from a 2-D pretraining ckpt.
    pretrained = cfg_model.get("pretrained_path", "")
    if pretrained and os.path.isfile(pretrained):
        print(f"Loading warm-start weights from {pretrained}")
        model.load_gazelle_state_dict(
            torch.load(pretrained, map_location=device, weights_only=True))

    for name, param in model.named_parameters():
        param.requires_grad = "backbone" not in name

    # Xavier-init only newly-introduced (non-backbone, non-loaded) params.
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

    # ---- Optimizer ---------------------------------------------------
    param_dicts = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "inout" in name:
            lr = config["train"]["inout_lr"]
        elif "depth" in name:
            lr = config["train"].get("depth_lr", config["train"]["lr"])
        elif "linear" in name or "transformer" in name:
            lr = config["train"]["fuse_lr"]
        else:
            lr = config["train"]["lr"]
        param_dicts.append({"params": param, "lr": lr})

    opt_name = config["train"]["optimizer"]
    wd       = float(config["train"]["weight_decay"])
    if opt_name == "Adam":
        optimizer = Adam(param_dicts, weight_decay=wd)
    elif opt_name == "AdamW":
        optimizer = AdamW(param_dicts, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    # ---- Scheduler ---------------------------------------------------
    sched = config["train"]["lr_scheduler"]
    if sched["type"] == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=sched["step_size"], eta_min=float(sched["min_lr"]))
    elif sched["type"] == "warmup":
        ws = sched["step_size"]
        scheduler = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, e / ws))
    else:
        scheduler = StepLR(
            optimizer, step_size=sched["step_size"], gamma=sched["gamma"])

    # ---- Transforms (Resize before ToTensor — see train_goo.py fix) --
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
    depth_dir     = config["data"].get("depth_dir", "depth")
    train_dataset = GOOSynthDepth(data_path, img_transform, split="train",
                                  depth_dir=depth_dir)
    test_dataset  = GOOSynthDepth(data_path, val_transform, split="test",
                                  depth_dir=depth_dir)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        collate_fn=collate_train,
        shuffle=True,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["eval"]["batch_size"],
        collate_fn=collate,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"]["pin_memory"],
    )

    # ---- Loss functions ----------------------------------------------
    if cfg_model.get("is_focal_loss", 0) == 1:
        inout_loss_fn = FocalLoss()
    else:
        inout_loss_fn = nn.BCELoss()
    heatmap_loss_fn = (nn.MSELoss() if cfg_model.get("pbce_loss") == "mse"
                       else nn.BCELoss())

    # ---- Checkpoint dir ----------------------------------------------
    ckpt_dir = os.path.join(
        config["logging"]["pre_dir"],
        "_".join([
            "depth",
            opt_name,
            "bs"  + str(config["train"]["batch_size"]),
            "lr"  + str(config["train"]["lr"]),
            "hw"  + str(w_heatmap),
            "iw"  + str(w_inout),
            "dw"  + str(w_depth),
        ]),
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints → {ckpt_dir}")

    # ---- Training loop -----------------------------------------------
    num_epochs = config["train"]["epochs"]
    save_every = config["logging"]["save_every"]
    best_l2    = float("inf")
    best_path  = None

    # Anchored depth loss selector — Fix 1 vs Fix 2's inner loss.
    depth_loss_name = str(cfg_model.get("depth_loss", "log_huber")).lower()
    lambda_si       = float(cfg_model.get("lambda_si", 0.5))
    print(f"Depth loss: {depth_loss_name}  (lambda_si={lambda_si} if si_log)")

    for epoch in range(num_epochs):
        model.train()
        sums = {"total": 0.0, "hm": 0.0, "io": 0.0, "depth": 0.0}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, bboxes, gazex, gazey, inout, gt_hm, gt_zg, gt_zh = batch

            preds       = model({"images": images.to(device), "bboxes": bboxes})
            pred_hm     = torch.cat(preds["heatmap"],    0)       # [B, 64, 64]
            pred_zg     = torch.cat(preds["depth_gaze"], 0)       # [B]
            pred_zh     = torch.cat(preds["depth_head"], 0)       # [B]
            pred_inouts = (torch.cat(preds["inout"], 0)
                           if preds["inout"] is not None else None)

            gt_hm_d  = gt_hm.to(device)
            gt_zg_d  = gt_zg.to(device)
            gt_zh_d  = gt_zh.to(device)
            gt_io_d  = torch.tensor([io[0] for io in inout],
                                    dtype=torch.float32, device=device)

            # ---- 1. Spatial heatmap loss (main task) -----------------
            l_hm = heatmap_loss_fn(pred_hm, gt_hm_d) * LOSS_SCALAR

            # ---- 2. In/out classification loss (auxiliary) -----------
            if pred_inouts is not None and w_inout > 0:
                l_io = inout_loss_fn(pred_inouts, gt_io_d)
            else:
                l_io = torch.zeros((), device=device)

            # ---- 3. Anchored depth loss (Fix 2; optionally Fix 1+2) --
            if depth_loss_name == "si_log":
                l_depth = anchored_si_log_loss(
                    pred_zg, pred_zh, gt_zg_d, gt_zh_d,
                    lambda_si=lambda_si)
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

        scheduler.step()
        n = len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}]  "
              f"Total={sums['total']/n:.6f}  "
              f"HM={sums['hm']/n:.6f}  "
              f"IO={sums['io']/n:.6f}  "
              f"Depth={sums['depth']/n:.6f}")

        metrics = evaluate(model, test_loader, device)

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

        if metrics["L2"] < best_l2 and epoch >= save_every:
            best_l2   = metrics["L2"]
            best_path = os.path.join(ckpt_dir, f"best_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler":            scheduler.state_dict(),
                "metrics":              metrics,
            }, best_path)
            print(f"  Best model → epoch {epoch + 1}  L2={best_l2:.4f}")

    if best_path is not None:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        with torch.no_grad():
            metrics = evaluate(model, test_loader, device)
        final_path = os.path.join(
            ckpt_dir,
            f"Best_ep{ckpt['epoch']}_l2{int(metrics['L2']*100)}"
            f"_auc{int(metrics['AUC']*100)}_d1{int(metrics['delta1']*100)}.pt",
        )
        torch.save({
            "model_state_dict": model.state_dict(),
            "metrics":          metrics,
        }, final_path)
        print(f"Final best model → {final_path}")


if __name__ == "__main__":
    main()
