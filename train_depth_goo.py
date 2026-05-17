"""
train_depth_goo.py — depth-aware gaze pretraining on GOOSynthV3.

Extends the train_goo.py pipeline with an additional depth-prediction head
(``GT3D``) trained against DepthAnythingV2 pseudo-labels cached under
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
import wandb
import yaml

from eval import eval_metrics, average_precision_score, vat_auc, vat_l2
from network.network_builder_gt3d import get_gt3d_model
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
# Dense depth distillation, 3-D ray-tracing, and anchored-consistency losses
# used by train_gazefollow-depthaware.py (GT3D multi-task curriculum).
# Centralised here so any future depth-aware trainer can import them too.
# --------------------------------------------------------------------------

def dense_si_log_loss(pred_dense, gt_dense, gaussian_mask=None,
                      alpha=0.5, lambda_si=0.5, eps=DEPTH_EPS):
    """Scale-invariant log-loss on a dense depth map (Eigen et al., 2014).

    Both ``pred_dense`` and ``gt_dense`` are assumed to be per-image min-max-
    normalised to [0, 1] (DepthAnythingV2 inverse-depth convention, no
    metric units), so only the *log-ratio of pixel pairs* — i.e. the SI-log
    quantity — is identifiable.

    If ``gaussian_mask`` (the target heatmap) is supplied and ``alpha > 0``,
    the loss is a convex combination of:
        L = α · weighted_SI-log   (focus on pixels around the gaze target)
          + (1 − α) · global_SI-log  (whole-image scene regularisation)

    Shapes:
        pred_dense, gt_dense : [B, H, W]  or  [B, 1, H, W]
        gaussian_mask        : [B, H, W]  (non-negative; per-image-normalised
                                            inside this function)
    """
    if pred_dense.dim() == 4:
        pred_dense = pred_dense.squeeze(1)
    if gt_dense.dim() == 4:
        gt_dense = gt_dense.squeeze(1)

    log_p = torch.log(pred_dense.clamp(min=eps))
    log_g = torch.log(gt_dense.clamp(min=eps))
    d     = log_p - log_g                                # [B, H, W]

    # Global SI-log over all pixels.
    d_mean_g  = d.flatten(1).mean(dim=1)                 # [B]
    d2_mean_g = (d * d).flatten(1).mean(dim=1)           # [B]
    L_global  = (d2_mean_g - float(lambda_si) * d_mean_g.pow(2)).mean()

    if gaussian_mask is None or alpha <= 0:
        return L_global

    # Weighted SI-log (focus near the gaze target).
    w = gaussian_mask.to(d.device).float()
    w = w / (w.flatten(1).sum(dim=1, keepdim=True)
             .unsqueeze(-1)
             .clamp(min=1e-8))                            # [B, H, W], rows sum to 1
    d_mean_w  = (w * d ).flatten(1).sum(dim=1)            # [B]
    d2_mean_w = (w * d * d).flatten(1).sum(dim=1)         # [B]
    L_weighted = (d2_mean_w - float(lambda_si) * d_mean_w.pow(2)).mean()

    return float(alpha) * L_weighted + (1.0 - float(alpha)) * L_global


def gaze_ray_3d_loss(pred_heatmap, pred_z_gaze, pred_z_head,
                     gt_gazex, gt_gazey, gt_z_gaze, gt_z_head,
                     head_cx, head_cy,
                     gamma=0.5, w_cos=1.0, w_l1=0.5,
                     temperature=0.05, eps=DEPTH_EPS, mask=None):
    """3-D head→target ray-tracing loss in (x, y, log-ratio) space.

    Generalises the 2-D cosine-angle loss currently used in
    train_gazefollow-depthaware.py: a near distractor and a far true target
    lying on the same image-plane ray have *different* 3-D ray angles, so
    the model can disambiguate them via depth.

    Differentiability: the (x, y) component is computed by soft-argmax over
    ``pred_heatmap`` (temperature ``temperature``), so gradients flow back
    into the heatmap head.  The z component is the head-anchored log-ratio
    of the predicted depth scalars, so gradients also flow into the depth
    heads.

    Inputs (all on the same device):
        pred_heatmap                  : [B, H, W]  sigmoid heatmap
        pred_z_gaze, pred_z_head      : [B]        sigmoid depth scalars
        gt_gazex, gt_gazey            : [B]        in [0, 1]
        gt_z_gaze, gt_z_head          : [B]        in [0, 1]
        head_cx, head_cy              : [B]        in [0, 1] (head bbox centre)
        mask                          : [B] bool   optional (e.g. inout==1)

    ``gamma`` rebalances the dynamic range of the Z axis (log-ratio,
    typically [-3, 3]) against the X/Y axes ([-1, 1]).  γ ≈ 0.5 is a good
    starting point.  Sweep in {0.25, 0.5, 1.0}.

    Returns a 0-D tensor.  If ``mask`` is all-False the loss is 0.
    """
    B, H, W = pred_heatmap.shape

    # Differentiable 2-D peak via soft-argmax in normalised [0, 1] coords.
    flat = pred_heatmap.view(B, -1)
    soft = F.softmax(flat / float(temperature), dim=-1).view(B, H, W)
    xr   = torch.linspace(0, 1, W, device=pred_heatmap.device).view(1, 1, W).expand(B, H, W)
    yr   = torch.linspace(0, 1, H, device=pred_heatmap.device).view(1, H, 1).expand(B, H, W)
    px   = (xr * soft).sum(dim=(1, 2))                    # [B]
    py   = (yr * soft).sum(dim=(1, 2))                    # [B]

    # Z component: head-anchored log-ratio (dimensionless, scale-invariant).
    pred_dz = (torch.log(pred_z_gaze.clamp(min=eps))
               - torch.log(pred_z_head.clamp(min=eps)))
    gt_dz   = (torch.log(gt_z_gaze.clamp(min=eps))
               - torch.log(gt_z_head.clamp(min=eps)))

    pred_vec = torch.stack([px       - head_cx,
                            py       - head_cy,
                            float(gamma) * pred_dz], dim=-1)   # [B, 3]
    gt_vec   = torch.stack([gt_gazex - head_cx,
                            gt_gazey - head_cy,
                            float(gamma) * gt_dz],   dim=-1)   # [B, 3]

    cos      = F.cosine_similarity(pred_vec, gt_vec, dim=-1)   # [B]
    l_cos    = (1.0 - cos)
    l_l1     = (pred_vec - gt_vec).abs().mean(dim=-1)          # [B]
    per_smp  = float(w_cos) * l_cos + float(w_l1) * l_l1       # [B]

    if mask is not None:
        m = mask.to(per_smp.device).float()
        denom = m.sum().clamp(min=1.0)
        return (per_smp * m).sum() / denom
    return per_smp.mean()


def anchored_consistency_loss(pred_dense, pred_z_gaze, pred_z_head,
                              gazex_norm, gazey_norm,
                              head_cx_norm, head_cy_norm):
    """L2 consistency between the dense decoder and the anchored scalar pair.

    Pulls the dense map's value at the gaze pixel toward ``pred_z_gaze`` and
    at the head pixel toward ``pred_z_head``.  Cheap regulariser that
    prevents the two depth pathways from drifting; default-OFF (turn on by
    setting ``w_consist > 0`` in the config).

    Inputs:
        pred_dense : [B, H, W] or [B, 1, H, W] in [0, 1]
        pred_z_gaze, pred_z_head : [B]
        gazex_norm, gazey_norm, head_cx_norm, head_cy_norm : [B] in [0, 1]
    """
    if pred_dense.dim() == 4:
        pred_dense = pred_dense.squeeze(1)
    B, H, W = pred_dense.shape

    gx = torch.as_tensor(gazex_norm,    device=pred_dense.device, dtype=torch.float32)
    gy = torch.as_tensor(gazey_norm,    device=pred_dense.device, dtype=torch.float32)
    hx = torch.as_tensor(head_cx_norm,  device=pred_dense.device, dtype=torch.float32)
    hy = torch.as_tensor(head_cy_norm,  device=pred_dense.device, dtype=torch.float32)

    ug = (gx * (W - 1)).round().long().clamp(0, W - 1)
    vg = (gy * (H - 1)).round().long().clamp(0, H - 1)
    uh = (hx * (W - 1)).round().long().clamp(0, W - 1)
    vh = (hy * (H - 1)).round().long().clamp(0, H - 1)

    idx       = torch.arange(B, device=pred_dense.device)
    z_at_gaze = pred_dense[idx, vg, ug]
    z_at_head = pred_dense[idx, vh, uh]
    return ((z_at_gaze - pred_z_gaze).pow(2)
            + (z_at_head - pred_z_head).pow(2)).mean()


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
# 3-D metrics aligned with the Privacy-Preserving 3-D paper
# (Tafasca et al., arXiv 2409.17886).  That paper defines:
#
#   • 3-D L2     : Euclidean distance in *metric* 3-D between the predicted
#                  and ground-truth gaze targets.  Requires camera intrinsics
#                  and a metric depth (e.g. ZoeDepth) for the head and target.
#   • 3-D Angle  : angular error between predicted and GT gaze rays
#                  (head → target).
#
# In our setting the depth pseudo-labels are per-image min-max-normalised
# (DepthAnythingV2), so absolute metric distance is not identifiable.  We
# therefore expose:
#
#   • ``gaze3d_angle_relative``: the angular error of the head→target ray
#     in a synthetic 3-D space whose z axis is the head-anchored log-ratio
#     log z = log z_target − log z_head.  This is meaningful for
#     model-vs-model comparison under our depth convention but is *not*
#     directly comparable to the metric-3-D angle reported in Tafasca 2024
#     (the relative scaling between (x, y) ∈ [0, 1]² and z ∈ ℝ differs).
#
#   • ``gaze3d_l2_metric`` / ``gaze3d_angle_metric``: stubs that raise
#     ``NotImplementedError`` until a metric depth and intrinsics pipeline
#     is wired in.  They document the exact inputs and definitions so that
#     a future evaluator (e.g. ZoeDepth + COLMAP focal estimate) can drop
#     in without changing the call sites.
# --------------------------------------------------------------------------

def gaze3d_angle_relative(pred_heatmap, pred_z_gaze, pred_z_head,
                          gt_gazex_norm, gt_gazey_norm,
                          gt_z_gaze, gt_z_head,
                          head_cx_norm, head_cy_norm,
                          eps=DEPTH_EPS):
    """Angular error of the head→target ray in (x, y, log-ratio) space.

    Inputs
    ------
    pred_heatmap   : tensor [H, W], sigmoid heatmap from the model.
    pred_z_gaze, pred_z_head : 0-D tensors (per-sample scalars).
    gt_gazex_norm, gt_gazey_norm : floats in [0, 1] — GT gaze pixel.
    gt_z_gaze, gt_z_head         : floats in [0, 1] — GT head-anchored depths.
    head_cx_norm, head_cy_norm   : floats in [0, 1] — head bbox centre.

    Output
    ------
    Angular error in **degrees** between predicted and GT head→target rays.

    Convention
    ----------
    The 3-D point of the gaze target is (x, y, z) where x, y are the
    normalised image coords and z is the head-anchored log-ratio
    ``log z_target − log z_head``.  The head sits at (head_cx, head_cy, 0)
    by definition, so the ray is (Δx, Δy, log z_target − log z_head).
    Because the (x, y) and z axes are not in the same physical units, this
    angle is **scale-coupled** to the depth normalisation; it is suitable
    for model-vs-model comparison under a fixed convention but not for
    cross-paper comparison against metric-3-D angle.
    """
    h, w = pred_heatmap.shape
    flat_idx = torch.argmax(pred_heatmap.flatten()).item()
    v_star = flat_idx // w
    u_star = flat_idx % w
    pred_x = u_star / max(w - 1, 1)
    pred_y = v_star / max(h - 1, 1)
    pred_lr = float((torch.log(pred_z_gaze.clamp(min=eps))
                     - torch.log(pred_z_head.clamp(min=eps))).item())
    gt_lr   = float((torch.log(torch.as_tensor(gt_z_gaze).clamp(min=eps))
                     - torch.log(torch.as_tensor(gt_z_head).clamp(min=eps))).item())

    pred_vec = np.array([pred_x         - float(head_cx_norm),
                         pred_y         - float(head_cy_norm),
                         pred_lr], dtype=np.float64)
    gt_vec   = np.array([float(gt_gazex_norm) - float(head_cx_norm),
                         float(gt_gazey_norm) - float(head_cy_norm),
                         gt_lr],   dtype=np.float64)
    pn = float(np.linalg.norm(pred_vec)) + 1e-8
    gn = float(np.linalg.norm(gt_vec))   + 1e-8
    cos = float(np.dot(pred_vec, gt_vec) / (pn * gn))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def gaze3d_l2_relative(pred_heatmap, pred_z_gaze, pred_z_head,
                       gt_gazex_norm, gt_gazey_norm,
                       gt_z_gaze, gt_z_head,
                       eps=DEPTH_EPS):
    """Alias of :func:`anchored_l2_3d` — head-anchored, scale-invariant 3-D L2.

    Reported under the same name as the Privacy-Preserving 3-D paper's
    metric 3-D L2 for table compatibility, but lives in (x, y, log-ratio)
    space rather than metric (X, Y, Z).
    """
    return anchored_l2_3d(pred_heatmap, pred_z_gaze, pred_z_head,
                          gt_gazex_norm, gt_gazey_norm,
                          gt_z_gaze, gt_z_head, eps=eps)


def gaze3d_l2_metric(*_args, **_kwargs):
    """Metric 3-D L2 in physical units.

    Requires (a) per-image camera focal length f_x, f_y, principal point,
    and (b) a metric depth estimate at both the head pixel and the gaze
    pixel.  Until those are wired into the dataloader (e.g. via ZoeDepth +
    a focal-length estimator), this stub intentionally raises so the call
    site fails loudly rather than silently reporting a misleading value.
    """
    raise NotImplementedError(
        "Metric 3-D L2 needs camera intrinsics and metric depth; not "
        "available under DepthAnythingV2 per-image min-max normalisation.")


def gaze3d_angle_metric(*_args, **_kwargs):
    """Metric 3-D angular error of the head→target ray.

    Same prerequisites as :func:`gaze3d_l2_metric`.  Provided as a stub so
    a future drop-in implementation does not require call-site changes.
    """
    raise NotImplementedError(
        "Metric 3-D angle needs camera intrinsics and metric depth; not "
        "available under DepthAnythingV2 per-image min-max normalisation.")


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    """Compute heatmap metrics + scale-invariant anchored depth metrics.

    Heatmap metrics  : AUC, L2 (existing GazeLLE convention), AP for inout.
    Depth metrics    : Ratio-MAE / Ratio-RMSE in log-ratio space, δ₁ at 1.25.
    3-D metrics      : Anchored-L2-3D (joint 2-D peak + log-ratio) and
                       Angle-3D-rel  (relative 3-D angular error of the
                       head→target ray under our depth convention).
    """
    model.eval()
    aucs, l2s, inout_preds, inout_gts = [], [], [], []
    pred_zg_all, pred_zh_all = [], []
    gt_zg_all,   gt_zh_all   = [], []
    l2_3d_all, angle_3d_all  = [], []

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
                    hm_cpu = preds["heatmap"][i][j].detach().cpu()
                    l2_3d_all.append(anchored_l2_3d(
                        hm_cpu,
                        pred_zg[sample_idx], pred_zh[sample_idx],
                        gazex[i][j][0], gazey[i][j][0],
                        gt_zg[sample_idx].item(), gt_zh[sample_idx].item(),
                    ))
                    bbox = bboxes[i][j]
                    head_cx = (float(bbox[0]) + float(bbox[2])) * 0.5
                    head_cy = (float(bbox[1]) + float(bbox[3])) * 0.5
                    angle_3d_all.append(gaze3d_angle_relative(
                        hm_cpu,
                        pred_zg[sample_idx], pred_zh[sample_idx],
                        gazex[i][j][0], gazey[i][j][0],
                        gt_zg[sample_idx].item(), gt_zh[sample_idx].item(),
                        head_cx, head_cy,
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
    L2_3D  = float(np.mean(l2_3d_all))    if l2_3d_all    else 0.0
    A3D    = float(np.mean(angle_3d_all)) if angle_3d_all else 0.0

    print(f"  AUC={AUC:.4f}  L2={L2:.4f}  AP={AP:.4f}  |  "
          f"RatioMAE={R_MAE:.4f}  RatioRMSE={R_RMSE:.4f}  "
          f"δ1={D1:.4f}  L2-3D={L2_3D:.4f}  Angle-3D={A3D:.2f}°")
    return {
        "AUC":         AUC,
        "L2":          L2,
        "AP":          AP,
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

    device    = config["hardware"]["device"] if torch.cuda.is_available() else "cpu"
    data_path = config["data"]["goo_path"]
    print(f"Device: {device}  |  Data: {data_path}")

    wandb.init(
        project=config["model"]["name"],
        name="train_depth_goo",
        config=config,
    )

    # ---- Loss weights (configurable, sane defaults) ------------------
    cfg_model    = config["model"]
    w_heatmap    = float(cfg_model.get("mse_weight",   1.0))
    w_inout      = float(cfg_model.get("bce_weight",   0.0))
    w_depth      = float(cfg_model.get("depth_weight", 0.5))
    print(f"Loss weights: heatmap={w_heatmap}  inout={w_inout}  depth={w_depth}")

    # ---- Model -------------------------------------------------------
    # Drop-in 3-D variant.  Override config['model']['name'] e.g. to
    # "gt3d_dinov2_vitl14_inout" to pick the depth-aware backbone.
    model, _ = get_gt3d_model(config)
    print(f"Model: {cfg_model['name']}")

    # Optionally warm-start the heatmap branch from a 2-D pretraining ckpt.
    pretrained = cfg_model.get("pretrained_path", "")
    if pretrained and os.path.isfile(pretrained):
        print(f"Loading warm-start weights from {pretrained}")
        model.load_gt3d_state_dict(
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
    # 4-bucket LR dispatch driven by canonical substrings in module names.
    # By design every trainable parameter lands in exactly one bucket; the
    # base ``lr`` arm should never be reached (kept as a safety net only):
    #   "inout"  → inout_lr   (inout_token, inout_head)
    #   "depth"  → depth_lr   (depth_token_*, depth_scalar_head, depth_to_feat,
    #                          dense_depth_decoder, depth_heatmap_head,
    #                          depth_refined_heatmap_head)
    #   "block"  → block_lr   (trunk_blocks, refine_blocks)
    #   "fuse"   → fuse_lr    (fuse_proj, fuse_head_token)
    # Order matters: depth/inout must be checked before block (in case of
    # any future module whose name contains both substrings).
    param_dicts = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "inout" in name:
            lr = config["train"]["inout_lr"]
        elif "depth" in name:
            lr = config["train"].get("depth_lr", config["train"]["lr"])
        elif "block" in name:
            lr = config["train"]["block_lr"]
        elif "fuse" in name:
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

    log_every = int(config["logging"].get("log_every",
                                          config["logging"]["save_every"]))

    for epoch in range(num_epochs):
        model.train()
        sums = {"total": 0.0, "hm": 0.0, "io": 0.0, "depth": 0.0}

        for cur_iter, batch in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
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

            if cur_iter % log_every == 0:
                wandb.log({
                    "train/loss":         total.item(),
                    "train/heatmap_loss": l_hm.item(),
                    "train/inout_loss":   float(l_io.item()),
                    "train/depth_loss":   l_depth.item(),
                })

        scheduler.step()
        n = len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}]  "
              f"Total={sums['total']/n:.6f}  "
              f"HM={sums['hm']/n:.6f}  "
              f"IO={sums['io']/n:.6f}  "
              f"Depth={sums['depth']/n:.6f}")

        metrics = evaluate(model, test_loader, device)
        wandb.log({
            "eval/auc":        metrics["AUC"],
            "eval/l2":         metrics["L2"],
            "eval/inout_ap":   metrics["AP"],
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

    wandb.finish()


if __name__ == "__main__":
    main()
