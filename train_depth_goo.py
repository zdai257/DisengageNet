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

    Each item:
        image     : [3, H, W] tensor (Resize→ToTensor→Normalize)
        bbox_norm : [[x1,y1,x2,y2]] in [0, 1]
        gazex/y   : [[x]], [[y]] in [0, 1]
        inout     : [int]
        heatmap   : [64, 64]   sigma=3 Gaussian GT (train only)
        gt_depth  : [64, 64]   per-image min-max-normalised relative depth
                                (train only)

    Test split skips the augmentation/heatmap/gt_depth fields.
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

    def _load_depth(self, image_rel_path, size):
        """Load the cached .npy depth and wrap it as a single-channel float PIL."""
        path = self._depth_path(image_rel_path)
        depth = np.load(path).astype(np.float32)        # H × W (any size)
        return Image.fromarray(depth, mode="F")         # PIL float image

    def __getitem__(self, idx):
        frame = self.frames[idx]
        image = Image.open(
            os.path.join(self.data_path, frame["path"])).convert("RGB")
        depth = self._load_depth(frame["path"], image.size)

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

        if not self.is_train:
            return image_t, [bbox_norm], [gazex_norm], [gazey_norm], [inout]

        # GT spatial heatmap (sigma=3 Gaussian on the 64×64 grid).
        gt_heatmap = get_heatmap(gazex_norm[0], gazey_norm[0], 64, 64)

        # GT depth map at 64×64; per-image min-max normalisation to [0, 1].
        depth_64 = depth.resize((64, 64), Image.BILINEAR)
        gt_depth = torch.from_numpy(np.asarray(depth_64, dtype=np.float32))
        d_min, d_max = float(gt_depth.min()), float(gt_depth.max())
        gt_depth = (gt_depth - d_min) / (d_max - d_min + 1e-8)

        return (image_t, [bbox_norm], [gazex_norm], [gazey_norm], [inout],
                gt_heatmap, gt_depth)


def collate_train(batch):
    images, bboxes, gazex, gazey, inout, heatmaps, gt_depths = zip(*batch)
    return (torch.stack(images),
            list(bboxes), list(gazex), list(gazey), list(inout),
            torch.stack(heatmaps), torch.stack(gt_depths))


def collate(batch):
    images, bboxes, gazex, gazey, inout = zip(*batch)
    return torch.stack(images), list(bboxes), list(gazex), list(gazey), list(inout)


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


def log_huber_elementwise(pred, target,
                          delta=DEPTH_HUBER_DELTA, eps=DEPTH_EPS):
    """Huber loss in log-depth space, element-wise (no reduction)."""
    diff   = torch.log(pred.clamp(min=eps)) - torch.log(target.clamp(min=eps))
    abs_d  = diff.abs()
    quad   = 0.5 * diff * diff
    linear = delta * (abs_d - 0.5 * delta)
    return torch.where(abs_d < delta, quad, linear)


def masked_depth_loss(pred_depth, gt_depth, mask, eps=1e-6):
    """Weighted log-Huber depth loss; ``mask`` is the GT spatial heatmap.

    Because the mask is a sigma=3 Gaussian peaking at 1.0 at the gaze target
    and ~0 outside a ±9-pixel disk, the depth loss only constrains the local
    neighbourhood of the gaze target — exactly the depth value that matters
    for 3-D gaze prediction.
    """
    per_px = log_huber_elementwise(pred_depth, gt_depth)   # [N, 64, 64]
    return (per_px * mask).sum() / (mask.sum() + eps)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    aucs, l2s, inout_preds, inout_gts = [], [], [], []
    for images, bboxes, gazex, gazey, inout in tqdm(loader, desc="Eval", leave=False):
        preds = model({"images": images.to(device), "bboxes": bboxes})
        for i in range(images.shape[0]):
            for j in range(len(bboxes[i])):
                if inout[i][j] == 1:
                    aucs.append(vat_auc(preds["heatmap"][i][j],
                                        gazex[i][j][0], gazey[i][j][0]))
                    l2s.append(vat_l2(preds["heatmap"][i][j],
                                      gazex[i][j][0], gazey[i][j][0]))
                inout_preds.append(preds["inout"][i][j].item()
                                   if preds["inout"] is not None else 1.0)
                inout_gts.append(inout[i][j])

    AUC = float(np.mean(aucs)) if aucs else 0.0
    L2  = float(np.mean(l2s))  if l2s  else 0.0
    AP  = average_precision_score(inout_gts, inout_preds)
    print(f"  AUC={AUC:.4f}  L2={L2:.4f}  AP={AP:.4f}")
    return AUC, L2, AP


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

    for epoch in range(num_epochs):
        model.train()
        sums = {"total": 0.0, "hm": 0.0, "io": 0.0, "depth": 0.0}

        for (images, bboxes, gazex, gazey, inout, gt_hm, gt_depth) in tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):

            preds         = model({"images": images.to(device), "bboxes": bboxes})
            pred_hm       = torch.cat(preds["heatmap"], 0)        # [B, 64, 64]
            pred_depth    = torch.cat(preds["depth"], 0)          # [B, 64, 64]
            pred_inouts   = (torch.cat(preds["inout"], 0)
                             if preds["inout"] is not None else None)

            gt_hm_d    = gt_hm.to(device)
            gt_depth_d = gt_depth.to(device)
            gt_io_d    = torch.tensor([io[0] for io in inout],
                                      dtype=torch.float32, device=device)

            # ---- 1. Spatial heatmap loss (main task) -----------------
            l_hm = heatmap_loss_fn(pred_hm, gt_hm_d) * LOSS_SCALAR

            # ---- 2. In/out classification loss (auxiliary) -----------
            if pred_inouts is not None and w_inout > 0:
                l_io = inout_loss_fn(pred_inouts, gt_io_d)
            else:
                l_io = torch.zeros((), device=device)

            # ---- 3. Depth loss — log-Huber masked by GT heatmap ------
            #    Only the gaze-target neighbourhood (~9-px radius) gets a
            #    non-negligible weight, so the depth head is forced to
            #    learn "depth at the gaze point", not a global depth map.
            l_depth = masked_depth_loss(pred_depth, gt_depth_d, gt_hm_d)

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

        AUC, L2, AP = evaluate(model, test_loader, device)

        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler":            scheduler.state_dict(),
                "auc": AUC, "l2": L2, "ap": AP,
            }, ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

        if L2 < best_l2 and epoch >= save_every:
            best_l2   = L2
            best_path = os.path.join(ckpt_dir, f"best_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler":            scheduler.state_dict(),
                "auc": AUC, "l2": L2, "ap": AP,
            }, best_path)
            print(f"  Best model → epoch {epoch + 1}  L2={best_l2:.4f}")

    if best_path is not None:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        with torch.no_grad():
            AUC, L2, AP = evaluate(model, test_loader, device)
        final_path = os.path.join(
            ckpt_dir,
            f"Best_ep{ckpt['epoch']}_l2{int(L2 * 100)}_auc{int(AUC * 100)}.pt",
        )
        torch.save({
            "model_state_dict": model.state_dict(),
            "auc": AUC, "l2": L2, "ap": AP,
        }, final_path)
        print(f"Final best model → {final_path}")


if __name__ == "__main__":
    main()
