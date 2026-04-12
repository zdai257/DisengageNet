import os
import json

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
from tqdm import tqdm
import yaml

from eval import eval_metrics, average_precision_score, vat_auc, vat_l2
from network.network_builder import get_gazelle_model
from network.network_builder_update2 import get_gazemoe_model
from network.utils import SoftArgmax2D, CosineL1, VectorL2Loss

LOSS_SCALAR = 1


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GOOSynth(torch.utils.data.Dataset):
    """
    Reads goosynth_{split}_preprocess.json produced by preprocess_goosynth.py.
    Returns the same (image, bboxes, gazex, gazey, inout) tuple used by train_vat.py.
    """

    def __init__(self, data_path, img_transform, split="train"):
        json_path = os.path.join(data_path, f"goosynth_{split}_preprocess.json")
        self.frames    = json.load(open(json_path, "rb"))
        self.data_path = data_path
        self.transform = img_transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame  = self.frames[idx]
        image  = Image.open(
            os.path.join(self.data_path, frame["path"])
        ).convert("RGB")
        image  = self.transform(image)
        bboxes = [head["bbox_norm"]  for head in frame["heads"]]
        gazex  = [head["gazex_norm"] for head in frame["heads"]]
        gazey  = [head["gazey_norm"] for head in frame["heads"]]
        inout  = [head["inout"]      for head in frame["heads"]]
        return image, bboxes, gazex, gazey, inout


def collate(batch):
    images, bboxes, gazex, gazey, inout = zip(*batch)
    return torch.stack(images), list(bboxes), list(gazex), list(gazey), list(inout)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.02/0.98, gamma=2.0, reduction='mean', apply_sigmoid=False):
        """
            alpha (float): Weighting factor for the minority class (e.g., 0.4/0.6 for VAT; 0.02/0.98 for GOO).
            gamma (float): Focusing parameter to down-weight easy examples (e.g., 2.0).
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.apply_sigmoid = apply_sigmoid

    def forward(self, inputs, targets):
        targets = targets.float()
        # Apply sigmoid if inputs are logits
        if self.apply_sigmoid:
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs
        # Compute binary cross-entropy (without reduction)
        bce = F.binary_cross_entropy(probs, targets, reduction='none')
        # Compute pt = exp(-bce) = probability of true class
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


# ---------------------------------------------------------------------------
# GT heatmap construction
# ---------------------------------------------------------------------------

def apply_dilation_blur(heatmap, dilation_kernel=5, blur_radius=0.8,
                        peak_val=1.0, min_val=0.0):
    heatmap = heatmap.unsqueeze(0)
    heatmap = F.max_pool2d(heatmap, kernel_size=dilation_kernel, stride=1,
                           padding=dilation_kernel // 2)
    ks = int(6 * blur_radius) | 1
    heatmap = TF.gaussian_blur(heatmap, [ks, ks], [blur_radius])
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / heatmap.max()
    heatmap = heatmap * (peak_val - min_val) + min_val
    return heatmap.squeeze(0)


def build_gt(bboxes, gazex, gazey, inout):
    """Convert per-batch annotation lists → stacked tensors for loss computation."""
    gt_heatmaps, gt_inouts, bbox_ctrs, gt_xys = [], [], [], []
    for bbxs, gtxs, gtys, ios in zip(bboxes, gazex, gazey, inout):
        h_maps, h_ios = [], []
        for bbx, gtx, gty, io in zip(bbxs, gtxs, gtys, ios):
            bbox_ctrs.append(torch.tensor(
                [(bbx[0] + bbx[2]) / 2, (bbx[1] + bbx[3]) / 2], dtype=torch.float32))
            gt_xys.append(torch.tensor([gtx[0], gty[0]], dtype=torch.float32))
            hm = torch.zeros(64, 64)
            hm[int(gty[0] * 63), int(gtx[0] * 63)] = 1
            hm = apply_dilation_blur(hm)
            h_maps.append(hm)
            h_ios.append(torch.tensor([io], dtype=torch.float32))
        gt_heatmaps.append(torch.stack(h_maps))
        gt_inouts.append(torch.cat(h_ios))
    return (
        torch.cat(gt_heatmaps),
        torch.cat(gt_inouts),
        torch.stack(bbox_ctrs),
        torch.stack(gt_xys),
    )


# ---------------------------------------------------------------------------
# Per-layer LR groups
# ---------------------------------------------------------------------------

def get_custom_lr_groups(model, base_lr, inout_lr, lr_decay=0.8):
    group_map = {"lower_lr": [], "low_lr": [], "base_lr": [], "inout_lr": []}
    for name, param in model.named_parameters():
        if not param.requires_grad or name.startswith("backbone"):
            continue
        if name.startswith("linear") or name.startswith("ms_fusion"):
            group_map["lower_lr"].append(param)
        elif name.startswith("transformer"):
            group_map["low_lr"].append(param)
        elif name.startswith("heatmap") or name.startswith("head"):
            group_map["base_lr"].append(param)
        elif name.startswith("inout"):
            group_map["inout_lr"].append(param)
        else:
            raise TypeError(f"Unknown layer name: {name}")
    n = len(group_map)
    param_groups = []
    for i, (key, val) in enumerate(group_map.items()):
        lr = inout_lr if key == "inout_lr" else base_lr * (lr_decay ** (n - i - 1))
        param_groups.append({"params": val, "lr": lr})
        print(f"  {key:<12} lr={lr:.6f}  params={sum(p.numel() for p in val):,}")
    return param_groups


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(config, model, loader, device, loss_fns):
    model.eval()
    bce_loss, pbce_loss, angle_loss, vec_loss, soft_argmax = loss_fns
    total_loss = 0.0
    aucs, l2s, inout_preds, inout_gts = [], [], [], []

    for images, bboxes, gazex, gazey, inout in tqdm(loader, desc="Eval", leave=False):
        preds = model({"images": images.to(device), "bboxes": bboxes})
        pred_inouts   = torch.cat(preds["inout"], 0)
        pred_heatmaps = torch.cat(preds["heatmap"], 0)

        # shapes mirror the training loop (see comments there)
        gt_hm, gt_io, bbox_ctrs, gt_xys = build_gt(bboxes, gazex, gazey, inout)
        pred_xys   = soft_argmax(pred_heatmaps)  # [N_total, 2]
        inout_mask = gt_io.to(device)             # [N_total]

        loss0    = bce_loss(pred_inouts, gt_io.to(device))
        pbce_raw = pbce_loss(pred_heatmaps, gt_hm.to(device)) * LOSS_SCALAR
        loss1    = (pbce_raw.mean([1, 2]) * inout_mask
                    if pbce_raw.dim() > 1 else pbce_raw * inout_mask)
        loss2    = angle_loss(pred_xys - bbox_ctrs.to(device),
                              gt_xys.to(device) - bbox_ctrs.to(device)) * inout_mask
        loss3    = vec_loss(pred_xys - bbox_ctrs.to(device),
                            gt_xys.to(device) - bbox_ctrs.to(device)) * inout_mask

        total_loss += (
            config["model"]["bce_weight"]   * loss0
            + config["model"]["mse_weight"]   * loss1.mean()
            + config["model"]["angle_weight"] * loss2.mean()
            + config["model"]["vec_weight"]   * loss3.mean()
        ).item()

        for i in range(images.shape[0]):
            for j in range(len(bboxes[i])):
                if inout[i][j] == 1:
                    aucs.append(vat_auc(preds["heatmap"][i][j], gazex[i][j][0], gazey[i][j][0]))
                    l2s.append(vat_l2(preds["heatmap"][i][j], gazex[i][j][0], gazey[i][j][0]))
                inout_preds.append(preds["inout"][i][j].item())
                inout_gts.append(inout[i][j])

    AUC = float(np.mean(aucs)) if aucs else 0.0
    L2  = float(np.mean(l2s))  if l2s  else 0.0
    AP  = average_precision_score(inout_gts, inout_preds)
    print(f"  AUC={AUC:.4f}  L2={L2:.4f}  AP={AP:.4f}")
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open("configuration.yaml", "r") as f:
        config = yaml.safe_load(f)

    device    = config["hardware"]["device"] if torch.cuda.is_available() else "cpu"
    data_path = config["data"]["goo_path"]
    print(f"Device: {device}  |  Data: {data_path}")

    # ---- Model (random init, backbone frozen) ------------------------
    model, _ = get_gazelle_model(config)
    #model, _ = get_gazemoe_model(config)

    for name, param in model.named_parameters():
        param.requires_grad = "backbone" not in name

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
    if config["train"]["layer_decay"]:
        param_dicts = get_custom_lr_groups(
            model, config["train"]["lr"], config["train"]["inout_lr"])
    else:
        param_dicts = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "inout" in name:
                lr = config["train"]["inout_lr"]
            elif "linear" in name or "ms_fusion" in name or "transformer" in name:
                lr = config["train"]["fuse_lr"]
            else:
                lr = config["train"]["lr"]
            param_dicts.append({"params": param, "lr": lr})

    opt_name = config["train"]["optimizer"]
    wd       = config["train"]["weight_decay"]
    if opt_name == "Adam":
        optimizer = Adam(param_dicts, weight_decay=wd)
    elif opt_name == "AdamW":
        optimizer = AdamW(param_dicts, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    # ---- Scheduler ---------------------------------------------------
    sched  = config["train"]["lr_scheduler"]
    if sched["type"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=sched["step_size"], eta_min=float(sched["min_lr"]))
    elif sched["type"] == "warmup":
        ws = sched["step_size"]
        scheduler = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, e / ws))
    else:
        scheduler = StepLR(optimizer, step_size=sched["step_size"], gamma=sched["gamma"])

    # ---- Transforms --------------------------------------------------
    res = config["data"]["input_resolution"]
    img_transform = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomApply([T.RandomGrayscale(p=0.2)], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Resize((res, res)),
    ])
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Resize((res, res)),
    ])

    # ---- Dataloaders -------------------------------------------------
    train_dataset = GOOSynth(data_path, val_transform, split="train")
    test_dataset  = GOOSynth(data_path, val_transform, split="test")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        collate_fn=collate,
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
    if config['model']['is_focal_loss'] == 1:
        inout_loss_fn = FocalLoss()
    else:
        inout_loss_fn = torch.nn.BCELoss()
    #bce_loss  = torch.nn.BCELoss(reduction="mean")
    pbce_loss = (torch.nn.MSELoss(reduction=config["model"]["reduction"])
                 if config["model"]["pbce_loss"] == "mse"
                 else torch.nn.BCELoss(reduction=config["model"]["reduction"]))
    angle_loss    = CosineL1()
    vec_loss      = VectorL2Loss()
    soft_argmax   = SoftArgmax2D()
    loss_fns = (inout_loss_fn, pbce_loss, angle_loss, vec_loss, soft_argmax)

    # ---- Checkpoint setup --------------------------------------------
    ckpt_dir = os.path.join(
        config["logging"]["log_dir"],
        "_".join([
            opt_name,
            "bs" + str(config["train"]["batch_size"]),
            str(config["train"]["lr"]),
            str(config["train"]["inout_lr"]),
            config["model"]["pbce_loss"],
            str(config["model"]["bce_weight"]),
            str(config["model"]["mse_weight"]),
            str(config["model"]["angle_weight"]),
            str(config["model"]["vec_weight"]),
        ]),
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints → {ckpt_dir}")

    # ---- Training loop -----------------------------------------------
    num_epochs           = config["train"]["epochs"]
    save_every           = config["logging"]["save_every"]
    best_val_loss        = float("inf")
    best_checkpoint_path = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, bboxes, gazex, gazey, inout in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            # images        : [B, 3, 448, 448]
            # bboxes/gazex/y: list[B] of list[N_i] of norm coords in [0,1]
            preds = model({"images": images.to(device), "bboxes": bboxes})
            # preds["heatmap"]: list[B] of [N_i, 64, 64]
            # preds["inout"] : list[B] of [N_i]
            pred_inouts   = torch.cat(preds["inout"], 0)    # [N_total]
            pred_heatmaps = torch.cat(preds["heatmap"], 0)  # [N_total, 64, 64]

            # gt_hm       : [N_total, 64, 64]   gt_io   : [N_total]
            # bbox_ctrs   : [N_total, 2]         gt_xys  : [N_total, 2]
            gt_hm, gt_io, bbox_ctrs, gt_xys = build_gt(bboxes, gazex, gazey, inout)
            # fixed: make pred_xys normalized in [0, 1], as those from build_gt()
            pred_xys   = soft_argmax(pred_heatmaps) / 63.0  # [N_total, 2]
            inout_mask = gt_io.to(device)             # [N_total]

            # loss0: scalar  (inout BCE, reduction='mean')
            loss0 = inout_loss_fn(pred_inouts, gt_io.to(device))
            # pbce_raw: [N_total, 64, 64] when reduction='none',
            #           scalar            when reduction='mean'
            #pbce_raw = pbce_loss(pred_heatmaps, gt_hm.to(device)) * LOSS_SCALAR
            #loss1 = (pbce_raw.mean([1, 2]) * inout_mask  # [N_total,64,64]→[N_total]
            #         if pbce_raw.dim() > 1 else pbce_raw)  # scalar path

            # fix PBCE loss with Gazelle-VAT's logic
            loss1 = pbce_loss(pred_heatmaps[inout_mask.bool()], gt_hm.to(device)[inout_mask.bool()]) * LOSS_SCALAR

            # CosineL1 → [N_total];  VectorL2Loss → scalar
            loss2 = angle_loss(pred_xys - bbox_ctrs.to(device),
                               gt_xys.to(device) - bbox_ctrs.to(device)) * inout_mask
            loss3 = vec_loss(pred_xys - bbox_ctrs.to(device),
                             gt_xys.to(device) - bbox_ctrs.to(device)) * inout_mask

            total_loss = (
                config["model"]["bce_weight"]   * loss0
                + config["model"]["mse_weight"]   * loss1
                + config["model"]["angle_weight"] * loss2.mean()  # .mean() needed?
                + config["model"]["vec_weight"]   * loss3.mean()  # .mean() needed?
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        scheduler.step()

        mean_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {mean_loss:.7f}")

        val_loss = evaluate(config, model, test_loader, device, loss_fns)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Val Loss:   {val_loss:.7f}")

        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": val_loss,
            }, ckpt_path)
            print(f"  Checkpoint saved → {ckpt_path}")

        if val_loss < best_val_loss and epoch > save_every:
            best_val_loss        = val_loss
            best_checkpoint_path = os.path.join(ckpt_dir, f"best_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": best_val_loss,
            }, best_checkpoint_path)
            print(f"  Best model → epoch {epoch+1}  loss={best_val_loss:.4f}")

    # ---- Final eval on best checkpoint -------------------------------
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        auc, l2, ap = eval_metrics(config, model, test_loader, device)

    final_path = os.path.join(
        ckpt_dir,
        f"Best_ep{checkpoint['epoch']}_l2{int(l2*100)}_ap{int(ap*100)}.pt",
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "loss": checkpoint["loss"],
        "auc": auc, "l2": l2, "ap": ap,
    }, final_path)
    print(f"Final best model → {final_path}")


if __name__ == "__main__":
    main()
