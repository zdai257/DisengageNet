import numpy as np
import os
from os.path import join
import json
import yaml
import random
import copy
from PIL import Image
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop, Adam, AdamW
import wandb
from tqdm import tqdm
from network.network_builder import get_gazelle_model
from network.network_builder_update2 import get_gt360_model, get_gazemoe_model
import network.utils as utils
from train_gazefollow import GazeDataset, collate_fn, HybridLoss
from eval import vat_auc, vat_l2


def load_data_vat(file, sample_rate):
    sequences = json.load(open(file, "r"))
    data = []
    for i in range(len(sequences)):
        for j in range(0, len(sequences[i]['frames']), sample_rate):
            data.append(sequences[i]['frames'][j])
    return data


def load_data_gazefollow(file):
    data = json.load(open(file, "r"))
    return data


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.4/0.6, gamma=2.0, reduction='mean', apply_sigmoid=False):
        """
            alpha (float): Weighting factor for the minority class (e.g., 0.25 for imbalanced datasets).
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


def main():
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)
    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    wandb.init(
        project=config['model']['name'],
        name="train_vat",
        config=config
    )

    checkpoint_dir = "_".join([
        "VAT",
        config['model']['name'],
        config['model']['moe_type'],
        str(config['model']['is_msf']),
        config['train']['optimizer'],
        "bs" + str(config['train']['batch_size']),
        config['model']['pbce_loss'],
        str(config['train']['lr']),
        str(config['train']['fuse_lr']),
        str(config['train']['block_lr']),
        str(config['train']['inout_lr']),
    ])
    exp_dir = os.path.join(config['logging']['log_dir'], checkpoint_dir)
    os.makedirs(exp_dir, exist_ok=True)
    print("VAT checkpoint saved at: ", exp_dir)

    # load pretrained model
    #model, transform = get_gazelle_model(config)
    model, transform = get_gazemoe_model(config)
    print("Loading model from {}".format(config['model']['pretrained_path']))

    ### initializing from ckpt without inout head ###
    model.load_gazelle_state_dict(torch.load(config['model']['pretrained_path'], weights_only=True, map_location=device))
    ### if loading model incl. backbone ###
    #model.load_gazelle_state_dict(
    #    torch.load(config['model']['pretrained_path'], weights_only=True, map_location=device), include_backbone=False)
    model.to(device)

    for name, param in model.named_parameters():  # Randomly initialize learnable parameters
        break
        if param.requires_grad:  # Only initialize unfrozen parameters
            if param.dim() > 1:  # Weights
                torch.nn.init.xavier_normal_(param)
            else:  # Biases
                torch.nn.init.zeros_(param)

    for param in model.backbone.parameters():  # freeze backbone
        param.requires_grad = False
    print(f"Learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_dataset = GazeDataset('videoattentiontarget', config['data']['train_path'], 'train', transform,
                                in_frame_only=False, sample_rate=6)
    train_dl = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=config['train']['batch_size'],
                                           shuffle=True,
                                           collate_fn=collate_fn,
                                           num_workers=config['hardware']['num_workers'])

    # Note this eval dataloader samples frames sparsely for efficiency - for final results, run eval_vat.py which uses sample rate 1
    eval_dataset = GazeDataset('videoattentiontarget', config['data']['test_path'], 'test', transform,
                               in_frame_only=False, sample_rate=1)  # sample_rate=6 / 1
    eval_dl = torch.utils.data.DataLoader(eval_dataset,
                                          batch_size=config['train']['batch_size'],
                                          shuffle=False,
                                          collate_fn=collate_fn,
                                          num_workers=config['hardware']['num_workers'])

    param_dicts = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if 'ms_fusion' in n or 'linear' in n:
                param_dicts.append({'params': p, 'lr': config['train']['fuse_lr']})
            elif 'transformer' in n:
                param_dicts.append({'params': p, 'lr': config['train']['block_lr']})
            elif 'inout' in n:
                param_dicts.append({'params': p, 'lr': config['train']['inout_lr']})
            else:  # heatmap head params
                param_dicts.append({'params': p, 'lr': config['train']['lr']})

    if config['train']['pre_optimizer'] == 'Adam':
        optimizer = Adam(param_dicts)
    elif config['train']['pre_optimizer'] == 'AdamW':
        optimizer = AdamW(param_dicts)
    else:
        raise TypeError("Optimizer not supported!")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=config['train']['lr_scheduler']['step_size'],
                                                           eta_min=float(config['train']['lr_scheduler']['min_lr']))

    ### Auxiliary Loss ###
    angle_loss_fn = utils.CosineL1()
    softArgmax_fn = utils.SoftArgmax2D()
    SCALAR = 1
    # MSEloss or BCELoss
    if config['model']['pbce_loss'] == "mse":
        SCALAR = 36
        heatmap_loss_fn = torch.nn.MSELoss(reduction=config['model']['reduction'])
    elif config['model']['pbce_loss'] == "bce":
        heatmap_loss_fn = torch.nn.BCELoss()
    elif config['model']['pbce_loss'] == "hybrid":
        heatmap_loss_fn = HybridLoss(bce_weight=config['model']['bce_weight'], mse_weight=0.0, kld_weight=config['model']['kld_weight'])
    else:
        raise TypeError("Loss not supported!")

    if config['model']['is_focal_loss'] == 1:
        inout_loss_fn = FocalLoss()
    else:
        inout_loss_fn = nn.BCELoss()

    for epoch in range(config['train']['epochs']):
        # TRAIN EPOCH
        model.train()
        for cur_iter, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch

            optimizer.zero_grad()
            preds = model({"images": imgs.to(device), "bboxes": [[bbox] for bbox in bboxes]})
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            inout_preds = torch.stack(preds['inout']).squeeze(dim=1)

            # compute heatmap loss only for in-frame gaze targets
            heatmap_loss = heatmap_loss_fn(heatmap_preds[inout.bool()], heatmaps[inout.bool()].to(device))
            inout_loss = inout_loss_fn(inout_preds, inout.float().to(device))

            ### Introduce gaze angle L1 loss ###
            pred_xys = softArgmax_fn(heatmap_preds)
            gt_xys = [torch.tensor([gtx, gty], dtype=torch.float32) for gtx, gty in zip(gazex, gazey)]
            gt_xys = torch.stack(gt_xys).squeeze(2)
            bbox_ctrs = [torch.tensor([(bbx[0] + bbx[2]) / 2, (bbx[1] + bbx[3]) / 2], dtype=torch.float32) for bbx in bboxes]
            bbox_ctrs = torch.stack(bbox_ctrs)
            # compute angle loss only for in-frame gaze targets
            pred_vec = pred_xys - bbox_ctrs.to(device)
            gt_vec = gt_xys.to(device) - bbox_ctrs.to(device)
            angle_loss = angle_loss_fn(pred_vec[inout.bool()], gt_vec[inout.bool()])
            # inout: bce_loss 0.3 ~ 0.6 ; focal_loss 0.07 ~ 0.15
            #print('hm_loss : {}, inout_loss: {}'.format(heatmap_loss, inout_loss))
            loss = SCALAR * heatmap_loss + 1.0 * inout_loss \
                   + config['model']['angle_weight'] * angle_loss.mean()
            loss.backward()
            optimizer.step()

            if cur_iter % config['logging']['save_every'] == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/heatmap_loss": heatmap_loss.item(),
                    "train/inout_loss": inout_loss.item()
                })
                #print("TRAIN EPOCH {}, iter {}/{}, loss={}".format(epoch, cur_iter, len(train_dl), round(loss.item(), 4)))

        scheduler.step()

        ckpt_path = os.path.join(exp_dir, 'epoch_{}.pt'.format(epoch))
        torch.save(model.get_gazelle_state_dict(), ckpt_path)
        print("Saved checkpoint to {}".format(ckpt_path))

        # EVAL
        model.eval()
        l2s = []
        aucs = []
        all_inout_preds = []
        all_inout_gts = []

        for cur_iter, batch in tqdm(enumerate(eval_dl), total=len(eval_dl)):
            imgs, bboxes, gazex, gazey, inout, heights, widths = batch

            with torch.no_grad():
                preds = model({"images": imgs.to(device), "bboxes": [[bbox] for bbox in bboxes]})

            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            inout_preds = torch.stack(preds['inout']).squeeze(dim=1)
            for i in range(heatmap_preds.shape[0]):
                if inout[i] == 1: # in-frame
                    auc = vat_auc(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    l2 = vat_l2(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    aucs.append(auc)
                    l2s.append(l2)
                all_inout_preds.append(inout_preds[i].item())
                all_inout_gts.append(inout[i])

        epoch_l2 = np.mean(l2s)
        epoch_auc = np.mean(aucs)
        epoch_inout_ap = average_precision_score(all_inout_gts, all_inout_preds)

        wandb.log({"eval/auc": epoch_auc, "eval/l2": epoch_l2, "eval/inout_ap": epoch_inout_ap, "epoch": epoch})
        print("EVAL EPOCH {}: AUC={}, L2={}, Inout AP={}".format(epoch,
                                                                 round(float(epoch_auc), 4),
                                                                 round(float(epoch_l2), 4),
                                                                 round(float(epoch_inout_ap), 4)))


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
