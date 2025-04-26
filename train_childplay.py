import numpy as np
import os
from os.path import join
import json
import yaml
import random
from PIL import Image
from sklearn.metrics import average_precision_score
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import RMSprop, Adam, AdamW
import wandb
from tqdm import tqdm
from network.network_builder import get_gazelle_model
from network.network_builder_update2 import get_gt360_model, get_gazemoe_model
import network.utils as utils
from train_gazefollow import GazeDataset, collate_fn, HybridLoss
from train_videoattentiontarget import FocalLoss
from eval import vat_auc, vat_l2


class ChildPlayDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, transform, dir_path="/Users/zhuangzhuangdai/repos/ChildPlay-gaze", split='train'):
        self.dir_path = dir_path
        self.split = split
        self.aug = self.split == "train"
        self.transform = transform

        df_clips = pd.read_csv(join(dir_path, "clips.csv"))

        self.data = []
        num_other_cls = 0

        for clip in os.listdir(join(dir_path, "annotations", split)):
            if clip.startswith('.') or not clip.endswith('.csv'):
                continue
            clip_name = clip[:11]

            if 1:
                annotation = join(dir_path, "annotations", split, clip)
                df = pd.read_csv(annotation)

                for id, frame in df.iterrows():
                    if frame['gaze_class'] == 'inside_visible':
                        inout = 1
                    elif frame['gaze_class'] == 'outside_frame':
                        inout = 0
                    else:
                        num_other_cls += 1
                        continue
                    frame_idx = int(frame['frame']) - 1
                    start_frame = int(clip.split('_')[-1].split('-')[0])
                    frame_idx += start_frame

                    img_idx = clip_name + f'_{frame_idx}.jpg'
                    path = join(dir_path, "images", clip.split('.')[0], img_idx)

                    xmin, ymin = float(frame['bbox_x']), float(frame['bbox_y'])
                    xmax = float(frame['bbox_x']) + float(frame['bbox_width'])
                    ymax = float(frame['bbox_y']) + float(frame['bbox_height'])

                    bbox = [xmin, ymin, xmax, ymax]
                    gazex, gazey = float(frame['gaze_x']), float(frame['gaze_y'])

                    self.data.append((path, bbox, gazex, gazey, inout))

        num_in = sum([item[4] for item in self.data])
        num_out = len(self.data) - num_in
        print("{} has {} total data; {} in; {} out; {} other classes.".format(split, len(self.data),
                                                                              num_in, num_out, num_other_cls))

    def __getitem__(self, idx):
        img_path, bbox, gazex, gazey, inout = self.data[idx]
        gazex, gazey = [gazex], [gazey]

        img = Image.open(img_path)
        img = img.convert("RGB")
        width, height = img.size

        gazex_norm = [gazex[0] / width]
        gazey_norm = [gazey[0] / height]

        if self.aug:
            if np.random.sample() <= 0.5:
                img, bbox, gazex, gazey = utils.random_crop(img, bbox, gazex, gazey, inout)
            if np.random.sample() <= 0.5:
                img, bbox, gazex, gazey = utils.horiz_flip(img, bbox, gazex, gazey, inout)
            if np.random.sample() <= 0.5:
                bbox = utils.random_bbox_jitter(img, bbox)
            # TODO more augmentations

            # update width and height and re-normalize
            width, height = img.size
            bbox_norm = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
            gazex_norm = [x / float(width) for x in gazex]
            gazey_norm = [y / float(height) for y in gazey]
        else:
            bbox_norm = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]

        img = self.transform(img)

        if self.split == "train":  # note for training set, there is only one annotation
            heatmap = utils.get_heatmap(gazex_norm[0], gazey_norm[0], 64, 64)
            return img, bbox_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width, heatmap
        else:
            return img, bbox_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width

    def __len__(self):
        return len(self.data)


def main():
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)
    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    wandb.init(
        project=config['model']['name'],
        name="train_gf360",
        config=config
    )

    checkpoint_dir = "_".join([
        "ChildPlay",
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
    print("ChildPlay checkpoint saved at: ", exp_dir)

    # load pretrained model
    # model, transform = get_gazelle_model(config)
    model, transform = get_gazemoe_model(config)

    print("Loading model from {}".format(config['model']['pretrained_path']))
    ### initializing from ckpt without inout head ###
    model.load_gazelle_state_dict(
        torch.load(config['model']['pretrained_path'], weights_only=True, map_location=device))
    ### if loading model incl. backbone ###
    # model.load_gazelle_state_dict(
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

    train_dataset = ChildPlayDataset(transform)
    train_dl = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=config['train']['batch_size'],
                                           shuffle=True,
                                           collate_fn=collate_fn,
                                           num_workers=config['hardware']['num_workers'])

    eval_dataset = ChildPlayDataset(transform, split='test')
    eval_dl = torch.utils.data.DataLoader(eval_dataset,
                                          batch_size=config['train']['batch_size'],
                                          shuffle=False,
                                          collate_fn=collate_fn,
                                          num_workers=config['hardware']['num_workers'])
    print(train_dataset.__len__(), eval_dataset.__len__())

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
        heatmap_loss_fn = HybridLoss(bce_weight=1.0, mse_weight=0.0, kld_weight=0.1)
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
            bbox_ctrs = [torch.tensor([(bbx[0] + bbx[2]) / 2, (bbx[1] + bbx[3]) / 2], dtype=torch.float32) for bbx in
                         bboxes]
            bbox_ctrs = torch.stack(bbox_ctrs)
            # compute angle loss only for in-frame gaze targets
            pred_vec = pred_xys - bbox_ctrs.to(device)
            gt_vec = gt_xys.to(device) - bbox_ctrs.to(device)
            angle_loss = angle_loss_fn(pred_vec[inout.bool()], gt_vec[inout.bool()])

            loss = SCALAR * heatmap_loss + config['model']['bce_weight'] * inout_loss \
                   + config['model']['angle_weight'] * angle_loss.mean()
            loss.backward()
            optimizer.step()

            if cur_iter % config['logging']['save_every'] == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/heatmap_loss": heatmap_loss.item(),
                    "train/inout_loss": inout_loss.item()
                })
                # print("TRAIN EPOCH {}, iter {}/{}, loss={}".format(epoch, cur_iter, len(train_dl), round(loss.item(), 4)))

        scheduler.step()

        ckpt_path = os.path.join(exp_dir, 'epoch_{}.pt'.format(epoch))
        torch.save(model.get_gazelle_state_dict(), ckpt_path)
        print("Saved checkpoint to {}".format(ckpt_path))

        # EVAL
        model.eval()
        l2s = []
        aucs = []

        for cur_iter, batch in tqdm(enumerate(eval_dl), total=len(eval_dl)):
            imgs, bboxes, gazex, gazey, inout, heights, widths = batch

            with torch.no_grad():
                preds = model({"images": imgs.to(device), "bboxes": [[bbox] for bbox in bboxes]})

            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            inout_preds = torch.stack(preds['inout']).squeeze(dim=1)
            for i in range(heatmap_preds.shape[0]):
                if inout[i] == 1:  # in-frame
                    auc = vat_auc(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    l2 = vat_l2(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    aucs.append(auc)
                    l2s.append(l2)

        epoch_l2 = np.mean(l2s)
        epoch_auc = np.mean(aucs)

        wandb.log({"eval/auc": epoch_auc, "eval/l2": epoch_l2, "epoch": epoch})
        print("EVAL EPOCH {}: AUC={}, Aver_L2={}".format(epoch,
                                                         round(float(epoch_auc), 4),
                                                         round(float(epoch_l2), 4)))


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
