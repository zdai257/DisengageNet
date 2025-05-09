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
from network.utils import SoftArgmax2D, CosineL1, visualize_heatmap, visualize_heatmap2, visualize_heatmap3
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, ToPILImage
from train_gazefollow import GazeDataset, collate_fn, HybridLoss
from train_videoattentiontarget import FocalLoss
from eval import vat_auc, vat_l2


class ChildPlayDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, transform, dir_path="./ChildPlay-gaze", split='train', aug_groups=None):
        self.dir_path = dir_path
        self.split = split
        self.aug = self.split == "train"
        self.aug_groups = aug_groups if aug_groups is not None else []
        self.transform = transform

        df_clips = pd.read_csv(join(dir_path, "clips.csv"))

        self.data = []
        num_other_cls = 0

        for clip in os.listdir(join(dir_path, "annotations", split)):
            if clip.startswith('.') or not clip.endswith('.csv'):
                continue
            clip_name = clip[:11]
            res = df_clips[df_clips['clip'] == clip.split('.csv')[0]]['resolution']

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
                    # cleanup data
                    if xmin > xmax:
                        temp = xmin
                        xmin = xmax
                        xmax = temp
                    if ymin > ymax:
                        temp = ymin
                        ymin = ymax
                        ymax = temp
                    
                    bbox = [xmin, ymin, xmax, ymax]
                    gazex, gazey = frame['gaze_x'], frame['gaze_y']
                    
                    self.data.append((path, bbox, gazex, gazey, inout, res.iloc[0]))

        num_in = sum([item[4] for item in self.data])
        num_out = len(self.data) - num_in
        print("{} has {} total data; {} in; {} out; {} other classes.".format(split, len(self.data),
                                                                              num_in, num_out, num_other_cls))

    def __getitem__(self, idx):
        img_path, bbox, gazex, gazey, inout, res = self.data[idx]
        gazex, gazey = [gazex], [gazey]
        
        img = Image.open(img_path)
        img = img.convert("RGB")
        if res == '720p':
            img = img.resize((1280, 720), Image.Resampling.LANCZOS)
        else:
            img = img.resize((1920, 1080), Image.Resampling.LANCZOS)
        
        width, height = img.size

        # move in out of frame bbox annotations
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, width)
        ymax = min(ymax, height)
        bbox = [xmin, ymin, xmax, ymax]
        
        if self.aug:
            if np.random.sample() <= 0.5:
                img, bbox, gazex, gazey = utils.random_crop(img, bbox, gazex, gazey, inout)
            if np.random.sample() <= 0.5:
                img, bbox, gazex, gazey = utils.horiz_flip(img, bbox, gazex, gazey, inout)
            if np.random.sample() <= 0.5:
                bbox = utils.random_bbox_jitter(img, bbox)
            # add more augmentations

            # update width and height and re-normalize
            width, height = img.size
            bbox_norm = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
            gazex_norm = [x / float(width) for x in gazex]
            gazey_norm = [y / float(height) for y in gazey]

            if np.random.sample() <= 0.5 and 'photometric' in self.aug_groups:
                photometric_transforms = transforms.Compose([
                    transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),
                    transforms.RandomGrayscale(p=0.2),
                    #transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # kernel size can be adjusted
                    #transforms.RandomSolarize(threshold=128, p=0.1),
                    #transforms.RandomPosterize(bits=4, p=0.1),  # Reduce to 4 bits per channel
                    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                    transforms.RandomAutocontrast(p=0.1),
                    #transforms.RandomEqualize(p=0.1),
                ])
                img = photometric_transforms(img)
        else:
            bbox_norm = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
            gazex_norm = [x / float(width) for x in gazex]
            gazey_norm = [y / float(height) for y in gazey]

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
        name="train_childplay",
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

    train_dataset = ChildPlayDataset(transform, aug_groups=config['data']['augmentations'])
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
        heatmap_loss_fn = HybridLoss(bce_weight=config['model']['bce_weight'], mse_weight=0.0, kld_weight=config['model']['kld_weight'])
    else:
        raise TypeError("Loss not supported!")

    if config['model']['is_focal_loss'] == 1:
        inout_loss_fn = FocalLoss(alpha=0.06, gamma=2.0)
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
            #print(heatmap_preds.shape, inout_preds)

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

            #print(inout, inout_preds)
            # DEBUG
            '''
            id = 0
            transform = ToPILImage()
            image = torch.clamp(imgs[id].detach().cpu(), 0, 1)
            image = transform(image)
            
            print(bboxes[id])
            print(gazex[id][0], gazey[id][0])
            torch.set_printoptions(threshold=10_000)
            #print(heatmaps[inout.bool()].to(device))
            viz = visualize_heatmap2(image, heatmaps[id], bbox=bboxes[id], xy=(gazex[id][0]*448, gazey[id][0]*448), dilation_kernel=6,
                                     blur_radius=1.3)  #, transparent_bg=None)
            plt.imshow(viz)
            plt.show()'
            '''
            
            loss = SCALAR * heatmap_loss + 0.1 * inout_loss \
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
        all_inout_preds = []
        all_inout_gts = []

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
                all_inout_preds.append(inout_preds[i].item())
                all_inout_gts.append(inout[i])

        epoch_l2 = np.mean(l2s)
        epoch_auc = np.mean(aucs)
        epoch_inout_ap = average_precision_score(all_inout_gts, all_inout_preds)

        wandb.log({"eval/auc": epoch_auc, "eval/l2": epoch_l2, "AP": epoch_inout_ap, "epoch": epoch})
        print("EVAL EPOCH {}: AUC={}, Aver_L2={}, AP={}".format(epoch,
                                                                round(float(epoch_auc), 4),
                                                                round(float(epoch_l2), 4), round(float(epoch_inout_ap), 4)))


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
