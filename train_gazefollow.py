import torch
import torch.nn.functional as F
from torch.optim import RMSprop, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import json
import os
from os.path import join
import copy
import numpy as np
from tqdm import tqdm
import yaml
import random
import wandb
from network.network_builder import get_gazelle_model
from network.network_builder_update2 import get_gt360_model, get_gazemoe_model
import network.utils as utils
from eval import eval_pretrain_gazefollow, gazefollow_auc, gazefollow_l2


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


# combine Mean Squared Error (MSE) with KL Divergence to ensure smooth heatmap predictions
def heatmap_kl_loss(pred, target):
    mse = torch.nn.MSELoss()(pred, target)
    kl = torch.nn.KLDivLoss(reduction="batchmean")(pred.log(), target)
    return mse + 0.1 * kl


# For in/out, use focal loss to handle class imbalance
def focal_loss(pred, target, alpha=0.25, gamma=2):
    bce = torch.nn.BCELoss(reduction="none")(pred, target)
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


class GazeDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_name, path, split, transform, in_frame_only=True, sample_rate=1):
        self.dataset_name = dataset_name
        self.path = path
        self.split = split
        self.aug = self.split == "train"
        self.transform = transform
        self.in_frame_only = in_frame_only
        self.sample_rate = sample_rate

        if dataset_name == "gazefollow":
            self.data = load_data_gazefollow(os.path.join(self.path, "{}_preprocessed.json".format(split)))
        elif dataset_name == "videoattentiontarget":
            self.data = load_data_vat(os.path.join(self.path, "{}_preprocessed.json".format(split)),
                                      sample_rate=sample_rate)
        else:
            raise ValueError("Invalid dataset: {}".format(dataset_name))

        self.data_idxs = []
        for i in range(len(self.data)):
            for j in range(len(self.data[i]['heads'])):
                if not self.in_frame_only or self.data[i]['heads'][j]['inout'] == 1:
                    self.data_idxs.append((i, j))

    def __getitem__(self, idx):
        img_idx, head_idx = self.data_idxs[idx]
        img_data = self.data[img_idx]
        head_data = copy.deepcopy(img_data['heads'][head_idx])
        bbox_norm = head_data['bbox_norm']
        gazex_norm = head_data['gazex_norm']
        gazey_norm = head_data['gazey_norm']
        inout = head_data['inout']

        img_path = os.path.join(self.path, img_data['path'])
        img = Image.open(img_path)
        img = img.convert("RGB")
        width, height = img.size

        if self.aug:
            bbox = head_data['bbox']
            gazex = head_data['gazex']
            gazey = head_data['gazey']

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

        img = self.transform(img)

        if self.split == "train":  # note for training set, there is only one annotation
            heatmap = utils.get_heatmap(gazex_norm[0], gazey_norm[0], 64, 64)
            return img, bbox_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width, heatmap
        else:
            return img, bbox_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width

    def __len__(self):
        return len(self.data_idxs)


def collate_fn(batch):
    transposed = list(zip(*batch))
    return tuple(
        torch.stack(items) if isinstance(items[0], torch.Tensor) else list(items)
        for items in transposed
    )


def main():
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)
    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    wandb.init(
        project=config['model']['name'],
        name="pretrain_gf",
        config=config
    )

    checkpoint_dir = "_".join([
        config['model']['name'],
        config['model']['moe_type'],
        str(config['model']['is_msf']),
        config['train']['pre_optimizer'],
        "bs" + str(config['train']['pre_batch_size']),
        config['model']['pbce_loss'],
        str(config['train']['pre_lr']),
        str(config['train']['pre_fuse_lr']),
        str(config['train']['pre_block_lr']),
    ])
    exp_dir = os.path.join(config['logging']['pre_dir'], checkpoint_dir)
    os.makedirs(exp_dir, exist_ok=True)
    print("Pretrained checkpoint saved at: ", exp_dir)

    #model, transform = get_gazelle_model(config)
    model, transform = get_gazemoe_model(config)
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

    train_dataset = GazeDataset('gazefollow', config['data']['pre_train_path'], 'train', transform)
    train_dl = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=config['train']['pre_batch_size'],
                                           shuffle=True,
                                           collate_fn=collate_fn,
                                           num_workers=config['hardware']['num_workers'])
    eval_dataset = GazeDataset('gazefollow', config['data']['pre_test_path'], 'test', transform)
    eval_dl = torch.utils.data.DataLoader(eval_dataset,
                                          batch_size=config['train']['pre_batch_size'],
                                          shuffle=False,
                                          collate_fn=collate_fn,
                                          num_workers=config['hardware']['num_workers'])

    param_dicts = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if 'ms_fusion' in n or 'linear' in n:
                param_dicts.append({'params': p, 'lr': config['train']['pre_fuse_lr']})
            elif 'transformer' in n:
                param_dicts.append({'params': p, 'lr': config['train']['pre_block_lr']})
            else:
                param_dicts.append({'params': p, 'lr': config['train']['pre_lr']})

    if config['train']['pre_optimizer'] == 'Adam':
        optimizer = Adam(param_dicts)
    elif config['train']['pre_optimizer'] == 'AdamW':
        optimizer = AdamW(param_dicts)
    else:
        raise TypeError("Optimizer not supported!")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=config['train']['pre_lr_scheduler']['step_size'],
                                                           eta_min=float(config['train']['pre_lr_scheduler']['min_lr']))

    ### Auxiliary Loss ###
    angle_loss_fn = utils.CosineL1()
    softArgmax_fn = utils.SoftArgmax2D()
    # MSEloss or BCELoss
    if config['model']['pbce_loss'] == "mse":
        loss_fn = torch.nn.MSELoss(reduction=config['model']['reduction'])
    elif config['model']['pbce_loss'] == "bce":
        loss_fn = torch.nn.BCELoss()
    else:
        raise TypeError("Loss not supported!")

    best_min_l2 = 1.0
    best_epoch = None

    for epoch in range(config['train']['pre_epochs']):
        # TRAIN EPOCH
        model.train()
        for cur_iter, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch

            optimizer.zero_grad()
            preds = model({"images": imgs.to(device), "bboxes": [[bbox] for bbox in bboxes]})
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)

            ### Introduce gaze angle L1 loss ###
            pred_xys = softArgmax_fn(heatmap_preds)
            gt_xys = [torch.tensor([gtx, gty], dtype=torch.float32) for gtx, gty in zip(gazex, gazey)]
            gt_xys = torch.stack(gt_xys).squeeze(2)
            bbox_ctrs = [torch.tensor([(bbx[0] + bbx[2]) / 2, (bbx[1] + bbx[3]) / 2], dtype=torch.float32) for bbx in bboxes]
            bbox_ctrs = torch.stack(bbox_ctrs)
            angle_loss = angle_loss_fn(pred_xys - bbox_ctrs.to(device),
                                       gt_xys.to(device) - bbox_ctrs.to(device))
            # heatmap-bce value: 0.04 ~ 0.1; cosine angle loss value: 0 ~ 2
            loss = loss_fn(heatmap_preds, heatmaps.to(device)) + config['model']['angle_weight'] * angle_loss.mean()
            loss.backward()
            optimizer.step()

            if cur_iter % config['logging']['save_every'] == 0:
                wandb.log({"train/loss": loss.item()})
                #print("TRAIN EPOCH {}, iter {}/{}, loss={}".format(epoch, cur_iter, len(train_dl), round(loss.item(), 4)))

        scheduler.step()

        ckpt_path = os.path.join(exp_dir, 'epoch_{}.pt'.format(epoch))
        torch.save(model.get_gazelle_state_dict(), ckpt_path)
        print("Saved checkpoint to {}".format(ckpt_path))

        # EVAL
        model.eval()
        avg_l2s = []
        min_l2s = []
        aucs = []
        for cur_iter, batch in tqdm(enumerate(eval_dl), total=len(eval_dl)):
            imgs, bboxes, gazex, gazey, inout, heights, widths = batch

            with torch.no_grad():
                preds = model({"images": imgs.to(device), "bboxes": [[bbox] for bbox in bboxes]})

            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            for i in range(heatmap_preds.shape[0]):
                auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
                avg_l2, min_l2 = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
                aucs.append(auc)
                avg_l2s.append(avg_l2)
                min_l2s.append(min_l2)

        epoch_avg_l2 = np.mean(avg_l2s)
        epoch_min_l2 = np.mean(min_l2s)
        epoch_auc = np.mean(aucs)

        wandb.log({"eval/auc": epoch_auc, "eval/min_l2": epoch_min_l2, "eval/avg_l2": epoch_avg_l2, "epoch": epoch})
        print("EVAL EPOCH {}: AUC={}, Min L2={}, Avg L2={}".format(epoch, round(float(epoch_auc), 4),
                                                                   round(float(epoch_min_l2), 4),
                                                                   round(float(epoch_avg_l2), 4)))

        if epoch_min_l2 < best_min_l2:
            best_min_l2 = epoch_min_l2
            best_epoch = epoch

    print("Completed training. Best Min L2 of {} obtained at epoch {}".format(round(best_min_l2, 4), best_epoch))


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
