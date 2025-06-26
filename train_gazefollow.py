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
from network.network_builder_update2 import get_gt360_model
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


# combine BCE/MSE with KL Divergence Loss
class HybridLoss(torch.nn.Module):
    def __init__(self, bce_weight=1.0, mse_weight=0.0, kld_weight=0.1, kld_reduction='batchmean'):
        super(HybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.kld_weight = kld_weight
        self.kld_reduction = kld_reduction
        self.bce_loss = torch.nn.BCELoss(reduction='mean')
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.kld_loss = torch.nn.KLDivLoss(reduction=kld_reduction)

    def forward(self, pred, target):
        loss = 0.0
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce_loss(pred, target)
            # print('bce_loss ', loss)
        if self.mse_weight > 0:
            loss += self.mse_weight * self.mse_loss(pred, target)
        if self.kld_weight > 0:
            # print(pred.shape, target.shape)
            pred_dist = pred / (pred.sum(dim=(1, 2), keepdim=True) + 1e-10)
            target_dist = target / (target.sum(dim=(1, 2), keepdim=True) + 1e-10)
            pred_log_dist = torch.log(pred_dist + 1e-10)
            kld_loss = self.kld_weight * self.kld_loss(pred_log_dist, target_dist)
            # bce_loss 0.05 ~ 0.1 ; kld_loss 2.0 ~ 2.4
            # print("kld_loss", kld_loss)
            loss += kld_loss
        return loss


class GazeDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_name, path, split, transform, in_frame_only=True, sample_rate=1, aug_groups=None):
        self.dataset_name = dataset_name
        self.path = path
        self.split = split
        self.aug = self.split == "train"
        self.transform = transform
        self.in_frame_only = in_frame_only
        self.sample_rate = sample_rate
        self.aug_groups = aug_groups if aug_groups is not None else []

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

            if np.random.sample() <= 0.5 and 'crop' in self.aug_groups:
                img, bbox, gazex, gazey = utils.random_crop(img, bbox, gazex, gazey, inout)
            if np.random.sample() <= 0.5 and 'crop' in self.aug_groups:
                img, bbox, gazex, gazey = utils.horiz_flip(img, bbox, gazex, gazey, inout)
            # more augmentations
            if np.random.sample() <= 0.5 and 'crop' in self.aug_groups:
                bbox = utils.random_bbox_jitter(img, bbox)

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
                    # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # kernel size can be adjusted
                    # transforms.RandomSolarize(threshold=128, p=0.1),
                    # transforms.RandomPosterize(bits=4, p=0.1),  # Reduce to 4 bits per channel
                    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                    transforms.RandomAutocontrast(p=0.1),
                    # transforms.RandomEqualize(p=0.1),
                ])
                img = photometric_transforms(img)

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
        name="pretrain_gf_GT360",
        config=config
    )

    checkpoint_dir = "_".join([
        "GT360",
        config['model']['name'],
        config['train']['pre_optimizer'],
        "bs" + str(config['train']['pre_batch_size']),
        str(config['train']['pre_lr']),
    ])
    exp_dir = os.path.join(config['logging']['pre_dir'], checkpoint_dir)
    os.makedirs(exp_dir, exist_ok=True)
    print("Pretrained checkpoint saved at: ", exp_dir)

    # model, transform = get_gazelle_model(config)
    model, transform = get_gt360_model(config)
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

    train_dataset = GazeDataset('gazefollow', config['data']['pre_train_path'], 'train', transform,
                                aug_groups=config['data']['augmentations'])
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
                param_dicts.append({'params': p, 'lr': config['train']['pre_lr']})
            elif 'transformer' in n:
                param_dicts.append({'params': p, 'lr': config['train']['pre_lr']})
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
    SCALAR = 1
    # MSEloss or BCELoss
    if config['model']['pbce_loss'] == "mse":
        SCALAR = 36
        loss_fn = torch.nn.MSELoss(reduction=config['model']['reduction'])
    elif config['model']['pbce_loss'] == "bce":
        loss_fn = torch.nn.BCELoss()
    elif config['model']['pbce_loss'] == "hybrid":
        loss_fn = HybridLoss(bce_weight=config['model']['bce_weight'], mse_weight=config['model']['mse_weight'],
                             kld_weight=config['model']['kld_weight'])
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

            loss = SCALAR * loss_fn(heatmap_preds, heatmaps.to(device))
            loss.backward()
            optimizer.step()

            if cur_iter % config['logging']['save_every'] == 0:
                wandb.log({"train/loss": loss.item()})
                # print("TRAIN EPOCH {}, iter {}/{}, loss={}".format(epoch, cur_iter, len(train_dl), round(loss.item(), 4)))

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
