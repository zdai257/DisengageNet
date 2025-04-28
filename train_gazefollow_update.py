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
    # def __init__(self, dataset_name, path, split, transform, in_frame_only=True, sample_rate=1):
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
                if not self.in_frame_only or (self.dataset_name == "videoattentiontarget" and self.data[i]['heads'][j]['inout'] == 1) or (self.dataset_name == "gazefollow" and self.data[i]['heads'][j]['inout'] == 1):
                # if not self.in_frame_only or self.data[i]['heads'][j]['inout'] == 1:
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

        if self.aug and self.aug_groups:
            
            # 1. Geometric Augmentations 
            
            if 'geometric' in self.aug_groups:
                bbox = head_data['bbox']
                gazex = head_data['gazex']
                gazey = head_data['gazey']
                if np.random.sample() <= 0.5:
                    img, bbox, gazex, gazey = utils.random_crop(img, bbox, gazex, gazey, inout)
                if np.random.sample() <= 0.5:
                    img, bbox, gazex, gazey = utils.horiz_flip(img, bbox, gazex, gazey, inout)
                if np.random.sample() <= 0.5:
                    bbox = utils.random_bbox_jitter(img, bbox)

                # New geometric augmentations - These need utils updates
                if np.random.sample() <= 0.5:
                    img, bbox, gazex, gazey = utils.random_scale(img, bbox, gazex, gazey, inout, scale_range=(0.8, 1.2))
                if np.random.sample() <= 0.5:
                    img, bbox, gazex, gazey = utils.random_rotation(img, bbox, gazex, gazey, inout, degrees=(-10, 10))
                if np.random.sample() <= 0.5:
                    img, bbox, gazex, gazey = utils.random_translation(img, bbox, gazex, gazey, inout, translate=(0.1, 0.1))
                if np.random.sample() <= 0.5:
                     img, bbox, gazex, gazey = utils.random_shear(img, bbox, gazex, gazey, inout, shear=(-10, 10))

            # update width and height and re-normalize
            width, height = img.size
            bbox_norm = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
            gazex_norm = [x / float(width) for x in gazex]
            gazey_norm = [y / float(height) for y in gazey]


            # 2. Photometric / Color Augmentations (PIL-based)
            if 'photometric' in self.aug_groups:
                photometric_transforms = transforms.Compose([
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.5),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # kernel size can be adjusted 
                    transforms.RandomSolarize(threshold=128, p=0.1),
                    transforms.RandomPosterize(bits=4, p=0.1), # Reduce to 4 bits per channel
                    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                    transforms.RandomAutocontrast(p=0.1),
                    transforms.RandomEqualize(p=0.1),
                ])
                img = photometric_transforms(img)

            img = self.transform(img) 


            # 3. Occlusion / Erasing Augmentations (Tensor-based)
            if 'occlusion' in self.aug_groups:
                # Random Erasing is typically applied after ToTensor and Normalization
                occlusion_transforms = transforms.Compose([
                    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)) # Adjust parameters as needed
                ])
                img = occlusion_transforms(img)

            # Random Noise (Tensor-based) - Requires custom implementation
            if 'photometric' in self.aug_groups: # Can add noise here if photometric group is active
                if np.random.sample() <= 0.1: # Apply with a certain probability
                    noise = torch.randn_like(img) * 0.05 # Adjust scale of noise
                    img = img + noise
                    img = torch.clamp(img, 0, 1) # Clamp values back to valid range



        # if self.aug:
        #     bbox = head_data['bbox']
        #     gazex = head_data['gazex']
        #     gazey = head_data['gazey']

        #     if np.random.sample() <= 0.5:
        #         img, bbox, gazex, gazey = utils.random_crop(img, bbox, gazex, gazey, inout)
        #     if np.random.sample() <= 0.5:
        #         img, bbox, gazex, gazey = utils.horiz_flip(img, bbox, gazex, gazey, inout)
        #     if np.random.sample() <= 0.5:
        #         bbox = utils.random_bbox_jitter(img, bbox)
        #     # more augmentations

        #     # update width and height and re-normalize
        #     width, height = img.size
        #     bbox_norm = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
        #     gazex_norm = [x / float(width) for x in gazex]
        #     gazey_norm = [y / float(height) for y in gazey]

        # img = self.transform(img)

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
        # ... config loading
        augmentation_groups_to_use = config['train'].get('augmentation_groups', None)
        if augmentation_groups_to_use:
            print(f"Using augmentation groups: {augmentation_groups_to_use}")
        else:
            print("No augmentation groups specified in config.")
    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    # enforce model_name as GazeFollow dataset has no INOUT branch!
    #config['model']['name'] = "gazelle_dinov2_vitl14"
    # if using GazeMoE model
    config['model']['name'] = "gazemoe_dinov2_vitl14_inout"

    wandb.init(
        project=config['model']['name'],
        name="train_gazemoe",
        config=config
    )


    aug_groups_str = "_".join(config['train'].get('augmentation_groups', ['none']))
    checkpoint_dir = "_".join([
        config['model']['name'],
        config['model'].get('moe_type', 'nomoe'),
        str(config['model'].get('is_msf', False)),
        config['train']['pre_optimizer'],
        "bs" + str(config['train']['pre_batch_size']),
        config['model'].get('pbce_loss', 'mse'),
        "lr" + str(config['train']['pre_lr']),
        "fuselr" + str(config['train'].get('pre_fuse_lr', config['train']['pre_lr'])),
        "blocklr" + str(config['train'].get('pre_block_lr', config['train']['pre_lr'])),
        "aug_" + aug_groups_str
    ])

    # checkpoint_dir = "_".join([
    #     config['model']['name'],
    #     config['model']['moe_type'],
    #     str(config['model']['is_msf']),
    #     config['train']['pre_optimizer'],
    #     "bs" + str(config['train']['pre_batch_size']),
    #     config['model']['pbce_loss'],
    #     str(config['train']['pre_lr']),
    #     str(config['train']['pre_fuse_lr']),
    #     str(config['train']['pre_block_lr']),
    # ])
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

    # train_dataset = GazeDataset('gazefollow', config['data']['pre_train_path'], 'train', transform)
    train_dataset = GazeDataset(
        'gazefollow',
        config['data']['pre_train_path'],
        'train',
        transform,
        aug_groups=augmentation_groups_to_use # Pass augmentation groups here
    )
    train_dl = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=config['train']['pre_batch_size'],
                                           shuffle=True,
                                           collate_fn=collate_fn,
                                           num_workers=config['hardware']['num_workers'])
    # eval_dataset = GazeDataset('gazefollow', config['data']['pre_test_path'], 'test', transform)
    eval_dataset = GazeDataset(
        'gazefollow',
        config['data']['pre_test_path'],
        'test',
        transform,
        aug_groups=None
    )

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

    # # MSEloss or BCELoss
    # if config['model']['pbce_loss'] == "mse":
    #     loss_fn = torch.nn.MSELoss(reduction=config['model']['reduction'])
    # elif config['model']['pbce_loss'] == "bce":
    #     loss_fn = torch.nn.BCELoss()

    # MSEloss or BCELoss
    if config['model']['pbce_loss'] == "mse":
        # Use heatmap_kl_loss which combines MSE and KL
        loss_fn = heatmap_kl_loss
    elif config['model']['pbce_loss'] == "bce":
        # If using BCE for heatmap
        loss_fn = torch.nn.BCELoss(reduction=config['model'].get('reduction', 'mean')) # Use .get for reduction
    else:
        # Defaulting to heatmap_kl_loss if pbce_loss is not mse or bce
        loss_fn = heatmap_kl_loss
        print(f"Warning: Unsupported pbce_loss '{config['model']['pbce_loss']}', defaulting to heatmap_kl_loss.")

    # best_min_l2 = 1.0
    best_min_l2 = float('inf') # Initialize with a very large value
    best_epoch = None

    for epoch in range(config['train']['pre_epochs']):
        # TRAIN EPOCH
        model.train()
        for cur_iter, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch

            optimizer.zero_grad()
            preds = model({"images": imgs.to(device), "bboxes": [[bbox] for bbox in bboxes]})
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)

            loss = loss_fn(heatmap_preds, heatmaps.to(device))
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