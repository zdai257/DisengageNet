import argparse
import torch
import torch.nn.functional as F
from torch.optim import RMSprop, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms.functional as TF
from PIL import Image
import json
import os
from os.path import join
import numpy as np
from tqdm import tqdm
import yaml

from network.network_builder import get_gazelle_model, get_gt360_model
from eval import eval_pretrain_gazefollow


class GazeFollow(torch.utils.data.Dataset):
    def __init__(self, root_path, img_transform, split='train'):
        self.frames = json.load(open(os.path.join(root_path, "{}_preprocessed.json".format(split)), "rb"))
        self.root_path = root_path
        self.transform = img_transform

    def __getitem__(self, idx):
        frame = self.frames[idx]

        image = Image.open(join(self.root_path, frame['path'])).convert("RGB")
        image = self.transform(image)
        h = frame['height']
        w = frame['width']
        bbox = frame['bbox_norm']
        gazex = frame['gazex_norm']
        gazey = frame['gazey_norm']

        return image, bbox, gazex, gazey, h, w

    def __len__(self):
        return len(self.frames)


def collate(batch):
    images, bbox, gazex, gazey, height, width = zip(*batch)
    return torch.stack(images), list(bbox), list(gazex), list(gazey), list(height), list(width)


@torch.no_grad()
def evaluate(config, model, val_loader, device):
    model.eval()
    batch_size = config['eval']['batch_size']

    val_pbce_loss = torch.nn.BCEWithLogitsLoss()
    validation_loss = 0.0
    val_total = len(val_loader)

    with tqdm(total=val_total) as pbar:
        for images, bboxes, gazex, gazey, h, w in val_loader:

            preds = model({"images": images.to(device), "bboxes": [bboxes]})

            # preds = a dict of{'heatmap': list of Batch_size*tensor[head_count, 64, 64],
            #                   'inout': list of Batch_size*tensor[head_count,] }

            pred_heatmap = preds['heatmap'][0]

            gt_gaze_xy = []
            for gtxs, gtys in zip(gazex, gazey):
                batch_size0 = len(gtxs)
                # for GazeFollow, len should always be 1 (?!), so gtxs is no list ??
                for i, (gtx, gty) in enumerate(zip(gtxs, gtys)):
                    gt_heatmap = torch.zeros((64, 64))
                    x_grid = int(gtx * 63)
                    y_grid = int(gty * 63)

                    gt_heatmap[y_grid, x_grid] = 1

                gt_gaze_xy.append(gt_heatmap)

            gt_heatmaps = torch.stack(gt_gaze_xy)

            loss = val_pbce_loss(pred_heatmap, gt_heatmaps)
            validation_loss += loss.item()

            pbar.update(1)

    return float(validation_loss / val_total)


def main():
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    # load GazeFollow config
    lr = config['train']['pre_lr']
    num_epochs = config['train']['pre_epochs']

    # TODO: my_net
    # my_net = get_gt360_model(config)
    model, gazelle_transform = get_gazelle_model(config)
    #model.load_gazelle_state_dict(torch.load(config['model']['pretrained_path'], weights_only=True),
    #                              include_backbone=False)

    # Verify the freezing and initialization
    for name, param in model.named_parameters():
        #print(f"{name}: requires_grad={param.requires_grad}")
        pass

    # Freeze 'backbone' parameters
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False  # Freeze these parameters
        else:
            param.requires_grad = True  # Keep these learnable

    # Randomly initialize learnable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only initialize unfrozen parameters
            if param.dim() > 1:  # Weights
                torch.nn.init.xavier_normal_(param)
            else:  # Biases
                torch.nn.init.zeros_(param)

    # Verify the freezing and initialization
    for name, param in model.named_parameters():
        #print(f"{name}: requires_grad={param.requires_grad}")
        pass

    model.to(device)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    # set LR
    param_dicts = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            param_dicts.append({'params': p, 'lr': lr})

    if config['train']['pre_optimizer'] == 'Adam':
        optimizer = Adam(param_dicts)
    elif config['train']['pre_optimizer'] == 'AdamW':
        optimizer = AdamW(param_dicts)
    else:
        raise TypeError("Optimizer not supported!")

    # Cosine/linear learning rate scheduler
    lr_step_size = config['train']['pre_lr_scheduler']['step_size']
    gamma = config['train']['pre_lr_scheduler']['gamma']  # Factor by which the learning rate will be reduced
    if config['train']['pre_lr_scheduler']['type'] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=100)
        # eta_min=config['train']['lr_scheduler']['min_lr'])
    else:  # linear lr scheduler
        scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

    input_resolution = config['data']['input_resolution']

    # transform
    # RandomCrop + flip + BBox Jitter (?)
    transform_list = []
    transform_list.append(transforms.Resize(input_resolution))
    #transform_list.append(transforms.RandomResizedCrop((input_resolution, input_resolution)))
    # TODO: deal with flipped bbox labels
    # transform_list.append(transforms.RandomHorizontalFlip(0.5))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    my_transform = transforms.Compose(transform_list)

    my_transform = gazelle_transform

    # apply my_transform
    train_dataset = GazeFollow(root_path=config['data']['pre_train_path'],
                               img_transform=my_transform,
                               split='train')
    test_dataset = GazeFollow(root_path=config['data']['pre_test_path'],
                              img_transform=my_transform,
                              split='test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['train']['pre_batch_size'],
        collate_fn=collate,
        # shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['eval']['batch_size'],
        collate_fn=collate,
        # shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )

    #train_length = train_dataset.__len__()

    #mse_loss = torch.nn.MSELoss(reduction='sum')  # Mean Squared Error Loss
    # Pixel wise binary CrossEntropy loss
    pbce_loss = torch.nn.BCEWithLogitsLoss(reduction="sum")

    # save dir for checkpoints
    os.makedirs(config['logging']['pre_dir'], exist_ok=True)

    best_loss = float('inf')
    early_stop_count = 0
    best_checkpoint_path = None
    batch_size = config['train']['pre_batch_size']

    # START TRAINING
    for epoch in range(config['train']['pre_epochs']):
        model.train(True)
        epoch_loss = 0.

        for batch, (images, bboxes, gazex, gazey, h, w) in tqdm(enumerate(train_loader), total=len(train_loader)):

            # forward pass
            preds = model({"images": images.to(device), "bboxes": [bboxes]})

            # preds = a dict of{'heatmap': list of head_count *tensor[Batch_size, 64, 64],
            #                   'inout': list of head_count *tensor[Batch_size,] }

            pred_heatmap = preds['heatmap'][0]

            gt_gaze_xy = []
            for gtxs, gtys in zip(gazex, gazey):
                batch_size0 = len(gtxs)
                # for GazeFollow, len should always be 1 (?!), so gtxs is no list ??
                for i, (gtx, gty) in enumerate(zip(gtxs, gtys)):
                    gt_heatmap = torch.zeros((64, 64))
                    x_grid = int(gtx * 63)
                    y_grid = int(gty * 63)

                    gt_heatmap[y_grid, x_grid] = 1

                gt_gaze_xy.append(gt_heatmap)

            gt_heatmaps = torch.stack(gt_gaze_xy)

            loss = pbce_loss(pred_heatmap, gt_heatmaps)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip exploding gradients
            optimizer.step()

            # Accumulate loss for reporting
            epoch_loss += loss.item()

        scheduler.step()

        mean_ep_loss = epoch_loss / len(train_loader)
        # Print epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {mean_ep_loss:.7f}")

        val_loss = evaluate(config, model, test_loader, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.7f}")

        # Save model every 5 epochs
        if (epoch + 1) % config['logging']['save_every'] == 0:
            checkpoint_path = os.path.join(config['logging']['pre_dir'], f"model_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': val_loss
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        # Save best model based on VAL_LOSS
        if val_loss < best_loss and epoch > 5:
            best_loss = val_loss
            checkpoint_path = os.path.join(config['logging']['pre_dir'], f"best_epoch_{epoch + 1}.pt")
            best_checkpoint_path = checkpoint_path
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': best_loss
            }, checkpoint_path)
            print(f"\nBest model updated at epoch {epoch + 1} with loss {best_loss:.4f}")

    # Quantitative EVAL: discuss per image, and per head in image
    # Test best.model on metrics
    checkpoint = torch.load(best_checkpoint_path)
    model_state_dict = checkpoint['model_state_dict']
    bst_ep = checkpoint['epoch']
    bst_loss = checkpoint['loss']

    with torch.no_grad():
        model.load_state_dict(model_state_dict)

        auc, meanl2, minl2 = eval_pretrain_gazefollow(config, model, test_loader, device)

        best_checkpoint = os.path.join(config['logging']['pre_dir'],
                                       f"Best_model_ep{bst_ep}_l2{int(meanl2 * 100)}_loss{int(best_loss * 100)}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss': bst_loss,
            'auc': auc,
            'l2': meanl2
        }, best_checkpoint)


if __name__ == "__main__":
    main()
