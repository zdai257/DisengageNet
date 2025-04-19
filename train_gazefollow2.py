import argparse
import torch
import torch.nn.functional as F
from torch.optim import RMSprop, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, ToPILImage
import torchvision.transforms.functional as TF
from PIL import Image
import json
import os
from os.path import join
import numpy as np
from tqdm import tqdm
import yaml

from network.network_builder import get_gazelle_model
from network.network_builder_update2 import get_gt360_model, get_gazemoe_model
from eval import eval_pretrain_gazefollow, gazefollow_auc, gazefollow_l2
from network.utils import SoftArgmax2D, CosineL1, visualize_heatmap, visualize_heatmap2, visualize_heatmap3
import matplotlib.pyplot as plt

LOSS_SCALAR = 1


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


# Chong's extended GazeFollow
class GazeFollowExtended(torch.utils.data.Dataset):
    """
    Format:
    [image_path,id,body_bbox_x,body_bbox_y,body_bbox_width,body_bbox_height,eye_x,eye_y,gaze_x,gaze_y,head_bbox_x_min,head_bbox_y_min,head_bbox_x_max,head_bbox_y_max,in_or_out,source,meta]
    Description: (NO 'in_or_out' field for test split)
    <head_bbox_x_min,head_bbox_y_min,head_bbox_x_max,head_bbox_y_max,in_or_out> are the added fields to GazeFollow.
    Head bounding box <head_bbox_x_min,head_bbox_y_min,head_bbox_x_max,head_bbox_y_max> is in the image pixel coordinate.
    <in_or_out> is 1 for inside, 0 for outside cases. There are a few -1's to denote invalid case (e.g., no human at all, upper body is cut, etc).
    """
    def __init__(self, root_path, img_transform, split='train'):
        self.frames = []
        with open(join(root_path, "{}_annotations_release.txt".format(split)), "r") as f:
            for line in f:
                frame = {}
                anno_lst = line.strip().split(',')

                if split == 'train':
                    frame['inout'] = float(anno_lst[14])
                    if frame['inout'] != 1:
                        continue
                else:
                    frame['inout'] = 1

                frame['path'] = anno_lst[0]
                frame['gazex_norm'] = [float(anno_lst[8])]
                frame['gazey_norm'] = [float(anno_lst[9])]
                frame['bbox_pixel'] = [float(anno_lst[10]), float(anno_lst[11]), float(anno_lst[12]), float(anno_lst[13])]

                self.frames.append(frame)
            print("Number of valid samples ", len(self.frames))

        self.root_path = root_path
        self.transform = img_transform

    def __getitem__(self, idx):
        frame = self.frames[idx]

        image = Image.open(join(self.root_path, frame['path'])).convert("RGB")
        w, h = image.width, image.height

        image = self.transform(image)
        # normalise bbox
        bbox = [frame['bbox_pixel'][0]/w, frame['bbox_pixel'][1]/h, frame['bbox_pixel'][2]/w, frame['bbox_pixel'][3]/h]
        gazex = frame['gazex_norm']
        gazey = frame['gazey_norm']

        # involve 'inout' in training??
        inout = frame['inout']
        return image, bbox, gazex, gazey, inout, h, w

    def __len__(self):
        return len(self.frames)


def collate(batch):
    images, bbox, gazex, gazey, inout, height, width = zip(*batch)
    return torch.stack(images), list(bbox), list(gazex), list(gazey), list(inout), list(height), list(width)


def apply_dilation_blur(heatmap, dilation_kernel=3, blur_radius=1.):
    """
    Args:
        heatmap (Tensor): Shape (64, 64), predicted heatmaps.
        dilation_kernel (int): Kernel size for dilation (must be odd).
        blur_radius (float): Standard deviation for Gaussian blur.
    Returns:
        Tensor: Processed heatmap of shape (64, 64).
    """
    H, W = heatmap.shape
    heatmap = heatmap.unsqueeze(0)  # Convert to (1, H, W) for processing
    # Dilation using max pooling
    heatmap_dilated = F.max_pool2d(heatmap, kernel_size=dilation_kernel, stride=1, padding=dilation_kernel//2)
    # Gaussian blur
    kernel_size = int(6 * blur_radius) | 1  # Ensure it's an odd number
    heatmap_blurred = TF.gaussian_blur(heatmap_dilated, kernel_size=[kernel_size,kernel_size], sigma=[blur_radius])
    return heatmap_blurred.squeeze(0)  # Convert back to (64, 64)


def apply_dilation_blur2(heatmap, dilation_kernel=3, blur_radius=0.7, peak_val=1.0, min_val=0.0):
    """
    Applies dilation followed by Gaussian blur and rescales values to maintain high confidence near the peak.

    Args:
        heatmap (Tensor): Shape (64, 64), a single ground-truth heatmap.
        dilation_kernel (int): Kernel size for dilation (must be odd).
        blur_radius (float): Standard deviation for Gaussian blur.
        peak_val (float): Maximum value for the peak (default=1.0).
        min_val (float): Minimum value in the blurred area (default=0.7).
    Returns:
        Tensor: Processed heatmap of shape (64, 64) with controlled confidence levels.
    """
    H, W = heatmap.shape
    heatmap = heatmap.unsqueeze(0)  # Convert to (1, H, W) for processing
    # Dilation using max pooling
    heatmap_dilated = F.max_pool2d(heatmap, kernel_size=dilation_kernel, stride=1, padding=dilation_kernel // 2)
    # Gaussian blur
    kernel_size = int(6 * blur_radius) | 1  # Ensure kernel size is an odd number
    heatmap_blurred = TF.gaussian_blur(heatmap_dilated, kernel_size=[kernel_size, kernel_size], sigma=[blur_radius])
    # Rescale: Normalize between [min_val, peak_val]
    heatmap_blurred = heatmap_blurred - heatmap_blurred.min()  # Shift minimum to 0
    heatmap_blurred = heatmap_blurred / heatmap_blurred.max()  # Normalize to [0,1]
    heatmap_blurred = heatmap_blurred * (peak_val - min_val) + min_val  # Scale to [min_val, peak_val]
    return heatmap_blurred.squeeze(0)  # Convert back to (64, 64)


@torch.no_grad()
def evaluate(config, model, val_loader, device):
    model.eval()
    batch_size = config['eval']['batch_size']

    # angle L1 loss
    angle_loss_fn = CosineL1()
    softArgmax_fn = SoftArgmax2D()
    # pixel-wise MSE / BCE loss
    if config['model']['pbce_loss'] == "mse":
        val_pbce_loss = torch.nn.MSELoss(reduction=config['model']['reduction'])
    else:
        val_pbce_loss = torch.nn.BCELoss(reduction=config['model']['reduction'])
    validation_loss = 0.0
    val_total = len(val_loader)

    aucs = []
    min_l2s = []
    avg_l2s = []
    with tqdm(total=val_total) as pbar:
        for images, bboxes, gazex, gazey, inouts, h, w in val_loader:

            preds = model({"images": images.to(device), "bboxes": [bboxes]})

            # preds = a dict of{'heatmap': list of Batch_size*tensor[head_count, 64, 64],
            #                   'inout': list of Batch_size*tensor[head_count,] }

            pred_heatmap = preds['heatmap'][0]

            if True:
                gt_gaze_xy = []
                gtxs = gazex
                gtys = gazey
                bbox_ctrs, gt_xys = [], []
                #print(gazex, gazey)
                batch_size0 = len(gtxs)
                # for GazeFollow, len(heads) should always be 1
                for i, (bbx, gtx, gty) in enumerate(zip(bboxes, gtxs, gtys)):
                    bbox_ctrs.append(torch.tensor([(bbx[0] + bbx[2]) / 2, (bbx[1] + bbx[3]) / 2], dtype=torch.float32))
                    gt_xys.append(torch.tensor([gtx[0], gty[0]], dtype=torch.float32))

                    gt_heatmap = torch.zeros((64, 64))
                    x_grid = int(gtx[0] * 63)
                    y_grid = int(gty[0] * 63)

                    gt_heatmap[y_grid, x_grid] = 1
                    # Gaussian blur
                    gt_heatmap = apply_dilation_blur2(gt_heatmap)
                    gt_gaze_xy.append(gt_heatmap)

            head_i = 0
            for j in range(images.shape[0]):
                auc = gazefollow_auc(preds['heatmap'][head_i][j], gazex[j], gazey[j], h[j], w[j])
                avg_l2, min_l2 = gazefollow_l2(preds['heatmap'][head_i][j], gazex[j], gazey[j])
                aucs.append(auc)
                avg_l2s.append(avg_l2)
                min_l2s.append(min_l2)

            gt_bbox_ctrs = torch.stack(bbox_ctrs)
            gt_gt_xys = torch.stack(gt_xys)
            gt_heatmaps = torch.stack(gt_gaze_xy)
            ### Introduce gaze angle L1 loss ###
            pred_xys = softArgmax_fn(pred_heatmap)
            # extend angle_loss to vector_l2 loss?
            val_angle_loss = angle_loss_fn(pred_xys - gt_bbox_ctrs.to(device),
                                           gt_gt_xys.to(device) - gt_bbox_ctrs.to(device))

            val_hp_loss = val_pbce_loss(pred_heatmap, gt_heatmaps.to(device)) * LOSS_SCALAR
            val_hp_loss = val_hp_loss.mean([1, 2])
            
            #gaze_in = torch.FloatTensor(inouts).to(device)
            #loss = torch.mul(loss, gaze_in)
            #loss = torch.sum(loss) / torch.sum(gaze_in)
            loss = config['model']['mse_weight'] * val_hp_loss.mean() + config['model']['angle_weight'] * val_angle_loss.mean()

            validation_loss += loss.item()

            pbar.update(1)

    auc = np.array(aucs).mean()
    meanl2 = np.array(avg_l2s).mean()
    minl2 = np.array(min_l2s).mean()
    print("Eval -- AUC: {}; Mean L2: {}; Min L2: {}".format(auc, meanl2, minl2))
    return float(validation_loss / val_total)


def main():
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    # load GazeFollow config
    lr = config['train']['pre_lr']
    num_epochs = config['train']['pre_epochs']

    # load network
    model, gazelle_transform = get_gazemoe_model(config)
    model.load_gazelle_state_dict(torch.load(config['model']['pretrained_path'], weights_only=True, map_location=device),
                                  include_backbone=False)
    # model.load_state_dict(torch.load(config['model']['pretrained_path'], weights_only=True, map_location=device))

    # Verify the freezing and initialization
    for name, param in model.named_parameters():
        # print(f"{name}: requires_grad={param.requires_grad}")
        pass

    # Freeze 'backbone' parameters
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False  # Freeze these parameters
        else:
            param.requires_grad = True  # Keep these learnable

    # Randomly initialize learnable parameters
    for name, param in model.named_parameters():
        #break
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
            if 'ms_fusion' in n or 'transformer' in n:
                param_dicts.append({'params': p, 'lr': config['train']['fuse_lr']})
            else:
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
                                                               T_max=10)
        # eta_min=config['train']['lr_scheduler']['min_lr'])
    else:  # linear lr scheduler
        scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

    input_resolution = config['data']['input_resolution']

    # RandomCrop + flip + BBox Jitter (?)
    transform_list = []
    # augmentation++
    transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    # transform_list.append(T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3))
    transform_list.append(transforms.RandomApply([transforms.RandomGrayscale(p=0.2)], p=0.3))
    # transform_list.append(transforms.RandomResizedCrop((input_resolution, input_resolution)))
    # TODO: deal with flipped bbox labels
    # transform_list.append(transforms.RandomHorizontalFlip(0.5))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    my_transform = transforms.Compose(transform_list)

    #my_transform = gazelle_transform

    # apply my_transform
    train_dataset = GazeFollowExtended(root_path=config['data']['pre_train_path'],
                                       img_transform=my_transform,
                                       split='train')
    test_dataset = GazeFollowExtended(root_path=config['data']['pre_test_path'],
                                      img_transform=my_transform,
                                      split='test')
    #print(train_dataset.__len__(), test_dataset.__len__())
    #train_dataset = GazeFollowExtended('gazefollow_extended', img_transform=my_transform, split='train')
    #test_dataset = GazeFollowExtended('gazefollow_extended', img_transform=my_transform, split='test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['train']['pre_batch_size'],
        collate_fn=collate,
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['eval']['batch_size'],
        collate_fn=collate,
        # shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        drop_last=True
    )
    # train_length = train_dataset.__len__()

    ### Loss ###
    angle_loss_fn = CosineL1()
    softArgmax_fn = SoftArgmax2D()
    # MSEloss or BCELoss
    if config['model']['pbce_loss'] == "mse":
        pbce_loss = torch.nn.MSELoss(reduction=config['model']['reduction'])
    else:
        pbce_loss = torch.nn.BCELoss(reduction=config['model']['reduction'])

    # save dir for checkpoints
    os.makedirs(config['logging']['pre_dir'], exist_ok=True)

    best_loss = float('inf')
    best_auc = 0.5
    best_meanl2 = float('inf')
    early_stop_count = 0
    best_checkpoint_path = None
    batch_size = config['train']['pre_batch_size']

    checkpoint_dir = "_".join([
        config['train']['pre_optimizer'],
        "bs" + str(config['train']['pre_batch_size']),
        str(config['train']['pre_lr']),
        str(config['train']['fuse_lr']),
        config['model']['pbce_loss'],
        str(config['model']['mse_weight']),
        str(config['model']['angle_weight'])
    ])
    os.makedirs(os.path.join(config['logging']['pre_dir'], checkpoint_dir), exist_ok=True)
    print("Pretrained checkpoint saved at: ", os.path.join(config['logging']['log_dir'], checkpoint_dir))
    # START TRAINING
    for epoch in range(config['train']['pre_epochs']):
        model.train(True)
        epoch_loss = 0.

        for batch, (images, bboxes, gazex, gazey, inouts, h, w) in tqdm(enumerate(train_loader), total=len(train_loader)):

            # forward pass
            preds = model({"images": images.to(device), "bboxes": [bboxes]})

            # preds = a dict of{'heatmap': list of head_count *tensor[Batch_size, 64, 64],
            #                   'inout': list of head_count *tensor[Batch_size,] }
            pred_heatmap = preds['heatmap'][0]
            
            if True:
                gt_gaze_xy = []
                gtxs = gazex
                gtys = gazey
                bbox_ctrs, gt_xys = [], []
                #print(gazex, gazey)
                batch_size0 = len(gtxs)
                # for GazeFollow, len(heads) should always be 1
                for i, (bbx, gtx, gty) in enumerate(zip(bboxes, gtxs, gtys)):
                    # bbx: (xmin, ymin, xmax, ymax)
                    bbox_ctrs.append(torch.tensor([(bbx[0] + bbx[2]) / 2, (bbx[1] + bbx[3]) / 2], dtype=torch.float32))
                    gt_xys.append(torch.tensor([gtx[0], gty[0]], dtype=torch.float32))

                    gt_heatmap = torch.zeros((64, 64))
                    x_grid = int(gtx[0] * 63)
                    y_grid = int(gty[0] * 63)

                    gt_heatmap[y_grid, x_grid] = 1
                    # Gaussian blur
                    gt_heatmap = apply_dilation_blur2(gt_heatmap)

                    # DEBUG
                    """
                    id = i
                    transform = ToPILImage()
                    image = torch.clamp(images[i].detach().cpu(), 0, 1)
                    image = transform(image)
                    
                    print(gazex[id][0], gazey[id][0])
                    torch.set_printoptions(threshold=10_000)
                    print(gt_heatmap)
                    viz = visualize_heatmap2(image, gt_heatmap, bbox=bboxes[id], xy=(gazex[id][0]*448, gazey[id][0]*448), dilation_kernel=6,
                                 blur_radius=1.3)  #, transparent_bg=None)
                    plt.imshow(viz)
                    plt.show()
                    """
                    
                    gt_gaze_xy.append(gt_heatmap)

            gt_bbox_ctrs = torch.stack(bbox_ctrs)
            gt_gt_xys = torch.stack(gt_xys)
            gt_heatmaps = torch.stack(gt_gaze_xy)
            ### Introduce gaze angle L1 loss ###
            pred_xys = softArgmax_fn(pred_heatmap)
            # extend angle_loss to vector_l2 loss?
            angle_loss = angle_loss_fn(pred_xys - gt_bbox_ctrs.to(device),
                                       gt_gt_xys.to(device) - gt_bbox_ctrs.to(device))

            hp_loss = pbce_loss(pred_heatmap, gt_heatmaps.to(device)) * LOSS_SCALAR
            hp_loss = hp_loss.mean([1, 2])

            #gaze_in = torch.FloatTensor(inouts).to(device)
            #loss = torch.mul(loss, gaze_in)
            #loss = torch.sum(loss) / torch.sum(gaze_in)

            # hp_loss around 0.26; angle_loss in [0, 2]
            #print(hp_loss, angle_loss)
            loss = config['model']['mse_weight'] * hp_loss.mean() + config['model']['angle_weight'] * angle_loss.mean()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip exploding gradients
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        mean_ep_loss = epoch_loss / len(train_loader)
        # Print epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {mean_ep_loss:.7f}")

        val_loss = evaluate(config, model, test_loader, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.7f}")

        # Save model every 5 epochs
        if (epoch + 1) % config['logging']['save_every'] == 0:
            checkpoint_path = os.path.join(config['logging']['pre_dir'], checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': val_loss
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        # Save best model based on VAL_LOSS
        if val_loss < best_loss and epoch > config['logging']['save_every']:
            best_loss = val_loss
            checkpoint_path = os.path.join(config['logging']['pre_dir'], checkpoint_dir, f"best_epoch_{epoch + 1}.pt")
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
