import argparse
import torch
from torch.optim import RMSprop, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms.functional as TF
from PIL import Image
import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import yaml

from network.network_builder import get_gazelle_model, get_gt360_model
# VAT native data_loader
#from dataset_builder import VideoAttTarget_video


class VideoAttentionTarget(torch.utils.data.Dataset):
    def __init__(self, path, img_transform, split='train'):
        self.sequences = json.load(open(os.path.join(path, "{}_preprocessed.json".format(split)), "rb"))
        self.frames = []
        for i in range(len(self.sequences)):
            for j in range(len(self.sequences[i]['frames'])):
                self.frames.append((i, j))
        self.path = path
        self.transform = img_transform

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.frames[idx]
        seq = self.sequences[seq_idx]
        frame = seq['frames'][frame_idx]
        image = self.transform(Image.open(os.path.join(self.path, frame['path'])).convert("RGB"))
        bboxes = [head['bbox_norm'] for head in frame['heads']]
        gazex = [head['gazex_norm'] for head in frame['heads']]
        gazey = [head['gazey_norm'] for head in frame['heads']]
        inout = [head['inout'] for head in frame['heads']]

        return image, bboxes, gazex, gazey, inout

    def __len__(self):
        return len(self.frames)


def collate(batch):
    images, bboxes, gazex, gazey, inout = zip(*batch)
    return torch.stack(images), list(bboxes), list(gazex), list(gazey), list(inout)


# VideoAttentionTarget calculates AUC on 64x64 heatmap, defining a rectangular tolerance region of 6*(sigma=3) + 1 (uses 2D Gaussian code but binary thresholds > 0 resulting in rectangle)
# References:
# https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_videoatttarget.py#L106
# https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/utils/imutils.py#L31
def vat_auc(heatmap, gt_gazex, gt_gazey):
    res = 64
    sigma = 3
    assert heatmap.shape[0] == res and heatmap.shape[1] == res
    target_map = np.zeros((res, res))
    gazex = gt_gazex * res
    gazey = gt_gazey * res
    ul = [max(0, int(gazex - 3 * sigma)), max(0, int(gazey - 3 * sigma))]
    br = [min(int(gazex + 3 * sigma + 1), res - 1), min(int(gazey + 3 * sigma + 1), res - 1)]
    target_map[ul[1]:br[1], ul[0]:br[0]] = 1
    auc = roc_auc_score(target_map.flatten(), heatmap.cpu().flatten())
    return auc


# Reference: https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_videoatttarget.py#L118
def vat_l2(heatmap, gt_gazex, gt_gazey):
    argmax = heatmap.flatten().argmax().item()
    pred_y, pred_x = np.unravel_index(argmax, (64, 64))
    pred_x = pred_x / 64.
    pred_y = pred_y / 64.

    l2 = np.sqrt((pred_x - gt_gazex) ** 2 + (pred_y - gt_gazey) ** 2)

    return l2


@torch.no_grad()
def evaluate(config, model, val_loader, device):
    model.eval()
    batch_size = config['eval']['batch_size']
    val_bce_loss = torch.nn.BCELoss(reduction='sum')
    val_mse_loss = torch.nn.MSELoss(reduction='sum')
    validation_loss = 0.0
    val_total = len(val_loader)

    with tqdm(total=val_total) as pbar:
        for images, bboxes, gazex, gazey, inout in val_loader:

            preds = model({"images": images.to(device), "bboxes": bboxes})

            # preds = a dict of{'heatmap': list of Batch_size*tensor[head_count, 64, 64],
            #                   'inout': list of Batch_size*tensor[head_count,] }

            classification_preds, regression_preds = [], []
            for b in range(0, batch_size):
                head_count = preds['inout'][b].shape[0]
                inout_list, xy_list = torch.empty((head_count,)), torch.empty((head_count, 2))
                for head_idx in range(0, head_count):
                    inout_list[head_idx] = preds['inout'][b][head_idx].clone()
                    heatmap_tensor = preds['heatmap'][b][head_idx].clone()
                    # convert pred_heatmap to (x, y) loc
                    argmax = heatmap_tensor.flatten().argmax().item()
                    pred_y, pred_x = np.unravel_index(argmax, (64, 64))
                    pred_x = pred_x / 64.
                    pred_y = pred_y / 64.
                    xy_list[head_idx] = torch.tensor([float(pred_x), float(pred_y)])

                classification_preds.append(inout_list)  # a list of Batch*[heads * <val> ]
                regression_preds.append(xy_list)  # a list of Batch*[heads * (2,) ]

            gt_gaze_xy = []
            for gtxs, gtys in zip(gazex, gazey):
                heads = len(gtxs)
                gt_per_img = torch.empty((heads, 2))
                for i, (gtx, gty) in enumerate(zip(gtxs, gtys)):
                    gt_per_img[i] = torch.tensor([gtx[0], gty[0]], dtype=torch.float32)
                gt_gaze_xy.append(gt_per_img)

            gt_inout = []
            for i in range(len(inout)):
                gt_inout.append(torch.tensor(inout[i], dtype=torch.float32))

            total_bce_loss = val_bce_loss(torch.cat(classification_preds, dim=0), torch.cat(gt_inout, dim=0))
            total_mse_loss = val_mse_loss(torch.cat(regression_preds, dim=0), torch.cat(gt_gaze_xy, dim=0))

            total_loss = config['model']['bce_weight'] * total_bce_loss + config['model']['mse_weight'] * total_mse_loss
            validation_loss += total_loss.item()

            pbar.update(1)

    return float(validation_loss / val_total)


def main():

    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    # load config
    lr = config['train']['lr']
    num_epochs = config['train']['epochs']

    #TODO: my_net
    #my_net = get_gt360_model(config)
    model, gazelle_transform = get_gazelle_model(config)
    model.load_gazelle_state_dict(torch.load(config['model']['pretrained_path'], weights_only=True),
                                  include_backbone=False)

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

    #exit()

    model.to(device)

    # optim
    if config['train']['optimizer'] == 'Adam':
        #optimizer = Adam(model.parameters(), lr=lr)
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif config['train']['optimizer'] == 'AdamW':
        #optimizer = AdamW(model.parameters(), lr=lr)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        raise TypeError("Optimizer not supported!")

    # linear learning rate scheduler
    lr_step_size = config['train']['lr_scheduler']['step_size']
    gamma = config['train']['lr_scheduler']['gamma']  # Factor by which the learning rate will be reduced
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

    input_resolution = config['data']['input_resolution']

    # transform
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    my_transform = transforms.Compose(transform_list)

    # apply my_transform or gazelle_transform
    train_dataset = VideoAttentionTarget(path=config['data']['train_path'],
                                         img_transform=my_transform,
                                         split='train')

    test_dataset = VideoAttentionTarget(path=config['data']['test_path'],
                                        img_transform=my_transform,
                                        split='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batch_size'], collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['eval']['batch_size'], collate_fn=collate)
    # val_loader?

    train_length = train_dataset.__len__()

    bce_loss = torch.nn.BCELoss(reduction='sum')  # Binary Cross-Entropy Loss
    mse_loss = torch.nn.MSELoss(reduction='sum')  # Mean Squared Error Loss

    # save dir for checkpoints
    os.makedirs(config['logging']['log_dir'], exist_ok=True)

    best_loss = float('inf')
    early_stop_count = 0

    # eval metrics
    aucs = []
    l2s = []
    inout_preds = []
    inout_gts = []

    batch_size = config['train']['batch_size']

    # START TRAINING
    for epoch in range(config['train']['epochs']):
        model.train(True)
        epoch_loss = 0.
        total_loss = None

        for batch, (images, bboxes, gazex, gazey, inout) in tqdm(enumerate(train_loader), total=len(train_loader)):

            # freeze batchnorm layers
            #for module in model.modules():
            #    if isinstance(module, torch.nn.modules.BatchNorm1d):
            #        module.eval()
            #    if isinstance(module, torch.nn.modules.BatchNorm2d):
            #        module.eval()
            #    if isinstance(module, torch.nn.modules.BatchNorm3d):
            #        module.eval()
            
            # forward pass
            preds = model({"images": images.to(device), "bboxes": bboxes})

            # preds = a dict of{'heatmap': list of Batch_size*tensor[head_count, 64, 64],
            #                   'inout': list of Batch_size*tensor[head_count,] }

            classification_preds, regression_preds = [], []
            for b in range(0, images.shape[0]):
                head_count = preds['inout'][b].shape[0]
                inout_list, xy_list = torch.empty((head_count,)), torch.empty((head_count, 2))
                for head_idx in range(0, head_count):
                    inout_list[head_idx] = preds['inout'][b][head_idx].clone()
                    heatmap_tensor = preds['heatmap'][b][head_idx].clone()
                    # convert pred_heatmap to (x, y) loc
                    argmax = heatmap_tensor.flatten().argmax().item()
                    pred_y, pred_x = np.unravel_index(argmax, (64, 64))
                    pred_x = pred_x / 64.
                    pred_y = pred_y / 64.
                    xy_list[head_idx] = torch.tensor([float(pred_x), float(pred_y)])

                classification_preds.append(inout_list)  # a list of Batch*[heads * <val> ]
                regression_preds.append(xy_list)  # a list of Batch*[heads * (2,) ]

            #print(len(preds['heatmap']), preds['heatmap'][0].shape)
            #print(len(preds['inout']), preds['inout'][0].shape)
            # GT = a list of Batch*[head_count*[pixel_norm ] ]
            #print(len(gazex), gazex[0])
            #print(len(inout), inout[0])

            gt_gaze_xy = []
            for gtxs, gtys in zip(gazex, gazey):
                heads = len(gtxs)
                gt_per_img = torch.empty((heads, 2))
                for i, (gtx, gty) in enumerate(zip(gtxs, gtys)):
                    gt_per_img[i] = torch.tensor([gtx[0], gty[0]], dtype=torch.float32)
                gt_gaze_xy.append(gt_per_img)

            #print(len(gt_gaze_xy), gt_gaze_xy[0])
            #print(len(regression_preds), regression_preds[0])
            #print(len(classification_preds), classification_preds[0])
            #print(len(inout), inout[0])
            gt_inout = []
            for i in range(len(inout)):
                gt_inout.append(torch.tensor(inout[i], dtype=torch.float32))

            # Iterate over the batch
            # Compute BCE loss for this sample (classification)
            total_bce_loss = bce_loss(torch.cat(classification_preds, dim=0), torch.cat(gt_inout, dim=0))
            total_mse_loss = mse_loss(torch.cat(regression_preds, dim=0), torch.cat(gt_gaze_xy, dim=0))

            total_loss = config['model']['bce_weight'] * total_bce_loss + config['model']['mse_weight'] * total_mse_loss
        
            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate loss for reporting
            epoch_loss += total_loss.item()

        scheduler.step()

        mean_ep_loss = epoch_loss / train_length
        # Print epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {mean_ep_loss:.4f}")

        val_loss = evaluate(config, model, test_loader, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        # Save model every 5 epochs
        if (epoch + 1) % config['logging']['save_every'] == 0:
            checkpoint_path = os.path.join(config['logging']['log_dir'], f"model_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': val_loss
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        # Save best model based on VAL_LOSS
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = os.path.join(config['logging']['log_dir'], f"best_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': best_loss
            }, checkpoint_path)
            print(f"\nBest model updated at epoch {epoch + 1} with loss {best_loss:.4f}")

    # discuss per image, and per head in image
    # Test on metrics
    """
        for i in range(images.shape[0]):  # per image
            for j in range(len(bboxes[i])):  # per head
                if inout[i][j] == 1:  # in frame
                    auc = vat_auc(preds['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    l2 = vat_l2(preds['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    aucs.append(auc)
                    l2s.append(l2)
                inout_preds.append(preds['inout'][i][j].item())
                inout_gts.append(inout[i][j])

    print("AUC: {}".format(np.array(aucs).mean()))
    print("Avg L2: {}".format(np.array(l2s).mean()))
    print("Inout AP: {}".format(average_precision_score(inout_gts, inout_preds)))
    """


if __name__ == "__main__":
    main()
