import argparse
import torch
from torch.utils.data import Dataset
from torch.optim import RMSprop, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import json
import os
from os.path import join
import numpy as np
from tqdm import tqdm
import yaml
import csv
import random
import math
import face_detection
from network.network_builder import get_gazelle_model, get_gt360_model
from eval import eval_metrics
# VAT native data_loader
#from dataset_builder import VideoAttTarget_video


class ColumbiaGazeDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, isTrain='train', num_subjects=56):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.dataset_path = join(self.dataset_dir, "ColumbiaGaze", "columbia_gaze_data_set", f"Columbia Gaze Data Set")
        if isTrain == "train":
            num_subjects = 55
        subject_lst = list(range(num_subjects))
        subject_id_str = [str(x).zfill(4) for x in subject_lst]

        # TODO: get image-gazevec-EClabel from the dataset
        self.gaze_data = []  # List of (image_name, gaze_vector, ec)

        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.jpg') and not file.startswith('.'):  # Check for .jpg files
                    full_path = join(root, file)
                    filename, _ = os.path.splitext(file)  # Remove extension
                    labels = filename.split('_')  # Split by '_'

                    # e.g. "0003_2m_-30P_10V_-10H.jpg": five head Poses, three Vertical gaze angles, seven Horizontal gaze angles
                    headpose = int(labels[2][:-1])
                    vertical = int(labels[3][:-1])
                    horizontal = int(labels[4][:-1])

                    ec = 0
                    gaze_vector = (0., 0.)
                    v_unit = 2 * math.tan(math.radians(10))
                    #h_unit = 2 * math.tan(math.radians(5))
                    if vertical == 0:
                        if headpose == horizontal:
                            ec = 1
                        elif headpose < horizontal:
                            gaze_vector = (1., 0.)
                        else:
                            gaze_vector = (-1., 0.)
                    elif vertical == 10:
                        if headpose == horizontal:
                            gaze_vector = (0., 1.)
                        else:
                            # TODO: geometry correct?!
                            gaze_vector = (2*math.tan(math.radians(horizontal - headpose)), v_unit)
                            magnitude = np.linalg.norm(np.array(gaze_vector))
                            gaze_vector = (gaze_vector[0]/magnitude, gaze_vector[1]/magnitude)
                    elif vertical == -10:
                        if headpose == horizontal:
                            gaze_vector = (0., -1.)
                        else:
                            # TODO: geometry correct?!
                            gaze_vector = (2*math.tan(math.radians(horizontal - headpose)), -v_unit)
                            magnitude = np.linalg.norm(np.array(gaze_vector))
                            gaze_vector = (gaze_vector[0] / magnitude, gaze_vector[1] / magnitude)

                    self.gaze_data.append((full_path, gaze_vector, ec))

    def __len__(self):
        return len(self.gaze_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, gaze_vector, ec = self.gaze_data[idx]

        # Load and process the image
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # TODO: Convert gaze vector to a heatmap / grid coordinate at boundary of (65, 65)
        gaze_vector = np.array(gaze_vector, dtype=np.float32)

        return image, gaze_vector, ec


class Gaze360(Dataset):
    def __init__(self, anno_path="data/annotations/", transform=None, data_split='train', data_split_name='',
                 image_mode='RGB', advanced=False):
        # TODO: load Gaze360 dataset
        # self.data_dir = data_dir
        assert data_split == 'train' or data_split == 'test'
        self.data_split = data_split
        self.advanced = advanced
        self.transform = transform
        self.image_mode = image_mode
        self.data_list = []  # list(csv.reader(file))

        for filename in os.listdir(anno_path + "train"):
            filename = os.path.join(anno_path, "train", filename)
            file = open(filename, 'r')
            self.data_list = self.data_list + list(csv.reader(file))

        random.Random(2022).shuffle(self.data_list)

        # Split data
        ratio = 1
        if data_split == 'train':
            self.data_list = self.data_list[:round(ratio * len(self.data_list))]
        elif data_split == 'val':
            self.data_list = self.data_list[round(ratio * len(self.data_list)):]
        else:
            self.data_list = []
            for filename in os.listdir(anno_path + "test"):
                filename = os.path.join(anno_path, "test", filename)
                file = open(filename, 'r')
                self.data_list = self.data_list + list(csv.reader(file))

        self.length = len(self.data_list)
        print(f"Dataset count total: {self.length}")

    def __getitem__(self, index):
        # Row:
        # filename,bbox_x1,bbox_y1,bbox_x2,bbox_y2,split,label
        data_row = self.data_list[index]
        base_path = os.getcwd()
        path = os.path.join(base_path, data_row[0])
        img = Image.open(path)
        img = img.convert(self.image_mode)
        width, height = img.size
        x_min, y_min, x_max, y_max = round(float(data_row[1])), round(float(data_row[2])), round(
            float(data_row[3])), round(float(data_row[4]))

        # k = 0.2 to 0.40
        if self.data_split == 'train':
            k = np.random.random_sample() * 0.6 + 0.2
            x_min -= 0.6 * k * abs(x_max - x_min)
            y_min -= 2 * k * abs(y_max - y_min)
            x_max += 0.6 * k * abs(x_max - x_min)
            y_max += 0.6 * k * abs(y_max - y_min)

        img_crop = img.crop((max(0, int(x_min)), max(0, int(y_min)), min(width, int(x_max)), min(height, int(y_max))))
        label = int(data_row[5])

        if self.transform is not None:
            img_crop = self.transform(img_crop)
        if label == 1:
            label = torch.FloatTensor([0, 1])  # Class 0: No eye contact, Class 1: Eye contact
        else:
            label = torch.FloatTensor([1, 0])

        if self.advanced:
            return img_crop, label, path  # (bbox_x2-bbox_x1,bbox_y2-bbox_y1)
        else:
            return img_crop, label

    def __len__(self):
        # 122,450
        return self.length


def collate(batch):
    images, bboxes, gazex, gazey, inout = zip(*batch)
    return torch.stack(images), list(bboxes), list(gazex), list(gazey), list(inout)


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

# TODO: pipeline to train EC
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

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    # set lr differently
    param_dicts = []
    for n, p in model.named_parameters():
        if p.requires_grad and "inout" not in n:
            param_dicts.append({'params': p, 'lr': config['train']['lr']})
        if p.requires_grad and "inout" in n:
            param_dicts.append({'params': p, 'lr': config['train']['inout_lr']})

    if config['train']['optimizer'] == 'Adam':
        optimizer = Adam(param_dicts)
    elif config['train']['optimizer'] == 'AdamW':
        optimizer = AdamW(param_dicts)
    else:
        raise TypeError("Optimizer not supported!")

    # linear learning rate scheduler
    lr_step_size = config['train']['lr_scheduler']['step_size']
    gamma = config['train']['lr_scheduler']['gamma']  # Factor by which the learning rate will be reduced
    if config['train']['lr_scheduler']['type'] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=100)
                                                               #eta_min=config['train']['lr_scheduler']['min_lr'])
    else: # linear lr scheduler
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        collate_fn=collate,
        #shuffle=True,
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
    # val_loader?

    train_length = train_dataset.__len__()

    bce_loss = torch.nn.BCELoss(reduction='sum')  # Binary Cross-Entropy Loss
    mse_loss = torch.nn.MSELoss(reduction='sum')  # Mean Squared Error Loss

    # save dir for checkpoints
    os.makedirs(config['logging']['log_dir'], exist_ok=True)

    best_loss = float('inf')
    early_stop_count = 0
    best_checkpoint_path = None
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

            # hide MSE lose when out-of-frame
            inout_mask = (gt_inout == 1).float()

            total_mse_loss = mse_loss(torch.cat(regression_preds, dim=0), torch.cat(gt_gaze_xy, dim=0))
            total_mse_loss = total_mse_loss * inout_mask.squeeze()


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
            best_checkpoint_path = checkpoint_path
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': best_loss
            }, checkpoint_path)
            print(f"\nBest model updated at epoch {epoch + 1} with loss {best_loss:.4f}")

    # Quantitative Eval: discuss per image, and per head in image
    # Test best.model on metrics
    checkpoint = torch.load(best_checkpoint_path)
    model_state_dict = checkpoint['model_state_dict']
    bst_ep = checkpoint['epoch']
    bst_loss = checkpoint['loss']

    with torch.no_grad():
        model.load_state_dict(model_state_dict)

        auc, l2, ap = eval_metrics(config, model, test_loader, device)

        best_checkpoint = os.path.join(config['logging']['log_dir'],
                                       f"Best_model_ep{bst_ep}_l2{int(l2*100)}_ap{int(ap*100)}.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss': bst_loss,
            'auc': auc,
            'l2': l2,
            'ap': ap
        }, best_checkpoint)


if __name__ == "__main__":
    main()
