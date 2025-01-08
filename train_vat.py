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
from dataset_builder import VideoAttTarget_video

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/videoattentiontarget")
parser.add_argument("--model_name", type=str, default="gazelle_dinov2_vitl14_inout")
parser.add_argument("--ckpt_path", type=str, default="./checkpoints/gazelle_dinov2_vitl14_inout.pt")
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()


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


class CustomLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Custom loss combining BCE loss for classification and MSE loss for regression.
        Args:
            alpha (float): Weight for the BCE loss.
            beta (float): Weight for the MSE loss.
        """
        super(CustomLoss, self).__init__()
        self.bce_loss = torch.nn.BCELoss(reduction='sum')  # Binary Cross-Entropy Loss
        self.mse_loss = torch.nn.MSELoss(reduce=False, reduction='sum')  # Mean Squared Error Loss
        self.alpha = alpha  # Weight for BCE
        self.beta = beta  # Weight for MSE

    def forward(self, classification_pred, classification_target, 
                regression_pred, regression_target):
        """
        Compute the combined loss for variable Num_items.
        Args:
            classification_pred (list of Tensors): Predicted probabilities for classification (BCE input), shape (B, Num_items).
            classification_target (list of Tensors): Ground truth for classification (BCE target), shape (B, Num_items).
            regression_pred (list of Tensors): Predicted distances for regression (MSE input), shape (B, Num_items, 2).
            regression_target (list of Tensors): Ground truth distances for regression (MSE target), shape (B, Num_items, 2).
        Returns:
            Tensor: Combined loss value.
        """
        total_bce_loss = 0.0
        total_mse_loss = 0.0
        total_items = 0

        # Iterate over the batch
        for cls_pred, cls_target, reg_pred, reg_target in zip(
                classification_pred, classification_target, regression_pred, regression_target):
            # Compute BCE loss for this sample (classification)
            total_bce_loss += self.bce_loss(cls_pred, cls_target)

            # Compute MSE loss for this sample (regression)
            total_mse_loss += self.mse_loss(reg_pred, reg_target)

            # Accumulate the total number of items for normalization
            total_items += cls_pred.size(0)

        # Normalize losses by total items
        total_bce_loss /= total_items
        total_mse_loss /= total_items

        # Combine losses with weights
        total_loss = self.alpha * total_bce_loss + self.beta * total_mse_loss

        return total_loss


def main():

    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    # load config
    lr = config['train']['lr']
    num_epochs = config['train']['epochs']

    #my_net = get_gt360_model(config)
    model = get_gazelle_model(config)
    model.load_gazelle_state_dict(torch.load(args.ckpt_path, weights_only=True))
    model.to(device)

    if config['train']['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif config['train']['optimizer'] == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=lr)
    else:
        raise TypeError("Optimizer not supported!")

    # linear learning rate scheduler
    lr_step_size = config['lr_scheduler']['step_size']
    gamma = config['lr_scheduler']['gamma']  # Factor by which the learning rate will be reduced
    scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

    input_resolution = 224

    # transform
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    my_transform = transforms.Compose(transform_list)

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

    # LOSS
    loss_fn = CustomLoss(alpha=1.0, beta=1.0)

    best_val_loss = float('inf')
    early_stop_count = 0

    aucs = []
    l2s = []
    inout_preds = []
    inout_gts = []

    # START TRAINING
    for epoch in range(config['train']['epochs']):
        epoch_loss = 0.

        for batch, (images, bboxes, gazex, gazey, inout) in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.train(True)
            # freeze batchnorm layers
            for module in model.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()
            
            # forward pass
            preds = model({"images": images.to(device), "bboxes": bboxes})


            classification_preds, regression_preds = [], []
            for idx in range(0, len(bboxes)):
                
                classification_preds.append(preds['inout'][:,idx,:])  # Shape: (B, )
                regression_preds.append(preds['heatmap'][:,idx,:])  # Shape: (B, 2)

            # TODO: Compute total loss
            loss = loss_fn(classification_preds, inout, regression_preds, torch.cat((gazex, gazey), 1))
        
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Accumulate loss for reporting
            epoch_loss += loss.item()

        # Print epoch loss
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / train_length:.4f}")


    # discuss per image, and per head in image
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
