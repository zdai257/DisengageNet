import os
from os.path import join
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from network.network_builder import get_gazelle_model
from tqdm import tqdm
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score


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
def eval_metrics(config, model, test_loader, device):

    aucs = []
    l2s = []
    inout_preds = []
    inout_gts = []

    for _, (images, bboxes, gazex, gazey, inout) in tqdm(enumerate(test_loader), desc="Evaluating",
                                                         total=len(test_loader)):
        preds = model.forward({"images": images.to(device), "bboxes": bboxes})

        # eval each instance (head)
        for i in range(images.shape[0]):  # per image
            for j in range(len(bboxes[i])):  # per head
                if inout[i][j] == 1:  # in frame
                    auc = vat_auc(preds['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    l2 = vat_l2(preds['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    aucs.append(auc)
                    l2s.append(l2)
                inout_preds.append(preds['inout'][i][j].item())
                inout_gts.append(inout[i][j])

    AUC = np.array(aucs).mean()
    L2_mean = np.array(l2s).mean()
    AP = average_precision_score(inout_gts, inout_preds)
    print("AUC: {}".format(AUC))
    print("Avg L2: {}".format(L2_mean))
    print("Inout AP: {}".format(AP))

    return AUC, L2_mean, AP

