import os
from os.path import join
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from network.network_builder import get_gazelle_model
from network.network_builder_update2 import get_gt360_model
from tqdm import tqdm
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib.pyplot as plt
from network.utils import visualize_heatmap, visualize_heatmap2, visualize_heatmap3


# GazeFollow calculates AUC using original image size with GT (x,y) coordinates set to 1 and everything else as 0
# References:
# https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_gazefollow.py#L78
# https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/utils/imutils.py#L67
# https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/utils/evaluation.py#L7
def gazefollow_auc(heatmap, gt_gazex, gt_gazey, height, width):
    target_map = np.zeros((height, width))
    for point in zip(gt_gazex, gt_gazey):
        if point[0] >= 0:
            x, y = map(int, [point[0] * float(width), point[1] * float(height)])
            x = min(x, width - 1)
            y = min(y, height - 1)
            target_map[y, x] = 1
    resized_heatmap = torch.nn.functional.interpolate(heatmap.unsqueeze(dim=0).unsqueeze(dim=0), (height, width),
                                                      mode='bilinear').squeeze()
    auc = roc_auc_score(target_map.flatten(), resized_heatmap.cpu().flatten())

    return auc


# Reference: https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_gazefollow.py#L81
def gazefollow_l2(heatmap, gt_gazex, gt_gazey):
    argmax = heatmap.flatten().argmax().item()
    pred_y, pred_x = np.unravel_index(argmax, (64, 64))
    pred_x = pred_x / 64.
    pred_y = pred_y / 64.

    gazex = np.array(gt_gazex)
    gazey = np.array(gt_gazey)

    avg_l2 = np.sqrt((pred_x - gazex.mean()) ** 2 + (pred_y - gazey.mean()) ** 2)
    all_l2s = np.sqrt((pred_x - gazex) ** 2 + (pred_y - gazey) ** 2)
    min_l2 = all_l2s.min().item()

    return avg_l2, min_l2


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


@torch.no_grad()
def eval_metrics_eyediap(config, model, test_loader, device):

    aucs = []
    l2s = []
    inout_preds = []
    inout_gts = []
    batch_size = config['eval']['batch_size']

    for _, (images, gazex, gazey, inout) in tqdm(enumerate(test_loader), desc="Evaluating",
                                                         total=len(test_loader)):
        if images.shape[0] != batch_size:
            batch_size = images.shape[0]

        preds = model.forward({"images": images.to(device), "bboxes": batch_size*[[None]]})

        # eval each instance (head)
        for i in range(images.shape[0]):  # per image
            for j in range(1):  # per head
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


@torch.no_grad()
def eval_metrics_eyediap2(config, model, test_loader, device):

    aucs = []
    l2s = []
    inout_preds = []
    inout_gts = []
    batch_size = config['eval']['batch_size']

    for _, (images, bboxes, gazex, gazey, inout) in tqdm(enumerate(test_loader), desc="Evaluating",
                                                         total=len(test_loader)):

        preds = model.forward({"images": images.to(device), "bboxes": bboxes})

        # eval each instance (head)
        for i in range(images.shape[0]):  # per image
            for j in range(1):  # per head
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


@torch.no_grad()
def eval_pretrain_gazefollow(config, model, test_loader, device):

    aucs = []
    min_l2s = []
    avg_l2s = []

    for _, (images, bboxes, gazex, gazey, inouts, height, width) in tqdm(enumerate(test_loader), desc="Evaluating",
                                                                 total=len(test_loader)):
        preds = model.forward({"images": images.to(device), "bboxes": [bboxes]})

        # eval each instance (head)
        # in our preprocessed GazeFollow, there is 1 head per image
        i = 0
        # len(preds['heatmap']) == 0, preds['heatmap'][0] is of (B, 64, 64)
        
        for j in range(images.shape[0]):
            auc = gazefollow_auc(preds['heatmap'][i][j], gazex[j], gazey[j], height[j], width[j])
            avg_l2, min_l2 = gazefollow_l2(preds['heatmap'][i][j], gazex[j], gazey[j])
            aucs.append(auc)
            avg_l2s.append(avg_l2)
            min_l2s.append(min_l2)

    auc = np.array(aucs).mean()
    meanl2 = np.array(avg_l2s).mean()
    minl2 = np.array(min_l2s).mean()
    print("AUC: {}".format(auc))
    print("Avg L2: {}".format(meanl2))
    print("Min L2: {}".format(minl2))

    return auc, meanl2, minl2


if __name__=="__main__":
    split = 'test'
    dataset_name = "gazefollow_extended"  #"GazeFollow"

    import json

    if dataset_name == "GazeFollow":
        frames = json.load(open(os.path.join(dataset_name, "{}_preprocessed.json".format(split)), "rb"))
        print("This GazeFollow split dataset size is: ", len(frames))
        # this huggingFace GazeFollow dataset may be problematic
        id = 876

        frame_dict = frames[id]

        img_source = join(dataset_name, frame_dict['path'])
        image = Image.open(img_source).convert("RGB")

        # convert a heatmap from label
        gazex_pixel = frame_dict['gazex']
        gazey_pixel = frame_dict['gazey']
        gazex = frame_dict['gazex_norm']
        gazey = frame_dict['gazey_norm']

        gt_heatmap = torch.zeros((64, 64))
        x_grid = int(gazex[0] * 63)
        y_grid = int(gazey[0] * 63)

        gt_heatmap[y_grid, x_grid] = 1

        bbox = frame_dict["bbox_norm"]

        viz = visualize_heatmap2(image, gt_heatmap, bbox=bbox, xy=(gazex_pixel[0], gazey_pixel[0]), dilation_kernel=6,
                                 blur_radius=1.3)
        plt.imshow(viz)
        plt.show()

        saved_path = join("processed", "demo_" + img_source.split('/')[-1])

        if 0:  # if saving
            viz.convert("RGB").save(saved_path)

    elif dataset_name == "gazefollow_extended":
        # Chong's extended dataset
        id = 1000
        frames = []
        with open(join(dataset_name, "{}_annotations_release.txt".format(split)), "r") as f:
            for line in f:
                frame = {}
                anno_lst = line.strip().split(',')
                if split == 'train':
                    frame['inout'] = float(anno_lst[14])
                    if frame['inout'] != 1:
                        continue
                else:
                    frame['inout'] = None
                frame['path'] = anno_lst[0]
                frame['gazex_norm'] = [float(anno_lst[8])]
                frame['gazey_norm'] = [float(anno_lst[9])]
                frame['bbox_pixel'] = [float(anno_lst[10]), float(anno_lst[11]), float(anno_lst[12]),
                                       float(anno_lst[13])]
                frames.append(frame)

        print("This GazeFollow_extended split dataset size is: ", len(frames))
        frame_dict = frames[id]
        print(frame_dict)

        img_source = join(dataset_name, frame_dict['path'])
        image = Image.open(img_source).convert("RGB")
        w, h = image.width, image.height
        print(w, h)
        # convert a heatmap from label
        gazex_pixel = frame_dict['gazex_norm'][0] * w
        gazey_pixel = frame_dict['gazey_norm'][0] * h
        gazex = frame_dict['gazex_norm'][0]
        gazey = frame_dict['gazey_norm'][0]

        gt_heatmap = torch.zeros((64, 64))
        x_grid = int(gazex * 63)
        y_grid = int(gazey * 63)
        gt_heatmap[y_grid, x_grid] = 1

        bbox = [frame_dict['bbox_pixel'][0]/w, frame_dict['bbox_pixel'][1]/h, frame_dict['bbox_pixel'][2]/w, frame_dict['bbox_pixel'][3]/h]
        print(bbox)

        viz = visualize_heatmap2(image, gt_heatmap, bbox=bbox, xy=(gazex_pixel, gazey_pixel), dilation_kernel=6, blur_radius=1.3)
        #viz = visualize_heatmap3(image, gt_heatmap, bbox=bbox, xy=(gazex_pixel, gazey_pixel), dilation_kernel=5, blur_radius=1.3, transparent_bg=True)
        plt.imshow(viz)
        plt.show()

        saved_path = join("processed", "demo_" + img_source.split('/')[-1])

        if 0:  # if saving
            viz.convert("RGB").save(saved_path)
