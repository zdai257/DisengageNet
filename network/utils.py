from __future__ import absolute_import
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torchvision
import random
from sklearn.metrics import roc_auc_score


def repeat_tensors(tensor, repeat_counts):
    repeated_tensors = [tensor[i:i+1].repeat(repeat, *[1] * (tensor.ndim - 1)) for i, repeat in enumerate(repeat_counts)]
    return torch.cat(repeated_tensors, dim=0)

def split_tensors(tensor, split_counts):
    indices = torch.cumsum(torch.tensor([0] + split_counts), dim=0)
    return [tensor[indices[i]:indices[i+1]] for i in range(len(split_counts))]

class SoftArgmax2D(torch.nn.Module):
    """Computes soft-argmax over a 2D heatmap to get differentiable coordinates."""
    def __init__(self, temperature=0.1):
        super(SoftArgmax2D, self).__init__()
        self.temperature = temperature  # Controls sharpness of softmax

    def forward(self, heatmap):
        B, H, W = heatmap.shape  # Batch, Height, Width

        # Apply softmax along spatial dimensions
        heatmap = heatmap.view(B, -1)  # Flatten (B, H*W)
        softmax_hm = F.softmax(heatmap / self.temperature, dim=-1)  # Apply softmax
        softmax_hm = softmax_hm.view(B, H, W)  # Reshape back to (B, H, W)

        # Create coordinate grids
        x_range = torch.linspace(0, W - 1, W, device=heatmap.device).view(1, 1, W).expand(B, H, W)
        y_range = torch.linspace(0, H - 1, H, device=heatmap.device).view(1, H, 1).expand(B, H, W)

        # Compute expected coordinates
        x_pred = torch.sum(x_range * softmax_hm, dim=(1, 2))  # Sum over H and W
        y_pred = torch.sum(y_range * softmax_hm, dim=(1, 2))  # Sum over H and W

        coords_pred = torch.stack([x_pred, y_pred], dim=1)  # Shape (B, 2)
        return coords_pred

class CosineL1(torch.nn.Module):
    def __init__(self, epsilon=1e-8):
        super(CosineL1, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred_coords, y_true_coords):
        y_pred_norm = y_pred_coords / (torch.norm(y_pred_coords, dim=1, keepdim=True) + self.epsilon)
        y_true_norm = y_true_coords / (torch.norm(y_true_coords, dim=1, keepdim=True) + self.epsilon)

        cosine_sim = torch.sum(y_pred_norm * y_true_norm, dim=1)
        # Loss is in range [0, 2]
        loss = 1 - cosine_sim
        return loss

class VectorL2Loss(torch.nn.Module):
    def __init__(self):
        super(VectorL2Loss, self).__init__()

    def forward(self, y_pred_coords, y_true_coords):
        return ((y_pred_coords - y_true_coords) ** 2).mean()

def visualize_heatmap(pil_image, heatmap, bbox=None):
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_image.size, Image.Resampling.BILINEAR)
    heatmap = plt.cm.jet(np.array(heatmap) / 255.)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).convert("RGBA")
    heatmap.putalpha(128)
    overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap)

    if bbox is not None:
        width, height = pil_image.size
        xmin, ymin, xmax, ymax = bbox
        draw = ImageDraw.Draw(overlay_image)
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline="green", width=3)
    return overlay_image

def clip_line_to_bbox(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    """
            Clips a line so that it starts from the bbox edge instead of the center.
            Uses a simple bounding box clipping approach.
    """
    if x2 < xmin:  # Left edge
        y1 = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
        x1 = xmin
    elif x2 > xmax:  # Right edge
        y1 = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
        x1 = xmax
    if y2 < ymin:  # Top edge
        x1 = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
        y1 = ymin
    elif y2 > ymax:  # Bottom edge
        x1 = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
        y1 = ymax
    return int(x1), int(y1)

def visualize_heatmap2(pil_image, heatmap, bbox=None, xy=None, dilation_kernel=2, blur_radius=5, color="lime", transparent_bg=None):
    dot_radius = 5

    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    # Convert heatmap to uint8 format
    heatmap = (heatmap * 255).astype(np.uint8)

    # Use a circular kernel for dilation (ensures rounded spread)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
    heatmap = cv2.dilate(heatmap, kernel, iterations=1)

    # Apply Gaussian blur to smooth the spread (avoids blocky grid effects)
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=blur_radius, sigmaY=blur_radius)

    # Convert heatmap back to PIL image
    heatmap = Image.fromarray(heatmap)

    heatmap = heatmap.resize(pil_image.size, Image.Resampling.BILINEAR)
    #heatmap = heatmap.resize(pil_image.size, Image.Resampling.BICUBIC)

    if transparent_bg is None:
        heatmap = plt.cm.jet(np.array(heatmap) / 255.)
        heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
        heatmap = Image.fromarray(heatmap).convert("RGBA")
        heatmap.putalpha(128)

    else:
        heatmap = np.array(heatmap) / 255.
        colormap = plt.cm.jet(heatmap)

        # Extract RGBA channels
        rgba = (colormap[:, :, :3] * 255).astype(np.uint8)  # RGB channels
        alpha = (colormap[:, :, 2] < colormap[:, :, 0]) * 64  # Less blue -> more opacity at most 50%
        heatmap = Image.fromarray(np.dstack((rgba, alpha)).astype(np.uint8), "RGBA")

    overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap)

    if bbox is not None:
        width, height = pil_image.size
        xmin, ymin, xmax, ymax = bbox
        draw = ImageDraw.Draw(overlay_image)
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline=color, width=3)
        if xy is not None:
            center_x = int((xmin + xmax) / 2 * width)
            center_y = int((ymin + ymax) / 2 * height)

            # Compute clipped start point at the bbox boundary
            start_x, start_y = clip_line_to_bbox(center_x, center_y, xy[0], xy[1],
                                                 int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height))
            # Draw the line from bbox center to gaze point
            draw.line([(start_x, start_y), (xy[0], xy[1])], fill=color, width=3)

            # Draw a dot at the gaze point
            draw.ellipse(
                [(xy[0] - dot_radius, xy[1] - dot_radius),
                 (xy[0] + dot_radius, xy[1] + dot_radius)],
                fill=color, outline=color
            )
    return overlay_image

def visualize_heatmap3(pil_image, heatmap, bbox=None, xy=None, dilation_kernel=2, blur_radius=5, color="lime", transparent_bg=None):
    dot_radius = 5

    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    # Convert heatmap to uint8 format
    heatmap = (heatmap * 255).astype(np.uint8)

    # Use a circular kernel for dilation (ensures rounded spread)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
    heatmap = cv2.dilate(heatmap, kernel, iterations=1)

    # Apply Gaussian blur to smooth the spread (avoids blocky grid effects)
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=blur_radius, sigmaY=blur_radius)

    # Convert heatmap back to PIL image
    heatmap = Image.fromarray(heatmap)

    heatmap = heatmap.resize(pil_image.size, Image.Resampling.BILINEAR)
    #heatmap = heatmap.resize(pil_image.size, Image.Resampling.BICUBIC)

    if transparent_bg is None:
        heatmap = plt.cm.jet(np.array(heatmap) / 255.)
        heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
        heatmap = Image.fromarray(heatmap).convert("RGBA")
        heatmap.putalpha(128)

    else:
        heatmap = np.array(heatmap) / 255.
        colormap = plt.cm.jet(heatmap)

        # Extract RGBA channels
        rgba = (colormap[:, :, :3] * 255).astype(np.uint8)  # RGB channels
        alpha = (colormap[:, :, 2] < colormap[:, :, 0]) * 64  # Less blue -> more opacity at most 50%
        heatmap = Image.fromarray(np.dstack((rgba, alpha)).astype(np.uint8), "RGBA")

    #overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap)
    trans_overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))

    if bbox is not None:
        width, height = pil_image.size
        xmin, ymin, xmax, ymax = bbox
        draw = ImageDraw.Draw(trans_overlay)
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], outline=color, width=3)
        if xy is not None:
            center_x = int((xmin + xmax) / 2 * width)
            center_y = int((ymin + ymax) / 2 * height)

            # Compute clipped start point at the bbox boundary
            start_x, start_y = clip_line_to_bbox(center_x, center_y, xy[0], xy[1],
                                                 int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height))
            # Draw the line from bbox center to gaze point
            draw.line([(start_x, start_y), (xy[0], xy[1])], fill=color, width=3)

            # Draw a dot at the gaze point
            draw.ellipse(
                [(xy[0] - dot_radius, xy[1] - dot_radius),
                 (xy[0] + dot_radius, xy[1] + dot_radius)],
                fill=color, outline=color
            )

    trans_overlay = Image.alpha_composite(trans_overlay.convert("RGBA"), heatmap)
    return trans_overlay

def stack_and_pad(tensor_list):
    max_size = max([t.shape[0] for t in tensor_list])
    padded_list = []
    for t in tensor_list:
        if t.shape[0] == max_size:
            padded_list.append(t)
        else:
            padded_list.append(torch.cat([t, torch.zeros(max_size - t.shape[0], *t.shape[1:])], dim=0))
    return torch.stack(padded_list)

def smooth_by_conv(window_size, df, col):
    padded_track = pd.concat([pd.DataFrame([[df.iloc[0][col]]]*(window_size//2), columns=[0]),
                     df[col],
                     pd.DataFrame([[df.iloc[-1][col]]]*(window_size//2), columns=[0])])
    smoothed_signals = np.convolve(padded_track.squeeze(), np.ones(window_size)/window_size, mode='valid')
    return smoothed_signals


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def unnorm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = np.array(std).reshape(3,1,1)
    mean = np.array(mean).reshape(3,1,1)
    return img * std + mean


def get_head_box_channel(x_min, y_min, x_max, y_max, width, height, resolution, coordconv=False):
    head_box = np.array([x_min/width, y_min/height, x_max/width, y_max/height])*resolution
    head_box = head_box.astype(int)
    head_box = np.clip(head_box, 0, resolution-1)
    if coordconv:
        unit = np.array(range(0,resolution), dtype=np.float32)
        head_channel = []
        for i in unit:
            head_channel.append([unit+i])
        head_channel = np.squeeze(np.array(head_channel)) / float(np.max(head_channel))
        head_channel[head_box[1]:head_box[3],head_box[0]:head_box[2]] = 0
    else:
        head_channel = np.zeros((resolution,resolution), dtype=np.float32)
        head_channel[head_box[1]:head_box[3],head_box[0]:head_box[2]] = 1
    head_channel = torch.from_numpy(head_channel)
    return head_channel


def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    img = img/np.max(img) # normalize heatmap so it has max value of 1
    return to_torch(img)


def multi_hot_targets(gaze_pts, out_res):
    w, h = out_res
    target_map = np.zeros((h, w))
    for p in gaze_pts:
        if p[0] >= 0:
            x, y = map(int,[p[0]*w.float(), p[1]*h.float()])
            x = min(x, w-1)
            y = min(y, h-1)
            target_map[y, x] = 1
    return target_map


def random_crop(img, bbox, gazex, gazey, inout):
    width, height = img.size
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox
    # determine feasible crop region (must include bbox and gaze target)
    crop_reg_xmin = min(bbox_xmin, min(gazex)) if inout else bbox_xmin
    crop_reg_ymin = min(bbox_ymin, min(gazey)) if inout else bbox_ymin
    crop_reg_xmax = max(bbox_xmax, max(gazex)) if inout else bbox_xmax
    crop_reg_ymax = max(bbox_ymax, max(gazey)) if inout else bbox_ymax

    try:
        xmin = random.randint(0, int(crop_reg_xmin))
        ymin = random.randint(0, int(crop_reg_ymin))
        xmax = random.randint(int(crop_reg_xmax), width)
        ymax = random.randint(int(crop_reg_ymax), height)
    except:
        import pdb
        pdb.set_trace()

    img = torchvision.transforms.functional.crop(img, ymin, xmin, ymax - ymin, xmax - xmin)
    bbox = [bbox_xmin - xmin, bbox_ymin - ymin, bbox_xmax - xmin, bbox_ymax - ymin]
    gazex = [x - xmin for x in gazex]
    gazey = [y - ymin for y in gazey]

    return img, bbox, gazex, gazey


def horiz_flip(img, bbox, gazex, gazey, inout):
    width, height = img.size
    img = torchvision.transforms.functional.hflip(img)
    xmin, ymin, xmax, ymax = bbox
    bbox = [width - xmax, ymin, width - xmin, ymax]
    if inout:
        gazex = [width - x for x in gazex]
    return img, bbox, gazex, gazey


def random_bbox_jitter(img, bbox):
    width, height = img.size
    xmin, ymin, xmax, ymax = bbox
    jitter = 0.2
    xmin_j = (np.random.random_sample() * (jitter * 2) - jitter) * (xmax - xmin)
    xmax_j = (np.random.random_sample() * (jitter * 2) - jitter) * (xmax - xmin)
    ymin_j = (np.random.random_sample() * (jitter * 2) - jitter) * (ymax - ymin)
    ymax_j = (np.random.random_sample() * (jitter * 2) - jitter) * (ymax - ymin)

    bbox = [max(0, xmin_j + xmin), max(0, ymin_j + ymin), min(width, xmax_j + xmax), min(height, ymax_j + ymax)]

    return bbox


def get_heatmap(gazex, gazey, height, width, sigma=3, htype="Gaussian"):
    # Adapted from https://github.com/ejcgt/attention-target-detection/blob/master/utils/imutils.py

    img = torch.zeros(height, width)
    if gazex < 0 or gazey < 0:  # return empty map if out of frame
        return img
    gazex = int(gazex * width)
    gazey = int(gazey * height)

    # Check that any part of the gaussian is in-bounds
    ul = [int(gazex - 3 * sigma), int(gazey - 3 * sigma)]
    br = [int(gazex + 3 * sigma + 1), int(gazey + 3 * sigma + 1)]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if htype == "Gaussian":
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif htype == "Cauchy":
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]: img_y[1], img_x[0]: img_x[1]] += g[g_y[0]: g_y[1], g_x[0]: g_x[1]]
    img = img / img.max()  # normalize heatmap so it has max value of 1
    return img


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
    pred_y, pred_x = np.unravel_index(argmax, (heatmap.shape[0], heatmap.shape[1]))
    pred_x = pred_x / float(heatmap.shape[1])
    pred_y = pred_y / float(heatmap.shape[0])

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
