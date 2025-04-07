from __future__ import absolute_import
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import cv2


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
