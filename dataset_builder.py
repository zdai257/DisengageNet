import os
from os.path import join
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import cv2
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms.functional as TF
import glob
import network.utils as utils

import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt


input_resolution = 224
output_resolution = 64

class ECDataset(Dataset):
    def __init__(self, data_file, transform=None):
        self.data = pd.read_json(data_file, lines=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image = Image.open(item['image_path']).convert('RGB')
        face = None

        label = int(item['label'])
        if self.transform:
            image = self.transform(image)
        return image, face, label


class VideoAttTarget_video(Dataset):
    def __init__(self, data_dir, annotation_dir, transform, input_size=input_resolution, output_size=output_resolution,
                 test=False, imshow=False, seq_len_limit=400):
        shows = glob.glob(os.path.join(annotation_dir, '*'))
        self.all_sequence_paths = []
        for s in shows:
            sequence_annotations = glob.glob(os.path.join(s, '*', '*.txt'))
            self.all_sequence_paths.extend(sequence_annotations)
        self.data_dir = data_dir
        self.transform = transform
        self.input_size = input_size
        self.output_size = output_size
        self.test = test
        self.imshow = imshow
        self.length = len(self.all_sequence_paths)
        self.seq_len_limit = seq_len_limit

    def __getitem__(self, index):
        sequence_path = self.all_sequence_paths[index]
        df = pd.read_csv(sequence_path, header=None, index_col=False,
                         names=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey'])
        show_name = sequence_path.split('/')[-3]
        clip = sequence_path.split('/')[-2]
        seq_len = len(df.index)

        # moving-avg smoothing
        window_size = 11 # should be odd number
        df['xmin'] = utils.smooth_by_conv(window_size, df, 'xmin')
        df['ymin'] = utils.smooth_by_conv(window_size, df, 'ymin')
        df['xmax'] = utils.smooth_by_conv(window_size, df, 'xmax')
        df['ymax'] = utils.smooth_by_conv(window_size, df, 'ymax')

        if not self.test:
            # cond for data augmentation
            cond_jitter = np.random.random_sample()
            cond_flip = np.random.random_sample()
            cond_color = np.random.random_sample()
            if cond_color < 0.5:
                n1 = np.random.uniform(0.5, 1.5)
                n2 = np.random.uniform(0.5, 1.5)
                n3 = np.random.uniform(0.5, 1.5)
            cond_crop = np.random.random_sample()

            # if longer than seq_len_limit, cut it down to the limit with the init index randomly sampled
            if seq_len > self.seq_len_limit:
                sampled_ind = np.random.randint(0, seq_len - self.seq_len_limit)
                seq_len = self.seq_len_limit
            else:
                sampled_ind = 0

            if cond_crop < 0.5:
                sliced_x_min = df['xmin'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_x_max = df['xmax'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_y_min = df['ymin'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_y_max = df['ymax'].iloc[sampled_ind:sampled_ind+seq_len]

                sliced_gaze_x = df['gazex'].iloc[sampled_ind:sampled_ind+seq_len]
                sliced_gaze_y = df['gazey'].iloc[sampled_ind:sampled_ind+seq_len]

                check_sum = sliced_gaze_x.sum() + sliced_gaze_y.sum()
                all_outside = check_sum == -2*seq_len

                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                if all_outside:
                    crop_x_min = np.min([sliced_x_min.min(), sliced_x_max.min()])
                    crop_y_min = np.min([sliced_y_min.min(), sliced_y_max.min()])
                    crop_x_max = np.max([sliced_x_min.max(), sliced_x_max.max()])
                    crop_y_max = np.max([sliced_y_min.max(), sliced_y_max.max()])
                else:
                    crop_x_min = np.min([sliced_gaze_x.min(), sliced_x_min.min(), sliced_x_max.min()])
                    crop_y_min = np.min([sliced_gaze_y.min(), sliced_y_min.min(), sliced_y_max.min()])
                    crop_x_max = np.max([sliced_gaze_x.max(), sliced_x_min.max(), sliced_x_max.max()])
                    crop_y_max = np.max([sliced_gaze_y.max(), sliced_y_min.max(), sliced_y_max.max()])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Get image size
                path = os.path.join(self.data_dir, show_name, clip, df['path'].iloc[0])
                img = Image.open(path)
                img = img.convert('RGB')
                width, height = img.size

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)
        else:
            sampled_ind = 0


        faces, images, head_channels, heatmaps, paths, gazes, imsizes, gaze_inouts = [], [], [], [], [], [], [], []
        index_tracker = -1
        for i, row in df.iterrows():
            index_tracker = index_tracker+1
            if not self.test:
                if index_tracker < sampled_ind or index_tracker >= (sampled_ind + self.seq_len_limit):
                    continue

            face_x1 = row['xmin']  # note: Already in image coordinates
            face_y1 = row['ymin']  # note: Already in image coordinates
            face_x2 = row['xmax']  # note: Already in image coordinates
            face_y2 = row['ymax']  # note: Already in image coordinates
            gaze_x = row['gazex']  # note: Already in image coordinates
            gaze_y = row['gazey']  # note: Already in image coordinates

            impath = os.path.join(self.data_dir, show_name, clip, row['path'])
            img = Image.open(impath)
            img = img.convert('RGB')

            width, height = img.size
            imsize = torch.FloatTensor([width, height])
            # imsizes.append(imsize)

            face_x1, face_y1, face_x2, face_y2 = map(float, [face_x1, face_y1, face_x2, face_y2])
            gaze_x, gaze_y = map(float, [gaze_x, gaze_y])
            if gaze_x == -1 and gaze_y == -1:
                gaze_inside = False
            else:
                if gaze_x < 0: # move gaze point that was sliglty outside the image back in
                    gaze_x = 0
                if gaze_y < 0:
                    gaze_y = 0
                gaze_inside = True

            if not self.test:
                ## data augmentation
                # Jitter (expansion-only) bounding box size.
                if cond_jitter < 0.5:
                    k = cond_jitter * 0.1
                    face_x1 -= k * abs(face_x2 - face_x1)
                    face_y1 -= k * abs(face_y2 - face_y1)
                    face_x2 += k * abs(face_x2 - face_x1)
                    face_y2 += k * abs(face_y2 - face_y1)
                    face_x1 = np.clip(face_x1, 0, width)
                    face_x2 = np.clip(face_x2, 0, width)
                    face_y1 = np.clip(face_y1, 0, height)
                    face_y2 = np.clip(face_y2, 0, height)

                # Random Crop
                if cond_crop < 0.5:
                    # Crop it
                    img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)

                    # Record the crop's (x, y) offset
                    offset_x, offset_y = crop_x_min, crop_y_min

                    # convert coordinates into the cropped frame
                    face_x1, face_y1, face_x2, face_y2 = face_x1 - offset_x, face_y1 - offset_y, face_x2 - offset_x, face_y2 - offset_y
                    if gaze_inside:
                        gaze_x, gaze_y = (gaze_x- offset_x), \
                                         (gaze_y - offset_y)
                    else:
                        gaze_x = -1; gaze_y = -1

                    width, height = crop_width, crop_height

                # Flip?
                if cond_flip < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    x_max_2 = width - face_x1
                    x_min_2 = width - face_x2
                    face_x2 = x_max_2
                    face_x1 = x_min_2
                    if gaze_x != -1 and gaze_y != -1:
                        gaze_x = width - gaze_x

                # Random color change
                if cond_color < 0.5:
                    img = TF.adjust_brightness(img, brightness_factor=n1)
                    img = TF.adjust_contrast(img, contrast_factor=n2)
                    img = TF.adjust_saturation(img, saturation_factor=n3)

            # Face crop
            face = img.copy().crop((int(face_x1), int(face_y1), int(face_x2), int(face_y2)))

            # Head channel image
            head_channel = utils.get_head_box_channel(face_x1, face_y1, face_x2, face_y2, width, height,
                                                        resolution=self.input_size, coordconv=False).unsqueeze(0)
            if self.transform is not None:
                img = self.transform(img)
                face = self.transform(face)

            # Deconv output
            if gaze_inside:
                gaze_x /= float(width) # fractional gaze
                gaze_y /= float(height)
                gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
                gaze_map = utils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                 3,
                                                 type='Gaussian')
                gazes.append(torch.FloatTensor([gaze_x, gaze_y]))
            else:
                gaze_map = torch.zeros(self.output_size, self.output_size)
                gazes.append(torch.FloatTensor([-1, -1]))
            faces.append(face)
            images.append(img)
            head_channels.append(head_channel)
            heatmaps.append(gaze_map)
            gaze_inouts.append(torch.FloatTensor([int(gaze_inside)]))

        if self.imshow:
            for i in range(len(faces)):
                fig = plt.figure(111)
                img = 255 - utils.unnorm(images[i].numpy()) * 255
                img = np.clip(img, 0, 255)
                plt.imshow(np.transpose(img, (1, 2, 0)))
                plt.imshow(cv2.resize(heatmaps[i], (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR), cmap='jet', alpha=0.3)
                plt.imshow(cv2.resize(1 - head_channels[i].squeeze(0), (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR), alpha=0.2)
                plt.savefig(os.path.join('debug', 'viz_%d_inout=%d.png' % (i, gaze_inouts[i])))
                plt.close('all')

        faces = torch.stack(faces)
        images = torch.stack(images)
        head_channels = torch.stack(head_channels)
        heatmaps = torch.stack(heatmaps)
        gazes = torch.stack(gazes)
        gaze_inouts = torch.stack(gaze_inouts)
        # imsizes = torch.stack(imsizes)
        # print(faces.shape, images.shape, head_channels.shape, heatmaps.shape)

        if self.test:
            return images, faces, head_channels, heatmaps, gazes, gaze_inouts
        else: # train
            return images, faces, head_channels, heatmaps, gaze_inouts

    def __len__(self):
        return self.length


# Infinite sampler for repeating smaller datasets
class InfiniteSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __iter__(self):
        while True:
            yield from torch.randperm(len(self.dataset))
    
    def __len__(self):
        return len(self.dataset)  # Not used but must be defined
    

# Alternating data loader with repetition handling
class AlternatingDataLoader:
    def __init__(self, loader_a, loader_b, repeat_smaller=True):
        self.loader_a = loader_a
        self.loader_b = loader_b
        self.repeat_smaller = repeat_smaller
        self.iterator_a = iter(loader_a)
        self.iterator_b = iter(loader_b)
    
    def __iter__(self):
        self.iterator_a = iter(self.loader_a)
        self.iterator_b = iter(self.loader_b)
        return self
    
    def __next__(self):
        try:
            batch_a = next(self.iterator_a)
        except StopIteration:
            if self.repeat_smaller:
                self.iterator_a = iter(self.loader_a)
                batch_a = next(self.iterator_a)
            else:
                raise StopIteration("Dataset A exhausted.")
        
        try:
            batch_b = next(self.iterator_b)
        except StopIteration:
            if self.repeat_smaller:
                self.iterator_b = iter(self.loader_b)
                batch_b = next(self.iterator_b)
            else:
                raise StopIteration("Dataset B exhausted.")
        
        return batch_a, batch_b

# Example usage
if __name__ == "__main__":
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    my_transform = transforms.Compose(transform_list)

    VAT_path = "/home/CAMPUS/daiz1/Documents/videoattentiontarget/images"
    VAT_train_label = "/home/CAMPUS/daiz1/Documents/videoattentiontarget/annotations/train"
    Columbia_path = "/home/CAMPUS/daiz1/Documents/ColumbiaGaze"

    # TODO: try load with .json
    dataset_vat = VideoAttTarget_video(data_dir=VAT_path, annotation_dir=VAT_train_label, transform=my_transform)  # Larger dataset

    for i, batch_vat in enumerate(dataset_vat):
        images, faces, head_channels, heatmaps, gaze_inouts = batch_vat
        print(head_channels, gaze_inouts)
        exit()

    dataset_b = ECDataset()   # Smaller dataset
    
    # Define dataloaders
    loader_a = DataLoader(dataset_vat, batch_size=8, shuffle=True)
    loader_b = DataLoader(dataset_b, batch_size=8, shuffle=True, sampler=InfiniteSampler(dataset_b))  # Infinite sampler for smaller dataset
    
    # Create the alternating data loader
    alternating_loader = AlternatingDataLoader(loader_a, loader_b, repeat_smaller=True)
    
    # Iterate through alternating batches
    for i, (batch_a, batch_b) in enumerate(alternating_loader):
        data_a, labels_a = batch_a
        data_b, labels_b = batch_b
        
        print(f"Batch {i}:")
        print(f"  Dataset A - Data shape: {data_a.shape}, Labels shape: {labels_a.shape}")
        print(f"  Dataset B - Data shape: {data_b.shape}, Labels shape: {labels_b.shape}")
        
        if i >= 10:  # Stop after 10 batches
            break
    
