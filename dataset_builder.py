import os
from os.path import join
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
#import cv2
from skimage.transform import resize
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
                    # Headpose appears to be independent from V / H, if V=0 & H=0: EC = True
                    headpose = int(labels[2][:-1])
                    vertical = int(labels[3][:-1])
                    horizontal = int(labels[4][:-1])
                    ec = 0
                    gaze_vector = (0., 0.)
                    v_unit = 2 * math.tan(math.radians(10))
                    #h_unit = 2 * math.tan(math.radians(5))
                    if vertical == 0:
                        if horizontal == 0:
                            ec = 1
                        elif horizontal > 0:
                            gaze_vector = (1., 0.)
                        else:
                            gaze_vector = (-1., 0.)
                    elif vertical == 10:
                        if horizontal == 0:
                            gaze_vector = (0., 1.)
                        else:
                            # gaze_vector = ( 2*tan(H), 2*tan(10) )
                            gaze_vector = (2 * math.tan(math.radians(horizontal)), v_unit)
                            magnitude = np.linalg.norm(np.array(gaze_vector))
                            gaze_vector = (gaze_vector[0]/magnitude, gaze_vector[1]/magnitude)
                    elif vertical == -10:
                        if horizontal == 0:
                            gaze_vector = (0., -1.)
                        else:
                            # gaze_vector = ( 2*tan(H), -2*tan(10) )
                            gaze_vector = (2 * math.tan(math.radians(horizontal)), -v_unit)
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
        # TODO: Convert gaze vector to a heatmap / grid coordinate at boundary of (66, 66)
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
                plt.imshow(resize(heatmaps[i], (self.input_size, self.input_size)), cmap='jet', alpha=0.3)
                plt.imshow(resize(1 - head_channels[i].squeeze(0), (self.input_size, self.input_size)), alpha=0.2)
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


def load_daisee(data_dir):
    for split in os.listdir(data_dir):
        if split == 'Test':
            for id in os.listdir(join(data_dir, split)):
                for root, dirs, files in os.walk(join(data_dir, split, id)):
                    print(root, dirs)
                    print(files)
                    for filename in files:
                        if filename.endswith('.csv'):
                            df = pd.read_csv(join(root, filename), header=0, sep=',').values
                            seq_length = df.shape[0]
                            print(seq_length, df.shape)

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
    
