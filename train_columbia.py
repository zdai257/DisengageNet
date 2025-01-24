import os
from os.path import join
import math
import numpy as np
import pandas as pd
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor
import torchvision.transforms.functional as TF
from torch.optim import RMSprop, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import yaml
#from dataset_builder_builder import *


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


if __name__ == "__main__":
    data_dir = "./output_DAiSEE/DataSet"

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


