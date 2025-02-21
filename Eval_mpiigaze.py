import os
from os.path import join
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import yaml
import json


data_path = "MPIIFaceGaze/p00/p00.txt"

with open(data_path, "r") as file:
    lines = file.readlines()

print(len(lines), lines[0])

for line in lines:
    path = line.split(' ')[0]
    fc = np.array([float(line.split(' ')[-7]), float(line.split(' ')[-6]), float(line.split(' ')[-5])])
    gt = np.array([float(line.split(' ')[-4]), float(line.split(' ')[-3]), float(line.split(' ')[-2])])

    gaze = gt - fc
    if abs(gaze[0])<15 and abs(gaze[1])<15 and gaze[2]<0:
        print(path, gaze)


