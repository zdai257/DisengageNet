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
from EYEDIAP.EYEDIAP.Scripts.EYEDIAP_misc import *


class Eyediap(torch.utils.data.Dataset):
    def __init__(self, root_path, img_transform=None, split='test'):
        labels = json.load(open(os.path.join(root_path, "sampled.json"), "rb"))

        self.frames = [{'path': k, 'gazex_norm': v[0], 'gazey_norm': v[1]} for k,v in labels.items()]
        print("EYEDIAP sampled dataset length: ", len(self.frames))
        self.root_path = root_path
        self.transform = img_transform

    def __getitem__(self, idx):
        frame = self.frames[idx]

        inout = 1

        image = Image.open(join(frame['path'])).convert("RGB")
        image = self.transform(image)
        
        #bbox = frame['bbox_norm']
        gazex = frame['gazex_norm']
        gazey = frame['gazey_norm']
        if gazex == 0 and gazey == 0:
            inout = 0

        return image, gazex, gazey, inout

    def __len__(self):
        return len(self.frames)

def collate(batch):
    images, gazex, gazey, inout = zip(*batch)
    return torch.stack(images), list(gazex), list(gazey), list(inout)


if __name__ == "__main__":

    eyediap = Eyediap('EYEDIAP')

    print(eyediap.frames[0])

