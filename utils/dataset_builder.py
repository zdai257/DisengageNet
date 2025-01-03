import os
from os.path import join
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose, ToTensor


class CustomDataset(Dataset):
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


def get_dataset_and_loader(data_path, config, train=True):
    transform = Compose([ToTensor()])
    dataset = CustomDataset(data_path, transform)
    loader = DataLoader(dataset, batch_size=config['train']['batch_size'], shuffle=train)
    return dataset, loader

