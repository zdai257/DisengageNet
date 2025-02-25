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
#from EYEDIAP.EYEDIAP.Scripts.EYEDIAP_misc import *

from network.network_builder import get_gazelle_model, get_gt360_model
#from network.network_builder_update import get_gazelle_model, get_gt360_model
from eval import eval_metrics, eval_metrics_eyediap


class Eyediap(torch.utils.data.Dataset):
    def __init__(self, root_path, img_transform, split='test'):
        labels = json.load(open(os.path.join(root_path, "sampled.json"), "rb"))

        self.frames = [{'path': k, 'gazex_norm': v[0], 'gazey_norm': v[1]} for k,v in labels.items()]
        print("EYEDIAP sampled dataset length: ", len(self.frames))
        self.root_path = root_path
        self.transform = img_transform

    def __getitem__(self, idx):
        frame = self.frames[idx]

        inout = 1

        image = Image.open(join(frame['path'])).convert("RGB")
        w, h = image.size
        image = self.transform(image)
        
        #bbox = frame['bbox_norm']
        gazex = frame['gazex_norm'] / w
        gazey = frame['gazey_norm'] / h
        if gazex == 0 and gazey == 0:
            inout = 0

        return image, [[gazex]], [[gazey]], [inout]

    def __len__(self):
        return len(self.frames)

def collate(batch):
    images, gazex, gazey, inout = zip(*batch)
    return torch.stack(images), list(gazex), list(gazey), list(inout)


if __name__ == "__main__":

    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    # select Network
    model, gazelle_transform = get_gazelle_model(config)
    # load a pre-trained model
    #model.load_state_dict(torch.load(config['model']['pretrained_path'], map_location=device, weights_only=True))
    # load from public pre-trained
    model.load_gazelle_state_dict(torch.load(config['model']['pretrained_path'],  weights_only=True, map_location=device), 
                                  include_backbone=False)
    
    model.to(device)

    input_resolution = config['data']['input_resolution']
    # transform
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    my_transform = transforms.Compose(transform_list)

    # build dataset
    eyediap = Eyediap('EYEDIAP', my_transform)

    oft = 0
    for item in eyediap.frames:
        x = item['gazex_norm']
        y = item['gazey_norm']
        if x==0 and y==0:
            oft += 1
    print(oft, oft / 1750)

    test_loader = torch.utils.data.DataLoader(
        eyediap,
        batch_size=config['eval']['batch_size'],
        collate_fn=collate,
        # shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )

    with torch.no_grad():
        model.eval()

        auc, l2, ap = eval_metrics_eyediap(config, model, test_loader, device)

        #print(auc, l2)
        #print(ap)
