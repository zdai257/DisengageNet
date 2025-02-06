import os
from os.path import join
import sys
import yaml
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import random
from PIL import Image, ImageDraw
import dlib
import copy
import json
from network.ec_network_builder import get_ec_model

MODEL_WEIGHTS = 'model/model_weights.pkl'


def load_vat(path, img_transform=None, split='train'):
    sequences = json.load(open(os.path.join(path, "{}_preprocessed.json".format(split)), "rb"))
    frames = []
    for i in range(len(sequences)):
        for j in range(len(sequences[i]['frames'])):
            frames.append((i, j))

    vat = []
    for seq_idx, frame_idx in frames:

        seq = sequences[seq_idx]
        frame = seq['frames'][frame_idx]
        image_path = os.path.join(path, frame['path'])
        bboxes = [head['bbox_norm'] for head in frame['heads']]
        gazex = [head['gazex_norm'] for head in frame['heads']]
        gazey = [head['gazey_norm'] for head in frame['heads']]
        inout = [head['inout'] for head in frame['heads']]

        vat.append((image_path, bboxes, gazex, gazey, inout))

    return vat


def main():
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    my_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model_weight = MODEL_WEIGHTS

    model = get_ec_model(config, model_weight)
    model_dict = model.state_dict()
    snapshot = torch.load(model_weight, map_location=torch.device(device))
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

    model.to(device)
    model.eval()

    vat = load_vat('./VAT')

    vat_w_ec = []

    for i, item in enumerate(vat):
        image = Image.open(item[0]).convert('RGB')

        img_tuple = tuple()
        for head_idx, bbox in enumerate(item[1]):

            img = image.crop(bbox)
            img = my_transforms(img)

            # bboxes are tuples of form (xmin, ymin, xmax, ymax) and are in [0,1] normalized image coordinates
            # .crop method expects: (left, top, right, bottom)
            output = model(img.to(device))

            score = F.sigmoid(output).item()

            img_tuple = (item[0], item[1], item[2], item[3], item[4], score)

        vat_w_ec.append(img_tuple)

    with open("VAT_w_EC.json", "w") as file:
        json.dump(vat_w_ec, file)


if __name__ == "__main__":
    main()

