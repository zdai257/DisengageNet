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
import sys
import dlib
from network.network_builder import get_gazelle_model
from network.network_builder_update2 import get_gt360_model
from eval import eval_metrics, eval_metrics_eyediap, eval_metrics_eyediap2


CNN_FACE_MODEL = 'model/mmod_human_face_detector.dat'  # from http://dlib.net/files/mmod_human_face_detector.dat.bz2
MODEL_WEIGHTS = 'model/model_weights.pkl'

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PREDICTOR_PATH = os.path.join(script_dir, "model", "shape_predictor_68_face_landmarks.dat")
if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()


class Eyediap(torch.utils.data.Dataset):
    def __init__(self, root_path, img_transform, split='test'):
        labels = json.load(open(os.path.join(root_path, "sampled2.json"), "rb"))
        self.frames = []
        for sample in labels:
            if sample['bbox_norm'][0] is not None:
                self.frames.append({'path': sample['path'], 'gazex_norm': sample['gazex_norm'],
                                    'gazey_norm': sample['gazey_norm'], 'bbox_norm': sample['bbox_norm']})
        print("EYEDIAP sampled2 dataset length: ", len(self.frames))
        self.root_path = root_path
        self.transform = img_transform

    def __getitem__(self, idx):
        frame = self.frames[idx]

        inout = 1

        image = Image.open(join(frame['path'])).convert("RGB")
        w, h = image.size
        image = self.transform(image)
        
        bbox = frame['bbox_norm']
        gazex = frame['gazex_norm'] / w
        gazey = frame['gazey_norm'] / h
        if gazex == 0 and gazey == 0:
            inout = 0

        return image, bbox, [[gazex]], [[gazey]], [inout]

    def __len__(self):
        return len(self.frames)

def collate(batch):
    images, gazex, gazey, inout = zip(*batch)
    return torch.stack(images), list(gazex), list(gazey), list(inout)


class Eyediap2(torch.utils.data.Dataset):
    def __init__(self, root_path, img_transform, split='test'):
        labels = json.load(open(os.path.join(root_path, "sampled.json"), "rb"))

        self.root_path = root_path
        self.transform = img_transform

        self.bbox_scalar = 0.2
        cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)

        self.frames = []

        for k, v in labels.items():
            # get bbox
            bbox = []
            frame = Image.open(k).convert("RGB")
            # resize frame for efficiency
            frame.thumbnail((448, 448))
            w, h = frame.size
            dets = cnn_face_detector(np.array(frame), 1)
            # assume one face in Eyediap
            if len(dets) != 0:

                for d in dets:
                    l = d.rect.left()
                    r = d.rect.right()
                    t = d.rect.top()
                    b = d.rect.bottom()
                    # expand a bit
                    l -= (r - l) * self.bbox_scalar
                    r += (r - l) * self.bbox_scalar
                    t -= (b - t) * self.bbox_scalar
                    b += (b - t) * self.bbox_scalar
                    # normalize
                    bbox.append((l / w, t / h, r / w, b / h))
                    break
            else:
                bbox.append(None)

            self.frames.append({'path': k, 'gazex_norm': v[0], 'gazey_norm': v[1], 'bbox_norm': bbox})

        print("EYEDIAP sampled dataset length: ", len(self.frames))

        with open('EYEDIAP/sampled2.json', "w") as json_file:
            json.dump(self.frames, json_file, indent=4)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        inout = 1

        image = Image.open(join(frame['path'])).convert("RGB")
        w, h = image.size

        image = self.transform(image)

        bbox = frame['bbox_norm']
        gazex = frame['gazex_norm'] / w
        gazey = frame['gazey_norm'] / h
        if gazex == 0 and gazey == 0:
            inout = 0

        return image, [bbox], [[gazex]], [[gazey]], [inout]

    def __len__(self):
        return len(self.frames)


def collate2(batch):
    images, bboxes, gazex, gazey, inout = zip(*batch)
    return torch.stack(images), list(bboxes), list(gazex), list(gazey), list(inout)


if __name__ == "__main__":

    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    # select Network
    #model, gazelle_transform = get_gazelle_model(config)
    model, gazelle_transform = get_gt360_model(config)
    # load a pre-trained model
    model.load_state_dict(torch.load(config['model']['pretrained_path'], map_location=device, weights_only=True))
    # load from public pre-trained
    #model.load_gazelle_state_dict(torch.load(config['model']['pretrained_path'],  weights_only=True, map_location=device), 
    #                              include_backbone=False)
    
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
        collate_fn=collate2,
        # shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )

    with torch.no_grad():
        model.eval()

        auc, l2, ap = eval_metrics_eyediap2(config, model, test_loader, device)

        #print(auc, l2)
        #print(ap)
