import os
from os.path import join
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import random
from torchvision import transforms
import json
from PIL import Image, ImageDraw
import dlib
import pickle
from network.ec_network_builder import get_ec_model

CNN_FACE_MODEL = 'model/mmod_human_face_detector.dat'  # from http://dlib.net/files/mmod_human_face_detector.dat.bz2

MODEL_WEIGHTS = 'model/model_weights.pkl'
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PREDICTOR_PATH = os.path.join(script_dir, "model", "shape_predictor_68_face_landmarks.dat")
if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()


def compute_metrics(tp, fp, tn, fn):
    """Compute precision, recall, and F1-score from TP, FP, TN, FN counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1_score": f1_score}


class ColumbiaTest(object):
    def __init__(self, root_path='', label_path='Columbia/labels.json', split='train'):
        self.root_path = root_path
        labels = json.load(open(label_path, "rb"))

        self.frames = []
        for k, v in labels.items():
            path = v[0]
            ec = v[-1]

            fullpath = path.split('/')[-1]
            labels = fullpath.split('_')  # Split by '_'

            subject_id = labels[0]
            vertical = int(labels[3][:-1])
            horizontal = int(labels[4].split('.')[0][:-1])
            #print(subject_id, horizontal)

            if subject_id == '0055':  # 0055
                split = 'test'
                fullpath = join('Columbia', 'cropped', split, path)
                self.frames.append([fullpath, ec])

            else:
                fullpath = join('Columbia', 'cropped', split, path)
                self.frames.append([fullpath, ec])


class MPIIData(object):
    def __init__(self, root_path='', label_path='MPIIFaceGaze/EC_labels.json', split='test'):
        self.root_path = root_path
        labels = json.load(open(label_path, "rb"))

        self.frames = []
        for k, v in labels.items():
            path = k
            ec = v

            id = k.split('/')[1]
            if 1:
                self.frames.append([path, ec])


def main(t=0.85, pretrained=True):

    #C = ColumbiaTest()
    C = MPIIData()

    # Load config file
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)

    ec_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if pretrained == True:
        model_weight = MODEL_WEIGHTS
    else:
        model_weight = False

    with open(MODEL_WEIGHTS, 'rb') as f:
        loaded = pickle.load(f)

    # load model weights
    model = get_ec_model(config, model_weight)
    model_dict = model.state_dict()
    snapshot = torch.load(model_weight, map_location=torch.device(device))
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

    model.to(device)
    model.eval()

    total = len(C.frames)
    print("total sample: ", total)
    tp, fp, tn, fn = 0, 0, 0, 0

    for i in C.frames:
        img_source = i[0]
        ec = i[1]

        frame = Image.open(img_source).convert("RGB")

        face = frame

        img = ec_transforms(face)
        img.unsqueeze_(0)

        output = model(img.to(device))

        score = F.sigmoid(output).item()

        #print(score, path)
        if score > t and ec == 1:
            tp += 1
        elif score > t and ec == 0:
            fp += 1
        elif score <= t and ec == 0:
            tn += 1
        else:
            fn += 1

    result = compute_metrics(tp, fp, tn, fn)
    print(result)
    return result


if __name__ == "__main__":

    main()

