import os
from os.path import join
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import json
import tqdm
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
            if id == 'p14':
                self.frames.append([path, ec])


class MPIIDataset(Dataset):
    def __init__(self, transform=None, root_path='MPIIFaceGaze', label_path='MPIIFaceGaze/EC_labels.json', split='train'):
        self.root_path = root_path
        self.transform = transform
        labels = json.load(open(label_path, "rb"))

        self.class_counts = [0., 0.]

        if split == 'train':
            ids = ['p' + str(i).zfill(2) for i in range(14)]
        else:
            ids = ['p14']

        self.frames = []
        for k, v in labels.items():
            path = k
            ec = v

            id = k.split('/')[1]
            if id in ids:
                self.frames.append([path, ec])

                if ec == 0:
                    self.class_counts[0] += 1
                elif ec == 1:
                    self.class_counts[1] += 1

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img_name = self.frames[idx][0]
        image = Image.open(img_name).convert("RGB")
        label = self.frames[idx][1]

        if self.transform:
            image = self.transform(image)

        return image, label


def train():
    #cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=True)
    #print(model)

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)

    for n, p in enumerate(model.parameters()):
        if p.requires_grad:
            #print(n)
            pass

    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MPIIDataset(transform=transform)
    test_dataset = MPIIDataset(transform=transform, split='test')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    total_samples = train_dataset.__len__()
    class_counts = train_dataset.class_counts

    class_weights = torch.tensor([total_samples / (2 * class_counts[0]), total_samples / (2 * class_counts[1])],
                                 dtype=torch.float).to(device)
    print(class_weights)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Training
    num_epochs = 0
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for b, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    model_path = "deepec_resnet50.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def main(model_name="GT360", t=0.85, pretrained=True):

    #C = ColumbiaTest()
    C = MPIIData()

    # Load config file
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = 'cuda:3' if torch.cuda.is_available() else "cpu"
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

    # SELECT MODEL
    if model_name == 'DEEPEC':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 2)

        model.load_state_dict(torch.load("deepec_resnet50.pth", map_location=device))
    elif model_name == 'GT360':
        model = get_ec_model(config, model_weight)
        model_dict = model.state_dict()
        snapshot = torch.load(model_weight, map_location=torch.device(device))
        model_dict.update(snapshot)
        model.load_state_dict(model_dict)
    else:
        raise Exception

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

        if model_name == 'DEEPEC':

            true_false = output[0].argmax().cpu()

            #print(output, true_false)
            if round(float(true_false)) == 0:
                score = 1
            else:
                score = 0
        else:
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

    #train()
    main(model_name='DEEPEC')

