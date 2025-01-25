import os
from os.path import join
import math
import numpy as np
import pandas as pd
import json
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor
import torchvision.transforms.functional as TF
from torch.optim import RMSprop, Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import yaml
import pickle
import dlib
from network.ec_network_builder import get_ec_model

CNN_FACE_MODEL = 'model/mmod_human_face_detector.dat'  # from http://dlib.net/files/mmod_human_face_detector.dat.bz2
MODEL_WEIGHTS = 'model/model_weights.pkl'


class ColumbiaGazeDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, isTrain='train', num_subjects=56):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.dataset_path = join(self.dataset_dir, "ColumbiaGaze", "columbia_gaze_data_set", f"Columbia Gaze Data Set")
        if isTrain == "train":
            num_subjects = 55
            subject_lst = list(range(num_subjects))
        elif isTrain == "test":
            num_subjects = 1
            subject_lst = list(range(55, 56))
        subject_id_str = [str(x).zfill(4) for x in subject_lst]

        # TODO: leave-one-out for test split
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


def collate(batch):
    image, gaze_vector, ec = zip(*batch)
    return torch.stack(image), list(gaze_vector), list(ec)


if __name__ == "__main__":
    pretrained = False

    # Load config file
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)

    if pretrained == True:
        model_weight = MODEL_WEIGHTS
    else:
        model_weight = False

    with open(MODEL_WEIGHTS, 'rb') as f:
        loaded = pickle.load(f)

    # load model weights
    model = get_ec_model(config, model_weight)

    model_dict = model.state_dict()
    # still necessary for random init?
    #snapshot = torch.load(model_weight, map_location=torch.device(device))
    #model_dict.update(snapshot)
    model.load_state_dict(model_dict)

    # Randomly initialize learnable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only initialize unfrozen parameters
            if param.dim() > 1:  # Weights
                torch.nn.init.xavier_normal_(param)
            else:  # Biases
                torch.nn.init.zeros_(param)

    model.to(device)

    lr = 0.0001
    num_epochs = 100

    # assign lr
    param_dicts = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            param_dicts.append({'params': p, 'lr': lr})

    optimizer = Adam(param_dicts)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    input_resolution = 224

    train_transforms = transforms.Compose([transforms.Resize(input_resolution),
                                           transforms.CenterCrop(input_resolution),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                                          )

    train_dataset = ColumbiaGazeDataset("", transform=None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        collate_fn=collate,
        # shuffle=True,
        num_workers=3,
        pin_memory=True
    )

    train_length = train_dataset.__len__()

    #bce_loss = torch.nn.BCELoss(reduction='sum')  # Binary Cross-Entropy Loss
    bce_loss = torch.nn.BCEWithLogitsLoss()

    # save dir for checkpoints
    os.makedirs('ec_checkpoints', exist_ok=True)

    best_loss = float('inf')

    # START TRAINING
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.

        for batch, (image, _, ec) in tqdm(enumerate(train_loader), total=len(train_loader)):
            bbox = []
            # TODO: face detector as processing, rather than training step
            # face detect for a Batch
            img_lst = [image[i].numpy() for i in range(image.shape[0])]
            dets_lst = cnn_face_detector(img_lst, 1)

            for d in dets_lst:
                # for Columbia, presume one face per image
                d = d[0]
                l = d.rect.left()
                r = d.rect.right()
                t = d.rect.top()
                b = d.rect.bottom()
                # expand a bit
                l -= (r - l) * 0.2
                r += (r - l) * 0.2
                t -= (b - t) * 0.2
                b += (b - t) * 0.2
                bbox.append([l, t, r, b])

            imgs = []
            for i, b in enumerate(bbox):
                face = Image.fromarray(img_lst[i]).crop((b))

                imgs.append(train_transforms(face))

            images = torch.stack(imgs)

            # forward pass
            preds = model(images.to(device))

            score = F.sigmoid(preds).item()

            # TODO: correct Loss func
            print(score, ec)
            continue
            ec_loss = bce_loss(score, ec)

            optimizer.zero_grad()
            ec_loss.backward()
            optimizer.step()

            # Accumulate loss for reporting
            epoch_loss += ec_loss.item()

        scheduler.step()

        mean_ep_loss = epoch_loss / train_length

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {mean_ep_loss:.4f}")

