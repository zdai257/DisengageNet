import os
from os.path import join
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import random
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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


def bbox_jitter(bbox_left, bbox_top, bbox_right, bbox_bottom):
    cx = (bbox_right+bbox_left)/2.0
    cy = (bbox_bottom+bbox_top)/2.0
    scale = random.uniform(0.8, 1.2)
    bbox_right = (bbox_right-cx)*scale + cx
    bbox_left = (bbox_left-cx)*scale + cx
    bbox_top = (bbox_top-cy)*scale + cy
    bbox_bottom = (bbox_bottom-cy)*scale + cy
    return bbox_left, bbox_top, bbox_right, bbox_bottom


def drawrect(drawcontext, xy, outline=None, width=0):
    (x1, y1), (x2, y2) = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)


def main(facemode='DLIB', pretrained=True, jitter=0):

    # Load config file
    with open('configuration.yaml', 'r') as file:
        config = yaml.safe_load(file)

    device = config['hardware']['device'] if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    # TODO: config ec training

    #if facemode == 'DLIB':
    cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)
    # alternative dlib face-detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    frame_cnt = 0

    test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if pretrained == True:
        model_weight = MODEL_WEIGHTS
    else:
        model_weight = False

    with open(MODEL_WEIGHTS, 'rb') as f:
        loaded = pickle.load(f)

    print(type(loaded))

    # load model weights
    model = get_ec_model(config, model_weight)
    model_dict = model.state_dict()
    snapshot = torch.load(model_weight, map_location=torch.device(device))
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

    model.to(device)
    model.eval()

    # example input
    # SELECT TEST IMAGE
    # img_source = "data/WALIexample0.png"
    #img_source = "data/0028_2m_30P_0V_0H.jpg"
    img_source = "data/0028_2m_-15P_10V_5H.jpg"

    #height, width, channels = frame.shape
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame = Image.open("data/joye.jpg").convert("RGB")
    frame = Image.open(img_source).convert("RGB")

    frame = frame.resize((864, 576))

    bbox = []
    if facemode == 'DLIB':
        # Alternative face detector
        dets = cnn_face_detector(np.array(frame), 1)
        #dets = detector(np.array(frame), 0)

        for d in dets:
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
    else:
        raise Exception

    print(bbox)
    #frame = Image.fromarray(frame)
    for b in bbox:
        face = frame.crop((b))
        img = test_transforms(face)
        img.unsqueeze_(0)
        if jitter > 0:
            for i in range(jitter):
                bj_left, bj_top, bj_right, bj_bottom = bbox_jitter(b[0], b[1], b[2], b[3])
                bj = [bj_left, bj_top, bj_right, bj_bottom]
                facej = frame.crop((bj))
                img_jittered = test_transforms(facej)
                img_jittered.unsqueeze_(0)
                img = torch.cat([img, img_jittered])

        # forward pass
        output = model(img.to(device))
        if jitter > 0:
            output = torch.mean(output, 0)
        score = F.sigmoid(output).item()

        coloridx = min(int(round(score * 10)), 9)
        draw = ImageDraw.Draw(frame)

        if 1:
            drawrect(draw, [(b[0], b[1]), (b[2], b[3])], width=5)
            draw.text((b[0], b[3]), str(round(score, 2)), fill=(255, 255, 255, 128))

            frame.show()
            saved_path = join("processed", "EC_" + img_source.split('/')[-1])
            frame.save(saved_path)


if __name__ == "__main__":
    main()
