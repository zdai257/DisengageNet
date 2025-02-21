import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import dlib
import os
from os.path import join
import sys
from network.network_builder import get_gazelle_model
from network.ec_network_builder import get_ec_model
from ec_inference import drawrect


CNN_FACE_MODEL = 'model/mmod_human_face_detector.dat'  # from http://dlib.net/files/mmod_human_face_detector.dat.bz2
MODEL_WEIGHTS = 'model/model_weights.pkl'

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PREDICTOR_PATH = os.path.join(script_dir, "model", "shape_predictor_68_face_landmarks.dat")
if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()


class DemoSys():
    def __init__(self, model_ec=MODEL_WEIGHTS, model_gt='./', facedetect=None, gazedetect='OpenFace'):
        config = {'model': {}}
        config['model']['name'] = "gazelle_dinov2_vitl14_inout"
        self.device = 'cpu'
        config = {'device': {'hardware': self.device}}

        self.test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # load EC detector
        if facedetect is None:
            self.cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)

        model_weight = model_ec
        
        model = get_ec_model(config, model_weight)
        model_dict = model.state_dict()
        snapshot = torch.load(model_weight, map_location=torch.device(self.device))
        model_dict.update(snapshot)
        model.load_state_dict(model_dict)
        
        self.model_ec = model
        self.model_ec.to(self.device)
        self.model_ec.eval()

        # load IFT/OFT detector
        model, transform = get_gazelle_model(config)
        model.load_gazelle_state_dict(torch.load("gazelle_dinov2_vitl14_inout.pt", weights_only=True))
        self.model_gt = model
        self.gt_transform = transform

        self.model_gt.to(self.device)
        self.model_gt.eval()

    def conditional_inference(self, input_data, threshold=0.7):
        frame = Image.open(input_data).convert("RGB")

        # resize image
        frame = frame.resize((448, 448))

        with torch.no_grad():
            ec_prob, bboxes = self.ec_infer(frame)

            outcome = {}
            bbox_lst = []
            # decide stage2 inference
            for prob, bbox in zip(ec_prob, bboxes):
                if float(prob) < threshold:
                    bbox_lst.append(bbox)
                    
                else:
                    outcome[bbox] = 1

            if len(bbox_lst) > 0:
                preds = self.gt_infer(frame, bbox_lst, self.gt_transform)

                for i in range(len(bbox_lst)):  # per head
                    inout = preds['inout'][0][i]
                    if inout < 0.5:  # out of frame
                        outcome[bbox_lst[i]] = 0
                    else:  # in frame
                        outcome[bbox_lst[i]] = preds['heatmap'][0][i].detach()
                    
            return outcome
        
    def ec_infer(self, frame):
        bbox = []
        scores = []
        dets = self.cnn_face_detector(np.array(frame), 1)

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
            bbox.append((l, t, r, b))

        for b in bbox:
            face = frame.crop((b))
            img = self.test_transforms(face)
            img.unsqueeze_(0)

            # forward pass
            output = self.model_ec(img.to(self.device))
            score = F.sigmoid(output).item()
            print(score, b)

            scores.append(score)

            coloridx = min(int(round(score * 10)), 9)
            draw = ImageDraw.Draw(frame)

            drawrect(draw, [(b[0], b[1]), (b[2], b[3])], width=5)
            draw.text((b[0], b[3]), str(round(score, 2)), fill=(255, 255, 255, 128))

            frame.show()
            #saved_path = join("processed", "EC_" + img_source.split('/')[-1])
            #frame.save(saved_path)

        return scores, bbox
    
    def gt_infer(self, image, bboxes, transform):
        
        input = {
            "images": transform(image).unsqueeze(dim=0).to(self.device),    # tensor of shape [1, 3, 448, 448]
            "bboxes": bboxes
        }
        
        output = self.model_gt(input)
        
        predicted_heatmap = output["heatmap"][0][0]        # access prediction for first person in first image. Tensor of size [64, 64]
        predicted_inout = output["inout"][0][0]            # in/out of frame score (1 = in frame) (output["inout"] will be None  for non-inout models)
        return output


if __name__ == "__main__":
    
    demo = DemoSys()

    img_path = "data/0028_2m_-15P_10V_5H.jpg"

    demo.conditional_inference(img_path)
    