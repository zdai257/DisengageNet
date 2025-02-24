import os
os.environ["XFORMERS_DISABLE_MEMORY_EFFICIENT_ATTENTION"] = "1"
from os.path import join
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import dlib
import matplotlib.pyplot as plt
from network.network_builder import get_gazelle_model
#from network.network_builder_update import get_gazelle_model
from network.ec_network_builder import get_ec_model
from network.utils import visualize_heatmap, visualize_heatmap2


CNN_FACE_MODEL = 'model/mmod_human_face_detector.dat'  # from http://dlib.net/files/mmod_human_face_detector.dat.bz2
MODEL_WEIGHTS = 'model/model_weights.pkl'

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PREDICTOR_PATH = os.path.join(script_dir, "model", "shape_predictor_68_face_landmarks.dat")
if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()


class DemoSys():
    def __init__(self, model_ec=MODEL_WEIGHTS, model_gt="gazelle_dinov2_vitl14_inout.pt", facedetect=None, gazedetect='OpenFace'):
        self.saved_path = "GT360output.png"
        self.savefigs = 1

        config = {'model': {}}
        config['model']['name'] = "gazelle_dinov2_vitl14_inout"

        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        config['hardware'] = {'device': self.device}

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
        model.load_gazelle_state_dict(torch.load(model_gt, weights_only=True))
        self.model_gt = model
        self.gt_transform = transform

        self.model_gt.to(self.device)
        self.model_gt.eval()

    def conditional_inference(self, input_data, threshold=0.8):
        frame = Image.open(input_data).convert("RGB")

        # resize image
        frame.thumbnail((896, 896))  # (896, 896)
        print(frame.width, frame.height)

        frame0 = frame.convert("RGBA")
        # Create a transparent overlay for drawing
        overlay = Image.new("RGBA", frame0.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        with torch.no_grad():
            
            ec_prob, bboxes = self.ec_infer(frame)

            ecs = {}
            heatmaps = {}
            bbox_lst = []
            # decide stage2 inference
            for prob, bbox in zip(ec_prob, bboxes):
                if float(prob) < threshold:
                    bbox_lst.append(bbox)
                    
                else:
                    ecs[bbox] = 1

            # only if some faces are non-EC
            if len(bbox_lst) > 0:
                # access prediction for first person in first image. Tensor of size [64, 64]
                # in/out of frame score (1 = in frame) (output["inout"] will be None  for non-inout models)
                preds = self.gt_infer(frame, bbox_lst, self.gt_transform)

                for i, b in enumerate(bbox_lst):  # per face
                    inout = preds['inout'][0][i]
                    if inout < 0.5:  # out of frame (OFT)
                        heatmaps[b] = 0

                        # Draw a semi-transparent red rectangle on the overlay
                        draw.rectangle([(b[0], b[1]), (b[2], b[3])], fill=(255, 0, 0, 80), outline=(0, 255, 0), width=7)

                    else:  # in frame (IFT)
                        heatmap = preds['heatmap'][0][i].detach()

                        heatmaps[b] = preds['heatmap'][0][i].detach()

                        # convert heatmap to argmax (x,y) coordinate points
                        w, h = frame.size
                        bbox_norm = (b[0]/w, b[1]/h, b[2]/w, b[3]/h)

                        argmax = heatmap.flatten().argmax().item()
                        pred_y, pred_x = np.unravel_index(argmax, (64, 64))
                        pred_x = pred_x / 64.
                        pred_y = pred_y / 64.
                        x, y = float(pred_x), float(pred_y)

                        viz = visualize_heatmap2(frame, heatmap, bbox=bbox_norm, xy=(x * w, y * h),
                                                 dilation_kernel=5, blur_radius=1.3)
                        plt.imshow(viz)
                        plt.show()
                        if self.savefigs:
                            viz.convert("RGB").save(join("processed", "ec_" + self.saved_path))
                        plt.close()
                        break

        for b in ecs.keys():

            # Draw a semi-transparent green rectangle on the overlay
            draw.rectangle([(b[0], b[1]), (b[2], b[3])], fill=(0, 255, 0, 80), outline=(0, 255, 0), width=7)

        frame2show = Image.alpha_composite(frame.convert('RGBA'), overlay)
        frame2show.show()
        if self.savefigs:
            frame2show.convert("RGB").save(join("processed", "ift_" + self.saved_path))
        return ecs, heatmaps
        
    def ec_infer(self, frame):
        bbox = []
        scores = []
        dets = self.cnn_face_detector(np.array(frame), 1)
        # fail to detect face if frame-aspect-ratio distorted
        #print(dets)

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
            print(f"face prob: {score}, bbox value: {b}")

            scores.append(score)

            #drawrect(draw, [(b[0], b[1]), (b[2], b[3])], width=5)

            #draw.text((b[0], b[3]), str(round(score, 2)), fill=(255, 255, 255, 128))
            #break

        #frame0.show()

        # saved_path = join("processed", "EC_" + img_source.split('/')[-1])
        # frame.save(saved_path)
        return scores, bbox
    
    def gt_infer(self, image, bboxes, transform):
        
        input = {
            "images": transform(image).unsqueeze(dim=0).to(self.device),    # tensor of shape [1, 3, 448, 448]
            "bboxes": [bboxes]
        }
        
        output = self.model_gt(input)

        # convert output to Visualizalbe heatmap

        return output


if __name__ == "__main__":
    
    demo = DemoSys()

    #img_path = "data/WALIexample0.png"
    #img_path = "data/joye.jpg"
    #img_path = "data/0028_2m_-15P_10V_5H.jpg"
    img_path = "data/0028_2m_30P_0V_0H.jpg"
    #img_path = "data/0018_2m_15P_0V_0H.jpg"
    #img_path = "data/GF_image.jpg"

    ec_results, heatmap_results = demo.conditional_inference(img_path)

    print(ec_results.keys(), heatmap_results.keys())

