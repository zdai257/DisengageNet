import os
os.environ["XFORMERS_DISABLE_MEMORY_EFFICIENT_ATTENTION"] = "1"
from os.path import join
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
from transformers import YolosFeatureExtractor, YolosForObjectDetection, pipeline
import matplotlib.pyplot as plt
from network.network_builder import get_gazelle_model
#from network.network_builder_update import get_gt360_model
from network.network_builder_update2 import get_gt360_model
from network.ec_network_builder import get_ec_model
from network.utils import visualize_heatmap, visualize_heatmap2, visualize_heatmap3


class DemoSys():
    def __init__(self, model_gt='gazelle_dinov2_vitl14_inout.pt'):
        self.saved_path = "GT360output.png"
        self.savefigs = 1

        config = {'model': {}}
        config['model']['name'] = "gazelle_dinov2_vitl14_inout"

        self.device = 'cuda' if torch.cuda.is_available() else "cpu"
        config['hardware'] = {'device': self.device}

        self.test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        # Load dedicated head detection model
        model_name = "nickmuchi/yolos-small-finetuned-masks"
        self.head_feature_extractor = YolosFeatureExtractor.from_pretrained(model_name)
        self.head_model = YolosForObjectDetection.from_pretrained(model_name)
        self.head_confidence_threshold = 0.85

        model_weight = 'model/model_weights.pkl'
        model = get_ec_model(config, model_weight)
        model_dict = model.state_dict()
        snapshot = torch.load(model_weight, map_location=torch.device(self.device))
        model_dict.update(snapshot)
        model.load_state_dict(model_dict)
        
        self.model_ec = model
        self.model_ec.to(self.device)
        self.model_ec.eval()

        # load IFT/OFT detector
        #model, transform = get_gazelle_model(config)
        model, transform = get_gt360_model(config)
        # load a pre-trained model
        #model.load_state_dict(torch.load(model_gt, map_location=self.device, weights_only=False)['model_state_dict'])
        model.load_gazelle_state_dict(torch.load(model_gt, weights_only=True, map_location=torch.device(self.device)))
        self.model_gt = model
        self.gt_transform = transform

        self.model_gt.to(self.device)
        self.model_gt.eval()

    def conditional_inference(self, input_data, threshold=0.85, outdir='processed', imgname=None):
        fig_saved_token = False

        frame = Image.open(input_data).convert("RGB")

        # resize image
        frame.thumbnail((896, 896))  # (896, 896)
        print(frame.width, frame.height)
        w, h = frame.width, frame.height

        frame0 = frame.convert("RGBA")

        final_overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))

        # Create a series of transparent overlay for drawing
        overlays = []
        viz_overlays = []

        gaze_targets = []

        with torch.no_grad():
            
            ec_prob, bboxes = self.ec_infer(frame, bbox_scalar=0.2)

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
                bbox_norm_lst = [(b[0]/w, b[1]/h, b[2]/w, b[3]/h) for b in bbox_lst]

                preds = self.gt_infer(frame, bbox_norm_lst, self.gt_transform)

                for i, b in enumerate(bbox_lst):  # per face
                    ### Carefully remove this if visual multi faces ###
                    #if i == 0:
                    #    continue

                    inout = preds['inout'][0][i]
                    if inout < 0.5:  # out of frame (OFT)
                        heatmaps[b] = 0

                        print("OFT with prob = ", inout)
                        overlays.append(Image.new("RGBA", frame0.size, (0, 0, 0, 0)))
                        draw = ImageDraw.Draw(overlays[-1])
                        # Draw a semi-transparent red rectangle on the overlay
                        draw.rectangle([(b[0], b[1]), (b[2], b[3])], fill=(255, 0, 0, 70), outline=(0, 255, 0), width=7)

                    else:  # in frame (IFT)
                        heatmap = preds['heatmap'][0][i].detach()

                        heatmaps[b] = heatmap

                        print("IFT with prob = ", inout)

                        # convert heatmap to argmax (x,y) coordinate points
                        bbox_norm = (b[0]/w, b[1]/h, b[2]/w, b[3]/h)

                        argmax = heatmap.flatten().argmax().item()
                        pred_y, pred_x = np.unravel_index(argmax, (64, 64))
                        pred_x = pred_x / 64.
                        pred_y = pred_y / 64.
                        x, y = float(pred_x), float(pred_y)

                        gaze_targets.append((x, y))

                        viz = visualize_heatmap3(frame, heatmap, bbox=bbox_norm, xy=(x * w, y * h),
                                                 dilation_kernel=5, blur_radius=1.3, transparent_bg=True)

                        viz_overlays.append(viz)
                        #plt.imshow(viz)
                        #plt.show()

                        '''
                        if self.savefigs and not fig_saved_token:
                            if imgname is None:
                                viz.convert("RGB").save(join(outdir, "ift_" + self.saved_path))
                            else:
                                viz.convert("RGB").save(join(outdir, imgname + '.png'))
                            fig_saved_token = True
                        '''
                        #plt.close()
                    #break

                for viz_overlay in viz_overlays:
                    final_overlay = Image.alpha_composite(final_overlay, viz_overlay)

        for b in ecs.keys():
            overlays.append(Image.new("RGBA", frame0.size, (0, 0, 0, 0)))
            draw = ImageDraw.Draw(overlays[-1])
            # Draw a semi-transparent green rectangle on the overlay
            draw.rectangle([(b[0], b[1]), (b[2], b[3])], fill=(0, 255, 0, 70), outline=(0, 255, 0), width=7)

        for overlay in overlays:
            # iteratively add overlays
            final_overlay = Image.alpha_composite(final_overlay, overlay)

        frame2show = Image.alpha_composite(frame.convert('RGBA'), final_overlay)
        #frame2show.show()

        if self.savefigs and not fig_saved_token:
            if imgname is None:
                frame2show.convert("RGB").save(join(outdir, "gt360_" + self.saved_path))
            else:
                frame2show.convert("RGB").save(join(outdir, imgname + '.png'))

        return ecs, heatmaps, gaze_targets
        
    def ec_infer(self, frame, bbox_scalar=0.2):
        bboxes = []
        scores = []
        
        inputs = self.head_feature_extractor(images=frame, return_tensors="pt")
    
        with torch.no_grad():
            outputs = self.head_model(**inputs)

        # Convert outputs to COCO API format
        target_sizes = torch.tensor([frame.size[::-1]])
        results = self.head_feature_extractor.post_process_object_detection(
            outputs, 
            threshold=self.head_confidence_threshold, 
            target_sizes=target_sizes
        )[0]

        for score, bbox in zip(results["scores"], results["boxes"]):
            if score.item() < self.head_confidence_threshold:
                continue
            l, t, r, b = [float(x) for x in bbox.tolist()]
            # expand a bit
            l -= (r - l) * bbox_scalar
            r += (r - l) * bbox_scalar
            t -= (b - t) * bbox_scalar
            b += (b - t) * bbox_scalar
            bboxes.append((l, t, r, b))
        
        for b in bboxes:
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
        return scores, bboxes
    
    def gt_infer(self, image, bboxes, transform):
        
        input = {
            "images": transform(image).unsqueeze(dim=0).to(self.device),    # tensor of shape [1, 3, 448, 448]
            "bboxes": [bboxes]
        }
        
        output = self.model_gt(input)

        # convert output to Visualizalbe heatmap

        return output


if __name__ == "__main__":
    
    demo = DemoSys(model_gt="GT360_vat.pt")

    #img_path = "data/WALIexample0.png"
    #img_path = "data/WALIHRIexample1.png"
    #img_path = "data/WALIHRIexample2.png"
    #img_path = "data/WALIHRIexample3.png"
    #img_path = "data/joye.jpg"
    #img_path = "data/0028_2m_-15P_10V_5H.jpg"
    #img_path = "data/0028_2m_30P_0V_0H.jpg"
    #img_path = "data/0018_2m_15P_0V_0H.jpg"
    #img_path = "data/example-16_A_FT_M.png"
    #img_path = "data/example-2_A_FT_M.png"
    #img_path = "data/example-8_A_FT_M.png"
    #img_path = "data/0041.jpg"
    #img_path = "data/0000867.jpg"  # interesting
    #img_path = "data/0000051.jpg"
    #img_path = "data/0000000.jpg"
    #img_path = "data/00004218.jpg"
    img_path = "data/00000033.jpg"

    ec_results, heatmap_results, gaze_targets = demo.conditional_inference(img_path)

    # naive depth
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")
    
    frame = Image.open(img_path).convert("RGB")
    height, width = frame.height, frame.width
    x_rel, y_rel = gaze_targets[0][0], gaze_targets[0][1]
    x_abs = min(max(int(x_rel * width), 0), width - 1)
    y_abs = min(max(int(y_rel * height), 0), height - 1)
    depth = pipe(frame)['depth']

    depth_value = np.array(depth)[y_abs, x_abs]
    print(f"Depth at ({x_rel}, {y_rel}) -> pixel ({x_abs}, {y_abs}): {depth_value}")

    print(ec_results.keys(), heatmap_results.keys())

