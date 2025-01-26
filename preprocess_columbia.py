import os
from os.path import join
import numpy as np
import math
import dlib
from PIL import Image
import cv2
import json

CNN_FACE_MODEL = 'model/mmod_human_face_detector.dat'  # from http://dlib.net/files/mmod_human_face_detector.dat.bz2
cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)


def apply_face_crop(img):
    print("Start face cropping")
    # for Columbia, presume one face per image
    dets = cnn_face_detector(np.array(img), 1)
    if len(dets) == 0:
        return img

    bbox = []
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
        break

    b = bbox[0]
    face = img.crop((b))
    print(face.size, type(face))
    return face


def load_columbia(dataset_dir='./', new_size=(864, 576), train_size=55, num_subjects=56):

    target_annotation = join(dataset_dir, "labels.json")
    target_path = join(dataset_dir, "Columbia", "cropped")
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        os.makedirs(join(target_path, 'train'))
        os.makedirs(join(target_path, 'test'))

    # self.dataset_path = join(self.dataset_dir, "ColumbiaGaze", "columbia_gaze_data_set", f"Columbia Gaze Data Set")
    dataset_path = join(dataset_dir, "Columbia", f"Columbia Gaze Data Set")

    subject_lst = list(range(train_size))
    subject_id_str = [str(x).zfill(4) for x in subject_lst]

    # leave-one-out for test split
    data = {}  # Dict of (image_name, gaze_vector, ec)
    data_id = 0
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg') and not file.startswith('.'):  # Check for .jpg files
                full_path = join(root, file)
                frame = Image.open(full_path).convert("RGB")

                frame = frame.resize(new_size)

                face_frame = apply_face_crop(frame)

                filename, _ = os.path.splitext(file)  # Remove extension
                labels = filename.split('_')  # Split by '_'

                subject_id = labels[0]
                #print(subject_id, subject_id_str)
                # save cropped face to file
                if subject_id in subject_id_str:
                    face_frame.save(join(target_path, 'train', file), format="JPEG")
                else:
                    face_frame.save(join(target_path, 'test', file), format="JPEG")

                # e.g. "0003_2m_-30P_10V_-10H.jpg": five head Poses, three Vertical gaze angles, seven Horizontal gaze angles
                # Headpose appears to be independent from V / H, if V=0 & H=0: EC = True
                headpose = int(labels[2][:-1])
                vertical = int(labels[3][:-1])
                horizontal = int(labels[4][:-1])
                ec = 0
                gaze_vector = (0., 0.)
                v_unit = 2 * math.tan(math.radians(10))
                # h_unit = 2 * math.tan(math.radians(5))
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
                        gaze_vector = (gaze_vector[0 ] /magnitude, gaze_vector[1 ] /magnitude)
                elif vertical == -10:
                    if horizontal == 0:
                        gaze_vector = (0., -1.)
                    else:
                        # gaze_vector = ( 2*tan(H), -2*tan(10) )
                        gaze_vector = (2 * math.tan(math.radians(horizontal)), -v_unit)
                        magnitude = np.linalg.norm(np.array(gaze_vector))
                        gaze_vector = (gaze_vector[0] / magnitude, gaze_vector[1] / magnitude)

                data[data_id] = (full_path, gaze_vector, ec)
                data_id += 1

    with open(target_annotation, "w") as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    load_columbia(dataset_dir='./')
