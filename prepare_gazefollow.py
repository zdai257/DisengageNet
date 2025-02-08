import os
from os.path import join
import pandas as pd
from PIL import Image
import numpy as np
import json
import io
import dlib
#from datasets import load_dataset
from huggingface_hub import hf_hub_download


REPO_ID = "ShijianDeng/gazefollow"
TRAINFILE = "gazefollow/train/train.parquet"
TESTFILE = "gazefollow/test/test.parquet"

REPO_ID0 = "vikhyatk/gazefollow"
TESTFILE0 = "data/test-00000-of-00001.parquet"

CNN_FACE_MODEL = 'model/mmod_human_face_detector.dat'  # from http://dlib.net/files/mmod_human_face_detector.dat.bz2
cnn_face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_MODEL)


def apply_face_detect(img, eye_centre):

    dets = cnn_face_detector(np.array(img), 1)
    if len(dets) == 0:
        return None

    l, r, t, b = None, None, None, None
    for d in dets:
        l = d.rect.left()
        r = d.rect.right()
        t = d.rect.top()
        b = d.rect.bottom()

        if r > eye_centre[0] > l and t > eye_centre[1] > b:
            # expand a bit
            l -= (r - l) * 0.2
            r += (r - l) * 0.2
            t -= (b - t) * 0.2
            b += (b - t) * 0.2
            break

    if l is not None:
        bbox = [l, t, r, b]
        return bbox
    else:
        return None


def get_coordinates(text, mode='eye'):
    start_index = text.find('(x, y): (')
    if mode == 'eye':
        end_index = text.find('). Provide')
    elif mode == 'target':
        end_index = text.find(').')
    else:
        raise TypeError("Mode unsupported")

    # Extract the substring between the parentheses
    tuple_str = text[start_index + 9:end_index-1]

    x_str, y_str = tuple_str.split(',')

    x = float(x_str.strip())
    y = float(y_str.strip())
    return x, y


def main(root_path):
    df_test = pd.read_parquet(
        hf_hub_download(repo_id=REPO_ID, filename=TESTFILE, repo_type="dataset")
    )

    df_train = pd.read_parquet(
        hf_hub_download(repo_id=REPO_ID, filename=TRAINFILE, repo_type="dataset")
    )

    print(df_train.head())

    # preprocess TRAIN split
    frames = []
    id = 0

    pre_image = None
    pre_eye_cent = None

    for idx, row in df_train.iterrows():
        image_raw = row['images'][0]['bytes']
        image = Image.open(io.BytesIO(image_raw))
        w, h = image.size

        labels = row['texts']
        is_gazefollow = labels[0]['source']
        if is_gazefollow != "gazefollow":
            continue
        eye_x, eye_y = get_coordinates(labels[0]['user'])
        gazex, gazey = get_coordinates(labels[0]['assistant'])

        eye_cent = (eye_x * float(w), eye_y * float(h))

        bbox = apply_face_detect(image, eye_cent)

        if bbox is None:
            continue
        elif eye_cent == pre_eye_cent and image == pre_image:
            continue
        else:
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
            gazex_pixel = gazex * float(w)
            gazey_pixel = gazey * float(h)

            if xmin > xmax:
                temp = xmin
                xmin = xmax
                xmax = temp
            if ymin > ymax:
                temp = ymin
                ymin = ymax
                ymax = temp

            # move in out of frame bbox annotations
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmax, w)
            ymax = min(ymax, h)

            img_path = "train/" + str(int(id)).zfill(7) + ".jpg"

            frames.append({
                'path': img_path,
                'bbox': [xmin, ymin, xmax, ymax],
                'bbox_norm': [xmin / float(w), ymin / float(h), xmax / float(w), ymax / float(h)],
                'gazex': [gazex_pixel],
                'gazey': [gazey_pixel],
                'gazex_norm': [gazex],
                'gazey_norm': [gazey],
                'height': h,
                'width': w
            })
            id += 1

            image.save(join(root_path, img_path), "JPEG")

            pre_image = image
            pre_eye_cent = eye_cent

    print("Complete preprocessing {} train samples".format(id))
    with open(os.path.join(root_path, "train_preprocessed.json"), "w") as out_file:
        json.dump(frames, out_file)

    # preprocess TEST split
    # columns: ['images', 'texts']
    frames = []
    # Continue counting the samples!
    #id = 0

    pre_image = None
    pre_eye_cent = None

    for idx, row in df_test.iterrows():
        image_raw = row['images'][0]['bytes']
        image = Image.open(io.BytesIO(image_raw))
        w, h = image.size

        labels = row['texts']
        is_gazefollow = labels[0]['source']
        if is_gazefollow != "gazefollow":
            continue
        eye_x, eye_y = get_coordinates(labels[0]['user'])
        gazex, gazey = get_coordinates(labels[0]['assistant'])

        eye_cent = (eye_x * float(w), eye_y * float(h))

        bbox = apply_face_detect(image, eye_cent)

        if bbox is None:
            continue
        elif eye_cent == pre_eye_cent and image == pre_image:
            continue
        else:
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
            gazex_pixel = gazex * float(w)
            gazey_pixel = gazey * float(h)

            if xmin > xmax:
                temp = xmin
                xmin = xmax
                xmax = temp
            if ymin > ymax:
                temp = ymin
                ymin = ymax
                ymax = temp

            # move in out of frame bbox annotations
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmax, w)
            ymax = min(ymax, h)

            img_path = "test/" + str(int(id)).zfill(7) + ".jpg"

            frames.append({
                'path': img_path,
                'bbox': [xmin, ymin, xmax, ymax],
                'bbox_norm': [xmin / float(w), ymin / float(h), xmax / float(w), ymax / float(h)],
                'gazex': [gazex_pixel],
                'gazey': [gazey_pixel],
                'gazex_norm': [gazex],
                'gazey_norm': [gazey],
                'height': h,
                'width': w
            })
            id += 1

            image.save(join(root_path, img_path), "JPEG")

            pre_image = image
            pre_eye_cent = eye_cent

    print("Complete preprocessing {} TOTAL samples".format(id))
    with open(os.path.join(root_path, "test_preprocessed.json"), "w") as out_file:
        json.dump(frames, out_file)


if __name__ == "__main__":
    root_path = "GazeFollow"
    main(root_path)
