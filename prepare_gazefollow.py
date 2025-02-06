import os
import pandas as pd
from PIL import Image
import io
#from datasets import load_dataset
from huggingface_hub import hf_hub_download


REPO_ID = "ShijianDeng/gazefollow"
TRAINFILE = "gazefollow/train/train.parquet"
TESTFILE = "gazefollow/test/test.parquet"

df_test = pd.read_parquet(
    hf_hub_download(repo_id=REPO_ID, filename=TESTFILE, repo_type="dataset")
)
df_train = pd.read_parquet(
    hf_hub_download(repo_id=REPO_ID, filename=TRAINFILE, repo_type="dataset")
)

idx = 0

image_data = df_train['images'].iloc[idx]  # Get the first image row

image_raw = image_data[0]['bytes']

labels = df_train['texts'].iloc[idx]


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


eye_x, eye_y = get_coordinates(labels[0]['user'])

tar_x, tar_y = get_coordinates(labels[0]['assistant'])

#print(eye_x, eye_y)
#print(tar_x, tar_y)

image = Image.open(io.BytesIO(image_raw))

#image.save('data/GF_image.jpg')
#image.show()  # Optionally display the image

# TODO: build GazeFollow dataloader


