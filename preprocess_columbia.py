import os
from os.path import join
import numpy as np
import pandas as pd
import json
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor


# preprocesses and compiles the annotations into a JSON file
def preprocess_dataset(input_dir, output_file, transform):
    data = []
    for label in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, label)
        if os.path.isdir(class_dir):
            for image_file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_file)
                if image_file.endswith(('.jpg', '.png')):
                    data.append({'image_path': image_path, 'label': label})

    df = pd.DataFrame(data)
    df.to_json(output_file, orient='records', lines=True)


if __name__ == "__main__":
    data_dir = "./output_DAiSEE/DataSet"

    for split in os.listdir(data_dir):
        if split == 'Test':
            for id in os.listdir(join(data_dir, split)):
                for root, dirs, files in os.walk(join(data_dir, split, id)):
                    print(root, dirs)
                    print(files)
                    for filename in files:
                        if filename.endswith('.csv'):
                            df = pd.read_csv(join(root, filename), header=0, sep=',').values
                            seq_length = df.shape[0]
                            print(seq_length, df.shape)


