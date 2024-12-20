import os
from os.path import join
import numpy as np
import pandas as pd

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
                        