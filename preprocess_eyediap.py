import os
from os.path import join
import numpy as np
import math
from PIL import Image
import cv2
import json


def extract_frames(video_path, xy, name, num_frames=50, output_folder='EYEDIAP/sampled'):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print(total_frames)
    
    if total_frames < num_frames:
        print(f"Warning: {video_path} has only {total_frames} frames, extracting all.")
        frame_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frame_count = 0
    label_dict= {}
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Jump to frame index
        ret, frame = cap.read()  # Read frame

        if ret:
            frame_filename = os.path.join(output_folder, f"{name}_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            label_dict[frame_filename] = (float(xy[frame_count][0]), float(xy[frame_count][1]))

            frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path} to {output_folder}")
    return label_dict
    

# Function to parse 2D coor data
def parse_coors(file_path):
    
    ball_track_vals = np.loadtxt(file_path, skiprows=1, delimiter=';')[:,-5:-3]
    return ball_track_vals


def load_eyediap(dataset_dir):

    label_json = {}

    for dirname in os.listdir(dataset_dir):

        if dirname.split('_')[2] == 'FT':

            labelfile = join(dataset_dir, dirname, "ball_tracking.txt")

            xy = parse_coors(labelfile)
            #print(xy.shape)

            vidfile = join(dataset_dir, dirname, "rgb_hd.mov")

            # sample frames and save by name
            labels = extract_frames(vidfile, xy, name=dirname)
            
            if labels is not None:
                label_json.update(labels)
            else:
                continue
            
    with open('EYEDIAP/sampled.json', "w") as json_file:
        json.dump(label_json, json_file, indent=4)


if __name__ == "__main__":

    EYEDIAP_path = 'EYEDIAP/EYEDIAP/Data'
    
    load_eyediap(dataset_dir=EYEDIAP_path)
