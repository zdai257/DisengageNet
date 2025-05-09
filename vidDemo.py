import cv2
import os
from os.path import join
import numpy as np

from Demo_sys import DemoSys


demo = DemoSys(model_gt="vatMoE.pt")

# Step 1: Extract a 15s clip (change start time as needed)
def extract_clip(input_videofile, output_clipfile, start_time=0, duration=10):
    start_time_sec = start_time
    duration_sec = duration

    # Open the input video
    cap = cv2.VideoCapture(input_videofile)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    start_frame = start_time_sec * fps  # Calculate start frame
    end_frame = (start_time_sec + duration_sec) * fps  # Calculate end frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_clipfile, fourcc, fps, (frame_width, frame_height))

    # Extract and save the desired frames
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_id > end_frame:
            break  # Stop when end_frame is reached or video ends

        if frame_id >= start_frame:
            out.write(frame)  # Write frames within the desired range

        frame_id += 1

    # Release resources
    cap.release()
    out.release()
    print("Extracted clip saved as:", output_clipfile)


# Step 2: Extract frames from the clipped video
def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"{frame_id:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_id += 1
    cap.release()


# Step 3: Process each frame with the deep learning model
def process_frames(input_folder, output_folder=None):
    os.makedirs(output_folder, exist_ok=True)
    i = 0
    for frame_file in sorted(os.listdir(input_folder)):
        #i += 1
        #if i<0:
            # break
            #continue
        frame_path = os.path.join(input_folder, frame_file)

        imgname = frame_file.split('.')[0]

        _, _ = demo.conditional_inference(frame_path, threshold=1.001, outdir=output_folder, imgname=imgname)


# Step 4: Convert processed frames back to video
def frames_to_video(frames_folder, output_video, fps=20):
    frame_files = sorted(os.listdir(frames_folder))
    if not frame_files:
        raise ValueError("No frames found in the directory")

    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[-1]))
    h, w, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()


if __name__ == "__main__":

    input_video = "trump_demo.mp4"  #"TVSeries_dataset_example.mp4"  #"master1.mkv"
    output_clip = "clip_" + input_video  #"youtube1_clip.mp4"  #"master1_clip.mp4"

    output_video = "moe_trump1.mp4"

    # Run the pipeline
    # 1. extract a section
    extract_clip(input_video, output_clip, start_time=215, duration=10)
    # 2. extract frames to a folder
    outdir = 'MoEFrames1'
    viddir = 'MoEdemo1'
    extract_frames(output_clip, output_folder=outdir)
    # 3. get inferred frames to a new clip

    process_frames(input_folder=outdir, output_folder=viddir)

    frames_to_video(frames_folder=join(viddir), output_video=output_video)

    print("Demo video saved as:", output_video)
