import os
import sys
import argparse
import csv
import cv2
import numpy as np
from PIL import Image

from Demo_sys import DemoSys


def pil_from_cv(frame_bgr):
	# cv2 uses BGR, PIL expects RGB
	return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def cv_from_pil(img_pil):
	# PIL is RGB, cv2 expects BGR
	return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def annotate_image_with_demo(demo, image_pil, threshold, outdir=None, imgname=None):
	# Replicate DemoSys.conditional_inference but accept an in-memory PIL image
	frame = image_pil.convert("RGB")
	frame.thumbnail((896, 896))

	# Save to a temp path only for DemoSys flow reuse if needed; here we inline logic by calling demo methods
	from PIL import ImageDraw
	import numpy as np

	w, h = frame.width, frame.height
	frame0 = frame.convert("RGBA")
	final_overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
	overlays = []
	viz_overlays = []

	import torch
	with torch.no_grad():
		ec_prob, bboxes = demo.ec_infer(frame, bbox_scalar=0.2)

		ecs = {}
		heatmaps = {}
		bbox_lst = []
		for prob, bbox in zip(ec_prob, bboxes):
			if float(prob) < threshold:
				bbox_lst.append(bbox)
			else:
				ecs[bbox] = 1

		if len(bbox_lst) > 0:
			bbox_norm_lst = [(b[0]/w, b[1]/h, b[2]/w, b[3]/h) for b in bbox_lst]
			preds = demo.gt_infer(frame, bbox_norm_lst, demo.gt_transform)

			from network.utils import visualize_heatmap3
			for i, b in enumerate(bbox_lst):
				inout = preds['inout'][0][i]
				if inout < 0.5:
					overlays.append(Image.new("RGBA", frame0.size, (0, 0, 0, 0)))
					draw = ImageDraw.Draw(overlays[-1])
					draw.rectangle([(b[0], b[1]), (b[2], b[3])], fill=(255, 0, 0, 70), outline=(0, 255, 0), width=7)
				else:
					heatmap = preds['heatmap'][0][i].detach()
					bbox_norm = (b[0]/w, b[1]/h, b[2]/w, b[3]/h)
					argmax = heatmap.flatten().argmax().item()
					pred_y, pred_x = np.unravel_index(argmax, (64, 64))
					pred_x = pred_x / 64.
					pred_y = pred_y / 64.
					viz = visualize_heatmap3(frame, heatmap, bbox=bbox_norm, xy=(pred_x * w, pred_y * h),
										   dilation_kernel=5, blur_radius=1.3, transparent_bg=True)
					viz_overlays.append(viz)

	for viz_overlay in viz_overlays:
		final_overlay = Image.alpha_composite(final_overlay, viz_overlay)

	from PIL import ImageDraw
	for b in ecs.keys():
		overlays.append(Image.new("RGBA", frame0.size, (0, 0, 0, 0)))
		draw = ImageDraw.Draw(overlays[-1])
		draw.rectangle([(b[0], b[1]), (b[2], b[3])], fill=(0, 255, 0, 70), outline=(0, 255, 0), width=7)

	for overlay in overlays:
		final_overlay = Image.alpha_composite(final_overlay, overlay)

	annotated = Image.alpha_composite(frame.convert('RGBA'), final_overlay).convert('RGB')

	# Build records for CSV: one entry per detected face
	records = []
	# Map bbox -> ec_prob for quick lookup
	bbox_to_prob = {bbox: prob for prob, bbox in zip(ec_prob, bboxes)}

	# Faces that were EC-only
	for b in ecs.keys():
		records.append({
			"status": "EC",
			"bbox_left": float(b[0]),
			"bbox_top": float(b[1]),
			"bbox_right": float(b[2]),
			"bbox_bottom": float(b[3]),
			"ec_prob": float(bbox_to_prob.get(b, float('nan'))),
			"inout_prob": float('nan'),
			"gaze_x": float('nan'),
			"gaze_y": float('nan')
		})

	# Faces that proceeded to GT stage: OFT or IFT
	if len(bbox_lst) > 0:
		preds = preds  # already computed above
		for i, b in enumerate(bbox_lst):
			inout = float(preds['inout'][0][i])
			if inout < 0.5:
				status = "OFT"
				gx, gy = float('nan'), float('nan')
			else:
				status = "IFT"
				heatmap = preds['heatmap'][0][i].detach()
				argmax = heatmap.flatten().argmax().item()
				pred_y, pred_x = np.unravel_index(argmax, (64, 64))
				gx = float((pred_x / 64.) * w)
				gy = float((pred_y / 64.) * h)
			records.append({
				"status": status,
				"bbox_left": float(b[0]),
				"bbox_top": float(b[1]),
				"bbox_right": float(b[2]),
				"bbox_bottom": float(b[3]),
				"ec_prob": float(bbox_to_prob.get(b, float('nan'))),
				"inout_prob": inout,
				"gaze_x": gx,
				"gaze_y": gy
			})

	return annotated, records


def process_image(demo, input_path, output_path, threshold, csv_output_path=None):
	img = Image.open(input_path).convert('RGB')
	annotated, records = annotate_image_with_demo(demo, img, threshold)
	os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
	annotated.save(output_path)

	# Write CSV alongside output image if requested
	csv_path = csv_output_path or os.path.splitext(output_path)[0] + '.csv'
	fieldnames = ["timestamp_sec", "track_id", "status", "bbox_left", "bbox_top", "bbox_right", "bbox_bottom", "ec_prob", "inout_prob", "gaze_x", "gaze_y"]
	with open(csv_path, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for idx, r in enumerate(records):
			row = {"timestamp_sec": 0.0, "track_id": idx}
			row.update(r)
			writer.writerow(row)


def process_video(demo, input_path, output_path, threshold, csv_output_path=None, max_seconds=None):
	# Try multiple backends to improve compatibility (e.g., .mov requires FFMPEG)
	cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
	if not cap.isOpened():
		cap.release()
		cap = cv2.VideoCapture(input_path, cv2.CAP_ANY)
	if not cap.isOpened():
		cap.release()
		cap = cv2.VideoCapture(input_path, cv2.CAP_GSTREAMER)
	if not cap.isOpened():
		raise RuntimeError(
			f"Failed to open video: {input_path}. If this is a .mov, ensure ffmpeg is installed and OpenCV is built with FFMPEG."
		)

	fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# Use mp4v for .mp4; fallback to XVID
	fourcc = cv2.VideoWriter_fourcc(*('mp4v' if output_path.lower().endswith('.mp4') else 'XVID'))
	writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
	if not writer.isOpened():
		raise RuntimeError(f"Failed to open writer: {output_path}")

	frame_idx = 0
	# Simple IoU-based tracker state: list of (track_id, bbox)
	next_track_id = 0
	active_tracks = []  # [(id, bbox)]

	def iou(a, b):
		ax1, ay1, ax2, ay2 = a
		bx1, by1, bx2, by2 = b
		ix1, iy1 = max(ax1, bx1), max(ay1, by1)
		ix2, iy2 = min(ax2, bx2), min(ay2, by2)
		inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
		area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
		area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
		union = area_a + area_b - inter + 1e-6
		return inter / union

	csv_path = csv_output_path or os.path.splitext(output_path)[0] + '.csv'
	fieldnames = ["timestamp_sec", "track_id", "status", "bbox_left", "bbox_top", "bbox_right", "bbox_bottom", "ec_prob", "inout_prob", "gaze_x", "gaze_y"]
	os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
	csv_file = open(csv_path, 'w', newline='')
	csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
	csv_writer.writeheader()
	try:
		while True:
			ok, frame_bgr = cap.read()
			if not ok:
				break
			# Keep native size for the writer; annotate on resized copy and then upscale back if needed
			img_pil = pil_from_cv(frame_bgr)
			annotated_pil, records = annotate_image_with_demo(demo, img_pil, threshold)
			annotated_bgr = cv_from_pil(annotated_pil)
			# Ensure size matches writer expected size
			if annotated_bgr.shape[1] != width or annotated_bgr.shape[0] != height:
				annotated_bgr = cv2.resize(annotated_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
			writer.write(annotated_bgr)

			# Assign track IDs based on IoU with previous active tracks
			assigned = set()
			new_active = []
			for r in records:
				bbox = (r["bbox_left"], r["bbox_top"], r["bbox_right"], r["bbox_bottom"]) 
				# find best match
				best_id = None
				best_iou = 0.0
				best_idx = -1
				for idx, (tid, tb) in enumerate(active_tracks):
					i = iou(bbox, tb)
					if i > best_iou:
						best_iou = i
						best_id = tid
						best_idx = idx
				if best_iou >= 0.3 and best_idx not in assigned:
					assigned.add(best_idx)
					tid = best_id
				else:
					tid = next_track_id
					next_track_id += 1
				new_active.append((tid, bbox))

			# update active tracks
			active_tracks = new_active

			# timestamp in seconds
			ts = frame_idx / fps if fps else frame_idx
			# If max_seconds is set, stop once we exceed it
			if max_seconds is not None and ts >= max_seconds:
				break
			for (tid, bbox), r in zip(active_tracks, records):
				row = {"timestamp_sec": ts, "track_id": tid}
				row.update(r)
				csv_writer.writerow(row)

			frame_idx += 1
	finally:
		cap.release()
		writer.release()
		csv_file.close()


def parse_args():
	parser = argparse.ArgumentParser(description="Run GT360 inference on image or video and save annotated output.")
	parser.add_argument('-i', '--input', required=True, help='Input image or video path')
	parser.add_argument('-o', '--output', required=True, help='Output image or video path')
	parser.add_argument('--model-gt', default='GT360_vat.pt', help='Path to GT360 gaze model weights')
	parser.add_argument('--model-ec', default='model/model_weights.pkl', help='Path to EC detector weights')
	parser.add_argument('--threshold', type=float, default=0.85, help='EC probability threshold for stage-2 GT')
	parser.add_argument('-x', '--csv-output', default=None, help='Optional CSV output path; defaults next to output media')
	parser.add_argument('--profile', action='store_true', help='Enable torch profiler (disabled by default)')
	parser.add_argument('--max-seconds', type=float, default=10.0, help='Max seconds to process for video inputs (default: 10.0)')
	parser.add_argument('--full-video', action='store_true', help='Process the entire video (overrides --max-seconds)')
	return parser.parse_args()


def main():
	args = parse_args()

	# Expand user paths
	args.input = os.path.expanduser(args.input)
	args.output = os.path.expanduser(args.output)
	args.model_gt = os.path.expanduser(args.model_gt)
	args.model_ec = os.path.expanduser(args.model_ec)
	if args.csv_output is not None:
		args.csv_output = os.path.expanduser(args.csv_output)

	# Initialize models once (profiling disabled unless --profile)
	demo = DemoSys(model_gt=args.model_gt, model_ec=args.model_ec, enable_profiler=args.profile)

	# Choose path by extension
	lower = args.input.lower()
	img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
	vid_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.mpg', '.mpeg', '.m4v'}
	ext = os.path.splitext(lower)[1]

	if ext in img_exts:
		process_image(demo, args.input, args.output, args.threshold, csv_output_path=args.csv_output)
	elif ext in vid_exts:
		max_seconds = None if args.full_video else (args.max_seconds if args.max_seconds is not None else 10.0)
		process_video(demo, args.input, args.output, args.threshold, csv_output_path=args.csv_output, max_seconds=max_seconds)
	else:
		raise ValueError(f"Unsupported file extension: {ext}")


if __name__ == '__main__':
	main()


