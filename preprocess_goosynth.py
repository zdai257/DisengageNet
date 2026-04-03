import argparse
import ast
import os
import json
import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# Head bbox is always the LAST entry in bboxes (per original GOO annotation format).
# hx / hy   : head/eye centre position in absolute pixel coordinates.
# gaze_cx/cy: gaze-target centre in absolute pixel coordinates.
# bboxes     : stored as a plain Python list in the HuggingFace parquet, but the
#              dataset card warns they may arrive as a string → we handle both.

parser = argparse.ArgumentParser(
    description="Preprocess GOOSynthV3 (HuggingFace cache) into the same JSON "
                "annotation format produced by preprocess_vat.py."
)
parser.add_argument(
    "--output_dir", type=str, default="./GOOSynthV3",
    help="Root directory to write images and annotation JSONs (default: ./GOOSynthV3).",
)
parser.add_argument(
    "--no_save_images", action="store_true",
    help="Skip saving images to disk (use when they are already extracted).",
)
parser.add_argument(
    "--image_format", type=str, default="jpg", choices=["jpg", "png"],
    help="Format used when saving images (default: jpg).",
)
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_bboxes(raw):
    """Return bboxes as a Python list, handling both list and string inputs."""
    if isinstance(raw, str):
        return ast.literal_eval(raw)
    return list(raw)  # already a list / sequence


def process_split(dataset_split, split_name, output_dir, save_images, image_format):
    images_dir = os.path.join(output_dir, "images", split_name)
    if save_images:
        os.makedirs(images_dir, exist_ok=True)

    frames = []
    num_skipped = 0

    for idx, sample in enumerate(tqdm(dataset_split, desc=f"Processing {split_name}")):
        width  = int(sample["width"])
        height = int(sample["height"])

        # ---- image path -------------------------------------------------- #
        img_filename = f"{idx:08d}.{image_format}"
        # Relative path stored in JSON (matches how VAT uses seq_img_path)
        img_rel_path = os.path.join("images", split_name, img_filename)
        img_abs_path = os.path.join(output_dir, img_rel_path)

        if save_images:
            image = sample["image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image))
            image.convert("RGB").save(
                img_abs_path,
                quality=95 if image_format == "jpg" else None,
            )

        # ---- head bounding box ------------------------------------------- #
        # Per original GOO annotation format, the head bbox is always the
        # last element appended to the bboxes list.
        raw_bboxes = parse_bboxes(sample["bboxes"])

        if len(raw_bboxes) == 0:
            num_skipped += 1
            continue

        head_raw = raw_bboxes[-1]  # [xmin, ymin, xmax, ymax] in absolute pixels
        xmin, ymin, xmax, ymax = (float(v) for v in head_raw)

        # Clamp within image bounds
        xmin = max(xmin, 0.0)
        ymin = max(ymin, 0.0)
        xmax = min(xmax, float(width))
        ymax = min(ymax, float(height))

        # ---- gaze target ------------------------------------------------- #
        # gaze_cx / gaze_cy are absolute pixel coordinates of the gaze target.
        gaze_cx = float(sample["gaze_cx"])
        gaze_cy = float(sample["gaze_cy"])

        # In-frame flag: GOOSynth gaze targets are always scene objects;
        # treat negative sentinel values as out-of-frame (matches VAT logic).
        inout = int(gaze_cx >= 0 and gaze_cy >= 0)

        # Clamp slightly-negative (boundary) coords to 0, matching VAT behaviour.
        if gaze_cx < 0:
            gaze_cx = 0.0
        if gaze_cy < 0:
            gaze_cy = 0.0

        head_entry = {
            "bbox":      [xmin, ymin, xmax, ymax],
            "bbox_norm": [
                xmin / float(width),
                ymin / float(height),
                xmax / float(width),
                ymax / float(height),
            ],
            "gazex":      [gaze_cx],
            "gazex_norm": [gaze_cx / float(width)],
            "gazey":      [gaze_cy],
            "gazey_norm": [gaze_cy / float(height)],
            "inout":      inout,
        }

        frames.append({
            "path":   img_rel_path,
            "width":  width,
            "height": height,
            "heads":  [head_entry],
        })

    print(f"{split_name}: {len(frames)} frames processed, {num_skipped} skipped (empty bboxes).")
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir  = args.output_dir
    save_images = not args.no_save_images
    os.makedirs(output_dir, exist_ok=True)

    print("Loading GOOSynthV3 dataset from HuggingFace cache …")
    dataset = load_dataset("markytools/goosyntheticv3")
    print(dataset)

    for split_name in ["train", "test"]:
        print(f"\n{'='*60}")
        print(f"Split: {split_name}")
        print(f"{'='*60}")

        frames = process_split(
            dataset[split_name],
            split_name,
            output_dir,
            save_images,
            args.image_format,
        )

        out_path = os.path.join(output_dir, f"goosynth_{split_name}_preprocess.json")
        with open(out_path, "w") as f:
            json.dump(frames, f)
        print(f"Saved → {out_path}  ({len(frames)} entries)")

    print("\nDone.")


if __name__ == "__main__":
    main()
