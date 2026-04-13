import argparse
import ast
import os
import json
import numpy as np
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# Head bbox convention (shared by GOOReal and GOOSynth):
#   The head / person bbox is always the LAST entry in the bboxes list.
# GOOReal-specific fields (stored as optional extras in the JSON):
#   hx / hy  : explicit head/eye centre in absolute pixels (GOOReal only).
# Common gaze fields:
#   gaze_cx / gaze_cy : gaze-target centre in absolute pixel coordinates.

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Preprocess GOOReal (default) or GOOSynth from HuggingFace cache into the "
                "same JSON annotation format used by train_goo.py / train_vat.py / train_childplay.py."
)
parser.add_argument(
    "--dataset", type=str, default="real", choices=["real", "synth"],
    help="Which GOO dataset variant to preprocess: "
         "'real' → markytools/goorealv3 (default), "
         "'synth' → markytools/goosyntheticv3.",
)
parser.add_argument(
    "--output_dir", type=str, default=None,
    help="Root directory for output images and annotation JSONs. "
         "Defaults to ./GOORealV3 (real) or ./GOOSynthV3 (synth).",
)
parser.add_argument(
    "--no_save_images", action="store_true",
    help="Skip writing images to disk (use when images are already extracted).",
)
parser.add_argument(
    "--image_format", type=str, default="jpg", choices=["jpg", "png"],
    help="Image format when saving to disk (default: jpg).",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Dataset-variant constants
# ---------------------------------------------------------------------------

IS_REAL = args.dataset == "real"

DATASET_ID  = "markytools/goorealv3"       if IS_REAL else "markytools/goosyntheticv3"
JSON_PREFIX = "gooreal"                    if IS_REAL else "goosynth"
DEFAULT_DIR = "./GOORealV3"                if IS_REAL else "./GOOSynthV3"

# GOOReal only has a 'test' split; GOOSynth has both 'train' and 'test'.
SPLITS = ["test"] if IS_REAL else ["train", "test"]

output_dir = args.output_dir or DEFAULT_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_bboxes(raw):
    """Return bboxes as a Python list; handles both list/array and string inputs."""
    if isinstance(raw, str):
        return ast.literal_eval(raw)
    return list(raw)


def process_split(dataset_split, split_name, out_dir, save_images, image_format, is_real):
    images_dir = os.path.join(out_dir, "images", split_name)
    if save_images:
        os.makedirs(images_dir, exist_ok=True)

    frames = []
    num_skipped = 0
    num_corrupt = 0

    # Index-based iteration so we can catch per-sample decode errors from the
    # HuggingFace / PIL layer (e.g. "broken data stream") without aborting the
    # whole run.  Iterator-based loops surface these errors inside next() where
    # a try/except around the loop body cannot catch them.
    total = len(dataset_split)
    for idx in tqdm(range(total), desc=f"Processing {split_name}"):
        try:
            sample = dataset_split[idx]
        except Exception as e:
            tqdm.write(f"  [WARN] idx {idx}: skipping corrupt sample ({type(e).__name__}: {e})")
            num_corrupt += 1
            continue

        try:
            width  = int(sample["width"])
            height = int(sample["height"])

            # ---- save image ---------------------------------------------- #
            img_filename = f"{idx:08d}.{image_format}"
            # Relative path stored in JSON – matches how loaders open images.
            img_rel_path = os.path.join("images", split_name, img_filename)
            img_abs_path = os.path.join(out_dir, img_rel_path)

            if save_images:
                image = sample["image"]
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(np.asarray(image))
                image.convert("RGB").save(
                    img_abs_path,
                    quality=95 if image_format == "jpg" else None,
                )

            # ---- head bounding box --------------------------------------- #
            # GOO annotation convention: head bbox is always the LAST entry.
            # For GOOReal, bboxes is always a string; for GOOSynth it may be either.
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

            # ---- gaze target --------------------------------------------- #
            gaze_cx = float(sample["gaze_cx"])
            gaze_cy = float(sample["gaze_cy"])

            # Treat negative sentinel values as out-of-frame (matches VAT/GazeFollow logic).
            inout = int(gaze_cx >= 0 and gaze_cy >= 0)

            # Clamp boundary-negative coords to 0
            if gaze_cx < 0:
                gaze_cx = 0.0
            if gaze_cy < 0:
                gaze_cy = 0.0

            # ---- build head annotation entry ----------------------------- #
            head_entry = {
                "bbox":       [xmin, ymin, xmax, ymax],
                "bbox_norm":  [
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

            # GOOReal provides an explicit head/eye centre (hx, hy) that is more
            # precise than the bbox centre.  Store as optional extras; core loaders
            # (train_goo.py, train_vat.py, …) ignore unknown keys automatically.
            if is_real:
                hx = float(sample["hx"])
                hy = float(sample["hy"])
                head_entry["hx"]      = hx
                head_entry["hy"]      = hy
                head_entry["hx_norm"] = hx / float(width)
                head_entry["hy_norm"] = hy / float(height)

            frames.append({
                "path":   img_rel_path,
                "width":  width,
                "height": height,
                "heads":  [head_entry],
            })

        except Exception as e:
            tqdm.write(f"  [WARN] idx {idx}: skipping due to processing error ({type(e).__name__}: {e})")
            num_skipped += 1

    print(f"  {split_name}: {len(frames)} frames processed, "
          f"{num_skipped} skipped (empty/bad bboxes), "
          f"{num_corrupt} skipped (corrupt image decode).")
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    save_images = not args.no_save_images
    os.makedirs(output_dir, exist_ok=True)

    variant_label = "GOOReal v3" if IS_REAL else "GOOSynth v3"
    print(f"Loading {variant_label} from HuggingFace cache …")
    print(f"  HuggingFace ID : {DATASET_ID}")
    print(f"  Output dir     : {output_dir}")
    print(f"  Splits         : {SPLITS}")
    print(f"  Save images    : {save_images}")

    dataset = load_dataset(DATASET_ID)
    print(dataset)

    for split_name in SPLITS:
        print(f"\n{'='*60}")
        print(f"Split : {split_name}")
        print(f"{'='*60}")

        frames = process_split(
            dataset[split_name],
            split_name,
            output_dir,
            save_images,
            args.image_format,
            IS_REAL,
        )

        out_json = os.path.join(output_dir, f"{JSON_PREFIX}_{split_name}_preprocess.json")
        with open(out_json, "w") as fh:
            json.dump(frames, fh)
        print(f"  Saved → {out_json}  ({len(frames)} entries)")

    print("\nDone.")


if __name__ == "__main__":
    main()
