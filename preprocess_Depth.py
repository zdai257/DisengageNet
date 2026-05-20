#!/usr/bin/env python
"""
preprocess_Depth.py — generate DepthAnythingV2 monocular depth maps for any
gaze-target dataset (GOOSynth, GOOReal, GazeFollow, VAT, ChildPlay, ...).

For every image under
    <data_path>/<image_dir>/**.jpg|png|...
the script writes a matching depth map to
    <data_path>/<depth_dir>/**.npy
with the relative directory layout preserved, so the depth file sits at the
exact same logical position as its source image.

Defaults target GOOSynthV3:
    ../GOOSynthV3/images/train/*.jpg  →  ../GOOSynthV3/depth/train/*.npy
    ../GOOSynthV3/images/test/*.jpg   →  ../GOOSynthV3/depth/test/*.npy

The DepthAnythingV2 engine is loaded on cuda:0 by default and falls back to
CPU if CUDA is unavailable or model construction fails.

Output format notes:
    • Saved as float16 .npy by default (≈half the disk footprint of float32)
      while preserving enough precision for all downstream gaze-heatmap use.
    • DepthAnythingV2 returns a *relative inverse depth* map — higher values
      mean points closer to the camera.  Normalise per-image at load time in
      the training dataloader (min-max to [0, 1]).
    • A matching .npy is written per image; an optional PNG16 format is
      provided for visual inspection (`--save_format png16`).
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# DepthAnythingV2 repo lives as a sibling checkout of this project.
sys.path.append("../Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2


# ---------------------------------------------------------------------------
# Model configuration (matches the user's cached inference engine)
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run DepthAnythingV2 over every image in a gaze-target "
                    "dataset and cache depth maps to <data_path>/<depth_dir>."
    )
    p.add_argument("--data_path", type=str, default="../GOOSynthV3",
                   help="Dataset root.  Defaults to the GOOSynthV3 directory.")
    p.add_argument("--image_dir", type=str, default="images",
                   help="Sub-folder under data_path holding raw images "
                        "(supports arbitrarily nested sub-splits, e.g. "
                        "'images/train', 'images/test', 'images/scene/0001').")
    p.add_argument("--depth_dir", type=str, default="depth",
                   help="Sub-folder under data_path where depth maps are written.")
    p.add_argument("--splits", type=str, nargs="*", default=None,
                   help="Optional list of sub-folders under image_dir to process "
                        "(e.g. --splits train test).  Default: recurse over all.")
    p.add_argument("--encoder", type=str, choices=list(MODEL_CONFIGS), default="vitb",
                   help="DepthAnythingV2 backbone 'base' variant as default.")
    p.add_argument("--ckpt_dir", type=str, default="../Depth-Anything-V2/checkpoints",
                   help="Directory containing depth_anything_v2_<encoder>.pth.")
    p.add_argument("--input_size", type=int, default=518,
                   help="DepthAnythingV2 inference input size (default 518).")
    p.add_argument("--save_format", choices=["npy", "png16"], default="npy",
                   help="'npy' stores raw floats (recommended for training); "
                        "'png16' stores per-image min-max normalised uint16 "
                        "(useful for visual inspection only).")
    p.add_argument("--dtype", choices=["float32", "float16"], default="float16",
                   help="Storage dtype for .npy output.")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device override (e.g. 'cuda:1', 'cpu').  "
                        "Default: cuda:0 if available, else cpu.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-compute depth even if the target .npy already exists.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device / model helpers
# ---------------------------------------------------------------------------

def pick_device(preferred):
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model(encoder, ckpt_dir, device):
    cfg = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**cfg)
    ckpt_path = os.path.join(ckpt_dir, f"depth_anything_v2_{encoder}.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"DepthAnythingV2 checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def load_model_with_fallback(encoder, ckpt_dir, device):
    """Build the model on the preferred device; fall back to CPU on any failure."""
    try:
        return build_model(encoder, ckpt_dir, device), device
    except Exception as e:
        if device.type == "cuda":
            print(f"[WARN] Could not load DepthAnythingV2 on {device} "
                  f"({type(e).__name__}: {e}); falling back to CPU.")
            return build_model(encoder, ckpt_dir, torch.device("cpu")), torch.device("cpu")
        raise


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def iter_images(image_root, splits):
    """Yield every image file beneath `image_root`.

    If `splits` is provided, restrict the walk to those top-level sub-folders.
    Relative paths are preserved in the output so that the <depth_dir> tree is
    a perfect mirror of the <image_dir> tree.
    """
    image_root = Path(image_root)
    if splits:
        roots = [image_root / s for s in splits]
        roots = [r for r in roots if r.is_dir()]
        if not roots:
            raise FileNotFoundError(f"None of the requested splits exist under {image_root}")
    else:
        roots = [image_root]

    for root in roots:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_depth(depth, out_path_noext, save_format, dtype):
    """Write depth to disk; creates parent dirs on demand.

    Returns the concrete filename that was written (with extension).
    """
    out_path_noext.parent.mkdir(parents=True, exist_ok=True)
    if save_format == "npy":
        arr = depth.astype(np.float16 if dtype == "float16" else np.float32)
        target = out_path_noext.with_suffix(".npy")
        np.save(target, arr)
    elif save_format == "png16":
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max > d_min:
            norm = (depth - d_min) / (d_max - d_min)
        else:
            norm = np.zeros_like(depth)
        target = out_path_noext.with_suffix(".png")
        cv2.imwrite(str(target), (norm * 65535.0).astype(np.uint16))
        # Per-image scale/offset so the relative depth can be recovered later.
        meta_path = out_path_noext.with_suffix(".json")
        meta_path.write_text(f'{{"d_min": {d_min}, "d_max": {d_max}}}')
    else:
        raise ValueError(f"Unknown save_format: {save_format}")
    return target


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"Requested device: {device}")

    model, device = load_model_with_fallback(args.encoder, args.ckpt_dir, device)
    print(f"Loaded DepthAnythingV2-{args.encoder} on {device}")

    src_root = Path(args.data_path) / args.image_dir
    dst_root = Path(args.data_path) / args.depth_dir
    if not src_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {src_root}")

    print(f"Source images: {src_root}")
    print(f"Depth output : {dst_root}")
    if args.splits:
        print(f"Splits filter: {args.splits}")

    images = list(iter_images(src_root, args.splits))
    print(f"Found {len(images)} images")
    if not images:
        return

    n_done, n_skipped, n_failed = 0, 0, 0

    with torch.inference_mode():
        for img_path in tqdm(images, desc="Depth", unit="img"):
            rel = img_path.relative_to(src_root)
            out_noext = (dst_root / rel).with_suffix("")
            final_ext = ".npy" if args.save_format == "npy" else ".png"
            final_path = out_noext.with_suffix(final_ext)

            if final_path.exists() and not args.overwrite:
                n_skipped += 1
                continue

            try:
                img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    raise IOError("cv2.imread returned None")
                # DepthAnythingV2.infer_image handles resize / normalization
                # internally and returns a H×W float32 map at the original
                # image resolution.
                depth = model.infer_image(img_bgr, args.input_size)
                save_depth(depth, out_noext, args.save_format, args.dtype)
                n_done += 1
            except Exception as e:
                tqdm.write(f"[WARN] {img_path}: {type(e).__name__}: {e}")
                n_failed += 1

    print(f"\nDone. {n_done} computed, {n_skipped} skipped (already existed), "
          f"{n_failed} failed.")


if __name__ == "__main__":
    main()
