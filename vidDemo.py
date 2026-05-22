#!/usr/bin/env python
"""
vidDemo.py — stream a gaze-target overlay demo video (no intermediate frame dumps).

Reads a source video frame-by-frame, runs :class:`DemoSys` inference + overlay
on each frame in memory, and writes directly to an output ``.mp4``.

Example::

    python vidDemo.py ../ChildPlay-gaze/clips/IMrKEA2j278_4931-5209.mp4 \\
        --model_gt vatMoE.pt \\
        --outdir processed \\
        --start 0 --duration 6 \\
        --fps 24 --max_size 896
"""

from __future__ import annotations

import argparse
import os
import sys
from os.path import join

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from Demo_sys import DemoSys, EC_THRES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gaze-target video demo — infer + overlay, stream to MP4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("video", nargs="?",
                   default="../ChildPlay-gaze/clips/IMrKEA2j278_4931-5209.mp4",
                   help="Input video path")
    p.add_argument("--model_gt", default="vatMoE.pt",
                   help="Gaze-model checkpoint passed to DemoSys")
    p.add_argument("--model_ec", default=None,
                   help="EC model weights (DemoSys default if omitted)")
    p.add_argument("--outdir", default="processed",
                   help="Directory for the output MP4")
    p.add_argument("--output", default=None,
                   help="Explicit output video path (overrides --outdir naming)")
    p.add_argument("--fps", type=float, default=0.0,
                   help="Output FPS; 0 = use source video FPS")
    p.add_argument("--max_size", type=int, default=896,
                   help="Thumbnail longest side for inference + overlay (0 = no resize)")
    p.add_argument("--start", type=float, default=0.0,
                   help="Start time in seconds")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Clip length in seconds; 0 = until end of file")
    p.add_argument("--stride", type=int, default=1,
                   help="Process every Nth frame (1 = all frames)")
    p.add_argument("--threshold", type=float, default=EC_THRES,
                   help="EC score threshold (auto-detect mode)")
    p.add_argument("--bbox", default=None,
                   help="Optional fixed head bbox for every frame: xmin,ymin,xmax,ymax "
                        "(normalised); skips face detector")
    p.add_argument("--codec", default="mp4v",
                   help="FourCC codec tag for cv2.VideoWriter (e.g. mp4v, avc1)")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-frame EC / inout logs")
    return p.parse_args()


def _resolve_output_path(args: argparse.Namespace) -> str:
    if args.output:
        return args.output
    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    return join(args.outdir, f"MoEdemo_{stem}.mp4")


def _pil_rgb_to_bgr(arr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def process_video(demo: DemoSys, args: argparse.Namespace) -> str:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out_fps = args.fps if args.fps > 0 else src_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_frame = int(max(0.0, args.start) * src_fps)
    end_frame = None
    if args.duration > 0:
        end_frame = start_frame + int(args.duration * src_fps)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    manual_bbox = None
    if args.bbox:
        manual_bbox = [float(x.strip()) for x in args.bbox.split(",")]
        if len(manual_bbox) != 4:
            raise ValueError("--bbox requires four comma-separated values")

    out_path = _resolve_output_path(args)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    writer = None
    frame_idx = start_frame
    written = 0
    pbar_total = None
    if total_frames > 0 and end_frame is not None:
        pbar_total = max(0, (end_frame - start_frame) // max(args.stride, 1))

    pbar = tqdm(total=pbar_total, desc="vidDemo", unit="frame")

    try:
        while cap.isOpened():
            if end_frame is not None and frame_idx >= end_frame:
                break

            ret, bgr = cap.read()
            if not ret:
                break

            process_this = ((frame_idx - start_frame) % max(args.stride, 1)) == 0
            frame_idx += 1
            if not process_this:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out_pil, _, _ = demo.render_frame(
                rgb,
                threshold=args.threshold,
                manual_bboxes_norm=manual_bbox,
                max_size=args.max_size,
                verbose=args.verbose,
            )
            out_bgr = _pil_rgb_to_bgr(np.asarray(out_pil))

            if writer is None:
                h, w = out_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*args.codec)
                writer = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Failed to open VideoWriter for {out_path!r} "
                        f"(codec={args.codec!r})")

            writer.write(out_bgr)
            written += 1
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()

    if written == 0:
        raise RuntimeError("No frames were processed — check --start/--duration/--stride")

    return out_path


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.video):
        print(f"[error] Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    demo_kw = {"model_gt": args.model_gt}
    if args.model_ec:
        demo_kw["model_ec"] = args.model_ec
    demo = DemoSys(**demo_kw)
    demo.savefigs = 0  # streaming mode — never write per-frame PNGs

    out_path = process_video(demo, args)
    print(f"Demo video saved: {out_path}")


if __name__ == "__main__":
    main()
