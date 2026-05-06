"""
visualise_depth.py — visualisation toolkit for the GT3D depth-aware gaze model.

Renders a 2×3 panel per sample suitable for figures in the paper manuscript:

    Panel A — RGB + bbox + GT marker (cyan ✕) + predicted target (magenta ✕)
              and head→target gaze rays for both pred (magenta) and GT (cyan).
    Panel B — RGB with predicted heatmap overlay (turbo colour map, alpha).
    Panel C — Cached / supplied depth map overlaid on the RGB (viridis).
    Panel D — Gaze-target-region depth: depth · heatmap, isolating the
              region the model attends to in 3-D.
    Panel E — Z-bar showing predicted z_head and z_gaze and their log-ratio
              alongside the GT values, anchored at the head depth.
    Panel F — Predicted dense depth map, when the model uses
              ``refine_cond="dense"``; otherwise blank with an annotation.

Usage modes
-----------
1. Single image + bbox::

    python visualise_depth.py \\
        --checkpoint  path/to/best.pt \\
        --image       path/to/image.jpg \\
        --bbox        x1 y1 x2 y2          # absolute pixels
        [--depth      path/to/depth.npy]   # optional, GT pseudo-label
        [--gt-gaze    gx gy]               # absolute pixels, optional
        --output      viz.png

2. Sample N items from a GOOSynthV3 split::

    python visualise_depth.py \\
        --checkpoint  path/to/best.pt \\
        --dataset     goosynth \\
        --data-path   ../GOOSynthV3 \\
        --split       test \\
        --num-samples 8 \\
        --output-dir  viz_goosynth/

3. Sample N items from GazeFollow::

    python visualise_depth.py \\
        --checkpoint  path/to/best.pt \\
        --dataset     gazefollow \\
        --data-path   ./gazefollow_extended \\
        --split       test \\
        --num-samples 8 \\
        --output-dir  viz_gf/

The script does *not* depend on the dataloaders in ``train_depth_*.py`` — it
loads images directly so it can also visualise arbitrary stills.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

from network.network_builder_gt3d import get_gt3d_model


# --------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------

DEPTH_EPS = 1e-4


def build_model(config_path: str, checkpoint_path: str, device: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model, _ = get_gt3d_model(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, config


def make_transform(input_resolution: int):
    return T.Compose([
        T.Resize((input_resolution, input_resolution)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def run_inference(model, image_pil: Image.Image, bbox_norm, transform, device):
    """Run a single-person forward pass.

    ``bbox_norm`` is a 4-tuple in [0, 1] (x1, y1, x2, y2).
    Returns a dict with numpy heatmap [64, 64], scalars z_gaze and z_head, and
    the predicted dense depth map (None if the model does not produce one).
    """
    img_t = transform(image_pil).unsqueeze(0).to(device)
    out = model({"images": img_t, "bboxes": [[list(bbox_norm)]]})
    hm   = out["heatmap"][0][0].detach().cpu().numpy()
    z_g  = float(out["depth_gaze"][0][0].item())
    z_h  = float(out["depth_head"][0][0].item())

    # Optional dense map — only present when refine_cond == "dense"; we
    # intercept it via a forward hook instead of changing the model API.
    dense_map = getattr(model, "_last_dense_depth", None)
    return {
        "heatmap":     hm,
        "z_gaze":      z_g,
        "z_head":      z_h,
        "dense_depth": dense_map,
    }


def _attach_dense_hook(model):
    """Cache the dense depth feature so we can visualise it.

    Mean-aggregates across the channel axis to a single (h, w) map and
    stashes it on ``model._last_dense_depth``.
    """
    if not getattr(model, "use_refine", False):
        return
    if getattr(model, "refine_cond", "scalar") != "dense":
        return

    def _hook(_module, _inp, out):
        # out: [B', K, 2h, 2w]
        m = out.detach().mean(dim=1).cpu().numpy()       # [B', 2h, 2w]
        model._last_dense_depth = m[0]                   # first sample

    model.dense_depth_decoder.register_forward_hook(_hook)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _normalise_for_show(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / max(hi - lo, 1e-8)


def _argmax_pixel(heatmap: np.ndarray, image_size_wh) -> Tuple[int, int]:
    """Return the (x, y) pixel of the heatmap argmax, in image coords."""
    H, W = heatmap.shape
    flat_idx = int(np.argmax(heatmap))
    v = flat_idx // W
    u = flat_idx %  W
    iw, ih = image_size_wh
    return (
        int(round(u / max(W - 1, 1) * (iw - 1))),
        int(round(v / max(H - 1, 1) * (ih - 1))),
    )


def render_panel(
    image_pil: Image.Image,
    bbox_pix: Tuple[int, int, int, int],
    heatmap: np.ndarray,
    z_gaze: float,
    z_head: float,
    *,
    depth_gt: Optional[np.ndarray] = None,
    dense_depth: Optional[np.ndarray] = None,
    gt_gaze_pix: Optional[Tuple[int, int]] = None,
    gt_z_gaze: Optional[float] = None,
    gt_z_head: Optional[float] = None,
    title: str = "",
) -> plt.Figure:
    """Build the 2×3 visualisation figure described in the module docstring."""
    iw, ih = image_pil.size
    rgb    = np.asarray(image_pil)
    pred_xy = _argmax_pixel(heatmap, (iw, ih))
    head_cx = (bbox_pix[0] + bbox_pix[2]) // 2
    head_cy = (bbox_pix[1] + bbox_pix[3]) // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(title, fontsize=11)
    axA, axB, axC, axD, axE, axF = axes.flatten()

    # -- A: RGB + bbox + rays ---------------------------------------------
    axA.imshow(rgb)
    axA.add_patch(Rectangle(
        (bbox_pix[0], bbox_pix[1]),
        bbox_pix[2] - bbox_pix[0], bbox_pix[3] - bbox_pix[1],
        fill=False, edgecolor="yellow", linewidth=2.0))
    axA.plot(*pred_xy, marker="x", markersize=14, mew=3, color="magenta",
             label="pred target")
    axA.plot([head_cx, pred_xy[0]], [head_cy, pred_xy[1]],
             color="magenta", lw=2.0, alpha=0.85)
    if gt_gaze_pix is not None:
        axA.plot(*gt_gaze_pix, marker="x", markersize=14, mew=3, color="cyan",
                 label="GT target")
        axA.plot([head_cx, gt_gaze_pix[0]], [head_cy, gt_gaze_pix[1]],
                 color="cyan", lw=2.0, alpha=0.85)
    axA.set_title("RGB + bbox + gaze rays")
    axA.legend(loc="lower right", fontsize=8)
    axA.axis("off")

    # -- B: heatmap overlay -----------------------------------------------
    hm_resized = np.asarray(
        Image.fromarray((_normalise_for_show(heatmap) * 255).astype(np.uint8))
        .resize((iw, ih), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    axB.imshow(rgb)
    axB.imshow(hm_resized, cmap="turbo", alpha=0.55)
    axB.plot(*pred_xy, marker="x", markersize=14, mew=3, color="white")
    axB.set_title("Predicted heatmap (turbo)")
    axB.axis("off")

    # -- C: depth overlay --------------------------------------------------
    if depth_gt is not None:
        d_resized = np.asarray(
            Image.fromarray(_normalise_for_show(depth_gt))
            .resize((iw, ih), Image.BILINEAR)
        )
        axC.imshow(rgb)
        axC.imshow(d_resized, cmap="viridis", alpha=0.55)
        axC.set_title("Cached depth (DepthAnythingV2)")
    else:
        axC.imshow(rgb)
        axC.set_title("No GT depth supplied")
    axC.axis("off")

    # -- D: gaze-target-region depth ---------------------------------------
    if depth_gt is not None:
        # Resize both to a common 64×64 grid for the product.
        d64 = np.asarray(
            Image.fromarray(depth_gt.astype(np.float32))
            .resize((64, 64), Image.BILINEAR), dtype=np.float32)
        d64 = _normalise_for_show(d64)
        h64 = _normalise_for_show(heatmap)
        masked = d64 * h64
        axD.imshow(masked, cmap="magma")
        axD.set_title("Gaze-region depth = heatmap · depth")
    else:
        h64 = _normalise_for_show(heatmap)
        axD.imshow(h64, cmap="magma")
        axD.set_title("Gaze-region heatmap (no depth GT)")
    axD.axis("off")

    # -- E: z-bar ----------------------------------------------------------
    _draw_zbar(axE, z_head, z_gaze, gt_z_head, gt_z_gaze)

    # -- F: dense depth (model-internal) -----------------------------------
    if dense_depth is not None:
        axF.imshow(_normalise_for_show(dense_depth), cmap="cividis")
        axF.set_title("Predicted dense depth feature\n(refine_cond=\"dense\")")
    else:
        axF.text(0.5, 0.5,
                 "Dense depth not available\n(model uses scalar/pair refine\n"
                 "or refinement is off)",
                 ha="center", va="center", fontsize=9,
                 transform=axF.transAxes)
        axF.set_title("Predicted dense depth")
    axF.axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _draw_zbar(ax, z_head, z_gaze, gt_z_head=None, gt_z_gaze=None):
    """Vertical bar chart contrasting predicted vs GT relative depth."""
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pred", "GT"] if gt_z_head is not None else ["pred", ""])

    # Pred column
    ax.bar(0, z_head, width=0.35, color="#888888", alpha=0.85,
           label=f"z_head={z_head:.3f}")
    ax.bar(0, z_gaze - z_head, width=0.35, bottom=z_head,
           color="#cc3366", alpha=0.85, label=f"z_gaze={z_gaze:.3f}")
    pred_lr = float(np.log(max(z_gaze, DEPTH_EPS))
                    - np.log(max(z_head, DEPTH_EPS)))
    ax.text(0, 1.02, f"log-ratio={pred_lr:+.3f}", ha="center", fontsize=9)

    # GT column
    if gt_z_head is not None and gt_z_gaze is not None:
        ax.bar(1, gt_z_head, width=0.35, color="#88cccc", alpha=0.85,
               label=f"GT z_head={gt_z_head:.3f}")
        ax.bar(1, gt_z_gaze - gt_z_head, width=0.35, bottom=gt_z_head,
               color="#3366cc", alpha=0.85, label=f"GT z_gaze={gt_z_gaze:.3f}")
        gt_lr = float(np.log(max(gt_z_gaze, DEPTH_EPS))
                      - np.log(max(gt_z_head, DEPTH_EPS)))
        ax.text(1, 1.02, f"log-ratio={gt_lr:+.3f}", ha="center", fontsize=9)

    ax.set_title("Head-anchored relative depth")
    ax.set_ylabel("z (per-image min-max-normalised)")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", alpha=0.5)


# --------------------------------------------------------------------------
# Helpers for the dataset modes
# --------------------------------------------------------------------------

def _load_cached_depth(image_path_abs: str, dataset_root: str,
                       depth_dir: str = "depth") -> Optional[np.ndarray]:
    """Resolve the cached .npy depth file next to the dataset root."""
    rel = os.path.relpath(image_path_abs, dataset_root)
    rel = rel.replace("images" + os.sep, depth_dir + os.sep, 1)
    if rel == os.path.relpath(image_path_abs, dataset_root):
        rel = os.path.join(depth_dir, rel)
    rel = os.path.splitext(rel)[0] + ".npy"
    full = os.path.join(dataset_root, rel)
    if os.path.isfile(full):
        return np.load(full).astype(np.float32)
    return None


def _sample_anchored_gt_z(depth_arr: np.ndarray, gx_norm: float, gy_norm: float,
                          bbox_norm) -> Tuple[float, float]:
    """Mirror of GOOSynthDepth._sample_anchored at 64×64 grid."""
    d_pil = Image.fromarray(depth_arr, mode="F").resize((64, 64), Image.BILINEAR)
    d = np.asarray(d_pil, dtype=np.float32)
    lo, hi = float(d.min()), float(d.max())
    n = (d - lo) / max(hi - lo, 1e-8)
    u = int(np.clip(round(gx_norm * 63), 0, 63))
    v = int(np.clip(round(gy_norm * 63), 0, 63))
    z_g = float(n[v, u])
    cx = (bbox_norm[0] + bbox_norm[2]) * 0.5
    cy = (bbox_norm[1] + bbox_norm[3]) * 0.5
    uh = int(np.clip(round(cx * 63), 0, 63))
    vh = int(np.clip(round(cy * 63), 0, 63))
    z_h = float(n[vh, uh])
    return z_g, z_h


def _iter_goosynth(data_path: str, split: str):
    js = os.path.join(data_path, f"goosynth_{split}_preprocess.json")
    with open(js, "r") as f:
        frames = json.load(f)
    for frame in frames:
        head = frame["heads"][0]
        yield {
            "rel_path": frame["path"],
            "bbox":     list(head["bbox"]),
            "gazex":    list(head["gazex"]),
            "gazey":    list(head["gazey"]),
            "inout":    head["inout"],
        }


def _iter_gazefollow(data_path: str, split: str):
    js = os.path.join(data_path, f"{split}_preprocessed.json")
    with open(js, "r") as f:
        items = json.load(f)
    for frame in items:
        for head in frame["heads"]:
            if head.get("inout", 1) != 1:
                continue
            yield {
                "rel_path":  frame["path"],
                "bbox":      list(head["bbox"]),
                "bbox_norm": list(head["bbox_norm"]),
                "gazex":     list(head["gazex"]),
                "gazey":     list(head["gazey"]),
                "gazex_norm": list(head["gazex_norm"]),
                "gazey_norm": list(head["gazey_norm"]),
                "inout":     head["inout"],
            }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",     default="configuration.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device",     default="cuda:0" if torch.cuda.is_available() else "cpu")

    # Mode 1: single image + bbox
    p.add_argument("--image",  default=None,
                   help="Path to a single RGB image.")
    p.add_argument("--bbox",   nargs=4, type=float, default=None,
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="Head bbox in absolute pixels.")
    p.add_argument("--depth",  default=None,
                   help="Optional cached depth .npy (per-image scale).")
    p.add_argument("--gt-gaze", nargs=2, type=float, default=None,
                   metavar=("GX", "GY"),
                   help="GT gaze pixel (absolute coords).")
    p.add_argument("--output", default="viz.png")

    # Mode 2: dataset sampling
    p.add_argument("--dataset",   choices=["goosynth", "gazefollow"], default=None)
    p.add_argument("--data-path", default=None)
    p.add_argument("--split",     default="test")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--depth-dir", default="depth")
    p.add_argument("--output-dir", default="viz/")
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def render_one_sample(model, transform, device, *,
                      image_path: str, bbox_pix, gt_gaze_pix=None,
                      depth_gt: Optional[np.ndarray] = None,
                      title: str = ""):
    image_pil = Image.open(image_path).convert("RGB")
    iw, ih = image_pil.size
    bbox_norm = (
        bbox_pix[0] / iw, bbox_pix[1] / ih,
        bbox_pix[2] / iw, bbox_pix[3] / ih,
    )

    res = transform.transforms[0].size
    if isinstance(res, (list, tuple)):
        pass                                       # already (h, w)

    pred = run_inference(model, image_pil, bbox_norm, transform, device)

    gt_z_g = gt_z_h = None
    if depth_gt is not None and gt_gaze_pix is not None:
        gx_n = gt_gaze_pix[0] / iw
        gy_n = gt_gaze_pix[1] / ih
        gt_z_g, gt_z_h = _sample_anchored_gt_z(
            depth_gt, gx_n, gy_n, bbox_norm)

    return render_panel(
        image_pil, tuple(int(round(v)) for v in bbox_pix),
        pred["heatmap"], pred["z_gaze"], pred["z_head"],
        depth_gt=depth_gt,
        dense_depth=pred["dense_depth"],
        gt_gaze_pix=(int(round(gt_gaze_pix[0])), int(round(gt_gaze_pix[1])))
                    if gt_gaze_pix is not None else None,
        gt_z_gaze=gt_z_g,
        gt_z_head=gt_z_h,
        title=title,
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model from {args.checkpoint}")
    model, config = build_model(args.config, args.checkpoint, args.device)
    _attach_dense_hook(model)

    res = int(config["data"]["input_resolution"])
    transform = make_transform(res)

    if args.image is not None:
        if args.bbox is None:
            raise SystemExit("--bbox is required with --image")
        depth_gt = (np.load(args.depth).astype(np.float32)
                    if args.depth and os.path.isfile(args.depth) else None)
        fig = render_one_sample(
            model, transform, args.device,
            image_path=args.image,
            bbox_pix=tuple(args.bbox),
            gt_gaze_pix=tuple(args.gt_gaze) if args.gt_gaze else None,
            depth_gt=depth_gt,
            title=f"{os.path.basename(args.image)} | "
                  f"refine={getattr(model, 'refine_cond', 'off')}",
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {args.output}")
        return

    if args.dataset is None:
        raise SystemExit("Provide either --image or --dataset.")

    if args.data_path is None:
        raise SystemExit("--data-path is required with --dataset.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    iterator = (_iter_goosynth(args.data_path, args.split)
                if args.dataset == "goosynth"
                else _iter_gazefollow(args.data_path, args.split))
    samples = list(iterator)
    if not samples:
        raise SystemExit(f"No samples found in {args.data_path} / {args.split}")
    chosen = random.sample(samples, k=min(args.num_samples, len(samples)))

    for k, s in enumerate(chosen):
        image_path = os.path.join(args.data_path, s["rel_path"])
        if not os.path.isfile(image_path):
            print(f"  [skip] missing {image_path}")
            continue
        depth_gt = _load_cached_depth(image_path, args.data_path, args.depth_dir)

        if args.dataset == "goosynth":
            iw, ih = Image.open(image_path).size
            bbox_pix = tuple(s["bbox"])
            gt_gaze_pix = (s["gazex"][0], s["gazey"][0]) if s.get("inout", 1) else None
        else:                                                    # gazefollow
            iw, ih = Image.open(image_path).size
            bb_n = s["bbox_norm"]
            bbox_pix = (bb_n[0] * iw, bb_n[1] * ih, bb_n[2] * iw, bb_n[3] * ih)
            gx_n = float(np.mean(s["gazex_norm"]))
            gy_n = float(np.mean(s["gazey_norm"]))
            gt_gaze_pix = (gx_n * iw, gy_n * ih)

        fig = render_one_sample(
            model, transform, args.device,
            image_path=image_path,
            bbox_pix=bbox_pix,
            gt_gaze_pix=gt_gaze_pix,
            depth_gt=depth_gt,
            title=f"{args.dataset}/{args.split}: {s['rel_path']} | "
                  f"refine={getattr(model, 'refine_cond', 'off')}",
        )
        out = out_dir / f"viz_{k:03d}_{Path(s['rel_path']).stem}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")

    print(f"Done.  Wrote {len(chosen)} figures to {out_dir}/")


if __name__ == "__main__":
    main()
