#!/usr/bin/env python
"""
test_models.py
==============

Self-contained evaluation script for gaze-target models, intentionally
*not* coupled to ``configuration.yaml``.  Every knob is a CLI argument
so the same script can run against any checkpoint on any dataset's test
split without touching the training-time YAML.

Currently supports:
    Architectures (--arch)
        gazelle_vitb14,  gazelle_vitb14_inout
        gazelle_vitl14,  gazelle_vitl14_inout        (default)
        gazemoe_vitl14_inout
        gt3d_vitl14_inout
        chong               (Chong et al., CVPR 2020 — per-frame variant)
        dam                 (Fang et al., Dual-Attention model)
        sharingan           (Tafasca et al., CVPR 2024)
        gatector            (Wang et al., CVPR 2022)

    Datasets (--dataset)
        gazefollow          ``<root>/test_preprocessed.json``
        vat                 ``<root>/test_preprocessed.json`` (or --split_file)
        childplay           ``<root>/test_preprocessed.json`` (or --split_file)
        goosynth            ``<root>/goosynth_test_preprocess.json``
        gooreal             ``<root>/gooreal_test_preprocess.json``

Metrics reported
    AUC          ─ standard gaze-target AUC (GazeFollow style for image
                   datasets, VAT-style rectangular AUC for video sets).
    AvgL2        ─ mean L2 between heatmap argmax and (mean-annotator) GT.
    L2-3D        ─ 3-D Euclidean distance in normalised (x, y, depth)
                   space.  Depth is sampled (nearest pixel) from the
                   per-image min-max-normalised DepthAnythingV2 map at
                   ``argmax(heatmap)`` and at the GT gaze pixel.
    Angle-3D     ─ angular error of the head→target ray in the same
                   normalised 3-D space, in degrees.
    DOA          ─ Depth-Order Accuracy.  Percentage of in-frame samples
                   for which sign(d̃_pred − d̃_head) == sign(d̃_gt − d̃_head),
                   excluding samples with |d̃_gt − d̃_head| < 1e-3.
    AP           ─ optional, in/out classification average precision.
                   Reported only when the model exposes an in/out head
                   *and* the dataset has in/out labels.

Depth source
    ``--depth_dir`` is searched relative to ``--data_root``.  If the
    corresponding ``.npy`` file is missing for a given image and
    ``--depth_on_the_fly`` is set, the script will spin up a single
    DepthAnythingV2 inference engine and compute the map at runtime
    (no caching — useful for ad-hoc evaluation runs).

The script prints a tabular summary and optionally writes a JSON / CSV
report to ``--out``.

Example
    python test_models.py \\
        --arch gazelle_vitl14_inout \\
        --checkpoint gazelle_dinov2_vitl14_inout.pt \\
        --dataset gazefollow \\
        --data_root ./gazefollow_extended \\
        --batch_size 60
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from eval import gazefollow_auc, gazefollow_l2, vat_auc, vat_l2


# ============================================================================
# 1.  CLI
# ============================================================================

ARCHES = [
    "gazelle_vitb14",
    "gazelle_vitb14_inout",
    "gazelle_vitl14",
    "gazelle_vitl14_inout",
    "gazemoe_vitl14_inout",
    "gt3d_vitl14_inout",
    "chong",
    "dam",
    "sharingan",
    "gatector",
]
DATASETS = ["gazefollow", "vat", "childplay", "goosynth", "gooreal"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--arch", choices=ARCHES, default="gazelle_vitl14_inout")
    p.add_argument("--checkpoint", type=str,
                   default="gazelle_dinov2_vitl14_inout.pt",
                   help="Path to .pt under the repo root.")
    p.add_argument("--include_backbone", action="store_true",
                   help="For Gazelle/GazeMoE/GT3D: also load backbone weights "
                        "from the checkpoint (default: backbone is the frozen "
                        "DINOv2 hub model, only the learnable parts are loaded).")
    p.add_argument("--dataset", choices=DATASETS, default="gazefollow")
    p.add_argument("--data_root", type=str, required=True,
                   help="Filesystem path to the dataset root.")
    p.add_argument("--split_file", type=str, default=None,
                   help="Override the default per-dataset annotations file.")
    p.add_argument("--depth_dir", type=str, default="",
                   help="Sub-folder under data_root holding .npy depth maps.  "
                        "Empty string, 'none', or 'off' disables the cache "
                        "lookup and auto-enables on-the-fly DepthAnythingV2 "
                        "inference.")
    p.add_argument("--depth_on_the_fly", action="store_true",
                   help="If a depth file is missing, run DepthAnythingV2 at "
                        "runtime instead of skipping the 3-D metrics.  "
                        "Auto-enabled when --depth_dir is empty / 'none' / 'off'.")
    p.add_argument("--no_depth", action="store_true",
                   help="Completely skip depth lookup AND on-the-fly DA2: only "
                        "2-D metrics (AUC, AvgL2) + optional AP are reported.")
    p.add_argument("--da2_encoder", choices=["vits", "vitb", "vitl"],
                   default="vitb",
                   help="DepthAnythingV2 backbone 'base' for on-the-fly depth.")
    p.add_argument("--da2_ckpt_dir", type=str,
                   default="../Depth-Anything-V2/checkpoints")
    p.add_argument("--batch_size", type=int, default=60)
    p.add_argument("--num_workers", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--input_resolution", type=int, default=448,
                   help="Resize the input image to this side length before "
                        "feeding the model.  Default 448 matches Gazelle / "
                        "GazeMoE / GT3D; Chong / DAM use 224, but each builder "
                        "overrides via the transform it returns.")
    p.add_argument("--limit", type=int, default=0,
                   help="Optional cap on the number of evaluated samples "
                        "(useful for quick smoke runs).  0 = no cap.")
    p.add_argument("--out", type=str, default=None,
                   help="Optional path to write a JSON report.")
    # ---- Reproduction-specific knobs (vendored upstream repos) ----
    p.add_argument("--upstream_root", type=str, default=None,
                   help="For reproduction models (sharingan / gatector): path "
                        "to the vendored upstream checkout.  Defaults to "
                        "'../sharingan' or '../GaTector-A-Unified-Framework-"
                        "for-Gaze-Object-Prediction' per builder.")
    p.add_argument("--gatector_input_size", type=int, default=224,
                   help="GaTector scene/head input resolution (default 224).")
    return p.parse_args()


# ============================================================================
# 2.  Dataset adapters
# ============================================================================
#
# All adapters emit a *flat* list of evaluation samples (one head per
# sample).  Each sample is a dict:
#
#   path        : str  — relative to data_root
#   bbox_norm   : [x1, y1, x2, y2]  in [0, 1]
#   gazex_norm  : list[float]  in [0, 1]  (per-annotator)
#   gazey_norm  : list[float]  in [0, 1]
#   inout       : int          (1 = in-frame, 0 = out-of-frame; None if unknown)
#   width, height : int        (original image resolution; used for AUC)
#   depth_path  : str | None   — resolved at iteration time
# ----------------------------------------------------------------------------

_DEPTH_DIR_OFF = {"", "none", "off", "no", "false"}


def _depth_cache_disabled(depth_dir: Optional[str]) -> bool:
    """A depth-dir value of ``""`` / ``none`` / ``off`` means no cache lookup."""
    return depth_dir is None or str(depth_dir).strip().lower() in _DEPTH_DIR_OFF


def _resolve_depth_path(data_root: str, image_rel_path: str,
                        depth_dir: Optional[str]) -> Optional[str]:
    """Translate ``images/foo/bar.jpg`` into ``<depth_dir>/foo/bar.npy``.

    Returns ``None`` when the depth cache has been disabled (empty / none / off),
    which lets the downstream code skip the cache lookup entirely and fall
    through to on-the-fly DepthAnythingV2 inference.
    """
    if _depth_cache_disabled(depth_dir):
        return None
    rel = image_rel_path.replace("images" + os.sep, depth_dir + os.sep, 1)
    if rel == image_rel_path:
        rel = os.path.join(depth_dir, image_rel_path)
    rel = os.path.splitext(rel)[0] + ".npy"
    return os.path.join(data_root, rel)


def _emit_head_sample(path: str, head: Dict[str, Any], w: int, h: int,
                       data_root: str, depth_dir: str) -> Dict[str, Any]:
    return {
        "path":       path,
        "bbox_norm":  list(head["bbox_norm"]),
        "gazex_norm": list(head.get("gazex_norm", [])),
        "gazey_norm": list(head.get("gazey_norm", [])),
        "inout":      int(head.get("inout", 1)),
        "width":      int(w),
        "height":     int(h),
        "depth_path": _resolve_depth_path(data_root, path, depth_dir),
    }


def _preprocessed_json_to_samples(json_path: str, data_root: str,
                                   depth_dir: str) -> List[Dict[str, Any]]:
    """Adapter for the preprocessed JSON formats used in this repo.

    Two schemas are auto-detected:

    1. **Flat (GazeFollow / GOO)**: top-level list of *images*, each
       with ``path`` + ``heads[*]`` and an image-level ``width / height``
       (or we PIL-open to recover it).
    2. **Nested (VAT)**: top-level list of *clips*, each with
       clip-level ``path``, ``width``, ``height`` and a nested
       ``frames[*]`` list; per-frame ``path`` + ``heads[*]``.

    Multi-head frames are flattened into one sample per head; out-of-frame
    heads are kept (so AP can still be computed) but excluded from spatial
    metrics in the eval loop.
    """
    with open(json_path, "r") as f:
        frames = json.load(f)

    samples: List[Dict[str, Any]] = []
    for frm in frames:
        # Nested (VAT) schema: clip with its own width/height + frames[].
        if "frames" in frm and isinstance(frm["frames"], list):
            w = int(frm.get("width", 0))
            h = int(frm.get("height", 0))
            for sub in frm["frames"]:
                spath = sub.get("path", "")
                if not spath:
                    continue
                ww, hh = w, h
                if ww <= 0 or hh <= 0:
                    try:
                        with Image.open(os.path.join(data_root, spath)) as im:
                            ww, hh = im.size
                    except (OSError, FileNotFoundError):
                        ww, hh = ww or 0, hh or 0
                for head in sub.get("heads", []):
                    samples.append(_emit_head_sample(
                        spath, head, ww, hh, data_root, depth_dir))
            continue

        # Flat (GazeFollow / GOO) schema.
        path = frm["path"]
        try:
            with Image.open(os.path.join(data_root, path)) as im:
                w, h = im.size
        except (OSError, FileNotFoundError):
            w = frm.get("width",  0)
            h = frm.get("height", 0)
        for head in frm.get("heads", []):
            samples.append(_emit_head_sample(
                path, head, w, h, data_root, depth_dir))
    return samples


_CHILDPLAY_RES_MAP = {
    "360p":  (640,  360),
    "480p":  (854,  480),
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
    "4k":    (3840, 2160),
}


def _childplay_csv_to_samples(data_root: str, split: str,
                               depth_dir: str) -> List[Dict[str, Any]]:
    """Load ChildPlay's per-clip CSV annotations directly (no preprocessed
    JSON required).

    Expected layout::

        <data_root>/clips.csv                                  (clip,...,resolution)
        <data_root>/annotations/<split>/<clip>.csv             (per-frame rows)
        <data_root>/images/<clip>/<video_id>_<abs_frame>.jpg   (downscaled images)

    Note: the annotation CSVs encode bbox / gaze coordinates in the *native*
    video resolution (read from ``clips.csv``'s ``resolution`` column),
    NOT the on-disk image size — the images are usually downscaled.  We
    therefore normalise by the native (W, H) so that bbox_norm / gaze_norm
    stay in [0, 1].

    Row columns: clip, frame, person_id, bbox_x, bbox_y, bbox_width,
    bbox_height, gaze_class, gaze_x, gaze_y, is_child.
    ``frame`` is 1-indexed within the clip; absolute frame number in the
    image filename is ``<clip-start> + frame - 1``.
    """
    import csv

    ann_dir = os.path.join(data_root, "annotations", split)
    if not os.path.isdir(ann_dir):
        raise FileNotFoundError(
            f"ChildPlay annotations dir not found: {ann_dir}")

    # ---- Map each clip → native (W, H) from clips.csv's resolution col --
    clips_csv_path = os.path.join(data_root, "clips.csv")
    clip_wh: Dict[str, tuple] = {}
    if os.path.isfile(clips_csv_path):
        with open(clips_csv_path) as f:
            for row in csv.DictReader(f):
                res = (row.get("resolution") or "").strip().lower()
                wh = _CHILDPLAY_RES_MAP.get(res)
                if wh is not None:
                    clip_wh[row["clip"]] = wh

    samples: List[Dict[str, Any]] = []
    # Fall back to PIL-opening the first frame if clips.csv didn't cover
    # the clip; the resulting dims will be the downscaled image's, which
    # works as long as the annotations are also in that frame.
    _img_size_cache: Dict[str, tuple] = {}

    for csv_name in sorted(os.listdir(ann_dir)):
        if not csv_name.endswith(".csv"):
            continue
        clip = csv_name[:-4]
        try:
            stem, frange = clip.rsplit("_", 1)
            start = int(frange.split("-")[0])
        except (ValueError, IndexError):
            stem, start = clip, 0

        with open(os.path.join(ann_dir, csv_name)) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rel_frame = int(row["frame"])
                except (KeyError, ValueError):
                    continue
                abs_frame = start + rel_frame - 1
                img_rel = os.path.join("images", clip,
                                        f"{stem}_{abs_frame}.jpg")

                w, h = clip_wh.get(clip, (0, 0))
                if w <= 0 or h <= 0:
                    if clip not in _img_size_cache:
                        try:
                            with Image.open(os.path.join(data_root, img_rel)) as im:
                                _img_size_cache[clip] = im.size
                        except (OSError, FileNotFoundError):
                            _img_size_cache[clip] = (0, 0)
                    w, h = _img_size_cache[clip]
                if w <= 0 or h <= 0:
                    continue

                try:
                    bx = float(row["bbox_x"]);     by = float(row["bbox_y"])
                    bw = float(row["bbox_width"]); bh = float(row["bbox_height"])
                    gx = float(row["gaze_x"]);     gy = float(row["gaze_y"])
                except (KeyError, ValueError):
                    continue
                x1, y1 = bx, by
                x2, y2 = bx + bw, by + bh
                bbox_norm = [
                    max(0.0, min(1.0, x1 / w)),
                    max(0.0, min(1.0, y1 / h)),
                    max(0.0, min(1.0, x2 / w)),
                    max(0.0, min(1.0, y2 / h)),
                ]

                gaze_class = row.get("gaze_class", "").lower()
                inout = 1 if (gaze_class == "inside_visible" and gx >= 0) else 0

                head = {
                    "bbox_norm":  bbox_norm,
                    "gazex_norm": [max(0.0, min(1.0, gx / w))] if inout == 1 else [],
                    "gazey_norm": [max(0.0, min(1.0, gy / h))] if inout == 1 else [],
                    "inout":      inout,
                }
                samples.append(_emit_head_sample(
                    img_rel, head, w, h, data_root, depth_dir))

    return samples


def load_dataset_samples(dataset: str, data_root: str, depth_dir: str,
                          split_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Resolve the annotations file for the requested dataset and return a
    flat list of evaluation samples.

    ChildPlay's official release ships per-clip CSV annotations rather than
    a single preprocessed JSON; we read those directly under
    ``<data_root>/annotations/test/<clip>.csv`` unless a ``--split_file``
    override is supplied.
    """
    if dataset == "childplay" and split_file is None:
        # Default to the raw upstream layout (annotations/test/*.csv).
        if not os.path.isfile(os.path.join(data_root, "test_preprocessed.json")):
            return _childplay_csv_to_samples(data_root, "test", depth_dir)
        # If a preprocessed JSON happens to exist next to the upstream
        # files, fall through to use it.

    if split_file is None:
        defaults = {
            "gazefollow": "test_preprocessed.json",
            "vat":        "test_preprocessed.json",
            "childplay":  "test_preprocessed.json",
            "goosynth":   "goosynth_test_preprocess.json",
            "gooreal":    "gooreal_test_preprocess.json",
        }
        split_file = defaults[dataset]
    json_path = (split_file if os.path.isabs(split_file)
                 else os.path.join(data_root, split_file))
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"Annotations not found: {json_path}.  Pass --split_file to override.")
    return _preprocessed_json_to_samples(json_path, data_root, depth_dir)


# ============================================================================
# 3.  Depth helpers — cache + on-the-fly DA2 fallback
# ============================================================================

class DepthSource:
    """Provide a per-sample [H_d, W_d] *per-image min-max-normalised* depth map.

    Resolution order:
        1. Cached ``.npy`` at ``sample['depth_path']``.
        2. On-the-fly DepthAnythingV2 inference (if --depth_on_the_fly).
        3. None — 3-D metrics are skipped for that sample.
    """

    def __init__(self, depth_on_the_fly: bool, da2_encoder: str,
                 da2_ckpt_dir: str, device: torch.device):
        self.depth_on_the_fly = depth_on_the_fly
        self.da2_encoder = da2_encoder
        self.da2_ckpt_dir = da2_ckpt_dir
        self.device = device
        self._da2 = None  # lazy-init

    def _ensure_da2(self):
        if self._da2 is not None:
            return
        sys.path.append("../Depth-Anything-V2")
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
        cfgs = {
            "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        model = DepthAnythingV2(**cfgs[self.da2_encoder])
        ckpt = os.path.join(self.da2_ckpt_dir,
                            f"depth_anything_v2_{self.da2_encoder}.pth")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"DA2 checkpoint not found: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self._da2 = model.to(self.device).eval()

    @staticmethod
    def _minmax_norm(d: np.ndarray) -> np.ndarray:
        d = d.astype(np.float32)
        d_min, d_max = float(d.min()), float(d.max())
        return (d - d_min) / ((d_max - d_min) + 1e-8)

    def get(self, sample: Dict[str, Any], data_root: str,
            target_hw: Tuple[int, int] = (64, 64)) -> Optional[np.ndarray]:
        path = sample.get("depth_path") or ""
        depth_full: Optional[np.ndarray] = None
        if path and os.path.isfile(path):
            depth_full = np.load(path).astype(np.float32)
        elif self.depth_on_the_fly:
            self._ensure_da2()
            import cv2  # type: ignore
            img_bgr = cv2.imread(os.path.join(data_root, sample["path"]),
                                  cv2.IMREAD_COLOR)
            if img_bgr is None:
                return None
            with torch.inference_mode():
                depth_full = self._da2.infer_image(img_bgr, 518)
        if depth_full is None:
            return None

        # Resize to the evaluation resolution (default 64×64), then per-image
        # min-max-normalise to [0, 1].
        H, W = target_hw
        if depth_full.shape != (H, W):
            t = torch.from_numpy(depth_full).float()
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0).unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False)
            depth_full = t.squeeze().numpy()
        return self._minmax_norm(depth_full)


# ============================================================================
# 4.  Eval dataset (model-agnostic) and collate
# ============================================================================

class EvalDataset(Dataset):
    """Returns a dict per sample; the model-specific handle is responsible
    for stacking / cropping the RGB tensor it actually needs.

    Item:
        image_pil  : PIL.Image (RGB)
        sample     : Dict[str, Any]  (see load_dataset_samples)
    """

    def __init__(self, samples: List[Dict[str, Any]], data_root: str):
        self.samples = samples
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(os.path.join(self.data_root, s["path"])).convert("RGB")
        return img, s


def collate_eval(batch):
    imgs, samples = zip(*batch)
    return list(imgs), list(samples)


# ============================================================================
# 5.  Model handles — one ``ModelHandle`` per architecture, all expose the
#     same ``predict(images, samples) -> {'heatmap': [B,64,64], 'inout': [B]}``.
# ============================================================================

@dataclass
class ModelHandle:
    arch: str
    model: nn.Module
    transform: Callable
    device: torch.device
    has_inout: bool = False
    input_size: int = 448
    # Per-handle override for predict(); set in build_model().
    _predict_fn: Optional[Callable] = None

    def predict(self, images: List[Image.Image],
                samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        assert self._predict_fn is not None, "predict_fn not set"
        return self._predict_fn(self, images, samples)


def _predict_gazelle_style(handle: ModelHandle,
                            images: List[Image.Image],
                            samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Forward path for Gazelle / GazeMoE / GT3D models — shared signature
    ``model({'images', 'bboxes'})`` returning ``{'heatmap': [List[B,64,64]],
    'inout': Optional[List[B]]}``.

    Test-time convention: one head per sample (already flattened by the
    EvalDataset).  We therefore wrap each bbox in a singleton list, as the
    in-house models expect ``bboxes`` to be ``List[List[bbox]]`` (per-image,
    per-head).
    """
    img_t = torch.stack([handle.transform(im) for im in images]).to(handle.device)
    bboxes = [[s["bbox_norm"]] for s in samples]
    with torch.no_grad():
        out = handle.model({"images": img_t, "bboxes": bboxes})
    # ``out['heatmap']``: list of length B, each [1, 64, 64] (one head).
    heatmaps = torch.stack([h[0].cpu() for h in out["heatmap"]])
    inout: Optional[torch.Tensor] = None
    if handle.has_inout and "inout" in out and out["inout"] is not None:
        inout = torch.stack([
            (o[0] if torch.is_tensor(o) and o.ndim >= 1 else o).detach().cpu()
            for o in out["inout"]
        ]).float().view(-1)
    return {"heatmap": heatmaps, "inout": inout}


def _predict_reproduction(handle: ModelHandle,
                           images: List[Image.Image],
                           samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Forward path for the four reproduced prior works.  Each model's
    ``forward_eval(images, samples)`` returns a dict with ``heatmap`` and
    optionally ``inout``.  The model classes live in
    ``network/network_builder_<arch>.py``.
    """
    if not hasattr(handle.model, "forward_eval"):
        raise NotImplementedError(
            f"{handle.arch} model lacks a ``forward_eval`` method.  "
            "Either implement it in the corresponding builder file, or "
            "switch the predict_fn in ``build_model``.")
    with torch.no_grad():
        out = handle.model.forward_eval(
            images=images, samples=samples,
            transform=handle.transform, device=handle.device)
    return out


def _make_minimal_config(arch: str) -> Dict[str, Any]:
    """Build the smallest possible config dict the existing in-house factories
    require, so we can keep this script free of configuration.yaml.
    """
    # The in-house GazeMoE / GT3D factories read several deep keys; mirror
    # the same names but with safe defaults.  Backbone choice is implied by
    # the arch name.
    base = {
        "encoder":  {"type": "DINOv2", "pretrained": True},
        "decoder":  {"hidden_size": 256, "depth": 3, "num_heads": 8, "dropout": 0.1},
        "mlp_ratio": 1,
        "num_experts": 4,
        "num_shared_experts": 1,
        "top_k": 2,
        "moe_type": "moe",
        "is_msf": 1,
        "use_anchored_depth": True,
        "use_dense_depth":    True,
    }
    if arch.startswith("gazelle"):
        # ``gazelle_vitl14_inout`` → ``gazelle_dinov2_vitl14_inout``.
        suffix = arch.replace("gazelle_", "")
        base["name"] = f"gazelle_dinov2_{suffix}"
    elif arch == "gazemoe_vitl14_inout":
        base["name"] = "gazemoe_dinov2_vitl14_inout"
    #elif arch == "gt3d_vitl14_inout":
    #    base["name"] = "gt3d_dinov2_vitl14_inout"
    else:
        base["name"] = arch
    return {"model": base}


def build_model(arch: str, checkpoint: Optional[str],
                device: torch.device, include_backbone: bool = False,
                upstream_root: Optional[str] = None,
                gatector_input_size: int = 224,
                ) -> ModelHandle:
    """Build the model + transform for the requested arch and load weights.

    Architectures handled here:
        gazelle_*          → network/network_builder.get_gazelle_model
        gazemoe_vitl14_inout → network/network_builder_update2.get_gazemoe_model
        gt3d_vitl14_inout  → network/network_builder_gt3d.get_gt3d_model
        chong              → network/network_builder_chong.get_chong_model
        dam                → network/network_builder_dam.get_dam_model
        sharingan          → network/network_builder_sharingan.get_sharingan_model
        gatector           → network/network_builder_gatector.get_gatector_model
    """
    cfg = _make_minimal_config(arch)
    has_inout = arch.endswith("_inout") or arch in {"chong", "dam", "sharingan", "gatector"}

    if arch.startswith("gazelle"):
        from network.network_builder import get_gazelle_model
        model, transform = get_gazelle_model(cfg)
        predict_fn = _predict_gazelle_style
    elif arch == "gazemoe_vitl14_inout":
        from network.network_builder_update2 import get_gazemoe_model
        model, transform = get_gazemoe_model(cfg)
        predict_fn = _predict_gazelle_style
    elif arch == "gt3d_vitl14_inout":
        from network.network_builder_gt3d import get_gt3d_model
        model, transform = get_gt3d_model(cfg)
        predict_fn = _predict_gazelle_style
    elif arch == "chong":
        from network.network_builder_chong import get_chong_model
        model, transform = get_chong_model()
        predict_fn = _predict_reproduction
    elif arch == "dam":
        from network.network_builder_dam import get_dam_model
        model, transform = get_dam_model()
        predict_fn = _predict_reproduction
    elif arch == "sharingan":
        from network.network_builder_sharingan import get_sharingan_model
        kw: Dict[str, Any] = {}
        if upstream_root:
            kw["upstream_root"] = upstream_root
        if checkpoint:
            kw["ckpt_path"] = checkpoint
        model, transform = get_sharingan_model(**kw)
        predict_fn = _predict_reproduction
        checkpoint = None      # factory has already loaded the ckpt
    elif arch == "gatector":
        from network.network_builder_gatector import get_gatector_model
        kw = {"input_size": gatector_input_size}
        if upstream_root:
            kw["upstream_root"] = upstream_root
        if checkpoint:
            kw["ckpt_path"] = checkpoint
        model, transform = get_gatector_model(**kw)
        predict_fn = _predict_reproduction
        checkpoint = None      # factory has already loaded the ckpt
    else:
        raise ValueError(f"Unknown arch: {arch}")

    # Generic loader for arches where the factory didn't already load the
    # checkpoint (in-house Gazelle / GazeMoE / GT3D, plus Chong / DAM).
    if checkpoint and os.path.isfile(checkpoint):
        sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if hasattr(model, "load_gazelle_state_dict"):
            model.load_gazelle_state_dict(sd, include_backbone=include_backbone)
        else:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f"[warn] partial state_dict load: "
                      f"missing={len(missing)}  unexpected={len(unexpected)}")
        print(f"Loaded checkpoint from {checkpoint}")
    elif checkpoint:
        print(f"[warn] checkpoint path '{checkpoint}' not found — "
              f"running with random-init weights.")

    model = model.to(device).eval()
    handle = ModelHandle(
        arch=arch, model=model, transform=transform, device=device,
        has_inout=has_inout,
    )
    handle._predict_fn = predict_fn
    return handle


# ============================================================================
# 6.  Metrics (2-D + 3-D-proxy) — operate on a single sample at a time.
# ============================================================================

def _sample_depth(depth_norm: np.ndarray, x_norm: float, y_norm: float) -> float:
    H, W = depth_norm.shape
    x = int(np.clip(int(float(x_norm) * W), 0, W - 1))
    y = int(np.clip(int(float(y_norm) * H), 0, H - 1))
    return float(depth_norm[y, x])


def _argmax_norm_xy(heatmap_np: np.ndarray) -> Tuple[float, float]:
    H, W = heatmap_np.shape
    idx = int(heatmap_np.flatten().argmax())
    py, px = np.unravel_index(idx, (H, W))
    return float(px) / W, float(py) / H


def l2_3d_proxy(heatmap_np: np.ndarray, gazex: List[float], gazey: List[float],
                depth_norm: np.ndarray) -> float:
    pred_x, pred_y = _argmax_norm_xy(heatmap_np)
    gt_x, gt_y = float(np.mean(gazex)), float(np.mean(gazey))
    d_pred = _sample_depth(depth_norm, pred_x, pred_y)
    d_gt   = _sample_depth(depth_norm, gt_x,   gt_y)
    return float(math.sqrt(
        (pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2 + (d_pred - d_gt) ** 2
    ))


def angle_3d_proxy(heatmap_np: np.ndarray, gazex: List[float], gazey: List[float],
                    head_cx: float, head_cy: float,
                    depth_norm: np.ndarray) -> float:
    pred_x, pred_y = _argmax_norm_xy(heatmap_np)
    gt_x, gt_y = float(np.mean(gazex)), float(np.mean(gazey))
    d_pred = _sample_depth(depth_norm, pred_x, pred_y)
    d_gt   = _sample_depth(depth_norm, gt_x,   gt_y)
    d_head = _sample_depth(depth_norm, head_cx, head_cy)
    pv = np.array([pred_x - head_cx, pred_y - head_cy, d_pred - d_head],
                  dtype=np.float64)
    gv = np.array([gt_x   - head_cx, gt_y   - head_cy, d_gt   - d_head],
                  dtype=np.float64)
    cos = float(np.dot(pv, gv) /
                ((np.linalg.norm(pv) + 1e-8) * (np.linalg.norm(gv) + 1e-8)))
    cos = max(-1.0, min(1.0, cos))
    return float(math.degrees(math.acos(cos)))


def doa_proxy(heatmap_np: np.ndarray, gazex: List[float], gazey: List[float],
              head_cx: float, head_cy: float,
              depth_norm: np.ndarray, eps: float = 1e-3) -> Tuple[bool, bool]:
    """Single-sample DOA — returns (is_correct, is_valid)."""
    pred_x, pred_y = _argmax_norm_xy(heatmap_np)
    gt_x, gt_y = float(np.mean(gazex)), float(np.mean(gazey))
    d_pred = _sample_depth(depth_norm, pred_x, pred_y)
    d_gt   = _sample_depth(depth_norm, gt_x,   gt_y)
    d_head = _sample_depth(depth_norm, head_cx, head_cy)
    if abs(d_gt - d_head) < eps:
        return False, False
    return (d_pred - d_head) * (d_gt - d_head) > 0.0, True


# ============================================================================
# 7.  Eval loop
# ============================================================================

@dataclass
class MetricSink:
    aucs:        List[float] = field(default_factory=list)
    avg_l2s:     List[float] = field(default_factory=list)
    l2_3ds:      List[float] = field(default_factory=list)
    angle_3ds:   List[float] = field(default_factory=list)
    doa_correct: int = 0
    doa_valid:   int = 0
    inout_pred:  List[float] = field(default_factory=list)
    inout_gt:    List[int]   = field(default_factory=list)
    n_eval:      int = 0
    n_no_depth:  int = 0


def _per_sample_eval(heatmap_np: np.ndarray, sample: Dict[str, Any],
                      depth_norm: Optional[np.ndarray],
                      dataset: str, sink: MetricSink) -> None:
    """Accumulate all metrics for a single (heatmap, sample) pair."""
    gazex, gazey = sample["gazex_norm"], sample["gazey_norm"]
    if not gazex or not gazey:
        return

    # --- 2-D ----------
    # Wrap AUC/L2 in a guard: degenerate annotations (e.g. ChildPlay
    # frames with gaze clipped to the image border) can collapse the
    # tolerance rectangle and trip ``roc_auc_score`` with a
    # "Only one class present in y_true" ValueError.  Skip the sample's
    # AUC contribution in that case; the L2 (argmax-based) still works.
    hm_t = torch.from_numpy(heatmap_np).float()
    if dataset in {"gazefollow", "goosynth", "gooreal"}:
        try:
            sink.aucs.append(gazefollow_auc(
                hm_t, gazex, gazey, sample["height"], sample["width"]))
        except ValueError:
            pass
        try:
            avg_l2, _ = gazefollow_l2(hm_t, gazex, gazey)
            sink.avg_l2s.append(avg_l2)
        except (ValueError, IndexError):
            pass
    else:  # vat / childplay — rectangular AUC, single-annotator L2
        try:
            sink.aucs.append(vat_auc(hm_t, gazex[0], gazey[0]))
        except ValueError:
            pass
        try:
            sink.avg_l2s.append(vat_l2(hm_t, gazex[0], gazey[0]))
        except (ValueError, IndexError):
            pass

    # --- 3-D ----------
    if depth_norm is None:
        sink.n_no_depth += 1
        return
    bbox = sample["bbox_norm"]
    hcx = (float(bbox[0]) + float(bbox[2])) * 0.5
    hcy = (float(bbox[1]) + float(bbox[3])) * 0.5
    sink.l2_3ds.append(l2_3d_proxy(heatmap_np, gazex, gazey, depth_norm))
    sink.angle_3ds.append(angle_3d_proxy(heatmap_np, gazex, gazey,
                                          hcx, hcy, depth_norm))
    is_corr, is_val = doa_proxy(heatmap_np, gazex, gazey,
                                 hcx, hcy, depth_norm)
    if is_val:
        sink.doa_valid   += 1
        sink.doa_correct += int(bool(is_corr))


def evaluate(handle: ModelHandle, samples: List[Dict[str, Any]],
              data_root: str, dataset: str, depth_src: DepthSource,
              batch_size: int, num_workers: int) -> Dict[str, Any]:
    ds = EvalDataset(samples, data_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                     num_workers=num_workers, collate_fn=collate_eval,
                     pin_memory=False)

    sink = MetricSink()
    t0 = time.time()
    for images, batch_samples in tqdm(dl, total=len(dl),
                                       desc=f"{handle.arch} on {dataset}"):
        out = handle.predict(images, batch_samples)
        heatmaps = out["heatmap"]   # CPU tensor [B, 64, 64]
        inout = out.get("inout", None)

        for i, sample in enumerate(batch_samples):
            inout_i = int(sample.get("inout", 1) or 0)
            if inout_i == 1:
                hm_np = heatmaps[i].numpy()
                depth_norm = depth_src.get(sample, data_root, target_hw=(64, 64))
                _per_sample_eval(hm_np, sample, depth_norm, dataset, sink)
                sink.n_eval += 1

            # Inout AP — only collect when the model has an inout head and
            # the dataset provides a ground-truth label.
            if inout is not None and sample.get("inout") is not None:
                sink.inout_pred.append(float(inout[i].item()))
                sink.inout_gt.append(int(sample["inout"]))

    elapsed = time.time() - t0
    AUC      = float(np.mean(sink.aucs))      if sink.aucs      else float("nan")
    AvgL2    = float(np.mean(sink.avg_l2s))   if sink.avg_l2s   else float("nan")
    L2_3D    = float(np.mean(sink.l2_3ds))    if sink.l2_3ds    else float("nan")
    Angle_3D = float(np.mean(sink.angle_3ds)) if sink.angle_3ds else float("nan")
    DOA      = (100.0 * sink.doa_correct / sink.doa_valid) if sink.doa_valid else float("nan")
    AP: Optional[float] = None
    if sink.inout_pred and sink.inout_gt and (set(sink.inout_gt) != {0} and set(sink.inout_gt) != {1}):
        AP = float(average_precision_score(sink.inout_gt, sink.inout_pred))

    return {
        "n_samples":      len(samples),
        "n_eval_in_frame": sink.n_eval,
        "n_no_depth":     sink.n_no_depth,
        "AUC":            AUC,
        "AvgL2":          AvgL2,
        "L2_3D":          L2_3D,
        "Angle_3D_deg":   Angle_3D,
        "DOA_percent":    DOA,
        "AP_inout":       AP,
        "elapsed_sec":    elapsed,
    }


# ============================================================================
# 8.  Pretty-printer
# ============================================================================

def print_summary(arch: str, dataset: str, checkpoint: str,
                   metrics: Dict[str, Any]) -> None:
    print()
    print("─" * 78)
    print(f"  {arch}  ─  {dataset}  ─  {checkpoint}")
    print("─" * 78)

    def _fmt(v, digits=4, suffix=""):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "  n/a"
        return f"{v:.{digits}f}{suffix}"

    print(f"  N samples (total)      : {metrics['n_samples']}")
    print(f"  N samples (in-frame)   : {metrics['n_eval_in_frame']}")
    if metrics["n_no_depth"]:
        print(f"  N samples (no depth)   : {metrics['n_no_depth']}")
    print()
    print(f"  AUC                    : {_fmt(metrics['AUC'])}")
    print(f"  AvgL2                  : {_fmt(metrics['AvgL2'])}")
    print(f"  L2-3D  (proxy)         : {_fmt(metrics['L2_3D'])}")
    print(f"  Angle-3D (proxy, deg)  : {_fmt(metrics['Angle_3D_deg'], 2, '°')}")
    print(f"  DOA       (%)          : {_fmt(metrics['DOA_percent'], 2, '%')}")
    if metrics["AP_inout"] is not None:
        print(f"  AP (in/out)            : {_fmt(metrics['AP_inout'])}")
    print(f"  Elapsed                : {metrics['elapsed_sec']:.1f}s")
    print("─" * 78)


# ============================================================================
# 9.  Main
# ============================================================================

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # ---- Samples -----
    samples = load_dataset_samples(
        args.dataset, args.data_root, args.depth_dir, args.split_file)
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]
        print(f"--limit applied: evaluating {len(samples)} samples only")
    print(f"Loaded {len(samples)} samples from "
          f"{os.path.join(args.data_root, args.split_file or '<default>')}")

    # ---- Depth source ----------------------------------------------------
    # 1. Empty / none / off depth_dir ⇒ auto-enable on-the-fly DA2 inference.
    # 2. Else, probe the first few samples' resolved .npy paths.  If none
    #    exists, on-the-fly DA2 is enabled with a clear warning (and an
    #    explanation if on-the-fly was not requested explicitly).
    cache_disabled = _depth_cache_disabled(args.depth_dir)
    depth_on_the_fly = bool(args.depth_on_the_fly)
    if args.no_depth:
        depth_on_the_fly = False
        cache_disabled   = True
        for s in samples:
            s["depth_path"] = None
        print("[depth] --no_depth set ⇒ skipping all depth lookup/inference "
              "(3-D metrics will report 'n/a').")
    elif cache_disabled and not depth_on_the_fly:
        depth_on_the_fly = True
        print("[depth] --depth_dir is empty/none/off ⇒ "
              "auto-enabling on-the-fly DepthAnythingV2 inference.")
    elif not cache_disabled:
        probed = [s for s in samples[:32]
                  if s.get("depth_path") and os.path.isfile(s["depth_path"])]
        n_probe = min(32, len(samples))
        if not probed and n_probe > 0:
            example = samples[0].get("depth_path", "<unresolved>")
            print(f"[depth] WARNING — no cached depth .npy found under "
                  f"--depth_dir='{args.depth_dir}' (probed {n_probe} samples, "
                  f"e.g. {example!r}).")
            if not depth_on_the_fly:
                depth_on_the_fly = True
                print("[depth] Auto-enabling on-the-fly DepthAnythingV2 "
                      "inference for this run.  Pass --depth_dir '' to make "
                      "this explicit, or run preprocess_Depth.py to cache "
                      "the maps for future runs.")
        else:
            print(f"[depth] Using cached depth at "
                  f"<data_root>/{args.depth_dir}/  "
                  f"({len(probed)}/{n_probe} probed samples have a .npy).")
    if depth_on_the_fly:
        print(f"[depth] On-the-fly engine: DepthAnythingV2-{args.da2_encoder} "
              f"(ckpt dir: {args.da2_ckpt_dir})")

    depth_src = DepthSource(
        depth_on_the_fly=depth_on_the_fly,
        da2_encoder=args.da2_encoder,
        da2_ckpt_dir=args.da2_ckpt_dir,
        device=device,
    )

    # ---- Model -----
    handle = build_model(
        arch=args.arch, checkpoint=args.checkpoint,
        device=device, include_backbone=args.include_backbone,
        upstream_root=args.upstream_root,
        gatector_input_size=args.gatector_input_size,
    )

    # ---- Eval -----
    metrics = evaluate(handle, samples, args.data_root, args.dataset,
                        depth_src, args.batch_size, args.num_workers)

    print_summary(args.arch, args.dataset, args.checkpoint, metrics)

    if args.out:
        report = {
            "arch": args.arch, "dataset": args.dataset,
            "checkpoint": args.checkpoint, "data_root": args.data_root,
            "split_file": args.split_file,  **metrics,
        }
        Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote JSON report to {args.out}")


if __name__ == "__main__":
    main()
