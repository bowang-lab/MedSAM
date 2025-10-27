# src/imgpipe/augment.py
from __future__ import annotations

"""
YOLO Augmenter with full-image random rotations + jitter/noise/color AND crops.

- Processes a YOLO dataset laid out as:
    data_root/{images,labels}/{train,val,test}
- Writes an augmented copy to out_root with the same layout.
- Per-image randomness: every image (and every augmented copy) is randomized;
  augmentations are NOT identical across images.

Functionality
------------
A) Full-image pipeline (randomized each time)
   - Flips (H/V)
   - Rotations: RandomRotate90 & arbitrary-angle Rotate
   - Mild affine jitter (scale/translate/rotate/shear)
   - Optional RandomResizedCrop (off by default)
   - Color jitter
   - Color "filter" (HSV or RGB shift)
   - Noise (Gaussian or ISO)

B) Crops (kept as requested)
   - Fixed-size tiling
   - Object-centric zoom crops (with jitter)
   - Multi-scale sliding zoom sweep
   Crops use a unified policy to decide whether to keep a crop and
   how to keep/clip boxes: object_policy ∈ {"full","partial","negative_ok"}.

C) Outputs for each input image:
   - Passthrough original
   - `full_rotate_n` extra full-image rotation-only variants
   - `multiplier` general augmented variants
   - If *_from_aug is True, crops can also be produced from each augmented image.

Usage
-----
from pathlib import Path
from src.imgpipe.augment import Augmentor

aug = Augmentor()  # uses DEFAULT_CFG below
aug.run_from_yolo_root(
    data_root=Path("/path/to/yolo_split"),
    out_root=Path("/path/to/yolo_aug"),
    splits=("train",),
    prefer_copy=False,
)
"""

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal

import albumentations as A
import cv2
import numpy as np
import yaml

from src.utils import ensure_dir


# ----------------------------- small I/O helpers -----------------------------

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _expand(p: Path | str | None) -> Optional[Path]:
    if p in (None, "", False):
        return None
    return Path(os.path.expanduser(str(p))).resolve()


def _list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in _IMG_EXTS])


def _read_image(path: Path) -> Optional["cv2.Mat"]:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _write_image(path: Path, image: "cv2.Mat") -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image)


# ----------------------------- YOLO label helpers ----------------------------

def _read_yolo_labels(lbl_path: Path) -> Tuple[List[List[float]], List[int]]:
    """
    Return (boxes, labels) where boxes are [cx,cy,w,h] normalized to [0,1].
    Missing file ⇒ empty labels.
    """
    boxes, labels = [], []
    if not lbl_path.exists():
        return boxes, labels
    lines = [ln.strip() for ln in lbl_path.read_text().splitlines() if ln.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        boxes.append([x, y, w, h])
        labels.append(cls)
    return boxes, labels


def _write_yolo_labels(lbl_path: Path, boxes: List[List[float]], labels: List[int]) -> None:
    ensure_dir(lbl_path.parent)
    if not boxes:
        lbl_path.write_text("")  # explicit negatives ok
        return
    lbl_path.write_text(
        "\n".join(f"{l} " + " ".join(f"{x:.6f}" for x in b) for b, l in zip(boxes, labels)) + "\n"
    )


def _sanitize_yolo_boxes(boxes: List[List[float]]) -> List[List[float]]:
    """Clamp to [0,1] via xyxy->clamp->yolo; drop degenerate."""
    def _clamp(v, lo, hi): return max(lo, min(hi, v))
    out = []
    for cx, cy, w, h in boxes:
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        x1 = _clamp(x1, 0.0, 1.0); y1 = _clamp(y1, 0.0, 1.0)
        x2 = _clamp(x2, 0.0, 1.0); y2 = _clamp(y2, 0.0, 1.0)
        w2 = max(0.0, x2 - x1); h2 = max(0.0, y2 - y1)
        if w2 <= 1e-6 or h2 <= 1e-6:
            continue
        out.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0, w2, h2])
    return out


# ----------------------------- geometry helpers -----------------------------

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _yolo_to_xyxy_abs(box_yolo, W, H):
    cx, cy, w, h = box_yolo
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return [x1, y1, x2, y2]


def _xyxy_abs_to_yolo(box_xyxy, W, H):
    x1, y1, x2, y2 = box_xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return None
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return [cx / W, cy / H, bw / W, bh / H]


def _box_intersection(a_xyxy, b_xyxy):
    ax1, ay1, ax2, ay2 = a_xyxy
    bx1, by1, bx2, by2 = b_xyxy
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, None
    return (x2 - x1) * (y2 - y1), [x1, y1, x2, y2]


def _square_crop_bounds(cx, cy, side, W, H):
    half = side / 2.0
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = x0 + int(round(side))
    y1 = y0 + int(round(side))
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > W:
        shift = x1 - W
        x0 = max(0, x0 - shift)
        x1 = W
    if y1 > H:
        shift = y1 - H
        y0 = max(0, y0 - shift)
        y1 = H
    x0 = _clamp(x0, 0, max(0, W - 1))
    y0 = _clamp(y0, 0, max(0, H - 1))
    x1 = _clamp(x1, 1, W)
    y1 = _clamp(y1, 1, H)
    return [x0, y0, x1, y1]

def _make_gaussian_noise(var_limit: Tuple[float, float] | List[float] | float, mean: float = 0.0):
    """
    Construct a Gaussian noise transform that works across Albumentations versions.
    Tries GaussNoise(var_limit=...) first, then GaussianNoise(sigma_limit=...).
    """
    # 1) Try GaussNoise(var_limit=..., mean=...)
    try:
        return A.GaussNoise(var_limit=var_limit, mean=mean, p=1.0)
    except Exception:
        pass

    # 2) Try GaussianNoise(sigma_limit=..., mean=...). Convert variance -> sigma if needed.
    try:
        if isinstance(var_limit, (list, tuple)) and len(var_limit) == 2:
            sigma_limit = (float(var_limit[0]) ** 0.5, float(var_limit[1]) ** 0.5)
        else:
            sigma = float(var_limit) ** 0.5
            sigma_limit = (sigma, sigma)
        return A.GaussianNoise(sigma_limit=sigma_limit, mean=mean, p=1.0)
    except Exception:
        pass

    # 3) Last resort: minimal constructor without args
    try:
        return A.GaussianNoise(p=1.0)
    except Exception:
        # If even this fails, return a no-op to avoid breaking the pipeline
        return A.NoOp(p=1.0)


# ----------------------------- in-code config -------------------------------

ObjectPolicy = Literal["full", "partial", "negative_ok"]

DEFAULT_CFG: Dict[str, Any] = {
    "out_ext": ".jpg",
    "multiplier": 3,                 # number of general augmented copies per image
    "full_rotate_n": 1,              # extra full-image rotation-only variants
    "include_images_without_labels": False,
    "seed": 1337,                    # set None for fully non-deterministic
    "write_yaml": True,

    # Unified crop policy
    #  - full        : require at least 1 fully-contained object; keep only fully contained boxes
    #  - partial     : allow clipped boxes; require ≥ min_visibility overlap with at least 1 object
    #  - negative_ok : allow crops with no objects (emit empty labels)
    "object_policy": "partial",
    "min_visibility": 0.20,

    # General transforms
    "transform": {
        "hflip_p": 0.5,
        "vflip_p": 0.0,

        # rotations
        "rotate90_p": 0.25,          # k*90° rotations
        "rot_p": 0.40,               # arbitrary-angle rotation probability
        "rot_limit": [-25, 25],      # degrees

        # mild geometric jitter
        "affine_p": 0.60,
        "affine_scale": [0.95, 1.05],
        "affine_translate": [-0.03, 0.03],
        "affine_rotate": [-10, 10],
        "affine_shear": [-4, 4],

        # RandomResizedCrop (off by default; keep full frame)
        "rrc_p": 0.0,
        "rrc_scale": [0.9, 1.0],
        "rrc_ratio": [0.9, 1.1],

        # photometrics: jitter + color "filter"
        "cj_p": 0.50,
        "cj_brightness": 0.20,
        "cj_contrast": 0.20,
        "cj_saturation": 0.20,
        "cj_hue": 0.05,

        "color_filter_p": 0.30,      # OneOf: HSV or RGB shift
        "noise_p": 0.30,             # OneOf: Gaussian / ISO noise
        "gauss_noise_var": [5.0, 25.0],
    },

    # CROPS
    "tiling": {
        "enable": True,
        "tile_size": 512,
        "overlap": 0.40,
        "from_aug": False,           # also tile augmented images
    },
    "zoom_crops": {
        "enable": True,
        "scales": [0.35, 0.5, 0.7],
        "per_obj": 2,
        "on": "both",                # disc | cup | both | any
        "out_size": 640,
        "jitter": 0.05,
        "from_aug": False,           # also do from augmented images
    },
    "zoom_sweep": {
        "enable": True,
        "scales": [0.35, 0.5, 0.7],
        "overlap": 0.25,
        "out_size": 640,
        "from_aug": False,           # also do from augmented images
    },

    # label names for data.yaml
    "names": ["disc", "cup"],
}


# ----------------------------- settings dataclass ----------------------------

@dataclass
class Settings:
    out_ext: str
    multiplier: int
    full_rotate_n: int
    include_images_without_labels: bool
    seed: Optional[int]
    write_yaml: bool

    object_policy: ObjectPolicy
    min_visibility: float

    # transform
    hflip_p: float
    vflip_p: float
    rotate90_p: float
    rot_p: float
    rot_limit: Tuple[int, int]
    affine_p: float
    affine_scale: Tuple[float, float]
    affine_translate: Tuple[float, float]
    affine_rotate: Tuple[float, float]
    affine_shear: Tuple[float, float]
    rrc_p: float
    rrc_scale: Tuple[float, float]
    rrc_ratio: Tuple[float, float]
    cj_p: float
    cj_brightness: float
    cj_contrast: float
    cj_saturation: float
    cj_hue: float
    color_filter_p: float
    noise_p: float
    gauss_noise_var: Tuple[float, float]

    # crops
    tiling_enable: bool
    tiling_tile_size: int
    tiling_overlap: float
    tiling_from_aug: bool

    zc_enable: bool
    zc_scales: Tuple[float, ...]
    zc_per_obj: int
    zc_on: str
    zc_out_size: int
    zc_jitter: float
    zc_from_aug: bool

    zs_enable: bool
    zs_scales: Tuple[float, ...]
    zs_overlap: float
    zs_out_size: int
    zs_from_aug: bool

    names: List[str]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Settings":
        t = d.get("transform", {}) or {}
        til = d.get("tiling", {}) or {}
        zc = d.get("zoom_crops", {}) or {}
        zs = d.get("zoom_sweep", {}) or {}
        return Settings(
            out_ext=str(d.get("out_ext", ".jpg")),
            multiplier=int(d.get("multiplier", 2)),
            full_rotate_n=int(d.get("full_rotate_n", 0)),
            include_images_without_labels=bool(d.get("include_images_without_labels", False)),
            seed=(int(d["seed"]) if d.get("seed", None) is not None else None),
            write_yaml=bool(d.get("write_yaml", True)),

            object_policy=str(d.get("object_policy", "partial")),
            min_visibility=float(d.get("min_visibility", 0.20)),

            hflip_p=float(t.get("hflip_p", 0.5)),
            vflip_p=float(t.get("vflip_p", 0.0)),
            rotate90_p=float(t.get("rotate90_p", 0.0)),
            rot_p=float(t.get("rot_p", 0.0)),
            rot_limit=tuple(t.get("rot_limit", [-20, 20])),
            affine_p=float(t.get("affine_p", 0.7)),
            affine_scale=tuple(t.get("affine_scale", [0.9, 1.1])),
            affine_translate=tuple(t.get("affine_translate", [-0.05, 0.05])),
            affine_rotate=tuple(t.get("affine_rotate", [-15, 15])),
            affine_shear=tuple(t.get("affine_shear", [-5, 5])),
            rrc_p=float(t.get("rrc_p", 0.5)),
            rrc_scale=tuple(t.get("rrc_scale", [0.9, 1.0])),
            rrc_ratio=tuple(t.get("rrc_ratio", [0.9, 1.1])),
            cj_p=float(t.get("cj_p", 0.5)),
            cj_brightness=float(t.get("cj_brightness", 0.2)),
            cj_contrast=float(t.get("cj_contrast", 0.2)),
            cj_saturation=float(t.get("cj_saturation", 0.2)),
            cj_hue=float(t.get("cj_hue", 0.1)),
            color_filter_p=float(t.get("color_filter_p", 0.3)),
            noise_p=float(t.get("noise_p", 0.3)),
            gauss_noise_var=tuple(t.get("gauss_noise_var", [5.0, 25.0])),

            tiling_enable=bool(til.get("enable", True)),
            tiling_tile_size=int(til.get("tile_size", 512)),
            tiling_overlap=float(til.get("overlap", 0.40)),
            tiling_from_aug=bool(til.get("from_aug", False)),

            zc_enable=bool(zc.get("enable", True)),
            zc_scales=tuple(float(x) for x in zc.get("scales", [0.35, 0.5, 0.7])),
            zc_per_obj=int(zc.get("per_obj", 2)),
            zc_on=str(zc.get("on", "both")),
            zc_out_size=int(zc.get("out_size", 640)),
            zc_jitter=float(zc.get("jitter", 0.05)),
            zc_from_aug=bool(zc.get("from_aug", False)),

            zs_enable=bool(zs.get("enable", True)),
            zs_scales=tuple(float(x) for x in zs.get("scales", [0.35, 0.5, 0.7])),
            zs_overlap=float(zs.get("overlap", 0.25)),
            zs_out_size=int(zs.get("out_size", 640)),
            zs_from_aug=bool(zs.get("from_aug", False)),

            names=list(d.get("names", ["disc", "cup"])),
        )


# ----------------------------- Augmentor -------------------------------------

class Augmentor:
    """
    Augments full images and generates crop-based variants.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, aug_yaml: Optional[Path] = None):
        """
        `config` (dict) overrides DEFAULT_CFG. `aug_yaml` is accepted for API
        compatibility but ignored (no external YAML is read).
        """
        cfg = dict(DEFAULT_CFG)
        if config:
            # shallow-merge for simplicity
            for k, v in config.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        self.settings = Settings.from_dict(cfg)

        # reproducibility if requested
        if self.settings.seed is not None:
            random.seed(self.settings.seed)
            np.random.seed(self.settings.seed)

    # -------- Albumentations builders --------

    def _build_general_transform(self, H: int, W: int) -> A.Compose:
        s = self.settings
        ops: List[A.BasicTransform] = []

        # Flips
        if s.hflip_p > 0:
            ops.append(A.HorizontalFlip(p=s.hflip_p))
        if s.vflip_p > 0:
            ops.append(A.VerticalFlip(p=s.vflip_p))

        # Rotations (both discrete and arbitrary angle)
        if s.rotate90_p > 0:
            ops.append(A.RandomRotate90(p=s.rotate90_p))
        if s.rot_p > 0:
            ops.append(A.Rotate(limit=s.rot_limit, border_mode=cv2.BORDER_REFLECT_101, p=s.rot_p))

        # Mild geometric jitter
        if s.affine_p > 0:
            ops.append(
                A.Affine(
                    scale=s.affine_scale,
                    translate_percent=s.affine_translate,
                    rotate=s.affine_rotate,
                    shear=s.affine_shear,
                    fit_output=False,
                    p=s.affine_p,
                )
            )

        # Optional: RandomResizedCrop (off by default here)
        if s.rrc_p > 0:
            ops.append(A.RandomResizedCrop(size=(H, W), scale=s.rrc_scale, ratio=s.rrc_ratio, p=s.rrc_p))

        # Color jitter
        if s.cj_p > 0:
            ops.append(
                A.ColorJitter(
                    brightness=s.cj_brightness,
                    contrast=s.cj_contrast,
                    saturation=s.cj_saturation,
                    hue=s.cj_hue,
                    p=s.cj_p,
                )
            )

        # Color "filter" (choose one style)
        if s.color_filter_p > 0:
            ops.append(
                A.OneOf(
                    [
                        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
                        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
                    ],
                    p=s.color_filter_p,
                )
            )

        # Noise
        if s.noise_p > 0:
            ops.append(
                A.OneOf(
                    [
                        _make_gaussian_noise(s.gauss_noise_var, mean=0.0),
                        A.ISONoise(intensity=(0.1, 0.5), color_shift=(0.01, 0.05), p=1.0),
                    ],
                    p=s.noise_p,
                )
            )

        return A.Compose(
            ops,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.2,  # guard against vanishing boxes after transforms
            ),
        )

    def _build_rotation_only_transform(self) -> A.Compose:
        """Extra full-image rotations; labels kept via bbox transform."""
        s = self.settings
        ops: List[A.BasicTransform] = []
        if s.rotate90_p > 0:
            ops.append(A.RandomRotate90(p=1.0))   # force a 90° rotation
        if s.rot_p > 0:
            ops.append(A.Rotate(limit=s.rot_limit, border_mode=cv2.BORDER_REFLECT_101, p=1.0))
        if not ops:
            ops = [A.NoOp(p=1.0)]
        return A.Compose(
            ops,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.0),
        )

    # -------- Crop policy (map & filter boxes) --------

    def _relocate_boxes_with_policy(
        self,
        abs_boxes_xyxy: List[List[float]],
        labels: List[int],
        roi_xyxy: List[int],
        *,
        policy: ObjectPolicy,
        min_vis: float,
        allow_empty: bool,
    ) -> Tuple[List[List[float]], List[int], bool]:
        """
        Map absolute boxes to ROI-local YOLO boxes and filter by policy.

        Returns (boxes_yolo_in_roi, labels_in_roi, accept_sample)
        """
        rx0, ry0, rx1, ry1 = roi_xyxy
        W_roi = rx1 - rx0
        H_roi = ry1 - ry0
        if W_roi <= 1 or H_roi <= 1:
            return [], [], False

        kept_boxes: List[List[float]] = []
        kept_labels: List[int] = []
        fully_contained_count = 0
        visible_count = 0

        for b_xyxy, cls in zip(abs_boxes_xyxy, labels):
            bx1, by1, bx2, by2 = b_xyxy
            box_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
            if box_area <= 0:
                continue

            inter_area, inter = _box_intersection(b_xyxy, [rx0, ry0, rx1, ry1])
            if inter_area <= 0:
                continue

            # containment / visibility
            vis = inter_area / (box_area + 1e-12)
            is_full = (bx1 >= rx0) and (by1 >= ry0) and (bx2 <= rx1) and (by2 <= ry1)

            if policy == "full":
                if not is_full:
                    continue
                fully_contained_count += 1
                # no clipping for "full"
                lx1, ly1, lx2, ly2 = bx1 - rx0, by1 - ry0, bx2 - rx0, by2 - ry0
            else:
                # partial / negative_ok: clip to ROI
                visible_count += (vis >= min_vis)
                ix1, iy1, ix2, iy2 = inter
                lx1, ly1, lx2, ly2 = ix1 - rx0, iy1 - ry0, ix2 - rx0, iy2 - ry0

            yolo_local = _xyxy_abs_to_yolo([lx1, ly1, lx2, ly2], W_roi, H_roi)
            if yolo_local is None:
                continue
            kept_boxes.append(yolo_local)
            kept_labels.append(cls)

        # Decide acceptance
        if policy == "full":
            accept = fully_contained_count > 0
        elif policy == "partial":
            accept = visible_count > 0
        else:  # negative_ok
            accept = True if (kept_boxes or allow_empty) else False

        if not accept:
            return [], [], False
        return kept_boxes, kept_labels, True

    # -------- Crop generators --------

    def _gen_tiles(self, im, *, tile_size=512, overlap=0.2) -> Iterable[Tuple[List[int], str]]:
        H, W = im.shape[:2]
        ts = max(16, int(tile_size))
        stride = max(1, int(round(ts * (1.0 - overlap))))
        for y0 in range(0, max(1, H - ts + 1), stride):
            for x0 in range(0, max(1, W - ts + 1), stride):
                x1 = min(W, x0 + ts)
                y1 = min(H, y0 + ts)
                if x1 - x0 <= 1 or y1 - y0 <= 1:
                    continue
                yield [x0, y0, x1, y1], f"t{y0}_{x0}"

    def _gen_multiscale(self, im, *, scales: List[float], overlap: float) -> Iterable[Tuple[List[int], str]]:
        H, W = im.shape[:2]
        short_side = min(W, H)
        for s in scales:
            ts = max(16, int(round(s * short_side)))
            stride = max(1, int(round(ts * (1.0 - overlap))))
            for y0 in range(0, max(1, H - ts + 1), stride):
                for x0 in range(0, max(1, W - ts + 1), stride):
                    x1 = min(W, x0 + ts)
                    y1 = min(H, y0 + ts)
                    if x1 - x0 <= 1 or y1 - y0 <= 1:
                        continue
                    yield [x0, y0, x1, y1], f"ms{int(round(s*100))}_{y0}_{x0}"

    def _gen_zoom_crops(
        self, im, abs_boxes_xyxy, labels, *, scales, per_obj, on, jitter
    ) -> Iterable[Tuple[List[int], str]]:
        H, W = im.shape[:2]
        short_side = min(W, H)

        def _select_indices():
            if on == "disc":
                return [i for i, l in enumerate(labels) if l == 0]
            if on == "cup":
                return [i for i, l in enumerate(labels) if l == 1]
            if on == "both":
                return list(range(len(labels)))
            if on == "any":
                return list(range(len(labels))) if labels else []
            return list(range(len(labels)))

        for i in _select_indices():
            x1, y1, x2, y2 = abs_boxes_xyxy[i]
            ocx = (x1 + x2) / 2.0
            ocy = (y1 + y2) / 2.0
            for s in scales:
                side = max(16, int(round(s * short_side)))
                for k in range(per_obj):
                    jx = (random.uniform(-jitter, jitter)) * side
                    jy = (random.uniform(-jitter, jitter)) * side
                    cx = ocx + jx
                    cy = ocy + jy
                    rx0, ry0, rx1, ry1 = _square_crop_bounds(cx, cy, side, W, H)
                    yield [rx0, ry0, rx1, ry1], f"z{i}_s{int(round(s*100))}_{k}"

    # -------- writers --------

    @staticmethod
    def _dump_variant(
        out_img_dir: Path,
        out_lbl_dir: Path,
        stem: str,
        suffix: str,
        ext: str,
        img,
        boxes,
        labels,
    ):
        img_p = out_img_dir / f"{stem}_{suffix}{ext}"
        lbl_p = out_lbl_dir / f"{stem}_{suffix}.txt"
        _write_image(img_p, img)
        _write_yolo_labels(lbl_p, boxes, labels)

    # -------- Single (image, label) processor --------

    def _process_one(
        self,
        img_path: Path,
        lbl_path: Path,
        out_img_dir: Path,
        out_lbl_dir: Path,
        *,
        prefer_copy: bool,
    ):
        s = self.settings

        im = _read_image(img_path)
        if im is None:
            print(f"[WARN] Failed to read image: {img_path}")
            return

        boxes, labels = _read_yolo_labels(lbl_path)
        boxes = _sanitize_yolo_boxes(boxes)
        stem = img_path.stem
        H, W = im.shape[:2]
        abs_boxes = [_yolo_to_xyxy_abs(b, W, H) for b in boxes]

        # (A) Original passthrough
        orig_img_out = out_img_dir / f"{stem}{s.out_ext}"
        if s.out_ext.lower() == img_path.suffix.lower() and prefer_copy:
            ensure_dir(orig_img_out.parent)
            shutil.copy2(img_path, orig_img_out)
        else:
            _write_image(orig_img_out, im)
        _write_yolo_labels(out_lbl_dir / f"{stem}.txt", boxes, labels)

        # Helper: emit crops from a generator of ROIs on a given image/boxes
        def _emit_crops(
            base_img, base_abs_boxes, base_labels, roi_iter, *,
            suffix_prefix: str, out_size_if_any: Optional[int] = None
        ):
            for roi_xyxy, suffix in roi_iter:
                rx0, ry0, rx1, ry1 = roi_xyxy
                crop = base_img[ry0:ry1, rx0:rx1].copy()
                if crop.size == 0:
                    continue
                k_boxes, k_labels, ok = self._relocate_boxes_with_policy(
                    base_abs_boxes, base_labels, roi_xyxy,
                    policy=s.object_policy,
                    min_vis=s.min_visibility,
                    allow_empty=(s.object_policy == "negative_ok"),
                )
                if not ok:
                    continue
                if out_size_if_any and out_size_if_any > 0:
                    crop = cv2.resize(crop, (int(out_size_if_any), int(out_size_if_any)), interpolation=cv2.INTER_AREA)
                self._dump_variant(out_img_dir, out_lbl_dir, stem, suffix, s.out_ext, crop, k_boxes, k_labels)

        # (B) Crops on ORIGINAL image
        if s.tiling_enable:
            _emit_crops(
                im, abs_boxes, labels,
                self._gen_tiles(im, tile_size=s.tiling_tile_size, overlap=s.tiling_overlap),
                suffix_prefix="t"
            )
        if s.zc_enable and boxes:
            _emit_crops(
                im, abs_boxes, labels,
                self._gen_zoom_crops(
                    im, abs_boxes, labels,
                    scales=list(s.zc_scales),
                    per_obj=s.zc_per_obj,
                    on=s.zc_on,
                    jitter=s.zc_jitter,
                ),
                suffix_prefix="z",
                out_size_if_any=s.zc_out_size
            )
        if s.zs_enable:
            _emit_crops(
                im, abs_boxes, labels,
                self._gen_multiscale(im, scales=list(s.zs_scales), overlap=s.zs_overlap),
                suffix_prefix="ms",
                out_size_if_any=s.zs_out_size
            )

        # (C) Extra full-image rotation-only variants (if requested)
        if s.full_rotate_n > 0:
            rot_tf = self._build_rotation_only_transform()
            for k in range(s.full_rotate_n):
                if boxes:
                    out = rot_tf(image=im, bboxes=boxes, class_labels=labels)
                    rim, rboxes, rlabels = out["image"], out["bboxes"], out["class_labels"]
                else:
                    out = rot_tf(image=im, bboxes=[], class_labels=[])
                    rim, rboxes, rlabels = out["image"], [], []
                self._dump_variant(out_img_dir, out_lbl_dir, stem, f"rot{k}", s.out_ext, rim, rboxes, rlabels)

                # crops from rotation-only variants
                if s.tiling_enable and s.tiling_from_aug:
                    abs2 = [_yolo_to_xyxy_abs(b, rim.shape[1], rim.shape[0]) for b in rboxes]
                    _emit_crops(
                        rim, abs2, rlabels,
                        self._gen_tiles(rim, tile_size=s.tiling_tile_size, overlap=s.tiling_overlap),
                        suffix_prefix="t"
                    )
                if s.zc_enable and s.zc_from_aug and rboxes:
                    abs2 = [_yolo_to_xyxy_abs(b, rim.shape[1], rim.shape[0]) for b in rboxes]
                    _emit_crops(
                        rim, abs2, rlabels,
                        self._gen_zoom_crops(
                            rim, abs2, rlabels,
                            scales=list(s.zc_scales),
                            per_obj=s.zc_per_obj,
                            on=s.zc_on,
                            jitter=s.zc_jitter,
                        ),
                        suffix_prefix="z",
                        out_size_if_any=s.zc_out_size
                    )
                if s.zs_enable and s.zs_from_aug:
                    abs2 = [_yolo_to_xyxy_abs(b, rim.shape[1], rim.shape[0]) for b in rboxes]
                    _emit_crops(
                        rim, abs2, rlabels,
                        self._gen_multiscale(rim, scales=list(s.zs_scales), overlap=s.zs_overlap),
                        suffix_prefix="ms",
                        out_size_if_any=s.zs_out_size
                    )

        # (D) General augmented variants
        aug_tf = self._build_general_transform(H, W)
        for k in range(int(s.multiplier)):
            if boxes:
                out = aug_tf(image=im, bboxes=boxes, class_labels=labels)
                aimg, aboxes, alabels = out["image"], out["bboxes"], out["class_labels"]
            else:
                out = aug_tf(image=im, bboxes=[], class_labels=[])
                aimg, aboxes, alabels = out["image"], [], []
            self._dump_variant(out_img_dir, out_lbl_dir, stem, f"aug{k}", s.out_ext, aimg, aboxes, alabels)

            # crops from augmented variants (optional)
            if s.tiling_enable and s.tiling_from_aug:
                abs2 = [_yolo_to_xyxy_abs(b, aimg.shape[1], aimg.shape[0]) for b in aboxes]
                _emit_crops(
                    aimg, abs2, alabels,
                    self._gen_tiles(aimg, tile_size=s.tiling_tile_size, overlap=s.tiling_overlap),
                    suffix_prefix="t"
                )
            if s.zc_enable and s.zc_from_aug and aboxes:
                abs2 = [_yolo_to_xyxy_abs(b, aimg.shape[1], aimg.shape[0]) for b in aboxes]
                _emit_crops(
                    aimg, abs2, alabels,
                    self._gen_zoom_crops(
                        aimg, abs2, alabels,
                        scales=list(s.zc_scales),
                        per_obj=s.zc_per_obj,
                        on=s.zc_on,
                        jitter=s.zc_jitter,
                    ),
                    suffix_prefix="z",
                    out_size_if_any=s.zc_out_size
                )
            if s.zs_enable and s.zs_from_aug:
                abs2 = [_yolo_to_xyxy_abs(b, aimg.shape[1], aimg.shape[0]) for b in aboxes]
                _emit_crops(
                    aimg, abs2, alabels,
                    self._gen_multiscale(aimg, scales=list(s.zs_scales), overlap=s.zs_overlap),
                    suffix_prefix="ms",
                    out_size_if_any=s.zs_out_size
                )

    # -------- split-level runner --------

    def _process_split(self, in_root: Path, out_root: Path, split: str, *, prefer_copy: bool) -> None:
        in_img_dir = in_root / "images" / split
        in_lbl_dir = in_root / "labels" / split
        out_img_dir = out_root / "images" / split
        out_lbl_dir = out_root / "labels" / split

        if not in_img_dir.exists():
            print(f"[WARN] Missing images split '{split}'; skipping.")
            return
        if not in_lbl_dir.exists():
            print(f"[WARN] Missing labels split '{split}'; skipping.")
            return

        ensure_dir(out_img_dir)
        ensure_dir(out_lbl_dir)

        count = 0
        for img_path in _list_images(in_img_dir):
            lbl_path = in_lbl_dir / f"{img_path.stem}.txt"

            if not lbl_path.exists() and not self.settings.include_images_without_labels:
                continue

            # If negatives are included, ensure an empty label file exists for consistency
            if not lbl_path.exists() and self.settings.include_images_without_labels:
                ensure_dir(lbl_path.parent)
                lbl_path.write_text("")

            self._process_one(
                img_path=img_path,
                lbl_path=lbl_path,
                out_img_dir=out_img_dir,
                out_lbl_dir=out_lbl_dir,
                prefer_copy=prefer_copy,
            )
            count += 1

        print(f"[OK] Split '{split}': processed {count} images → {out_img_dir.parent}")

    # -------- public API --------

    def run_from_yolo_root(
        self,
        *,
        data_root: Path,
        out_root: Path,
        splits: Iterable[str],
        prefer_copy: bool = False,
    ) -> None:
        print(f"[AUG] Input : {data_root}")
        print(f"[AUG] Output: {out_root}")
        for s in ("images", "labels"):
            ensure_dir(out_root / s / "train")  # prime dirs

        for split in splits:
            self._process_split(data_root, out_root, split, prefer_copy=prefer_copy)

        self._maybe_write_data_yaml(out_root)
        print(f"[OK] Augmented dataset written to: {out_root}")

    def _maybe_write_data_yaml(self, out_root: Path) -> None:
        if not self.settings.write_yaml:
            return
        ensure_dir(out_root)
        yaml_path = out_root / "data.yaml"
        ds = {
            "path": str(out_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": self.settings.names,
        }
        test_img_dir = out_root / "images" / "test"
        test_lbl_dir = out_root / "labels" / "test"
        if test_img_dir.exists() and any(test_img_dir.iterdir()) and test_lbl_dir.exists() and any(test_lbl_dir.iterdir()):
            ds["test"] = "images/test"
        yaml_path.write_text(yaml.safe_dump(ds, sort_keys=False))
        print(f"[OK] Wrote data.yaml → {yaml_path}")