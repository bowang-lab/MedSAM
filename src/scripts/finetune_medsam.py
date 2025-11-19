#!/usr/bin/env python3
# src/scripts/finetune_medsam.py
# Fundu-style fine-tuning for MedSAM with:
# - SAM image encoder + adapters (last few blocks, lighter bottleneck)
# - Spatial & channel CBAM
# - Polar-coordinate training for OD/OC
# - Joint OD/OC loss with containment term
# - Multi-GPU (DDP) support and ImageFactory/YOLO-backed splits
# - Detector-guided prompts (Option B): GT/DET box mixing with cache

from __future__ import annotations

import os
import json
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Iterable
from contextlib import nullcontext

import numpy as np
from PIL import Image as PILImage, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# Project imports
from src.imgpipe.image_factory import ImageFactory
from src.imgpipe.image import Image as IMG
from src.imgpipe.normalized_box import NormalizedBox
from src.imgpipe.enums import Structure, LabelType  # expects Structure.DISC/CUP, LabelType.GT/PRED

# Optional dependencies
try:
    from ultralytics import YOLO
except Exception as _e:
    YOLO = None
    _YOLO_ERR = _e

try:
    from segment_anything import sam_model_registry
except Exception as _e:
    sam_model_registry = None
    _SAM_ERR = _e

try:
    import yaml  # for YOLO data.yaml
except Exception:
    yaml = None

try:
    import cv2
except Exception as _e:
    cv2 = None
    _CV2_ERR = _e


def _ensure_cv2_available():
    if cv2 is None:
        raise RuntimeError(
            f"OpenCV (cv2) not available: {_CV2_ERR!r}. "
            "Install: pip install opencv-python"
        )


# =========================================================
# Global defaults (single source for numeric hyperparams)
# =========================================================
DEFAULT_MODEL = "vit_b"
DEFAULT_IMGSZ = 1024
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 8
DEFAULT_WORKERS = 8
DEFAULT_LR = 1e-4
DEFAULT_WD = 1e-4
SEED = 42

DEFAULT_YOLO_DEVICE = "cuda:0"
DEFAULT_YOLO_IMGSZ = 640
DEFAULT_YOLO_CONF = 0.25
DEFAULT_YOLO_IOU = 0.50

DEFAULT_USE_DET_PROB = 0.5
DEFAULT_PAD_JITTER = 0.30
DEFAULT_BOX_TR = 0.05
DEFAULT_BOX_SC = 0.10

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

PIXEL_MEAN = torch.tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.120, 57.375]).view(3, 1, 1)


# =========================================================
# Small utils
# =========================================================
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on CPU/MPS
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_sam_available():
    if sam_model_registry is None:
        raise RuntimeError(
            f"segment-anything not available: {_SAM_ERR!r}. "
            "Install: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )


def _ensure_yolo_available():
    if YOLO is None:
        raise RuntimeError(
            f"Ultralytics YOLO not available: {_YOLO_ERR!r}. "
            "Install: pip install ultralytics"
        )


def _to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_device(t, device) for t in x)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    return x


# =========================================================
# Distributed context
# =========================================================
@dataclass
class DistConfig:
    backend: str = "nccl"
    init_method: str = "env://"
    world_size: int = int(os.environ.get("WORLD_SIZE", "1"))
    rank: int = int(os.environ.get("RANK", "0"))
    local_rank: int = int(os.environ.get("LOCAL_RANK", "0"))

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def device(self) -> torch.device:
        # Prefer CUDA, then MPS, then CPU
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}" if self.distributed else "cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


class DistributedContext:
    def __init__(self, cfg: DistConfig):
        self.cfg = cfg
        self._initialized = False

    def setup(self):
        if self.cfg.distributed:
            torch.cuda.set_device(self.cfg.local_rank)
            torch.distributed.init_process_group(
                backend=self.cfg.backend,
                init_method=self.cfg.init_method,
                world_size=self.cfg.world_size,
                rank=self.cfg.rank,
            )
            self._initialized = True

    def cleanup(self):
        if self._initialized:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            self._initialized = False

    @property
    def is_main(self) -> bool:
        return self.cfg.rank == 0

    def barrier(self):
        if self.cfg.distributed:
            torch.distributed.barrier()

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.cfg.distributed:
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return tensor


# =========================================================
# YOLO split helpers
# =========================================================
def _parse_data_yaml(yaml_path: Path) -> Dict[str, str]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")
    if yaml is not None:
        obj = yaml.safe_load(yaml_path.read_text())
        return {k: str(obj.get(k)) for k in ("train", "val", "test") if k in obj}
    # fallback
    lines = [ln.strip() for ln in yaml_path.read_text().splitlines()]
    out: Dict[str, str] = {}
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("train:"):
            out["train"] = ln.split(":", 1)[1].strip()
        if ln.startswith("val:"):
            out["val"] = ln.split(":", 1)[1].strip()
        if ln.startswith("test:"):
            out["test"] = ln.split(":", 1)[1].strip()
    return out


def _read_list(p: Path) -> List[Path]:
    return [Path(ln.strip()) for ln in p.read_text().splitlines() if ln.strip()]


def _resolve_split_images(yolo_ds: Path, entry: str) -> List[Path]:
    p = Path(entry)
    if not p.is_absolute():
        p = (yolo_ds / p).resolve()
    if p.suffix.lower() == ".txt":
        return _read_list(p)
    if p.is_dir():
        return sorted([q for q in p.rglob("*") if q.suffix.lower() in _IMG_EXTS])
    return [p] if p.suffix.lower() in _IMG_EXTS else []


# =========================================================
# Build Image objects and map to splits
# =========================================================
def build_image_index(data_root: Path, exclude: Optional[List[str]] = None) -> Dict[str, IMG]:
    fac = ImageFactory(root=data_root, auto_scan=True)
    fac.filter_empty_masks()  # require GT masks present
    if exclude:
        fac.filter_datasets(exclude=exclude)
    images = fac.make_images()
    idx: Dict[str, IMG] = {Path(im.image_path).stem: im for im in images}
    if not idx:
        raise RuntimeError("No images discovered by ImageFactory.")
    return idx


def images_from_yolo_splits(
    yolo_ds: Path, data_root: Path, exclude: Optional[List[str]] = None
) -> Tuple[List[IMG], List[IMG], List[IMG]]:
    mapping = _parse_data_yaml(yolo_ds / "data.yaml")
    train_imgs = _resolve_split_images(yolo_ds, mapping["train"])
    val_imgs = _resolve_split_images(yolo_ds, mapping["val"])
    test_imgs = _resolve_split_images(yolo_ds, mapping["test"])

    idx = build_image_index(data_root, exclude)
    miss = 0

    def to_images(ps: List[Path]) -> List[IMG]:
        nonlocal miss
        out: List[IMG] = []
        for p in ps:
            im = idx.get(Path(p).stem)
            if im is None:
                miss += 1
                continue
            im.set_split(None)
            out.append(im)
        return out

    tr, va, te = to_images(train_imgs), to_images(val_imgs), to_images(test_imgs)
    if miss:
        print(f"[WARN] {miss} YOLO-split images not found in ImageFactory index; skipped.")
    print(f"[INFO] YOLO split sizes → train={len(tr)} val={len(va)} test={len(te)}")
    return tr, va, te


# =========================================================
# Detector → normalized boxes with disk cache (Option B core)
# =========================================================
class YOLOBoxProvider:
    """Returns at most one NormalizedBox per class ({disc, cup}) with per-run cache."""

    def __init__(self, weights: Path, device: str = DEFAULT_YOLO_DEVICE,
                 imgsz: int = DEFAULT_YOLO_IMGSZ, conf: float = DEFAULT_YOLO_CONF,
                 iou: float = DEFAULT_YOLO_IOU):
        _ensure_yolo_available()
        # Important for memory: we will typically run this on "cpu"
        self.model = YOLO(str(weights))
        self.cfg = dict(device=device, imgsz=imgsz, conf=conf, iou=iou)

    def predict_one(self, img_path: Path, W: int, H: int) -> Dict[str, Optional[Tuple[float, float, float, float]]]:
        r = self.model.predict(source=str(img_path), verbose=False, **self.cfg)[0]
        out: Dict[str, Optional[Tuple[float, float, float, float]]] = {"disc": None, "cup": None}
        if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for b, c in zip(xyxy, cls):
            x1, y1, x2, y2 = map(float, b)
            nb = NormalizedBox.from_xyxy(x1, y1, x2, y2, W, H).as_tuple()
            if c == 0 and out["disc"] is None:
                out["disc"] = nb
            elif c == 1 and out["cup"] is None:
                out["cup"] = nb
        return out


def attach_detector_boxes_with_cache(
    images: Iterable[IMG],
    cache_path: Optional[Path],
    dist: DistributedContext,
    provider: Optional[YOLOBoxProvider] = None,
) -> None:
    """
    Implements Option B:
      - Rank-0 runs YOLO once and writes a JSONL cache:
          {"stem":..., "disc": (xc,yc,w,h) or null, "cup": (...)}
      - All ranks read the same cache and attach NormalizedBox(PRED) to each Image.
      - During training, datasets can mix GT vs DET boxes on-the-fly.
    """
    images = list(images)
    mapping: Dict[str, Dict[str, Optional[Tuple[float, float, float, float]]]] = {}

    if cache_path and cache_path.exists():
        # Load existing cache
        for line in cache_path.read_text().splitlines():
            rec = json.loads(line)
            mapping[rec["stem"]] = {"disc": rec.get("disc"), "cup": rec.get("cup")}
    else:
        if provider is None:
            if dist.is_main:
                print("[INFO] No provider and no cache; skipping detector attachment.")
        else:
            # Compute on main, write cache
            if dist.is_main:
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    jf = cache_path.open("w")
                else:
                    jf = open(os.devnull, "w")

                with jf:
                    for im in images:
                        rec = {"stem": Path(im.image_path).stem}
                        preds = provider.predict_one(Path(im.image_path), im.width, im.height)
                        rec["disc"] = preds["disc"]
                        rec["cup"] = preds["cup"]
                        jf.write(json.dumps(rec) + "\n")
                        mapping[rec["stem"]] = {"disc": rec["disc"], "cup": rec["cup"]}

        dist.barrier()
        # Non-main ranks load the newly created cache
        if not mapping and cache_path and cache_path.exists():
            for line in cache_path.read_text().splitlines():
                rec = json.loads(line)
                mapping[rec["stem"]] = {"disc": rec.get("disc"), "cup": rec.get("cup")}

    # Attach to Image objects
    for im in images:
        stem = Path(im.image_path).stem
        m = mapping.get(stem)
        if not m:
            continue
        if m.get("disc") is not None:
            im.set_box(Structure.DISC, LabelType.PRED, NormalizedBox(*m["disc"]))
        if m.get("cup") is not None:
            im.set_box(Structure.CUP, LabelType.PRED, NormalizedBox(*m["cup"]))


# =========================================================
# Geometry / preprocessing
# =========================================================
class LetterboxToSquare:
    def __init__(self, size: int = 1024):
        self.size = size

    def _compute_pad(self, w: int, h: int) -> Tuple[float, int, int, int, int]:
        scale = min(self.size / h, self.size / w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        pad_w = self.size - new_w
        pad_h = self.size - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        return scale, new_w, new_h, pad_left, pad_top

    def __call__(
        self, img: PILImage.Image, mask: PILImage.Image, box_xyxy: np.ndarray
    ) -> Tuple[PILImage.Image, PILImage.Image, np.ndarray]:
        w, h = img.size
        scale, new_w, new_h, pad_left, pad_top = self._compute_pad(w, h)
        img_r = img.resize((new_w, new_h), PILImage.BILINEAR)
        mask_r = mask.resize((new_w, new_h), PILImage.NEAREST)
        new_img = PILImage.new("RGB", (self.size, self.size))
        new_mask = PILImage.new("L", (self.size, self.size))
        new_img.paste(img_r, (pad_left, pad_top))
        new_mask.paste(mask_r, (pad_left, pad_top))
        x0, y0, x1, y1 = box_xyxy
        x0 = x0 * scale + pad_left
        y0 = y0 * scale + pad_top
        x1 = x1 * scale + pad_left
        y1 = y1 * scale + pad_top
        box_t = np.array([x0, y0, x1, y1], dtype=np.float32)
        return new_img, new_mask, box_t


def mask_to_tight_box(mask_np: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask_np)
    if ys.size == 0 or xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def pad_box(box: np.ndarray, pad_frac: float, img_w: int, img_h: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    px = w * pad_frac
    py = h * pad_frac
    xx0 = max(0.0, x0 - px)
    yy0 = max(0.0, y0 - py)
    xx1 = min(float(img_w - 1), x1 + px)
    yy1 = min(float(img_h - 1), y1 + py)
    return np.array([xx0, yy0, xx1, yy1], dtype=np.float32)


def jitter_box_xyxy(box: np.ndarray, img_w: int, img_h: int, tr: float = 0.05, sc: float = 0.10) -> np.ndarray:
    x0, y0, x1, y1 = box
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    dx = (2 * random.random() - 1) * tr * w
    dy = (2 * random.random() - 1) * tr * h
    s = 1.0 + (2 * random.random() - 1) * sc
    nw = max(1.0, w * s)
    nh = max(1.0, h * s)
    cx += dx
    cy += dy
    nx0 = max(0.0, cx - nw / 2)
    ny0 = max(0.0, cy - nh / 2)
    nx1 = min(float(img_w - 1), cx + nw / 2)
    ny1 = min(float(img_h - 1), cy + nh / 2)
    return np.array([nx0, ny0, nx1, ny1], dtype=np.float32)


def preprocess_for_sam(img_pil: PILImage.Image) -> torch.Tensor:
    x = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float()  # RGB 0..255
    return (x - PIXEL_MEAN) / PIXEL_STD


def nbox_to_xyxy(nbox: NormalizedBox, W: int, H: int) -> np.ndarray:
    x1, y1, x2, y2 = nbox.to_pixel_xyxy(W, H)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def color_jitter_safe(img: PILImage.Image, brightness: float = 0.10, contrast: float = 0.10) -> PILImage.Image:
    if brightness > 0:
        img = ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-brightness, brightness))
    if contrast > 0:
        img = ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-contrast, contrast))
    return img


# =========================================================
# Polar transform helpers
# =========================================================
def cartesian_to_polar(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Convert Cartesian image (H,W or H,W,C) to polar coordinates.
    Output is (out_h, out_w, C) or (out_h, out_w).
    Center = image center, radius = min(H,W)/2.
    """
    _ensure_cv2_available()
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    radius = min(h, w) / 2.0
    flags = cv2.WARP_POLAR_LINEAR
    polar = cv2.warpPolar(img, (out_w, out_h), center, radius, flags)
    # Rotate so angle axis becomes x-axis consistently
    polar = cv2.rotate(polar, cv2.ROTATE_90_CLOCKWISE)
    return polar


def polar_to_cartesian(img_polar: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Invert polar image back to Cartesian coordinates of size (out_h, out_w).
    Assumes same center/radius convention as cartesian_to_polar with H=W=out_h=out_w.
    """
    _ensure_cv2_available()
    center = (out_w / 2.0, out_h / 2.0)
    radius = min(out_h, out_w) / 2.0
    p = cv2.rotate(img_polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
    cart = cv2.warpPolar(p, (out_w, out_h), center, radius, flags)
    return cart


# =========================================================
# Original single-structure dataset types (kept for reference)
# =========================================================
@dataclass(frozen=True)
class SegItem:
    image: IMG
    structure: str  # "disc" | "cup"


def make_items(images: List[IMG], require_gt: bool = True) -> List[SegItem]:
    out: List[SegItem] = []
    for im in images:
        im.ensure_boxes_from_masks()
        if not require_gt:
            out.extend([SegItem(im, "disc"), SegItem(im, "cup")])
            continue
        if im.gt_disc_mask is not None and getattr(im.gt_disc_mask, "path", None):
            out.append(SegItem(im, "disc"))
        if im.gt_cup_mask is not None and getattr(im.gt_cup_mask, "path", None):
            out.append(SegItem(im, "cup"))
    return out


class MedSAMDataset(Dataset):
    """
    Original single-structure dataset (disc OR cup).
    Not used in Fundu pipeline, kept for reference.
    """

    def __init__(
        self,
        items: List[SegItem],
        img_size: int = DEFAULT_IMGSZ,
        train: bool = True,
        use_det_prob: float = DEFAULT_USE_DET_PROB,
        pad_jitter: float = DEFAULT_PAD_JITTER,
        box_tr: float = DEFAULT_BOX_TR,
        box_sc: float = DEFAULT_BOX_SC,
        prompt_mode: str = "mix",  # "mix" (train only), "gt", "det"
    ):
        self.items = items
        self.img_size = img_size
        self.train = train
        self.use_det_prob = float(np.clip(use_det_prob, 0.0, 1.0))
        self.pad_jitter = pad_jitter
        self.box_tr = box_tr
        self.box_sc = box_sc
        self.prompt_mode = prompt_mode
        self.letterbox = LetterboxToSquare(img_size)

    def __len__(self) -> int:
        return len(self.items)

    def _choose_nbox(self, im: IMG, struct: str) -> Optional[NormalizedBox]:
        gt = im.gt_disc_box if struct == "disc" else im.gt_cup_box
        det = im.inter_pred_disc_box if struct == "disc" else im.inter_pred_cup_box
        if self.train and self.prompt_mode == "mix":
            return det if (random.random() < self.use_det_prob and det is not None) else gt
        if self.prompt_mode == "det":
            return det if det is not None else gt
        return gt

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        im: IMG = item.image
        struct = item.structure

        img = PILImage.open(im.image_path).convert("RGB")
        mref = im.gt_disc_mask if struct == "disc" else im.gt_cup_mask
        if mref is None or getattr(mref, "path", None) is None:
            raise RuntimeError(f"Missing GT mask for {struct}: {im.image_path}")
        m = PILImage.open(mref.path).convert("L")

        if self.train:
            img = color_jitter_safe(img, brightness=0.10, contrast=0.10)

        base_nbox = self._choose_nbox(im, struct)
        if base_nbox is None:
            tb = mask_to_tight_box(np.array(m, dtype=np.uint8))
            box_xyxy = tb if tb is not None else np.array(
                [0, 0, img.size[0] - 1, img.size[1] - 1],
                dtype=np.float32,
            )
        else:
            box_xyxy = nbox_to_xyxy(base_nbox, im.width, im.height)

        pad_frac = (random.uniform(0.0, self.pad_jitter) if self.train else 0.0)
        box_p = pad_box(box_xyxy, pad_frac, im.width, im.height)
        if self.train and (self.box_tr > 0.0 or self.box_sc > 0.0):
            box_p = jitter_box_xyxy(box_p, im.width, im.height, tr=self.box_tr, sc=self.box_sc)

        img_r, mask_r, box_t = self.letterbox(img, m, box_p)
        x = preprocess_for_sam(img_r)
        y = torch.from_numpy((np.array(mask_r, dtype=np.uint8) > 0).astype(np.float32)).unsqueeze(0)
        b = torch.from_numpy(box_t.astype(np.float32))
        meta = {"image_path": str(im.image_path), "structure": struct, "stem": Path(im.image_path).stem}
        return {"image": x, "mask": y, "box": b, "meta": meta}


# =========================================================
# Fundu-style joint OD/OC dataset with polar coordinates
# =========================================================
def make_joint_images(images: List[IMG]) -> List[IMG]:
    """
    Filter images to those that have BOTH disc and cup GT masks.
    """
    out: List[IMG] = []
    for im in images:
        if (
            im.gt_disc_mask is not None
            and getattr(im.gt_disc_mask, "path", None)
            and im.gt_cup_mask is not None
            and getattr(im.gt_cup_mask, "path", None)
        ):
            out.append(im)
    return out


class FunduSAMDataset(Dataset):
    """
    Joint OD/OC dataset in polar space with Option B prompts:
    - Uses YOLO-predicted disc boxes (PRED) mixed with GT boxes,
      controlled by use_det_prob and prompt_mode.
    - Letterbox full fundus to (img_size, img_size)
    - Convert image + masks from Cartesian → polar
    - Returns:
        image: (3, H, W) normalized SAM-style, in polar coords
        mask:  (2, H, W) binary [disc, cup], in polar coords
    """

    def __init__(
        self,
        images: List[IMG],
        img_size: int = DEFAULT_IMGSZ,
        train: bool = True,
        use_det_prob: float = DEFAULT_USE_DET_PROB,
        pad_jitter: float = DEFAULT_PAD_JITTER,
        box_tr: float = DEFAULT_BOX_TR,
        box_sc: float = DEFAULT_BOX_SC,
        prompt_mode: str = "mix",  # "mix" (train only), "gt", "det"
    ):
        _ensure_cv2_available()
        self.images = images
        self.img_size = img_size
        self.train = train
        self.use_det_prob = float(np.clip(use_det_prob, 0.0, 1.0))
        self.pad_jitter = pad_jitter
        self.box_tr = box_tr
        self.box_sc = box_sc
        self.prompt_mode = prompt_mode
        self.letterbox = LetterboxToSquare(img_size)

    def __len__(self) -> int:
        return len(self.images)

    def _choose_disc_box(self, im: IMG) -> Optional[NormalizedBox]:
        gt = im.gt_disc_box
        det = im.inter_pred_disc_box
        if self.train and self.prompt_mode == "mix":
            return det if (random.random() < self.use_det_prob and det is not None) else gt
        if self.prompt_mode == "det":
            return det if det is not None else gt
        return gt

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        im: IMG = self.images[idx]
        im.ensure_boxes_from_masks()

        img = PILImage.open(im.image_path).convert("RGB")
        mref_disc = im.gt_disc_mask
        mref_cup = im.gt_cup_mask
        if mref_disc is None or getattr(mref_disc, "path", None) is None:
            raise RuntimeError(f"Missing GT disc mask: {im.image_path}")
        if mref_cup is None or getattr(mref_cup, "path", None) is None:
            raise RuntimeError(f"Missing GT cup mask: {im.image_path}")

        m_disc = PILImage.open(mref_disc.path).convert("L")
        m_cup = PILImage.open(mref_cup.path).convert("L")

        if self.train:
            img = color_jitter_safe(img, brightness=0.10, contrast=0.10)

        base_nbox = self._choose_disc_box(im)
        if base_nbox is None:
            # fallback to tight disc box
            tb = mask_to_tight_box(np.array(m_disc, dtype=np.uint8))
            box_xyxy = tb if tb is not None else np.array(
                [0, 0, img.size[0] - 1, img.size[1] - 1],
                dtype=np.float32,
            )
        else:
            box_xyxy = nbox_to_xyxy(base_nbox, im.width, im.height)

        pad_frac = (random.uniform(0.0, self.pad_jitter) if self.train else 0.0)
        box_p = pad_box(box_xyxy, pad_frac, im.width, im.height)
        if self.train and (self.box_tr > 0.0 or self.box_sc > 0.0):
            box_p = jitter_box_xyxy(box_p, im.width, im.height, tr=self.box_tr, sc=self.box_sc)

        # Letterbox full fundus + masks to square
        img_cart, disc_cart, box_t = self.letterbox(img, m_disc, box_p)
        _, cup_cart, _ = self.letterbox(img, m_cup, box_p)

        # Cartesian → polar for image and masks
        img_np = np.array(img_cart)
        disc_np = np.array(disc_cart, dtype=np.uint8)
        cup_np = np.array(cup_cart, dtype=np.uint8)

        img_pol = cartesian_to_polar(img_np, out_h=self.img_size, out_w=self.img_size)
        disc_pol = cartesian_to_polar(disc_np, out_h=self.img_size, out_w=self.img_size)
        cup_pol = cartesian_to_polar(cup_np, out_h=self.img_size, out_w=self.img_size)

        img_pol_pil = PILImage.fromarray(img_pol)
        x = preprocess_for_sam(img_pol_pil)

        disc_bin = (disc_pol > 0).astype(np.float32)
        cup_bin = (cup_pol > 0).astype(np.float32)
        y = torch.from_numpy(np.stack([disc_bin, cup_bin], axis=0))  # (2,H,W)

        b = torch.from_numpy(box_t.astype(np.float32))
        meta = {"image_path": str(im.image_path), "stem": Path(im.image_path).stem}
        return {"image": x, "mask": y, "box": b, "meta": meta}


# =========================================================
# Loss & metrics
# =========================================================
class BCEDice(nn.Module):
    """
    Original single-class BCE+Dice loss (kept for reference).
    """

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        num = 2 * (prob * target).sum(dim=(2, 3)) + 1e-6
        den = (prob.pow(2) + target.pow(2)).sum(dim=(2, 3)) + 1e-6
        dice = 1.0 - (num / den)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice.mean()


def dice_coef_prob(prob: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    pred = (prob >= thresh).float()
    inter = (pred * target).sum().item()
    den = pred.sum().item() + target.sum().item()
    return (2.0 * inter) / den if den > 0 else 1.0


class FunduJointLoss(nn.Module):
    """
    Fundu-style joint loss:
    L = w1 * L_disc + w2 * L_cup + w3 * L_contain
    where:
      - L_disc, L_cup are BCE losses for disc and cup
      - L_contain penalizes cup pixels outside the disc
    """

    def __init__(self, w_disc: float = 1.0, w_cup: float = 1.0, w_contain: float = 1.0):
        super().__init__()
        self.ce = nn.BCEWithLogitsLoss()
        self.w_disc = w_disc
        self.w_cup = w_cup
        self.w_contain = w_contain

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 2, H, W)
        target: (B, 2, H, W)
        """
        log_disc = logits[:, 0]
        log_cup = logits[:, 1]
        gt_disc = target[:, 0]
        gt_cup = target[:, 1]

        L_disc = self.ce(log_disc, gt_disc)
        L_cup = self.ce(log_cup, gt_cup)

        prob_disc = torch.sigmoid(log_disc)
        prob_cup = torch.sigmoid(log_cup)
        # cup outside disc
        L_contain = torch.mean(prob_cup * (1.0 - prob_disc))

        return self.w_disc * L_disc + self.w_cup * L_cup + self.w_contain * L_contain


# =========================================================
# Adapters + CBAM + FunduSAM Finetuner with lighter head
# =========================================================
class Adapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 32):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(bottleneck, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        return x + self.up(self.act(self.down(x)))


class FunduBlock(nn.Module):
    """
    Wraps a SAM ViT block, inserting adapters after attention and FFN.
    Assumes the wrapped block has attributes: norm1, attn, norm2, mlp.
    """

    def __init__(self, block: nn.Module, dim: int, bottleneck: int = 32):
        super().__init__()
        self.block = block
        self.adapter_after_attn = Adapter(dim, bottleneck)
        self.adapter_after_ffn = Adapter(dim, bottleneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.block
        x = x + b.attn(b.norm1(x))
        x = self.adapter_after_attn(x)
        x = x + b.mlp(b.norm2(x))
        x = self.adapter_after_ffn(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        avg = torch.mean(x, dim=(2, 3), keepdim=False)
        max_ = torch.amax(x, dim=(2, 3), keepdim=False)
        attn = self.mlp(avg) + self.mlp(max_)
        attn = self.sigmoid(attn).unsqueeze(-1).unsqueeze(-1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, max_], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


class ConvBlock(nn.Module):
    """
    Simple 2-layer conv block with BN + ReLU.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FunduSegHead(nn.Module):
    """
    Heavier segmentation head built on top of SAM encoder features:
    - Several ConvBlocks with residual-style skip
    - Final 1x1 conv to num_classes
    """

    def __init__(self, in_ch: int, num_classes: int = 2, width: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, width, kernel_size=1, bias=False)
        self.block1 = ConvBlock(width, width)
        self.block2 = ConvBlock(width, width)
        self.block3 = ConvBlock(width, width)
        self.out_conv = nn.Conv2d(width, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        logits = self.out_conv(x)
        return logits


class FunduSAMFinetuner(nn.Module):
    """
    Fundu-style model:
    - SAM image encoder with adapters in the last few blocks (lighter)
    - Spatial CBAM on input image
    - Channel CBAM on encoder feature map
    - Segmentation head -> 2-channel OD/OC mask in polar space
    - No prompt encoder / mask decoder usage
    """

    def __init__(
        self,
        sam_type: str,
        checkpoint: Path,
        freeze_encoders: bool = True,
        adapter_bottleneck: int = 32,
        num_classes: int = 2,
        last_n_adapted: int = 4,
        seg_width: int = 128,
    ):
        super().__init__()
        _ensure_sam_available()
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"MedSAM checkpoint not found: {checkpoint}")

        sam = sam_model_registry[sam_type](checkpoint=str(checkpoint))

        # Extract image encoder
        enc = sam.image_encoder

        # --------- Infer token dim for adapters (robustly) ----------
        if hasattr(enc, "blocks") and len(enc.blocks) > 0:
            blk0 = enc.blocks[0]
            mlp = getattr(blk0, "mlp", None)
            if mlp is None:
                raise RuntimeError("Unexpected SAM encoder block: missing 'mlp' submodule.")

            dim = None
            # Common cases: fc1 (some ViTs), linear1 (SAM's MLPBlock)
            if hasattr(mlp, "fc1"):
                dim = mlp.fc1.in_features
            elif hasattr(mlp, "linear1"):
                dim = mlp.linear1.in_features
            else:
                # Fallback: first Linear inside the MLP
                for m in mlp.modules():
                    if isinstance(m, nn.Linear):
                        dim = m.in_features
                        break

            if dim is None:
                raise RuntimeError("Could not infer token dimension from encoder MLP.")
        else:
            raise RuntimeError("Unexpected SAM image encoder structure: no blocks found.")

        # Wrap only the last few blocks with adapters (memory-friendly)
        n_blocks = len(enc.blocks)
        wrapped_blocks = []
        for i, b in enumerate(enc.blocks):
            if i >= n_blocks - max(1, last_n_adapted):
                wrapped_blocks.append(FunduBlock(b, dim=dim, bottleneck=adapter_bottleneck))
            else:
                wrapped_blocks.append(b)
        enc.blocks = nn.ModuleList(wrapped_blocks)
        self.image_encoder = enc

        # CBAM
        out_chans = getattr(enc, "out_chans", 256)
        self.spatial_attn = SpatialAttention()
        self.channel_attn = ChannelAttention(out_chans)

        # Segmentation head (still low-res)
        self.seg_head = FunduSegHead(in_ch=out_chans, num_classes=num_classes, width=seg_width)

        # Freeze encoder weights if requested, but keep adapters & seg_head trainable
        if freeze_encoders:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for blk in self.image_encoder.blocks:
                if isinstance(blk, FunduBlock):
                    for m in [blk.adapter_after_attn, blk.adapter_after_ffn]:
                        for p in m.parameters():
                            p.requires_grad = True

        for p in self.seg_head.parameters():
            p.requires_grad = True

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = batch["image"]  # (B,3,H,W) in polar coords
        # Spatial attention at input
        x = self.spatial_attn(x)

        # Image encoder
        feat = self.image_encoder(x)  # (B,C,h,w)

        # Channel attention on features
        feat = self.channel_attn(feat)

        # Segmentation head (still low-res)
        logits_low = self.seg_head(feat)  # (B,2,h,w)

        # Upsample back to input polar resolution
        logits = F.interpolate(
            logits_low,
            size=(x.shape[2], x.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        iou_pred = None  # not used in this setup
        return logits, iou_pred


# Backwards alias for any old references
MedSAMFinetuner = FunduSAMFinetuner


# =========================================================
# TrainConfig and Trainer
# =========================================================
@dataclass
class TrainConfig:
    # Data (no defaults)
    data_root: Path
    yolo_ds: Path
    out_dir: Path
    run_dir: Path
    run_name: str
    yolo_weights: Path  # non-default field

    # Now fields with defaults (all numeric defaults via constants above)
    exclude_datasets: Optional[List[str]] = None

    # Detector prompts
    yolo_device: str = DEFAULT_YOLO_DEVICE
    yolo_imgsz: int = DEFAULT_YOLO_IMGSZ
    yolo_conf: float = DEFAULT_YOLO_CONF
    yolo_iou: float = DEFAULT_YOLO_IOU
    det_cache: Optional[Path] = None  # path to JSONL detector cache

    # Model/opt
    model: str = DEFAULT_MODEL
    ckpt: Path = Path("")
    unfreeze_encoders: bool = False
    save_full: bool = False  # kept for CLI compatibility

    epochs: int = DEFAULT_EPOCHS
    batch: int = DEFAULT_BATCH
    imgsz: int = DEFAULT_IMGSZ
    lr: float = DEFAULT_LR
    wd: float = DEFAULT_WD
    workers: int = DEFAULT_WORKERS
    amp: bool = True              # enable AMP by default for memory savings
    grad_acc_steps: int = 1
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    seed: int = SEED

    # Prompt/augmentation (Option B tuning knobs)
    use_det_prob: float = DEFAULT_USE_DET_PROB
    pad_jitter: float = DEFAULT_PAD_JITTER
    box_jitter_tr: float = DEFAULT_BOX_TR
    box_jitter_sc: float = DEFAULT_BOX_SC

    # Modes
    do_train: bool = True
    do_test: bool = True
    test_prompt: str = "det"  # "det" or "gt"
    test_weights: Optional[Path] = None
    resume: Optional[Path] = None


class Trainer:
    def __init__(self, cfg: TrainConfig, dist: DistributedContext):
        self.cfg = cfg
        self.dist = dist
        self.device = dist.cfg.device()
        # Output dirs
        self.run_path = Path(cfg.run_dir) / cfg.run_name
        self.weights_dir = self.run_path / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.weights_dir / "best.pth"
        self.last_path = self.weights_dir / "last.pth"
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

        # Build splits from YOLO data.yaml (shared with detection)
        train_imgs, val_imgs, test_imgs = images_from_yolo_splits(
            cfg.yolo_ds, cfg.data_root, exclude=cfg.exclude_datasets
        )

        # Detector cache: rank-0 builds on CPU; all ranks load/attach
        provider = None
        if self.dist.is_main:
            # Use CPU for YOLO to avoid stealing GPU memory from MedSAM
            provider = YOLOBoxProvider(
                cfg.yolo_weights,
                device="cpu",
                imgsz=cfg.yolo_imgsz,
                conf=cfg.yolo_conf,
                iou=cfg.yolo_iou,
            )

        det_cache_path = cfg.det_cache if cfg.det_cache else (self.run_path / "detector_cache.jsonl")
        attach_detector_boxes_with_cache(
            train_imgs + val_imgs + test_imgs,
            det_cache_path,
            self.dist,
            provider=provider,
        )

        # Free YOLO model and clear any GPU cache
        del provider
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Joint OD/OC image lists (require both disc & cup GT masks)
        joint_train = make_joint_images(train_imgs)
        joint_val = make_joint_images(val_imgs)
        joint_test = make_joint_images(test_imgs)

        # Datasets/Loaders (polar, Option B mixing on train)
        self.ds_train = FunduSAMDataset(
            joint_train,
            img_size=cfg.imgsz,
            train=True,
            use_det_prob=cfg.use_det_prob,
            pad_jitter=cfg.pad_jitter,
            box_tr=cfg.box_jitter_tr,
            box_sc=cfg.box_jitter_sc,
            prompt_mode="mix",  # GT/DET mix
        )
        self.ds_val = FunduSAMDataset(
            joint_val,
            img_size=cfg.imgsz,
            train=False,
            use_det_prob=0.0,
            pad_jitter=0.0,
            box_tr=0.0,
            box_sc=0.0,
            prompt_mode="gt",
        )
        self.ds_test = FunduSAMDataset(
            joint_test,
            img_size=cfg.imgsz,
            train=False,
            use_det_prob=0.0,
            pad_jitter=0.0,
            box_tr=0.0,
            box_sc=0.0,
            prompt_mode=("det" if cfg.test_prompt == "det" else "gt"),
        )

        # Samplers
        self.sampler_train = (
            DistributedSampler(
                self.ds_train,
                num_replicas=dist.cfg.world_size,
                rank=dist.cfg.rank,
                shuffle=True,
            )
            if dist.cfg.distributed
            else None
        )
        self.sampler_val = (
            DistributedSampler(
                self.ds_val,
                num_replicas=dist.cfg.world_size,
                rank=dist.cfg.rank,
                shuffle=False,
            )
            if dist.cfg.distributed
            else None
        )

        # Loaders
        self.dl_train = DataLoader(
            self.ds_train,
            batch_size=cfg.batch,
            shuffle=(self.sampler_train is None),
            num_workers=cfg.workers,
            pin_memory=True,
            drop_last=False,
            sampler=self.sampler_train,
        )
        self.dl_val = DataLoader(
            self.ds_val,
            batch_size=max(1, cfg.batch // 2),
            shuffle=False,
            num_workers=cfg.workers,
            pin_memory=True,
            sampler=self.sampler_val,
        )
        # Test loader is rank-0 only (simplifies writing artifacts)
        self.dl_test = DataLoader(
            self.ds_test,
            batch_size=max(1, cfg.batch // 2),
            shuffle=False,
            num_workers=cfg.workers,
            pin_memory=True,
        )

        # Model
        self.model = FunduSAMFinetuner(
            sam_type=cfg.model,
            checkpoint=cfg.ckpt,
            freeze_encoders=not cfg.unfreeze_encoders,
        ).to(self.device)

        if dist.cfg.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[dist.cfg.local_rank],
                output_device=dist.cfg.local_rank,
                gradient_as_bucket_view=True,
                find_unused_parameters=cfg.find_unused_parameters,
                bucket_cap_mb=cfg.bucket_cap_mb,
            )

        # Fundu joint loss
        self.criterion = FunduJointLoss(w_disc=1.0, w_cup=2.0, w_contain=1.0)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.wd,
        )

        # AMP scaler: only on CUDA; on MPS/CPU it is disabled
        use_cuda = self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.cfg.amp and use_cuda))

        # Resume
        if cfg.resume and Path(cfg.resume).exists():
            self._load_resume(cfg.resume)
            if self.dist.is_main:
                print(f"[INFO] Resumed from {cfg.resume}")

        self.best_val = -1.0
        self.history: Dict[str, List[float]] = {"train_loss": [], "train_dice": [], "val_dice": []}

    # ---------- Checkpoint IO ----------
    def _unwrap(self) -> FunduSAMFinetuner:
        return self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

    def _save_ckpt(self, path: Path) -> None:
        mod = self._unwrap()
        state = {"model": mod.state_dict()}
        torch.save(state, str(path))

    def _load_resume(self, path: Path) -> None:
        state = torch.load(str(path), map_location="cpu")
        mod = self._unwrap()
        if "model" in state:
            mod.load_state_dict(state["model"], strict=True)
        elif "sam" in state:
            own_state = mod.state_dict()
            sam_state = state["sam"]
            sam_state = {k: v for k, v in sam_state.items() if k in own_state}
            own_state.update(sam_state)
            mod.load_state_dict(own_state, strict=False)
        else:
            mod.load_state_dict(state, strict=False)

    # ---------- One epoch ----------
    def _run_epoch(self, train: bool) -> Dict[str, float]:
        model = self.model
        crit = self.criterion
        opt = self.optimizer
        dl = self.dl_train if train else self.dl_val
        if self.sampler_train and train:
            self.sampler_train.set_epoch(self._epoch)  # type: ignore[attr-defined]

        model.train(mode=train)
        total_loss = 0.0
        total_dice = 0.0
        n = 0

        grad_acc = max(1, self.cfg.grad_acc_steps)
        use_cuda = self.device.type == "cuda"
        amp_enabled = self.cfg.amp and use_cuda

        for step, batch in enumerate(dl):
            batch = {k: v for k, v in batch.items() if k != "meta"}
            batch = _to_device(batch, self.device)

            if use_cuda:
                autocast_ctx = torch.cuda.amp.autocast(enabled=amp_enabled)
            else:
                autocast_ctx = nullcontext()

            with torch.set_grad_enabled(train):
                with autocast_ctx:
                    logits, _ = model(batch)  # (B,2,H,W) in polar space
                    loss = crit(logits, batch["mask"])

                if train:
                    if amp_enabled:
                        self.scaler.scale(loss / grad_acc).backward()
                    else:
                        (loss / grad_acc).backward()

                    if (step + 1) % grad_acc == 0:
                        if amp_enabled:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            opt.step()
                        opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                prob = torch.sigmoid(logits)
                for i in range(prob.shape[0]):
                    # average disc+cup dice in polar domain vs polar GT
                    d_disc = dice_coef_prob(prob[i, 0:1], batch["mask"][i, 0:1])
                    d_cup = dice_coef_prob(prob[i, 1:2], batch["mask"][i, 1:2])
                    total_dice += 0.5 * (d_disc + d_cup)
                    n += 1
                total_loss += float(loss.detach()) * batch["mask"].shape[0]

        # Reduce across ranks
        loss_tensor = torch.tensor([total_loss, float(n)], device=self.device)
        dice_tensor = torch.tensor([total_dice, float(n)], device=self.device)
        loss_tensor = self.dist.all_reduce_sum(loss_tensor)
        dice_tensor = self.dist.all_reduce_sum(dice_tensor)

        loss_mean = (loss_tensor[0].item() / max(1.0, loss_tensor[1].item()))
        dice_mean = (dice_tensor[0].item() / max(1.0, dice_tensor[1].item()))
        return {"loss": loss_mean, "dice": dice_mean}

    # ---------- Public loops ----------
    def fit(self):
        if not self.cfg.do_train:
            return
        if self.dist.is_main:
            print("[INFO] Starting training…")
        for epoch in range(1, self.cfg.epochs + 1):
            self._epoch = epoch  # for sampler
            tr = self._run_epoch(train=True)
            va = self._run_epoch(train=False)
            self.history["train_loss"].append(tr["loss"])
            self.history["train_dice"].append(tr["dice"])
            self.history["val_dice"].append(va["dice"])

            if self.dist.is_main:
                print(
                    f"[E{epoch:03d}] loss={tr['loss']:.4f} | dice(tr)={tr['dice']:.4f} | dice(val)={va['dice']:.4f}"
                )
                # Save last + best
                self._save_ckpt(self.last_path)
                if va["dice"] > self.best_val:
                    self.best_val = va["dice"]
                    self._save_ckpt(self.best_path)
            self.dist.barrier()

        if self.dist.is_main:
            metrics_path = Path(self.cfg.out_dir) / f"{self.cfg.run_name}_metrics.json"
            out = {"history": self.history, "best_val_dice": float(self.best_val)}
            metrics_path.write_text(json.dumps(out, indent=2))
            print(f"[OK] Training complete. Best val Dice={self.best_val:.4f} -> {self.best_path}")

    def test(self) -> Dict[str, Any]:
        if not self.cfg.do_test:
            return {}
        # Only rank-0 performs testing + file outputs
        if not self.dist.is_main:
            self.dist.barrier()
            return {}

        ckpt = (
            self.cfg.test_weights
            if self.cfg.test_weights
            else (self.best_path if self.best_path.exists() else self.last_path)
        )
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint found for testing: {ckpt}")
        self._load_resume(ckpt)

        self._unwrap().eval()
        out_dir = Path(self.cfg.out_dir)
        disc_dir = out_dir / "pred_masks" / "disc"
        cup_dir = out_dir / "pred_masks" / "cup"
        disc_dir.mkdir(parents=True, exist_ok=True)
        cup_dir.mkdir(parents=True, exist_ok=True)
        jl_path = out_dir / "test_predictions.jsonl"

        summary = {"disc_sum": 0.0, "cup_sum": 0.0, "disc_n": 0, "cup_n": 0}
        with jl_path.open("w") as jf, torch.no_grad():
            for batch in self.dl_test:
                metas = batch["meta"]
                batch = {k: v for k, v in batch.items() if k != "meta"}
                batch = _to_device(batch, self.device)
                logits, _ = self._unwrap()(batch)  # (B,2,H,W) in polar
                prob = torch.sigmoid(logits)

                for i in range(prob.shape[0]):
                    stem = metas[i]["stem"]

                    # Dice in polar domain
                    d_disc = dice_coef_prob(prob[i, 0:1], batch["mask"][i, 0:1])
                    d_cup = dice_coef_prob(prob[i, 1:2], batch["mask"][i, 1:2])

                    # Threshold in polar
                    disc_pol = (prob[i, 0].cpu().numpy() >= 0.5).astype(np.uint8) * 255
                    cup_pol = (prob[i, 1].cpu().numpy() >= 0.5).astype(np.uint8) * 255

                    # Convert back to Cartesian (letterboxed 1024×1024)
                    disc_cart = polar_to_cartesian(disc_pol, out_h=self.cfg.imgsz, out_w=self.cfg.imgsz)
                    cup_cart = polar_to_cartesian(cup_pol, out_h=self.cfg.imgsz, out_w=self.cfg.imgsz)

                    disc_cart = disc_cart.astype(np.uint8)
                    cup_cart = cup_cart.astype(np.uint8)

                    save_disc = disc_dir / f"{stem}.png"
                    save_cup = cup_dir / f"{stem}.png"
                    PILImage.fromarray(disc_cart).save(str(save_disc))
                    PILImage.fromarray(cup_cart).save(str(save_cup))

                    rec = {
                        "image": metas[i]["image_path"],
                        "stem": stem,
                        "dice_disc": float(d_disc),
                        "dice_cup": float(d_cup),
                        "prompt": self.cfg.test_prompt,
                        "mask_disc_path": str(save_disc),
                        "mask_cup_path": str(save_cup),
                    }
                    jf.write(json.dumps(rec) + "\n")

                    summary["disc_sum"] += float(d_disc)
                    summary["cup_sum"] += float(d_cup)
                    summary["disc_n"] += 1
                    summary["cup_n"] += 1

        disc_mean = summary["disc_sum"] / max(1, summary["disc_n"])
        cup_mean = summary["cup_sum"] / max(1, summary["cup_n"])
        overall = (summary["disc_sum"] + summary["cup_sum"]) / max(
            1, summary["disc_n"] + summary["cup_n"]
        )
        test_summary = {
            "disc_mean_dice": disc_mean,
            "cup_mean_dice": cup_mean,
            "overall_mean_dice": overall,
            "counts": {"disc": summary["disc_n"], "cup": summary["cup_n"]},
            "checkpoint": str(ckpt),
        }
        # Append to metrics
        metrics_path = Path(self.cfg.out_dir) / f"{self.cfg.run_name}_metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
        else:
            m = {}
        m["test"] = test_summary
        metrics_path.write_text(json.dumps(m, indent=2))
        print(f"[OK] Test summary: {test_summary}")
        self.dist.barrier()
        return test_summary


# =========================================================
# Public programmatic API
# =========================================================
def run_finetune(cfg: TrainConfig) -> Dict[str, Any]:
    dist_cfg = DistConfig()
    dist = DistributedContext(dist_cfg)
    set_global_seed(cfg.seed + dist_cfg.rank)
    dist.setup()
    try:
        trainer = Trainer(cfg, dist)
        trainer.fit()
        result = trainer.test()
        return {"best_val_dice": trainer.best_val, "test": result}
    finally:
        dist.cleanup()


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Fundu-style MedSAM (multi-GPU ready) with adapters+CBAM, polar coords, joint OD/OC loss."
    )
    # Data / IO
    p.add_argument("--data-root", type=Path, required=True, help="Root datasets scanned by ImageFactory.")
    p.add_argument("--yolo-ds", type=Path, required=True, help="YOLO dataset directory containing data.yaml.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for metrics/preds.")
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory for checkpoints.")
    p.add_argument("--run-name", type=str, default="FunduSAMFinetune")
    p.add_argument("--exclude-ds", nargs="*", default=None, help="Datasets to exclude by substring.")

    # Detector
    p.add_argument("--yolo-weights", type=Path, required=True, help="Path to trained YOLO weights.")
    p.add_argument(
        "--yolo-device",
        type=str,
        default=DEFAULT_YOLO_DEVICE,
        help=f"YOLO device (default: {DEFAULT_YOLO_DEVICE})",
    )
    p.add_argument(
        "--yolo-imgsz",
        type=int,
        default=DEFAULT_YOLO_IMGSZ,
        help=f"YOLO inference size (default: {DEFAULT_YOLO_IMGSZ})",
    )
    p.add_argument(
        "--yolo-conf",
        type=float,
        default=DEFAULT_YOLO_CONF,
        help=f"YOLO confidence threshold (default: {DEFAULT_YOLO_CONF})",
    )
    p.add_argument(
        "--yolo-iou",
        type=float,
        default=DEFAULT_YOLO_IOU,
        help=f"YOLO NMS IoU threshold (default: {DEFAULT_YOLO_IOU})",
    )
    p.add_argument("--det-cache", type=Path, default=None, help="Optional JSONL cache path for detector boxes.")

    # Model/opt
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"SAM backbone type (default: {DEFAULT_MODEL})",
    )
    p.add_argument("--ckpt", type=Path, required=True, help="MedSAM checkpoint.")
    p.add_argument("--unfreeze-encoders", action="store_true")
    p.add_argument(
        "--save-full",
        action="store_true",
        help="Deprecated: kept for CLI compatibility; new checkpoints always save full model.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of epochs (default: {DEFAULT_EPOCHS})",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Batch size (default: {DEFAULT_BATCH})",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Input resolution (default: {DEFAULT_IMGSZ})",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate (default: {DEFAULT_LR})",
    )
    p.add_argument(
        "--wd",
        type=float,
        default=DEFAULT_WD,
        help=f"Weight decay (default: {DEFAULT_WD})",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Dataloader workers (default: {DEFAULT_WORKERS})",
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-acc-steps", type=int, default=1)
    p.add_argument("--bucket-cap-mb", type=int, default=25)
    p.add_argument("--find-unused-parameters", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)

    # Prompt/augmentation (Option B controls)
    p.add_argument(
        "--use-det-prob",
        type=float,
        default=DEFAULT_USE_DET_PROB,
        help=f"Prob. of using DET box vs GT during train (default: {DEFAULT_USE_DET_PROB})",
    )
    p.add_argument(
        "--pad-jitter",
        type=float,
        default=DEFAULT_PAD_JITTER,
        help=f"Box padding jitter fraction (default: {DEFAULT_PAD_JITTER})",
    )
    p.add_argument(
        "--box-jitter-tr",
        type=float,
        default=DEFAULT_BOX_TR,
        help=f"Box translation jitter (default: {DEFAULT_BOX_TR})",
    )
    p.add_argument(
        "--box-jitter-sc",
        type=float,
        default=DEFAULT_BOX_SC,
        help=f"Box scale jitter (default: {DEFAULT_BOX_SC})",
    )

    # Modes
    p.add_argument("--train", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--test-prompt", choices=["det", "gt"], default="det")
    p.add_argument("--test-weights", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        data_root=args.data_root,
        yolo_ds=args.yolo_ds,
        out_dir=args.out_dir,
        run_dir=args.run_dir,
        run_name=args.run_name,
        yolo_weights=args.yolo_weights,
        exclude_datasets=args.exclude_ds,
        yolo_device=args.yolo_device,
        yolo_imgsz=args.yolo_imgsz,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        det_cache=args.det_cache,
        model=args.model,
        ckpt=args.ckpt,
        unfreeze_encoders=args.unfreeze_encoders,
        save_full=args.save_full,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr=args.lr,
        wd=args.wd,
        workers=args.workers,
        amp=args.amp or True,  # prefer AMP unless explicitly disabled downstream
        grad_acc_steps=args.grad_acc_steps,
        bucket_cap_mb=args.bucket_cap_mb,
        find_unused_parameters=args.find_unused_parameters,
        seed=args.seed,
        use_det_prob=args.use_det_prob,
        pad_jitter=args.pad_jitter,
        box_jitter_tr=args.box_jitter_tr,
        box_jitter_sc=args.box_jitter_sc,
        do_train=args.train,
        do_test=args.test,
        test_prompt=args.test_prompt,
        test_weights=args.test_weights,
        resume=args.resume,
    )
    res = run_finetune(cfg)
    if DistConfig().rank == 0:
        print(json.dumps({"result": res, "config": asdict(cfg)}, indent=2))


if __name__ == "__main__":
    main()