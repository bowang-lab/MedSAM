#!/usr/bin/env python3
# src/scripts/finetune_medsam.py
# Fine-tune MedSAM for OD/OC segmentation using box prompts.
# - Uses YOLO dataset splits (train/val/test) from --yolo-ds
# - Trains on a MIX of YOLO-predicted boxes and GT-tight boxes (probability --use-det-prob)
# - Validates on GT-tight boxes only for stable model selection
# - Tests on detector boxes by default (or GT via --test-prompt), saves masks and per-class Dice
# - Stores paths/boxes/predictions in your Image dataclass

from __future__ import annotations

import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import inspect

import numpy as np
from PIL import Image as PILImage, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Your project imports
from src.imgpipe.image_factory import ImageFactory
from src.imgpipe.image import Image as IMG
from src.imgpipe.normalized_box import NormalizedBox
from src.imgpipe.enums import Structure, LabelType  # expects Structure.DISC/CUP, LabelType.GT/PRED

# Detector
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    _YOLO_ERR = e

# SAM
try:
    from segment_anything import sam_model_registry
except Exception as e:
    sam_model_registry = None
    _SAM_IMPORT_ERR = e

# YAML (for YOLO data.yaml)
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

# =========================
# Defaults
# =========================

DEFAULT_DATA_ROOT = Path("/Users/carlosperez/Library/CloudStorage/OneDrive-UBC/Ipek_Carlos/GlaucomaDatasets/All_Datasets_Organized")
DEFAULT_OUT_DIR   = Path("/Users/carlosperez/PycharmProjects/MedSAM/MEDSAM_TRAIN")
DEFAULT_RUN_DIR   = Path("/Users/carlosperez/PycharmProjects/MedSAM/runs/runs_medsam")
DEFAULT_MODEL     = "vit_b"  # SAM backbone key for registry
DEFAULT_CKPT      = Path("/Users/carlosperez/PycharmProjects/MedSAM/work_dir/MedSAM/medsam_vit_b.pth")

DEFAULT_DEVICE: Optional[str | int] = "mps"
DEFAULT_IMGSZ = 1024
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 8
DEFAULT_WORKERS = 8
SEED = 42

# =========================
# Utilities
# =========================
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
    except Exception:
        pass

def _ensure_sam_available():
    if sam_model_registry is None:
        raise RuntimeError(
            f"segment-anything not available. Import error: {_SAM_IMPORT_ERR!r}\n"
            "Install: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

def _ensure_yolo_available():
    if YOLO is None:
        raise RuntimeError(
            f"Ultralytics YOLO not available. Import error: {_YOLO_ERR!r}\n"
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

# =========================
# YOLO split helpers
# =========================
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _parse_data_yaml(yaml_path: Path) -> Dict[str, str]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")
    if yaml:
        data = yaml.safe_load(yaml_path.read_text())
        return {k: str(data.get(k)) for k in ("train", "val", "test") if k in data}
    # Fallback minimal parse
    out: Dict[str, str] = {}
    for ln in yaml_path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"): continue
        if ln.startswith("train:"): out["train"] = ln.split(":", 1)[1].strip()
        if ln.startswith("val:"):   out["val"]   = ln.split(":", 1)[1].strip()
        if ln.startswith("test:"):  out["test"]  = ln.split(":", 1)[1].strip()
    return out

def _read_list(p: Path) -> List[Path]:
    with p.open("r") as f:
        return [Path(ln.strip()) for ln in f if ln.strip()]

def _resolve_split_images(yolo_ds: Path, entry: str) -> List[Path]:
    p = Path(entry)
    if not p.is_absolute():
        p = (yolo_ds / p).resolve()
    if p.suffix.lower() == ".txt":
        return _read_list(p)
    # directory of images
    if p.is_dir():
        return sorted([q for q in p.rglob("*") if q.suffix.lower() in _IMG_EXTS])
    # single image
    return [p] if p.suffix.lower() in _IMG_EXTS else []

# =========================
# Build Image objects and map to splits
# =========================
def build_image_index(data_root: Path, exclude: Optional[List[str]] = None) -> Dict[str, IMG]:
    fac = ImageFactory(root=data_root, auto_scan=True)
    fac.filter_empty_masks()
    if exclude:
        fac.filter_datasets(exclude=exclude)
    images = fac.make_images()
    idx: Dict[str, IMG] = {}
    for im in images:
        idx[Path(im.image_path).stem] = im
    if not idx:
        raise RuntimeError("No images discovered by ImageFactory.")
    return idx

def images_from_yolo_splits(yolo_ds: Path, data_root: Path, exclude: Optional[List[str]] = None
                            ) -> Tuple[List[IMG], List[IMG], List[IMG]]:
    mapping = _parse_data_yaml(yolo_ds / "data.yaml")
    train_imgs = _resolve_split_images(yolo_ds, mapping["train"])
    val_imgs   = _resolve_split_images(yolo_ds, mapping["val"])
    test_imgs  = _resolve_split_images(yolo_ds, mapping["test"])

    print(f"[INFO] YOLO split sizes → train={len(train_imgs)} val={len(val_imgs)} test={len(test_imgs)}")

    idx = build_image_index(data_root, exclude)
    miss = 0

    def to_images(ps: List[Path]) -> List[IMG]:
        nonlocal miss
        out: List[IMG] = []
        for p in ps:
            stem = Path(p).stem
            im = idx.get(stem)
            if im is None:
                miss += 1
                continue
            im.set_split(None)  # we manage split externally
            out.append(im)
        return out

    tr = to_images(train_imgs)
    va = to_images(val_imgs)
    te = to_images(test_imgs)
    if miss:
        print(f"[WARN] {miss} YOLO-split images not found in ImageFactory index; skipped.")
    return tr, va, te

# =========================
# YOLO detector → boxes
# =========================
class YOLOBoxProvider:
    """Caches YOLO predictions; returns NormalizedBox per structure (0=disc, 1=cup)."""
    def __init__(self, weights: Path, device: str = "cpu", imgsz: int = 640,
                 conf: float = 0.25, iou: float = 0.5):
        _ensure_yolo_available()
        self.model = YOLO(str(weights))
        self.device, self.imgsz, self.conf, self.iou = device, imgsz, conf, iou
        self.cache: Dict[str, Dict[str, Optional[NormalizedBox]]] = {}

    def __call__(self, image: IMG) -> Dict[str, Optional[NormalizedBox]]:
        key = str(image.image_path)
        if key in self.cache:
            return self.cache[key]
        r = self.model.predict(
            source=str(image.image_path),
            device=self.device, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            verbose=False
        )[0]
        out = {"disc": None, "cup": None}
        if getattr(r, "boxes", None) is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            for b, c in zip(xyxy, cls):
                x1, y1, x2, y2 = map(float, b)
                nb = NormalizedBox.from_xyxy(x1, y1, x2, y2, image.width, image.height)
                if c == 0 and out["disc"] is None:
                    out["disc"] = nb
                elif c == 1 and out["cup"] is None:
                    out["cup"] = nb
        self.cache[key] = out
        return out

def attach_detector_boxes(images: List[IMG], det: Optional[YOLOBoxProvider]) -> None:
    """Populate image.inter_pred_disc_box/inter_pred_cup_box using YOLO."""
    if det is None:
        return
    for im in images:
        preds = det(im)
        if preds["disc"] is not None:
            im.set_box(Structure.DISC, LabelType.PRED, preds["disc"])
        if preds["cup"] is not None:
            im.set_box(Structure.CUP, LabelType.PRED, preds["cup"])

# =========================
# Geometry / preprocessing
# =========================
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
        pad_top  = pad_h // 2
        return scale, new_w, new_h, pad_left, pad_top

    def __call__(self, img: PILImage.Image, mask: PILImage.Image, box_xyxy: np.ndarray
                 ) -> Tuple[PILImage.Image, PILImage.Image, np.ndarray]:
        w, h = img.size
        scale, new_w, new_h, pad_left, pad_top = self._compute_pad(w, h)
        img_r  = img.resize((new_w, new_h), PILImage.BILINEAR)
        mask_r = mask.resize((new_w, new_h), PILImage.NEAREST)
        new_img  = PILImage.new("RGB", (self.size, self.size))
        new_mask = PILImage.new("L",   (self.size, self.size))
        new_img.paste(img_r,  (pad_left, pad_top))
        new_mask.paste(mask_r,(pad_left, pad_top))
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
    s  = 1.0 + (2 * random.random() - 1) * sc
    nw = max(1.0, w * s)
    nh = max(1.0, h * s)
    cx += dx; cy += dy
    nx0 = max(0.0, cx - nw / 2)
    ny0 = max(0.0, cy - nh / 2)
    nx1 = min(float(img_w - 1), cx + nw / 2)
    ny1 = min(float(img_h - 1), cy + nh / 2)
    return np.array([nx0, ny0, nx1, ny1], dtype=np.float32)

PIXEL_MEAN = torch.tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
PIXEL_STD  = torch.tensor([58.395, 57.120, 57.375]).view(3, 1, 1)

def preprocess_for_sam(img_pil: PILImage.Image) -> torch.Tensor:
    x = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float()  # RGB 0..255
    return (x - PIXEL_MEAN) / PIXEL_STD

def nbox_to_xyxy(nbox: NormalizedBox, W: int, H: int) -> np.ndarray:
    x1, y1, x2, y2 = nbox.to_pixel_xyxy(W, H)
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def color_jitter_safe(img: PILImage.Image, brightness: float = 0.10, contrast: float = 0.10) -> PILImage.Image:
    # light photometric jitter (fundus-safe ranges)
    if brightness > 0:
        fac = 1.0 + random.uniform(-brightness, brightness)
        img = ImageEnhance.Brightness(img).enhance(fac)
    if contrast > 0:
        fac = 1.0 + random.uniform(-contrast, contrast)
        img = ImageEnhance.Contrast(img).enhance(fac)
    return img

# =========================
# Dataset over Image objects
# =========================
@dataclass(frozen=True)
class SegItem:
    image: IMG
    structure: str  # "disc" | "cup"

def make_items(images: List[IMG], require_gt: bool = True) -> List[SegItem]:
    out: List[SegItem] = []
    for im in images:
        # ensure boxes from masks are available in normalized space
        im.ensure_boxes_from_masks()
        if not require_gt:
            out.append(SegItem(im, "disc"))
            out.append(SegItem(im, "cup"))
            continue
        # add only if GT mask for the structure exists
        if im.gt_disc_mask is not None and getattr(im.gt_disc_mask, "path", None):
            out.append(SegItem(im, "disc"))
        if im.gt_cup_mask is not None and getattr(im.gt_cup_mask, "path", None):
            out.append(SegItem(im, "cup"))
    return out

class MedSAMDataset(Dataset):
    def __init__(self,
                 items: List[SegItem],
                 img_size: int = 1024,
                 train: bool = True,
                 use_det_prob: float = 0.5,
                 pad_jitter: float = 0.30,
                 box_tr: float = 0.05,
                 box_sc: float = 0.10,
                 prompt_mode: str = "mix"  # "mix" (train only), "gt", "det"
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
        # preferred normalized box choice per mode
        if struct == "disc":
            gt = im.gt_disc_box
            det = im.inter_pred_disc_box
        else:
            gt = im.gt_cup_box
            det = im.inter_pred_cup_box

        if self.train and self.prompt_mode == "mix":
            use_det = (random.random() < self.use_det_prob) and (det is not None)
            return det if use_det else gt
        if self.prompt_mode == "det":
            return det if det is not None else gt
        # default "gt"
        return gt

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        im: IMG = item.image
        struct = item.structure
        img = PILImage.open(im.image_path).convert("RGB")
        # pick mask path by structure
        mref = im.gt_disc_mask if struct == "disc" else im.gt_cup_mask
        if mref is None or getattr(mref, "path", None) is None:
            raise RuntimeError(f"Missing GT mask for {struct}: {im.image_path}")
        m = PILImage.open(mref.path).convert("L")

        # optional photometric jitter (train only)
        if self.train:
            img = color_jitter_safe(img, brightness=0.10, contrast=0.10)

        # pick base normalized box per mode
        base_nbox = self._choose_nbox(im, struct)

        # fallback: compute from GT mask if absent
        if base_nbox is None:
            mask_np = np.array(m, dtype=np.uint8)
            tb = mask_to_tight_box(mask_np)
            if tb is None:
                # ultimate fallback: whole image
                tb = np.array([0, 0, img.size[0] - 1, img.size[1] - 1], dtype=np.float32)
            box_xyxy = tb
        else:
            box_xyxy = nbox_to_xyxy(base_nbox, im.width, im.height)

        # pad + (train-only) jitter
        pad_frac = (random.uniform(0.0, self.pad_jitter) if self.train else 0.0)
        box_p = pad_box(box_xyxy, pad_frac, im.width, im.height)
        if self.train and (self.box_tr > 0.0 or self.box_sc > 0.0):
            box_p = jitter_box_xyxy(box_p, im.width, im.height, tr=self.box_tr, sc=self.box_sc)

        # resize & map box
        img_r, mask_r, box_t = self.letterbox(img, m, box_p)

        x = preprocess_for_sam(img_r)
        mask_arr = np.array(mask_r, dtype=np.uint8)
        y = torch.from_numpy((mask_arr > 0).astype(np.float32)).unsqueeze(0)
        b = torch.from_numpy(box_t.astype(np.float32))
        meta = {
            "image_path": str(im.image_path),
            "structure": struct,
            "W": im.width,
            "H": im.height,
            "stem": Path(im.image_path).stem,
        }
        return {"image": x, "mask": y, "box": b, "meta": meta}

# =========================
# Losses / Metrics
# =========================
class BCEDice(nn.Module):
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
        dice = dice.mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice

def dice_coef_prob(prob: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    pred = (prob >= thresh).float()
    inter = (pred * target).sum().item()
    den = pred.sum().item() + target.sum().item()
    if den == 0:
        return 1.0
    return (2.0 * inter) / den

# =========================
# SAM wrapper
# =========================
class MedSAMFinetuner(nn.Module):
    def __init__(self, sam_type: str, checkpoint: Path, freeze_encoders: bool = True):
        super().__init__()
        _ensure_sam_available()
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"MedSAM checkpoint not found: {checkpoint}")
        self.sam = sam_model_registry[sam_type](checkpoint=str(checkpoint))
        self.sam.eval()

        # Freeze encoders by default (decoder-only finetune)
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = not freeze_encoders
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = not freeze_encoders
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch["image"]  # (B,3,1024,1024)
        b = batch["box"]    # (B,4)
        B = x.shape[0]

        image_embeddings = self.sam.image_encoder(x)
        dense_pe = self.sam.prompt_encoder.get_dense_pe()

        boxes = b[:, None, :]
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=boxes, masks=None
        )

        md = self.sam.mask_decoder
        sig = inspect.signature(md.forward)
        params = set(sig.parameters.keys())
        md_kwargs = dict(
            image_embeddings=image_embeddings,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        zero_prior = torch.zeros((B, 1, 256, 256), device=x.device, dtype=image_embeddings.dtype)
        if "mask_input" in params:
            md_kwargs["mask_input"] = zero_prior
        elif "low_res_mask" in params:
            md_kwargs["low_res_mask"] = zero_prior
        elif "low_res_masks" in params:
            md_kwargs["low_res_masks"] = zero_prior

        low_res_masks, iou_pred = md(**md_kwargs)
        logits_1024 = F.interpolate(low_res_masks, size=(x.shape[2], x.shape[3]),
                                    mode="bilinear", align_corners=False)
        return logits_1024, iou_pred

# =========================
# Train / Val
# =========================
def run_one_epoch(model: MedSAMFinetuner, loader: DataLoader, optim: torch.optim.Optimizer,
                  loss_fn: nn.Module, device: torch.device, train: bool) -> Dict[str, float]:
    model.train(mode=train)
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    for batch in loader:
        meta = batch.get("meta", None)
        batch = {k: v for k, v in batch.items() if k != "meta"}  # keep tensors only
        batch = _to_device(batch, device)
        with torch.set_grad_enabled(train):
            logits, _ = model(batch)
            loss = loss_fn(logits, batch["mask"])
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                for i in range(prob.shape[0]):
                    total_dice += dice_coef_prob(prob[i], batch["mask"][i])
                    n += 1
            total_loss += loss.item() * batch["mask"].shape[0]
    return {"loss": total_loss / max(1, n), "dice": total_dice / max(1, n)}

def evaluate(model: MedSAMFinetuner, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_dice = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            meta = batch.get("meta", None)
            batch = {k: v for k, v in batch.items() if k != "meta"}
            batch = _to_device(batch, device)
            logits, _ = model(batch)
            prob = torch.sigmoid(logits)
            for i in range(prob.shape[0]):
                total_dice += dice_coef_prob(prob[i], batch["mask"][i])
                n += 1
    return {"dice": total_dice / max(1, n)}

# =========================
# Test: save masks + per-class metrics
# =========================
def predict_and_save(model: MedSAMFinetuner, loader: DataLoader, device: torch.device,
                     out_dir: Path, prompt_used: str) -> Dict[str, Any]:
    model.eval()
    out_dir = Path(out_dir)
    disc_dir = out_dir / "pred_masks" / "disc"
    cup_dir  = out_dir / "pred_masks" / "cup"
    (disc_dir).mkdir(parents=True, exist_ok=True)
    (cup_dir).mkdir(parents=True, exist_ok=True)

    jl_path = out_dir / "test_predictions.jsonl"
    summary: Dict[str, float] = {
        "disc_dice_sum": 0.0, "disc_n": 0,
        "cup_dice_sum": 0.0,  "cup_n": 0,
    }

    with jl_path.open("w") as jf:
        with torch.no_grad():
            for batch in loader:
                metas = batch["meta"]
                batch = {k: v for k, v in batch.items() if k != "meta"}
                batch = _to_device(batch, device)
                logits, _ = model(batch)
                prob = torch.sigmoid(logits)

                B = prob.shape[0]
                for i in range(B):
                    meta = metas if isinstance(metas, dict) else metas[i]
                    structure = meta["structure"]
                    stem = meta["stem"]

                    # Dice
                    d = dice_coef_prob(prob[i], batch["mask"][i])

                    # Save mask
                    mask_bin = (prob[i, 0].cpu().numpy() >= 0.5).astype(np.uint8) * 255
                    save_p = (disc_dir if structure == "disc" else cup_dir) / f"{stem}.png"
                    PILImage.fromarray(mask_bin).save(str(save_p))

                    # Update per-class aggregates
                    if structure == "disc":
                        summary["disc_dice_sum"] += float(d)
                        summary["disc_n"] += 1
                    else:
                        summary["cup_dice_sum"] += float(d)
                        summary["cup_n"] += 1

                    # JSONL record
                    rec = {
                        "image": meta["image_path"],
                        "stem": stem,
                        "class": structure,
                        "dice": float(d),
                        "prompt": prompt_used,
                        "mask_path": str(save_p),
                    }
                    jf.write(json.dumps(rec) + "\n")

    # finalize summary
    disc_mean = (summary["disc_dice_sum"] / max(1, summary["disc_n"]))
    cup_mean  = (summary["cup_dice_sum"]  / max(1, summary["cup_n"]))
    overall   = (summary["disc_dice_sum"] + summary["cup_dice_sum"]) / max(1, summary["disc_n"] + summary["cup_n"])
    return {
        "disc_mean_dice": disc_mean,
        "cup_mean_dice": cup_mean,
        "overall_mean_dice": overall,
        "counts": {"disc": summary["disc_n"], "cup": summary["cup_n"]},
    }

# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune MedSAM with a mix of YOLO-predicted and GT boxes.")
    # Modes
    p.add_argument("--train", action="store_true", help="Enable training.")
    p.add_argument("--test",  action="store_true", help="Enable testing after training or with --test-weights.")
    p.add_argument("--test-weights", type=Path, help="Checkpoint (*.pth) to load for testing (if not using best).")

    # Paths/config
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root datasets scanned by ImageFactory.")
    p.add_argument("--yolo-ds",   type=Path, required=True, help="YOLO dataset directory containing data.yaml.")
    p.add_argument("--out-dir",   type=Path, default=DEFAULT_OUT_DIR, help="Output dir for metrics/preds.")
    p.add_argument("--run-dir",   type=Path, default=DEFAULT_RUN_DIR, help="Runs dir for checkpoints.")
    p.add_argument("--model",     type=str,  default=DEFAULT_MODEL,   help="SAM backbone key (e.g., vit_b).")
    p.add_argument("--ckpt",      type=Path, default=DEFAULT_CKPT,    help="MedSAM ViT-B checkpoint (.pth).")
    p.add_argument("--run-name",  type=str,  default="MedSAMFinetune", help="Run name for directory naming.")

    # Training knobs
    p.add_argument("--device", type=str, default=str(DEFAULT_DEVICE), help="'cpu', 'mps', 'cuda:0', etc.")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch",  type=int, default=DEFAULT_BATCH)
    p.add_argument("--imgsz",  type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--wd",     type=float, default=1e-4)
    p.add_argument("--resume", type=Path, help="Resume from trainable checkpoint (*.pth).")

    # Prompt/augmentation knobs
    p.add_argument("--use-det-prob", type=float, default=0.5, help="Prob. to use detector box during TRAIN.")
    p.add_argument("--pad-jitter",   type=float, default=0.30, help="Random padding (fraction of box size).")
    p.add_argument("--box-jitter-tr",type=float, default=0.05, help="Translation jitter fraction.")
    p.add_argument("--box-jitter-sc",type=float, default=0.10, help="Scale jitter fraction.")

    # Prompt mode for VAL/TEST
    p.add_argument("--test-prompt", choices=["det", "gt"], default="det",
                   help="Prompt type during TEST. Validation always uses GT.")
    # Freezing
    p.add_argument("--unfreeze-encoders", action="store_true",
                   help="If set, unfreeze image+prompt encoders (full finetune). Default: decoder-only.")

    # Detector config
    p.add_argument("--yolo-weights", type=Path, required=True, help="YOLO .pt weights for detector boxes.")
    p.add_argument("--yolo-device",  type=str, default="cuda:0")
    p.add_argument("--yolo-imgsz",   type=int, default=640)
    p.add_argument("--yolo-conf",    type=float, default=0.25)
    p.add_argument("--yolo-iou",     type=float, default=0.50)

    return p.parse_args()

# =========================
# Main
# =========================
def main():
    os.environ.setdefault("PYTHONHASHSEED", "0")
    set_global_seed(SEED)
    args = parse_args()

    device = torch.device(args.device if args.device != "cpu" else "cpu")
    (args.out_dir).mkdir(parents=True, exist_ok=True)
    (args.run_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] DATA_ROOT = {args.data_root}")
    print(f"[INFO] YOLO_DS   = {args.yolo_ds}")
    print(f"[INFO] OUT_DIR   = {args.out_dir}")
    print(f"[INFO] RUN_DIR   = {args.run_dir}")
    print(f"[INFO] MODEL/CKPT= {args.model} / {args.ckpt}")
    print(f"[INFO] DEVICE    = {device}")
    print(f"[INFO] EPOCHS    = {args.epochs} | BATCH = {args.batch} | IMGSZ = {args.imgsz}")
    print(f"[INFO] use-det-prob={args.use_det_prob:.2f} | pad-jitter={args.pad_jitter:.2f} | tr={args.box_jitter_tr:.2f} | sc={args.box_jitter_sc:.2f}")

    # Build Image lists for each split and attach detector boxes
    train_imgs, val_imgs, test_imgs = images_from_yolo_splits(args.yolo_ds, args.data_root, exclude=["PAPILA"])

    det = YOLOBoxProvider(args.yolo_weights, device=args.yolo_device, imgsz=args.yolo_imgsz,
                          conf=args.yolo_conf, iou=args.yolo_iou)

    # Attach detector normalized boxes to Image objects for all splits
    for split_list in (train_imgs, val_imgs, test_imgs):
        attach_detector_boxes(split_list, det)

    # Create items (one per {image, structure} that has a GT mask)
    train_items = make_items(train_imgs, require_gt=True)
    val_items   = make_items(val_imgs,   require_gt=True)
    test_items  = make_items(test_imgs,  require_gt=True)

    print(f"[INFO] Items → train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    # Datasets/Loaders
    ds_train = MedSAMDataset(
        train_items, img_size=args.imgsz, train=True, use_det_prob=args.use_det_prob,
        pad_jitter=args.pad_jitter, box_tr=args.box_jitter_tr, box_sc=args.box_jitter_sc,
        prompt_mode="mix"
    )
    ds_val = MedSAMDataset(
        val_items, img_size=args.imgsz, train=False, use_det_prob=0.0,
        pad_jitter=0.0, box_tr=0.0, box_sc=0.0, prompt_mode="gt"  # always GT for validation
    )
    ds_test = MedSAMDataset(
        test_items, img_size=args.imgsz, train=False, use_det_prob=0.0,
        pad_jitter=0.0, box_tr=0.0, box_sc=0.0,
        prompt_mode=("det" if args.test_prompt == "det" else "gt")
    )

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=DEFAULT_WORKERS, pin_memory=True, drop_last=False)
    dl_val   = DataLoader(ds_val,   batch_size=max(1, args.batch // 2), shuffle=False,
                          num_workers=DEFAULT_WORKERS, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=max(1, args.batch // 2), shuffle=False,
                          num_workers=DEFAULT_WORKERS, pin_memory=True)

    # Model
    model = MedSAMFinetuner(
        sam_type=args.model,
        checkpoint=args.ckpt,
        freeze_encoders=not args.unfreeze_encoders,
    ).to(device)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)
    loss_fn = BCEDice(bce_weight=0.5)

    # Run paths
    run_path = Path(args.run_dir) / args.run_name
    weights_dir = run_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_path = weights_dir / "best.pth"
    last_path = weights_dir / "last.pth"
    metrics_path = Path(args.out_dir) / f"{args.run_name}_metrics.json"

    # Optionally resume
    if args.resume and Path(args.resume).exists():
        print(f"[INFO] Resuming from: {args.resume}")
        state = torch.load(str(args.resume), map_location="cpu")
        model.sam.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)

    # TRAIN
    best_val = -1.0
    history: Dict[str, List[float]] = {"train_dice": [], "val_dice": [], "train_loss": []}
    if args.train:
        print("[INFO] Starting training…")
        for epoch in range(1, args.epochs + 1):
            tr = run_one_epoch(model, dl_train, optimizer, loss_fn, device, train=True)
            va = evaluate(model, dl_val, device)
            history["train_loss"].append(tr["loss"])
            history["train_dice"].append(tr["dice"])
            history["val_dice"].append(va["dice"])
            print(f"[E{epoch:03d}] loss={tr['loss']:.4f} | dice(tr)={tr['dice']:.4f} | dice(val)={va['dice']:.4f}")

            # Save last
            torch.save({"mask_decoder": model.sam.mask_decoder.state_dict()}, str(last_path))

            # Best by val Dice
            if va["dice"] > best_val:
                best_val = va["dice"]
                torch.save({"mask_decoder": model.sam.mask_decoder.state_dict()}, str(best_path))

        # Save training history
        with open(metrics_path, "w") as f:
            json.dump({"history": history, "best_val_dice": float(best_val)}, f, indent=2)
        print(f"[OK] Training complete. Best val Dice={best_val:.4f} saved to {best_path}")

    # TEST
    if args.test:
        ckpt = args.test_weights if args.test_weights else (best_path if best_path.exists() else last_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint found for testing: {ckpt}")
        print(f"[INFO] Testing with weights: {ckpt}")
        state = torch.load(str(ckpt), map_location="cpu")
        model.sam.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)

        test_summary = predict_and_save(
            model, dl_test, device=device, out_dir=args.out_dir, prompt_used=args.test_prompt
        )
        # append to metrics
        if metrics_path.exists():
            m = json.loads(Path(metrics_path).read_text())
        else:
            m = {}
        m["test"] = test_summary
        Path(metrics_path).write_text(json.dumps(m, indent=2))
        print(f"[OK] Test summary: {test_summary}")

    if not args.train and not args.test:
        print("[WARN] No mode selected. Use --train and/or --test.")

if __name__ == "__main__":
    main()