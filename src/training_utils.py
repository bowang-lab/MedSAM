#!/usr/bin/env python3
# src/training_utils.py
"""
Shared utilities for MedSAM/YOLO training pipelines.

This module consolidates common training functions used across:
- finetune_medsam.py
- finetune_decoder.py
- other training scripts

Contents:
- YOLO data.yaml parsing
- Image preprocessing (letterbox, SAM normalization)
- Box manipulation (padding, jittering)
- Loss functions (BCE+Dice)
- Metrics (Dice coefficient)
- Data augmentation
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage, ImageEnhance

# YAML loading with fallback
try:
    import yaml
except ImportError:
    yaml = None


# ============================================================================
# Constants
# ============================================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# SAM normalization constants (ImageNet means/stds)
PIXEL_MEAN = torch.tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.120, 57.375]).view(3, 1, 1)


# ============================================================================
# YOLO data.yaml parsing
# ============================================================================

def parse_data_yaml(yaml_path: Path) -> Dict[str, str]:
    """
    Parse a YOLO data.yaml file and return paths for train/val/test splits.
    
    Args:
        yaml_path: Path to data.yaml file.
        
    Returns:
        Dict with keys 'train', 'val', 'test' (if present) mapping to their paths.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")
    
    if yaml is not None:
        data = yaml.safe_load(yaml_path.read_text())
        return {k: str(data.get(k)) for k in ("train", "val", "test") if k in data}
    
    # Fallback minimal parse without pyyaml
    out: Dict[str, str] = {}
    for ln in yaml_path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("train:"):
            out["train"] = ln.split(":", 1)[1].strip()
        if ln.startswith("val:"):
            out["val"] = ln.split(":", 1)[1].strip()
        if ln.startswith("test:"):
            out["test"] = ln.split(":", 1)[1].strip()
    return out


def read_file_list(p: Path) -> List[Path]:
    """Read a text file containing one path per line."""
    return [Path(ln.strip()) for ln in p.read_text().splitlines() if ln.strip()]


def resolve_split_images(yolo_ds: Path, entry: str) -> List[Path]:
    """
    Resolve split entry from data.yaml to actual image paths.
    
    Args:
        yolo_ds: YOLO dataset root directory.
        entry: Value from data.yaml (can be dir path, txt file, or image path).
        
    Returns:
        List of image paths.
    """
    p = Path(entry)
    if not p.is_absolute():
        p = (yolo_ds / p).resolve()
    
    if p.suffix.lower() == ".txt":
        return read_file_list(p)
    if p.is_dir():
        return sorted([q for q in p.rglob("*") if q.suffix.lower() in IMG_EXTS])
    return [p] if p.suffix.lower() in IMG_EXTS else []


# ============================================================================
# Image preprocessing
# ============================================================================

class LetterboxToSquare:
    """
    Letterbox an image and mask to a square size, preserving aspect ratio.
    Also transforms bounding boxes accordingly.
    """
    
    def __init__(self, size: int = 1024):
        self.size = size

    def _compute_pad(self, w: int, h: int) -> Tuple[float, int, int, int, int]:
        """Compute scale and padding for letterboxing."""
        scale = min(self.size / h, self.size / w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        pad_w = self.size - new_w
        pad_h = self.size - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        return scale, new_w, new_h, pad_left, pad_top

    def __call__(
        self,
        img: PILImage.Image,
        mask: PILImage.Image,
        box_xyxy: np.ndarray,
    ) -> Tuple[PILImage.Image, PILImage.Image, np.ndarray]:
        """
        Apply letterbox transform to image, mask, and bounding box.
        
        Args:
            img: RGB image.
            mask: Single-channel mask.
            box_xyxy: Bounding box as [x1, y1, x2, y2] in pixel coords.
            
        Returns:
            Tuple of (transformed_image, transformed_mask, transformed_box).
        """
        w, h = img.size
        scale, new_w, new_h, pad_left, pad_top = self._compute_pad(w, h)
        
        img_r = img.resize((new_w, new_h), PILImage.BILINEAR)
        mask_r = mask.resize((new_w, new_h), PILImage.NEAREST)
        
        new_img = PILImage.new("RGB", (self.size, self.size))
        new_mask = PILImage.new("L", (self.size, self.size))
        new_img.paste(img_r, (pad_left, pad_top))
        new_mask.paste(mask_r, (pad_left, pad_top))
        
        # Transform box
        x0, y0, x1, y1 = box_xyxy
        x0 = x0 * scale + pad_left
        y0 = y0 * scale + pad_top
        x1 = x1 * scale + pad_left
        y1 = y1 * scale + pad_top
        box_t = np.array([x0, y0, x1, y1], dtype=np.float32)
        
        return new_img, new_mask, box_t


def preprocess_for_sam(img_pil: PILImage.Image) -> torch.Tensor:
    """
    Preprocess PIL image for SAM model input.
    
    Args:
        img_pil: RGB PIL image.
        
    Returns:
        Normalized tensor of shape (3, H, W).
    """
    x = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float()  # RGB 0..255
    return (x - PIXEL_MEAN) / PIXEL_STD


# ============================================================================
# Box manipulation
# ============================================================================

def mask_to_tight_box(mask_np: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute tight bounding box from binary mask.
    
    Args:
        mask_np: Binary mask array.
        
    Returns:
        Box as [x0, y0, x1, y1] in pixel coords, or None if mask is empty.
    """
    ys, xs = np.nonzero(mask_np)
    if ys.size == 0 or xs.size == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def pad_box(
    box: np.ndarray,
    pad_frac: float,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """
    Pad a bounding box by a fraction of its size.
    
    Args:
        box: Box as [x0, y0, x1, y1].
        pad_frac: Fraction of box size to pad on each side.
        img_w: Image width (for clamping).
        img_h: Image height (for clamping).
        
    Returns:
        Padded box as [x0, y0, x1, y1].
    """
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


def jitter_box_xyxy(
    box: np.ndarray,
    img_w: int,
    img_h: int,
    tr: float = 0.05,
    sc: float = 0.10,
) -> np.ndarray:
    """
    Apply random translation and scale jitter to a bounding box.
    
    Args:
        box: Box as [x0, y0, x1, y1].
        img_w: Image width (for clamping).
        img_h: Image height (for clamping).
        tr: Translation jitter as fraction of box size.
        sc: Scale jitter as fraction (1.0 = no jitter).
        
    Returns:
        Jittered box as [x0, y0, x1, y1].
    """
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


def nbox_to_xyxy(nbox: Any, W: int, H: int) -> np.ndarray:
    """
    Convert a NormalizedBox to pixel xyxy coordinates.
    
    Args:
        nbox: NormalizedBox instance with to_pixel_xyxy method.
        W: Image width.
        H: Image height.
        
    Returns:
        Box as numpy array [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = nbox.to_pixel_xyxy(W, H)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ============================================================================
# Data augmentation
# ============================================================================

def color_jitter_safe(
    img: PILImage.Image,
    brightness: float = 0.10,
    contrast: float = 0.10,
) -> PILImage.Image:
    """
    Apply light photometric jitter suitable for fundus images.
    
    Args:
        img: PIL image.
        brightness: Brightness jitter range.
        contrast: Contrast jitter range.
        
    Returns:
        Augmented PIL image.
    """
    if brightness > 0:
        fac = 1.0 + random.uniform(-brightness, brightness)
        img = ImageEnhance.Brightness(img).enhance(fac)
    if contrast > 0:
        fac = 1.0 + random.uniform(-contrast, contrast)
        img = ImageEnhance.Contrast(img).enhance(fac)
    return img


# ============================================================================
# Loss functions
# ============================================================================

class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice loss for segmentation.
    
    Args:
        bce_weight: Weight for BCE component (dice weight = 1 - bce_weight).
    """
    
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            logits: Model output logits, shape (B, C, H, W).
            target: Ground truth masks, shape (B, C, H, W).
            
        Returns:
            Scalar loss tensor.
        """
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        num = 2 * (prob * target).sum(dim=(2, 3)) + 1e-6
        den = (prob.pow(2) + target.pow(2)).sum(dim=(2, 3)) + 1e-6
        dice = 1.0 - (num / den)
        dice = dice.mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice


# Alias for backward compatibility
BCEDice = BCEDiceLoss


# ============================================================================
# Metrics
# ============================================================================

def dice_coef_prob(
    prob: torch.Tensor,
    target: torch.Tensor,
    thresh: float = 0.5,
) -> float:
    """
    Compute Dice coefficient from probability tensor.
    
    Args:
        prob: Predicted probabilities.
        target: Ground truth binary mask.
        thresh: Threshold for binarization.
        
    Returns:
        Dice coefficient as float.
    """
    pred = (prob >= thresh).float()
    inter = (pred * target).sum().item()
    den = pred.sum().item() + target.sum().item()
    if den == 0:
        return 1.0
    return (2.0 * inter) / den


def dice_from_masks(
    pred: Optional[np.ndarray],
    gt: Optional[np.ndarray],
) -> Optional[float]:
    """
    Compute Dice coefficient between two binary masks.
    
    Args:
        pred: Predicted mask.
        gt: Ground truth mask.
        
    Returns:
        Dice coefficient, or None if inputs are invalid.
    """
    if pred is None or gt is None:
        return None
    predb = (pred > 0).astype(np.uint8)
    gtb = (gt > 0).astype(np.uint8)
    inter = int((predb & gtb).sum())
    s = int(predb.sum() + gtb.sum())
    if s == 0:
        return None
    return 2.0 * inter / float(s)


# ============================================================================
# Device utilities
# ============================================================================

def to_device(x: Any, device: torch.device) -> Any:
    """
    Recursively move tensor/collection to device.
    
    Args:
        x: Tensor, list, tuple, or dict of tensors.
        device: Target device.
        
    Returns:
        Same structure with tensors moved to device.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(t, device) for t in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x

