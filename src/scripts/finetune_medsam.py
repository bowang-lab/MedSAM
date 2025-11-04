#!/usr/bin/env python3
# src/scripts/finetune_medsam.py
# Fine-tune MedSAM with robustness to imperfect YOLO boxes.
# - Optionally loads a YOLO detector to provide imperfect boxes during training
# - Adds translation/scale noise + padding jitter to boxes
# - (Optional) point prompts (1 positive + K negatives) per sample
# - Weighted BCE outside the (dilated) box to reduce spill-over
# - Saves only trainable parts (decoder and optional LoRA) as compact checkpoints

from __future__ import annotations

import os
import json
import math
import random
import argparse
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from PIL import Image as PILImage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Project imports
from src.imgpipe.image_factory import ImageFactory

# --- SAM imports (lazy-checked) ---
try:
    from segment_anything import sam_model_registry
except Exception as _sam_err:
    sam_model_registry = None
    _SAM_IMPORT_ERR = _sam_err

# --- Optional YOLO imports ---
try:
    from ultralytics import YOLO as _YOLO
except Exception:
    _YOLO = None


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
SEED = 42

# Splits (per Sample, not per image)
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10


# =========================
# Utilities
# =========================
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
    except Exception:
        pass

def _ensure_sam_available():
    if sam_model_registry is None:
        raise RuntimeError(
            f"[ERR] segment-anything not available. Import error: {_SAM_IMPORT_ERR!r}\n"
            "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
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
# Data adapter
# =========================
@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path
    structure: str  # "disc" or "cup"

def gather_samples_via_imagefactory(root: Path, exclude: Optional[List[str]] = None) -> List[Sample]:
    """
    Build samples strictly from persisted paths in src/imgpipe/image.Image:
      - image_path
      - gt_disc_mask.path
      - gt_cup_mask.path
    Each disc/cup becomes a separate binary-sample.
    """
    print("[INFO] Scanning datasets with ImageFactory…")
    fac = ImageFactory(root=root, auto_scan=True)
    fac.filter_empty_masks()
    if exclude:
        fac.filter_datasets(exclude=exclude)
    images = fac.make_images()
    if not images:
        raise RuntimeError("No images available after filtering.")
    print(f"[INFO] Found {len(images)} images with masks (pre-split).")

    samples: List[Sample] = []
    missing_paths = 0

    for im in images:
        img_p: Path = im.image_path

        # Disc
        disc_ref = im.gt_disc_mask
        if disc_ref is not None and getattr(disc_ref, "path", None):
            dp = Path(disc_ref.path)
            if dp.exists():
                samples.append(Sample(img_p, dp, "disc"))
            else:
                missing_paths += 1

        # Cup
        cup_ref = im.gt_cup_mask
        if cup_ref is not None and getattr(cup_ref, "path", None):
            cp = Path(cup_ref.path)
            if cp.exists():
                samples.append(Sample(img_p, cp, "cup"))
            else:
                missing_paths += 1

    if not samples:
        raise RuntimeError("No (image, mask) pairs with valid on-disk paths. Check your serialized Image records.")
    if missing_paths:
        print(f"[WARN] {missing_paths} mask references were present but missing on disk; skipped.")

    print(f"[INFO] Formed {len(samples)} segmentation samples (disc+cup).")
    return samples

def split_samples(samples: List[Sample], train=0.8, val=0.1, test=0.1, seed=42) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    assert math.isclose(train + val + test, 1.0, rel_tol=1e-6)
    rng = random.Random(seed)
    samples = samples.copy()
    rng.shuffle(samples)
    n = len(samples)
    n_train = int(n * train)
    n_val   = int(n * val)
    train_s = samples[:n_train]
    val_s   = samples[n_train:n_train + n_val]
    test_s  = samples[n_train + n_val:]
    return train_s, val_s, test_s


# =========================
# Geometry / preprocessing
# =========================
class LetterboxToSquare:
    """Resize to square (size) with centered padding; also maps boxes and points."""
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

    def __call__(
        self,
        img: PILImage.Image,
        mask: PILImage.Image,
        box_xyxy: np.ndarray,
        pts_xy: Optional[np.ndarray] = None,  # (N,2) in original coords
    ) -> Tuple[PILImage.Image, PILImage.Image, np.ndarray, Optional[np.ndarray]]:
        w, h = img.size
        scale, new_w, new_h, pad_left, pad_top = self._compute_pad(w, h)
        # Resize
        img_r  = img.resize((new_w, new_h), PILImage.BILINEAR)
        mask_r = mask.resize((new_w, new_h), PILImage.NEAREST)
        # Pad to center
        new_img  = PILImage.new("RGB", (self.size, self.size))
        new_mask = PILImage.new("L",   (self.size, self.size))
        new_img.paste(img_r,  (pad_left, pad_top))
        new_mask.paste(mask_r,(pad_left, pad_top))
        # Box transform
        x0, y0, x1, y1 = box_xyxy
        x0 = x0 * scale + pad_left
        y0 = y0 * scale + pad_top
        x1 = x1 * scale + pad_left
        y1 = y1 * scale + pad_top
        box_t = np.array([x0, y0, x1, y1], dtype=np.float32)
        # Points transform
        pts_t = None
        if pts_xy is not None:
            px = pts_xy[:, 0] * scale + pad_left
            py = pts_xy[:, 1] * scale + pad_top
            pts_t = np.stack([px, py], axis=1).astype(np.float32)
        return new_img, new_mask, box_t, pts_t

def mask_to_tight_box(mask_np: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask_np)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)

def pad_box(box: np.ndarray, pad_frac: float, img_w: int, img_h: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    pad_x = w * pad_frac
    pad_y = h * pad_frac
    xx0 = max(0.0, x0 - pad_x)
    yy0 = max(0.0, y0 - pad_y)
    xx1 = min(float(img_w - 1), x1 + pad_x)
    yy1 = min(float(img_h - 1), y1 + pad_y)
    return np.array([xx0, yy0, xx1, yy1], dtype=np.float32)

def jitter_box_xyxy(box, img_w, img_h, tr=0.05, sc=0.10):
    # tr: max ±5% translation; sc: max ±10% scale change
    x0,y0,x1,y1 = box
    w = max(1.0, x1-x0); h = max(1.0, y1-y0)
    cx = (x0+x1)/2; cy = (y0+y1)/2

    # random translation
    dx = (2*random.random()-1) * tr * w
    dy = (2*random.random()-1) * tr * h
    cx += dx; cy += dy

    # random isotropic scaling
    s = 1.0 + (2*random.random()-1) * sc
    nw = max(1.0, w * s); nh = max(1.0, h * s)

    nx0 = max(0.0, cx - nw/2); ny0 = max(0.0, cy - nh/2)
    nx1 = min(float(img_w-1), cx + nw/2); ny1 = min(float(img_h-1), cy + nh/2)
    return np.array([nx0, ny0, nx1, ny1], dtype=np.float32)

PIXEL_MEAN = torch.tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
PIXEL_STD  = torch.tensor([58.395, 57.120, 57.375]).view(3, 1, 1)

def preprocess_for_sam(img_pil: PILImage.Image) -> torch.Tensor:
    """RGB PIL -> (3,1024,1024) float32 normalized as in SAM."""
    x = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float()  # RGB, 0..255
    return (x - PIXEL_MEAN) / PIXEL_STD


# =========================
# YOLO box provider (optional)
# =========================
class YOLOBoxProvider:
    """
    Runs a YOLO model to get disc/cup boxes per image. Caches results.
    Expects classes: disc = yolo_disc_id, cup = yolo_cup_id.
    """
    def __init__(self, weights: Path, device: str = "cpu", imgsz: int = 640,
                 yolo_disc_id: int = 0, yolo_cup_id: int = 1,
                 conf: float = 0.05, iou: float = 0.6):
        if _YOLO is None:
            raise RuntimeError("Ultralytics not installed. pip install ultralytics")
        self.model = _YOLO(str(weights))
        self.device = device
        self.imgsz = imgsz
        self.disc_id = yolo_disc_id
        self.cup_id  = yolo_cup_id
        self.conf = conf
        self.iou = iou
        self._cache: Dict[str, Dict[str, Optional[np.ndarray]]] = {}

    def _predict_one(self, image_path: Path) -> Dict[str, Optional[np.ndarray]]:
        res = self.model.predict(
            source=str(image_path),
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )
        if not res:
            return {"disc": None, "cup": None}
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            return {"disc": None, "cup": None}

        xyxy = r.boxes.xyxy.cpu().numpy()  # (N,4) in original pixels
        cls  = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
        conf = r.boxes.conf.cpu().numpy()  # (N,)

        out = {"disc": None, "cup": None}
        for target, cid in (("disc", self.disc_id), ("cup", self.cup_id)):
            idx = np.where(cls == cid)[0]
            if idx.size == 0:
                out[target] = None
            else:
                # pick highest confidence
                j = idx[np.argmax(conf[idx])]
                out[target] = xyxy[j].astype(np.float32)  # [x0,y0,x1,y1]
        return out

    def get_box(self, image_path: Path, structure: str) -> Optional[np.ndarray]:
        key = str(image_path)
        if key not in self._cache:
            self._cache[key] = self._predict_one(image_path)
        return self._cache[key].get(structure, None)


# =========================
# Dataset
# =========================
class MedSAMDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        img_size: int = 1024,
        jitter_pad: float = 0.3,
        jitter_trans: float = 0.05,
        jitter_scale: float = 0.10,
        train: bool = True,
        use_points: bool = False,
        neg_points: int = 3,
        yolo_provider: Optional[YOLOBoxProvider] = None,
        use_det_prob: float = 0.5,
    ):
        self.samples = samples
        self.img_size = img_size
        self.letterbox = LetterboxToSquare(img_size)
        self.jitter_pad = jitter_pad
        self.jitter_trans = jitter_trans
        self.jitter_scale = jitter_scale
        self.train = train
        self.use_points = use_points
        self.neg_points = max(0, int(neg_points))
        self.yolo = yolo_provider
        self.use_det_prob = use_det_prob

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _sample_points(mask_np: np.ndarray, k_neg: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Return (points Nx2, labels Nx1) in original coordinates. 1 pos + k neg."""
        ys, xs = np.nonzero(mask_np)
        if len(xs) == 0:
            # degenerate: return negatives only (all zeros)
            H, W = mask_np.shape
            pts = []
            while len(pts) < max(1, k_neg):
                x = np.random.randint(W); y = np.random.randint(H)
                pts.append([x, y])
            labels = np.zeros((len(pts), 1), dtype=np.float32)
            return np.asarray(pts, np.float32), labels

        # one positive
        i = np.random.randint(len(xs))
        pos = np.array([[xs[i], ys[i]]], dtype=np.float32)
        # negatives
        H, W = mask_np.shape
        neg = []
        while len(neg) < k_neg:
            x = np.random.randint(W); y = np.random.randint(H)
            if mask_np[y, x] == 0:
                neg.append([x, y])
        neg = np.asarray(neg, dtype=np.float32) if k_neg > 0 else np.zeros((0,2), np.float32)
        pts = np.vstack([pos, neg]) if k_neg > 0 else pos
        labels = np.array([1] + [0]*k_neg, dtype=np.float32).reshape(-1,1) if k_neg > 0 else np.array([[1.0]], dtype=np.float32)
        return pts, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        img = PILImage.open(s.image_path).convert("RGB")
        m   = PILImage.open(s.mask_path).convert("L")

        W, H = img.size
        mask_np = np.array(m, dtype=np.uint8)

        # Base box: GT-tight
        base_box = mask_to_tight_box(mask_np)
        if base_box is None:
            base_box = np.array([0, 0, W - 1, H - 1], dtype=np.float32)

        # Optionally replace with detector box (simulating real inference)
        if self.train and (self.yolo is not None) and (random.random() < self.use_det_prob):
            det = self.yolo.get_box(s.image_path, s.structure)
            if det is not None:
                # Clip to image bounds just in case
                x0,y0,x1,y1 = det
                base_box = np.array([
                    max(0.0, x0), max(0.0, y0),
                    min(float(W-1), x1), min(float(H-1), y1)
                ], dtype=np.float32)

        # Noise: pad + translation/scale (curriculum-friendly if you reduce early)
        pad_frac = random.uniform(0.0, self.jitter_pad) if self.train else 0.0
        box_p = pad_box(base_box, pad_frac, W, H)
        if self.train:
            box_p = jitter_box_xyxy(box_p, W, H, tr=self.jitter_trans, sc=self.jitter_scale)

        # Points (in original coords) → transformed with letterbox
        pts_xy = lbl = None
        if self.use_points:
            pts_xy, lbl = self._sample_points(mask_np, k_neg=self.neg_points)

        # Resize/pad to square + map box/points
        img_r, mask_r, box_t, pts_t = self.letterbox(img, m, box_p, pts_xy)

        # Tensors
        x = preprocess_for_sam(img_r)
        mask_arr = np.array(mask_r, dtype=np.uint8)
        y = torch.from_numpy((mask_arr > 0).astype(np.float32)).unsqueeze(0)  # (1,H,W)
        b = torch.from_numpy(box_t.astype(np.float32))  # (4,)

        result: Dict[str, torch.Tensor] = {"image": x, "mask": y, "box": b}
        if self.use_points and (pts_t is not None) and (lbl is not None):
            result["points"] = torch.from_numpy(pts_t.astype(np.float32))            # (1+K,2)
            result["labels"] = torch.from_numpy(lbl.astype(np.float32)).squeeze(1)   # (1+K,)
        return result


# =========================
# Losses / Metrics
# =========================
def outside_box_weight(H, W, box_xyxy, lam=0.3, dilate=0.05):
    """Weight map >1 outside a slightly dilated box (penalize spill)."""
    x0,y0,x1,y1 = box_xyxy
    # dilate in pixels proportional to box size
    dx = (x1-x0) * dilate
    dy = (y1-y0) * dilate
    x0 = max(0, int(x0 - dx)); y0 = max(0, int(y0 - dy))
    x1 = min(W-1, int(x1 + dx)); y1 = min(H-1, int(y1 + dy))
    w = np.ones((H, W), dtype=np.float32)
    if y0 > 0: w[:y0, :] += lam
    if y1 < H-1: w[y1+1:, :] += lam
    if x0 > 0: w[:, :x0] += lam
    if x1 < W-1: w[:, x1+1:] += lam
    return torch.from_numpy(w)

class BCEDiceWeighted(nn.Module):
    def __init__(self, bce_weight=0.5, spill_lam=0.3, spill_dilate=0.05):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_weight = bce_weight
        self.spill_lam = spill_lam
        self.spill_dilate = spill_dilate

    def forward(self, logits, target, boxes=None):
        # logits: (B,1,H,W), target: (B,1,H,W), boxes: (B,4) in 1024 coords
        bce = self.bce(logits, target)  # (B,1,H,W)
        if boxes is not None:
            B, _, H, W = logits.shape
            Wmaps = []
            for i in range(B):
                wmap = outside_box_weight(
                    H, W, boxes[i].detach().cpu().numpy(),
                    lam=self.spill_lam, dilate=self.spill_dilate
                )
                Wmaps.append(wmap)
            Wt = torch.stack(Wmaps, 0).to(logits.device).unsqueeze(1)  # (B,1,H,W)
            bce = (bce * Wt).mean()
        else:
            bce = bce.mean()

        prob = torch.sigmoid(logits)
        num = 2 * (prob * target).sum(dim=(2,3)) + 1e-6
        den = (prob.pow(2) + target.pow(2)).sum(dim=(2,3)) + 1e-6
        dice = 1.0 - (num / den)
        dice = dice.mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice

def dice_coef(prob: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    pred = (prob >= thresh).float()
    inter = (pred * target).sum().item()
    den = pred.sum().item() + target.sum().item()
    if den == 0:
        return 1.0
    return (2.0 * inter) / den


# =========================
# Model wrapper
# =========================
class MedSAMFinetuner(nn.Module):
    """
    Wraps a SAM model for decoder-only fine-tuning (default).
    """
    def __init__(self, sam_type: str, checkpoint: Path, freeze_encoders: bool = True, use_lora: bool = False, lora_r: int = 8):
        super().__init__()
        _ensure_sam_available()
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"MedSAM checkpoint not found: {checkpoint}")
        self.sam = sam_model_registry[sam_type](checkpoint=str(checkpoint))
        self.sam.eval()  # start in eval; we selectively unfreeze below

        # Freeze encoders if requested
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = not freeze_encoders
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = not freeze_encoders

        # Mask decoder always trainable
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True

        # Optional LoRA on image encoder (requires peft)
        self._using_lora = False
        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except Exception as e:
                raise RuntimeError(f"[ERR] --lora requested but 'peft' not available: {e!r}")
            # Heuristic target selection: all Linear layers in image encoder
            lora_targets = [n for n, m in self.sam.image_encoder.named_modules() if isinstance(m, nn.Linear)]
            peft_cfg = LoraConfig(
                r=lora_r, lora_alpha=2*lora_r, target_modules=lora_targets,
                lora_dropout=0.0, bias="none", task_type="FEATURE_EXTRACTION"
            )
            self.sam.image_encoder = get_peft_model(self.sam.image_encoder, peft_cfg)
            # Keep LoRA trainable
            for p in self.sam.image_encoder.parameters():
                p.requires_grad = True
            self._using_lora = True

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits_1024, iou_pred).
        batch keys: image (B,3,1024,1024), mask (B,1,1024,1024), box (B,4)
                    optional: points (B,N,2) and labels (B,N)
        """
        x = batch["image"]  # (B,3,1024,1024)
        b = batch["box"]    # (B,4)
        B = x.shape[0]

        # Encode image
        image_embeddings = self.sam.image_encoder(x)  # (B,256,64,64)
        dense_pe = self.sam.prompt_encoder.get_dense_pe()

        # Prepare prompts
        boxes = b[:, None, :]  # (B,1,4)
        pe_kwargs: Dict[str, Any] = dict(points=None, boxes=boxes, masks=None)

        if ("points" in batch) and ("labels" in batch):
            pts = batch["points"]  # (B,N,2)
            lbs = batch["labels"]  # (B,N)
            # SAM expects Tuple[Tensor, Tensor] with shapes (B,N,2) and (B,N)
            pe_kwargs["points"] = (pts, lbs)

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(**pe_kwargs)

        # Determine kwargs for mask decoder compatibility
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

        # If a prior/low-res mask is supported, pass zeros
        zero_prior = torch.zeros((B, 1, 256, 256), device=x.device, dtype=image_embeddings.dtype)
        if "mask_input" in params:
            md_kwargs["mask_input"] = zero_prior
        elif "low_res_mask" in params:
            md_kwargs["low_res_mask"] = zero_prior
        elif "low_res_masks" in params:
            md_kwargs["low_res_masks"] = zero_prior

        low_res_masks, iou_pred = md(**md_kwargs)
        logits_1024 = F.interpolate(
            low_res_masks, size=(x.shape[2], x.shape[3]),
            mode="bilinear", align_corners=False
        )
        return logits_1024, iou_pred


# =========================
# Training / Eval loops
# =========================
def run_one_epoch(model: MedSAMFinetuner, loader: DataLoader, optim: torch.optim.Optimizer,
                  loss_fn: nn.Module, device: torch.device, train: bool) -> Dict[str, float]:
    model.train(mode=train)
    total_loss = 0.0
    total_dice = 0.0
    n = 0
    for batch in loader:
        batch = _to_device(batch, device)
        with torch.set_grad_enabled(train):
            logits, _ = model(batch)
            # Pass boxes for spill penalty
            loss = loss_fn(logits, batch["mask"], boxes=batch.get("box"))
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                total_dice += dice_coef(prob, batch["mask"])
                n += 1
            total_loss += loss.item()
    return {"loss": total_loss / max(1, n), "dice": total_dice / max(1, n)}

@torch.no_grad()
def evaluate(model: MedSAMFinetuner, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_dice = 0.0
    n = 0
    for batch in loader:
        batch = _to_device(batch, device)
        logits, _ = model(batch)
        prob = torch.sigmoid(logits)
        total_dice += dice_coef(prob, batch["mask"])
        n += 1
    return {"dice": total_dice / max(1, n)}


# =========================
# I/O helpers
# =========================
def save_checkpoint(model: MedSAMFinetuner, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "mask_decoder": model.sam.mask_decoder.state_dict(),
        "using_lora": model._using_lora,
    }
    if model._using_lora:
        state["image_encoder"] = model.sam.image_encoder.state_dict()
    torch.save(state, str(path))

def load_trainable_parts(model: MedSAMFinetuner, path: Path) -> None:
    state = torch.load(str(path), map_location="cpu")
    model.sam.mask_decoder.load_state_dict(state["mask_decoder"])
    if state.get("using_lora", False) and model._using_lora and "image_encoder" in state:
        model.sam.image_encoder.load_state_dict(state["image_encoder"], strict=False)


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune MedSAM with robustness to imperfect YOLO boxes.")

    # Modes
    p.add_argument("--train", action="store_true", help="Enable training mode.")
    p.add_argument("--test-weights", type=Path, dest="test_weights", help="Path to a saved trainable checkpoint (*.pth).")

    # Paths/config
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--out-dir",   type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--run-dir",   type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--model",     type=str,  default=DEFAULT_MODEL)
    p.add_argument("--ckpt",      type=Path, default=DEFAULT_CKPT)

    # Training knobs
    p.add_argument("--device", type=str, default=str(DEFAULT_DEVICE))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch",  type=int, default=DEFAULT_BATCH)
    p.add_argument("--imgsz",  type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--wd",     type=float, default=1e-4)
    p.add_argument("--resume", type=Path, help="Resume from a trainable checkpoint (*.pth).")
    p.add_argument("--run-name", type=str, help="Run name for directory naming.")

    # Prompt jitter / noise
    p.add_argument("--pad-jitter", type=float, default=0.30, help="Random box padding fraction (0..).")
    p.add_argument("--box-trans",  type=float, default=0.05, help="±translation as fraction of box width/height.")
    p.add_argument("--box-scale",  type=float, default=0.10, help="±isotropic scale as fraction of box size.")

    # Points
    p.add_argument("--use-points", action="store_true", help="Include 1 positive + K negative points as prompts.")
    p.add_argument("--neg-points", type=int, default=3, help="Number of negative points per sample when --use-points.")

    # Spill penalty
    p.add_argument("--spill-lam",    type=float, default=0.30, help="Penalty weight outside box region.")
    p.add_argument("--spill-dilate", type=float, default=0.05, help="Relative dilation of the box before penalizing.")

    # Freezing / LoRA
    p.add_argument("--unfreeze-encoders", action="store_true", help="Unfreeze image+prompt encoders (full finetune).")
    p.add_argument("--lora", action="store_true", help="Enable LoRA adapters on image encoder.")
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank.")

    # Splits
    p.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    p.add_argument("--val-ratio",   type=float, default=VAL_RATIO)
    p.add_argument("--test-ratio",  type=float, default=TEST_RATIO)

    # YOLO integration
    p.add_argument("--yolo-weights", type=Path, help="Path to YOLO *.pt weights to sample detector boxes.")
    p.add_argument("--yolo-device",  type=str, help="Device for YOLO (defaults to --device).")
    p.add_argument("--yolo-imgsz",   type=int, default=640)
    p.add_argument("--yolo-disc-id", type=int, default=0)
    p.add_argument("--yolo-cup-id",  type=int, default=1)
    p.add_argument("--use-det-prob", type=float, default=0.50, help="Probability of using detector box (vs GT box) in training.")

    return p.parse_args()


# =========================
# Main
# =========================
def main():
    os.environ.setdefault("PYTHONHASHSEED", "0")
    set_global_seed(SEED)
    args = parse_args()

    device = torch.device(args.device if args.device != "cpu" else "cpu")

    print(f"[INFO] DATA_ROOT = {args.data_root}")
    print(f"[INFO] OUT_DIR   = {args.out_dir}")
    print(f"[INFO] RUN_DIR   = {args.run_dir}")
    print(f"[INFO] MODEL     = {args.model}")
    print(f"[INFO] CKPT      = {args.ckpt}")
    print(f"[INFO] DEVICE    = {device}")
    print(f"[INFO] EPOCHS    = {args.epochs}")
    print(f"[INFO] BATCH     = {args.batch}")
    print(f"[INFO] IMGSZ     = {args.imgsz}")
    print(f"[INFO] MODES     = train={args.train}, test={'yes' if args.test_weights else 'no'}")
    if args.yolo_weights:
        print(f"[INFO] YOLO      = {args.yolo_weights} (use_det_prob={args.use_det_prob:.2f})")

    # Build samples
    samples = gather_samples_via_imagefactory(args.data_root, exclude=["PAPILA"])
    train_s, val_s, test_s = split_samples(samples, args.train_ratio, args.val_ratio, args.test_ratio, SEED)
    print(f"[INFO] Split sizes -> train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    # YOLO provider (optional)
    yolo_provider: Optional[YOLOBoxProvider] = None
    if args.yolo_weights:
        yolo_dev = args.yolo_device or args.device
        yolo_provider = YOLOBoxProvider(
            weights=args.yolo_weights,
            device=yolo_dev,
            imgsz=args.yolo_imgsz,
            yolo_disc_id=args.yolo_disc_id,
            yolo_cup_id=args.yolo_cup_id
        )

    # Datasets
    ds_train = MedSAMDataset(
        train_s,
        img_size=args.imgsz,
        jitter_pad=args.pad_jitter,
        jitter_trans=args.box_trans,
        jitter_scale=args.box_scale,
        train=True,
        use_points=args.use_points,
        neg_points=args.neg_points,
        yolo_provider=yolo_provider,
        use_det_prob=args.use_det_prob,
    )
    ds_val = MedSAMDataset(
        val_s,
        img_size=args.imgsz,
        jitter_pad=0.0,
        jitter_trans=0.0,
        jitter_scale=0.0,
        train=False,
        use_points=False,
        yolo_provider=None,
        use_det_prob=0.0,
    )
    ds_test = MedSAMDataset(
        test_s,
        img_size=args.imgsz,
        jitter_pad=0.0,
        jitter_trans=0.0,
        jitter_scale=0.0,
        train=False,
        use_points=False,
        yolo_provider=None,
        use_det_prob=0.0,
    )

    pin_mem = (device.type != "mps")  # avoid pin_memory warnings on MPS
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=pin_mem, drop_last=False)
    dl_val   = DataLoader(ds_val,   batch_size=max(1, args.batch//2), shuffle=False, num_workers=2, pin_memory=pin_mem)
    dl_test  = DataLoader(ds_test,  batch_size=max(1, args.batch//2), shuffle=False, num_workers=2, pin_memory=pin_mem)

    # Model
    model = MedSAMFinetuner(
        sam_type=args.model,
        checkpoint=args.ckpt,
        freeze_encoders=not args.unfreeze_encoders,
        use_lora=args.lora,
        lora_r=args.lora_r,
    ).to(device)

    # Optimizer and loss
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)
    loss_fn = BCEDiceWeighted(bce_weight=0.5, spill_lam=args.spill_lam, spill_dilate=args.spill_dilate)

    # Paths
    run_name = args.run_name or "MedSAMTrain"
    run_path = Path(args.run_dir) / run_name
    weights_dir = run_path / "weights"
    metrics_path = Path(args.out_dir) / "medsam_metrics.json"
    best_path = weights_dir / "best.pth"
    last_path = weights_dir / "last.pth"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    if args.resume:
        print(f"[INFO] Resuming from: {args.resume}")
        load_trainable_parts(model, args.resume)

    # TRAIN
    best_val = -1.0
    history: Dict[str, List[float]] = {"train_dice": [], "val_dice": [], "train_loss": []}

    if args.train:
        print("[INFO] Starting training… (decoder-only unless --unfreeze-encoders or --lora)")
        for epoch in range(1, args.epochs + 1):
            tr = run_one_epoch(model, dl_train, optimizer, loss_fn, device, train=True)
            va = evaluate(model, dl_val, device)
            history["train_loss"].append(tr["loss"])
            history["train_dice"].append(tr["dice"])
            history["val_dice"].append(va["dice"])

            print(f"[E{epoch:03d}] loss={tr['loss']:.4f} | dice(train)={tr['dice']:.4f} | dice(val)={va['dice']:.4f}")

            # Save last
            save_checkpoint(model, last_path)

            # Save best (val dice)
            if va["dice"] > best_val:
                best_val = va["dice"]
                save_checkpoint(model, best_path)

        # Save metrics
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump({"history": history, "best_val_dice": best_val}, f, indent=2)
        print(f"[OK] Training done. Best val Dice={best_val:.4f}. Weights saved to: {best_path}")

    # TEST
    test_ckpt = args.test_weights or (best_path if best_path.exists() else None)
    if test_ckpt:
        print(f"[INFO] Testing with weights: {test_ckpt}")
        load_trainable_parts(model, test_ckpt)
        te = evaluate(model, dl_test, device)
        print(f"[OK] Test Dice={te['dice']:.4f}")
        try:
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    m = json.load(f)
            else:
                m = {}
            m["test_dice"] = te["dice"]
            with open(metrics_path, "w") as f:
                json.dump(m, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not write test metrics: {e!r}")
    elif not args.train:
        print("[WARN] No mode selected (use --train and/or --test-weights).")


if __name__ == "__main__":
    main()