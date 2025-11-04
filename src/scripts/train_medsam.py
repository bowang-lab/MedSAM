#!/usr/bin/env python3
# src/scripts/train_medsam.py
# Fine-tune MedSAM (decoder-only by default) for OD/OC segmentation using box prompts.
# - Scans datasets with ImageFactory (retaining your pipeline style)
# - Splits into train/val/test with ratios matching your YOLO script
# - Trains BCE+Dice on predicted masks from box prompts
# - Saves best checkpoint and a JSON metrics file
#
# Requirements (Python):
#   torch, torchvision, numpy, pillow
#   segment-anything (pip install git+https://github.com/facebookresearch/segment-anything.git)
# Optional for LoRA: peft (pip install peft)
#
# Notes:
# - By default, image & prompt encoders are frozen; only mask decoder trains (safe for small data).
# - Set --lora to enable LoRA on the image encoder (in addition to the decoder).
# - If your Image dataclass uses different attribute names, adjust _extract_paths().

from __future__ import annotations

import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from PIL import Image as PILImage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.imgpipe.image_factory import ImageFactory
from src.imgpipe.enums import Structure  # DISC / CUP

# Try SAM imports
try:
    from segment_anything import sam_model_registry
except Exception as e:
    sam_model_registry = None
    _SAM_IMPORT_ERR = e

# =========================
# Defaults (parallel to your YOLO script)
# =========================
DEFAULT_DATA_ROOT = Path("/Users/carlosperez/Library/CloudStorage/OneDrive-UBC/Ipek_Carlos/GlaucomaDatasets/All_Datasets_Organized")
DEFAULT_OUT_DIR   = Path("/Users/carlosperez/PycharmProjects/MedSAM/TRAINING_DS_TOY")
DEFAULT_RUN_DIR   = Path("/Users/carlosperez/PycharmProjects/MedSAM/runs_medsam")
DEFAULT_MODEL     = "vit_b"  # SAM backbone key for registry; MedSAM is ViT-B
DEFAULT_CKPT      = Path("/Users/carlosperez/PycharmProjects/MedSAM/checkpoints/medsam_vit_b.pth")  # <- set to your MedSAM checkpoint

DEFAULT_DEVICE: Optional[str | int] = "mps"
DEFAULT_IMGSZ = 1024
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 8
DEFAULT_WORKERS = 8
SEED = 42

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

def _extract_paths(img_obj) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Best-effort extraction from your Image dataclass.
    Adjust if your attribute names differ.
    Returns: (fundus_path, disc_mask_path, cup_mask_path)
    """
    # Common field names
    candidates_img  = ["fundus_path", "image_path", "rgb_path", "path"]
    candidates_disc = ["disc_mask_path", "mask_disc_path", "disc_path", "od_mask_path"]
    candidates_cup  = ["cup_mask_path", "mask_cup_path", "cup_path", "oc_mask_path"]

    def _first(attr_list):
        for a in attr_list:
            if hasattr(img_obj, a):
                p = getattr(img_obj, a)
                if p:
                    return Path(p)
        # Try mapping dicts if present
        if hasattr(img_obj, "masks") and isinstance(img_obj.masks, dict):
            # keys may be enums or strings
            for key, val in img_obj.masks.items():
                key_str = str(key).lower()
                if "disc" in key_str and "disc" in attr_list[0]:
                    return Path(val)
                if "cup" in key_str and "cup" in attr_list[0]:
                    return Path(val)
        return None

    img_p  = _first(candidates_img)
    disc_p = _first(candidates_disc)
    cup_p  = _first(candidates_cup)
    return img_p, disc_p, cup_p

def gather_samples_via_imagefactory(root: Path, exclude: Optional[List[str]] = None) -> List[Sample]:
    print("[INFO] Scanning datasets with ImageFactory…")
    fac = ImageFactory(root=root, auto_scan=True)
    fac.filter_empty_masks()
    if exclude:
        fac.filter_datasets(exclude=exclude)
    images = fac.make_images()
    if not images:
        raise RuntimeError("No images available after filtering.")
    print(f"[INFO] Found {len(images)} images with masks.")
    samples: List[Sample] = []
    for im in images:
        img_p, disc_p, cup_p = _extract_paths(im)
        if img_p is None:
            continue
        if disc_p and Path(disc_p).exists():
            samples.append(Sample(img_p, Path(disc_p), "disc"))
        if cup_p and Path(cup_p).exists():
            samples.append(Sample(img_p, Path(cup_p), "cup"))
    if not samples:
        raise RuntimeError("No (image,mask) pairs formed. Check your Image dataclass path extraction.")
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
# Image/Mask preprocessing and box prompts
# =========================
class LetterboxToSquare:
    """Resize to square (size) with centered padding; also maps boxes accordingly."""
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
        return new_img, new_mask, box_t

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

PIXEL_MEAN = torch.tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
PIXEL_STD  = torch.tensor([58.395, 57.120, 57.375]).view(3, 1, 1)

def preprocess_for_sam(img_pil: PILImage.Image) -> torch.Tensor:
    """RGB PIL -> (3,1024,1024) float32 normalized as in SAM."""
    x = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float()  # RGB, 0..255
    return (x - PIXEL_MEAN) / PIXEL_STD

# =========================
# Dataset
# =========================
class MedSAMDataset(Dataset):
    def __init__(self, samples: List[Sample], img_size: int = 1024, jitter_pad: float = 0.3, train: bool = True):
        self.samples = samples
        self.img_size = img_size
        self.letterbox = LetterboxToSquare(img_size)
        self.jitter_pad = jitter_pad
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        img = PILImage.open(s.image_path).convert("RGB")
        m   = PILImage.open(s.mask_path).convert("L")

        # tight box from original mask
        mask_np = np.array(m, dtype=np.uint8)
        box = mask_to_tight_box(mask_np)
        if box is None:
            # rare, skip or synthesize minimal box
            box = np.array([0, 0, img.size[0]-1, img.size[1]-1], dtype=np.float32)

        # jitter padding during training
        pad_frac = random.uniform(0.0, self.jitter_pad) if self.train else 0.0
        box_p = pad_box(box, pad_frac, img.size[0], img.size[1])

        # resize+pad to square + map box
        img_r, mask_r, box_t = self.letterbox(img, m, box_p)

        # tensors
        x = preprocess_for_sam(img_r)
        y = torch.from_numpy(np.array(mask_r > 0, dtype=np.float32))[None, :, :]  # (1,H,W)
        b = torch.from_numpy(box_t.astype(np.float32))  # (4,)

        return {"image": x, "mask": y, "box": b}

# =========================
# Losses / Metrics
# =========================
class BCEDice(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B,1,H,W), target: (B,1,H,W) in {0,1}
        bce = self.bce(logits, target)
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
        self.sam.eval()  # start eval; we'll unfreeze parts below

        # Freeze encoders if requested
        for p in self.sam.image_encoder.parameters():
            p.requires_grad = not freeze_encoders
        for p in self.sam.prompt_encoder.parameters():
            p.requires_grad = not freeze_encoders

        # Mask decoder always trainable
        for p in self.sam.mask_decoder.parameters():
            p.requires_grad = True

        # Optional LoRA on image encoder attention projections (requires peft)
        self._using_lora = False
        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except Exception as e:
                raise RuntimeError(f"[ERR] --lora requested but 'peft' not available: {e!r}")
            # Identify attention projection module names heuristically
            lora_targets = []
            for name, module in self.sam.image_encoder.named_modules():
                if isinstance(module, nn.Linear) and (name.endswith(".q_proj") or name.endswith(".k_proj") or name.endswith(".v_proj") or name.endswith(".proj")):
                    lora_targets.append(name)
            if not lora_targets:
                # Fallback: all Linear layers in image encoder
                lora_targets = [n for n, m in self.sam.image_encoder.named_modules() if isinstance(m, nn.Linear)]
            peft_cfg = LoraConfig(r=lora_r, lora_alpha=2*lora_r, target_modules=lora_targets, lora_dropout=0.0, bias="none", task_type="FEATURE_EXTRACTION")
            self.sam.image_encoder = get_peft_model(self.sam.image_encoder, peft_cfg)
            # Keep LoRA trainable even if base is frozen
            for p in self.sam.image_encoder.parameters():
                p.requires_grad = True
            self._using_lora = True

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns (logits_1024, iou_pred).
        batch keys: image (3,1024,1024), mask (1,1024,1024), box (4,)
        """
        x = batch["image"]  # (B,3,1024,1024), SAM-normalized
        b = batch["box"]    # (B,4) in 1024 coords

        # Encode image (grad on/off depending on freeze)
        image_embeddings = self.sam.image_encoder(x)
        # Dense positional encoding
        dense_pe = self.sam.prompt_encoder.get_dense_pe()
        # Prepare boxes (B,1,4)
        boxes = b[:, None, :]  # (B,1,4)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(points=None, boxes=boxes, masks=None)

        # Zero mask input (no previous mask)
        B = x.shape[0]
        mask_input = torch.zeros((B, 1, 256, 256), device=x.device, dtype=image_embeddings.dtype)

        low_res_masks, iou_pred = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            mask_input=mask_input,
        )
        # Upsample to 1024 for loss
        logits_1024 = F.interpolate(low_res_masks, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
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
            logits, _ = model(batch)  # (B,1,1024,1024)
            loss = loss_fn(logits, batch["mask"])
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                for i in range(prob.shape[0]):
                    total_dice += dice_coef(prob[i], batch["mask"][i])
                    n += 1
            total_loss += loss.item() * batch["mask"].shape[0]
    return {"loss": total_loss / max(1, n), "dice": total_dice / max(1, n)}

def evaluate(model: MedSAMFinetuner, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_dice = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            logits, _ = model(batch)
            prob = torch.sigmoid(logits)
            for i in range(prob.shape[0]):
                total_dice += dice_coef(prob[i], batch["mask"][i])
                n += 1
    return {"dice": total_dice / max(1, n)}

# =========================
# I/O helpers
# =========================
def save_checkpoint(model: MedSAMFinetuner, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save only trainable parts to keep it small
    state = {
        "mask_decoder": model.sam.mask_decoder.state_dict(),
        "using_lora": model._using_lora,
    }
    # If LoRA used, also save image_encoder (PEFT) adapters
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
    p = argparse.ArgumentParser(description="Fine-tune MedSAM on OD/OC masks using box prompts.")
    # Modes
    p.add_argument("--train", action="store_true", help="Enable training mode.")
    p.add_argument("--test-weights", type=Path, dest="test_weights", help="Path to a saved trainable checkpoint (*.pth).")

    # Paths/config
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help=f"Root datasets (default: {DEFAULT_DATA_ROOT})")
    p.add_argument("--out-dir",   type=Path, default=DEFAULT_OUT_DIR,   help=f"Output dir for splits/metrics (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--run-dir",   type=Path, default=DEFAULT_RUN_DIR,   help=f"Runs dir (default: {DEFAULT_RUN_DIR})")
    p.add_argument("--model",     type=str,  default=DEFAULT_MODEL,     help=f"SAM backbone key (default: {DEFAULT_MODEL})")
    p.add_argument("--ckpt",      type=Path, default=DEFAULT_CKPT,      help=f"MedSAM ViT-B checkpoint (.pth)")

    # Training knobs
    p.add_argument("--device", type=str, default=str(DEFAULT_DEVICE), help=f"'cpu', 'mps', 'cuda:0', etc. (default: {DEFAULT_DEVICE})")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,      help=f"Epochs (default: {DEFAULT_EPOCHS})")
    p.add_argument("--batch",  type=int, default=DEFAULT_BATCH,       help=f"Batch size (default: {DEFAULT_BATCH})")
    p.add_argument("--imgsz",  type=int, default=DEFAULT_IMGSZ,       help=f"Image size (default: {DEFAULT_IMGSZ})")
    p.add_argument("--lr",     type=float, default=1e-4,              help="Learning rate (default: 1e-4)")
    p.add_argument("--wd",     type=float, default=1e-4,              help="Weight decay (default: 1e-4)")
    p.add_argument("--resume", type=Path, help="Resume from a trainable checkpoint (*.pth).")

    # Prompt jitter
    p.add_argument("--pad-jitter", type=float, default=0.3, help="Random box padding fraction during training (default: 0.3)")

    # Freezing / LoRA
    p.add_argument("--unfreeze-encoders", action="store_true", help="Unfreeze image+prompt encoders (full finetune).")
    p.add_argument("--lora", action="store_true", help="Enable LoRA adapters on image encoder.")
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank (default: 8)")

    # Splits
    p.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    p.add_argument("--val-ratio",   type=float, default=VAL_RATIO)
    p.add_argument("--test-ratio",  type=float, default=TEST_RATIO)

    # Run naming
    p.add_argument("--run-name", type=str, help="Run name (for directory naming).")

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

    # Gather samples
    samples = gather_samples_via_imagefactory(args.data_root, exclude=["PAPILA"])
    train_s, val_s, test_s = split_samples(samples, args.train_ratio, args.val_ratio, args.test_ratio, SEED)
    print(f"[INFO] Split sizes -> train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    # Datasets/Loaders
    ds_train = MedSAMDataset(train_s, img_size=args.imgsz, jitter_pad=args.pad_jitter, train=True)
    ds_val   = MedSAMDataset(val_s,   img_size=args.imgsz, jitter_pad=0.0,           train=False)
    ds_test  = MedSAMDataset(test_s,  img_size=args.imgsz, jitter_pad=0.0,           train=False)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,  num_workers=DEFAULT_WORKERS, pin_memory=True, drop_last=False)
    dl_val   = DataLoader(ds_val,   batch_size=max(1, args.batch//2), shuffle=False, num_workers=DEFAULT_WORKERS, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=max(1, args.batch//2), shuffle=False, num_workers=DEFAULT_WORKERS, pin_memory=True)

    # Model
    model = MedSAMFinetuner(
        sam_type=args.model,
        checkpoint=args.ckpt,
        freeze_encoders=not args.unfreeze_encoders,
        use_lora=args.lora,
        lora_r=args.lora_r,
    ).to(device)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)
    loss_fn = BCEDice(bce_weight=0.5)

    # Optionally resume trainable parts
    run_name = args.run_name or "MedSAMTrain"
    run_path = Path(args.run_dir) / run_name
    weights_dir = run_path / "weights"
    metrics_path = Path(args.out_dir) / "medsam_metrics.json"
    best_path = weights_dir / "best.pth"
    last_path = weights_dir / "last.pth"
    weights_dir.mkdir(parents=True, exist_ok=True)

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

            # Save best on val dice
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
        # append to metrics
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