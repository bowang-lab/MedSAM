#!/usr/bin/env python3
#src.scripts.train_yolo.py

from __future__ import annotations

import json
import os
import random
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

from src.imgpipe.image_factory import ImageFactory
from src.imgpipe.yolo_splits import create_yolo_dataset

# =========================
# Default Configuration (will be overridden by Slurm CLI)
# =========================
DEFAULT_DATA_ROOT = Path("/Users/carlosperez/Library/CloudStorage/OneDrive-UBC/Ipek_Carlos/GlaucomaDatasets/All_Datasets_Organized")
DEFAULT_OUT_DIR = Path("/Users/carlosperez/PycharmProjects/MedSAM/TRAINING_DS_TOY")
DEFAULT_RUN_DIR = Path("/Users/carlosperez/PycharmProjects/MedSAM/runs")
DEFAULT_CFG = Path("/Users/carlosperez/PycharmProjects/MedSAM/src/configs/train_custom.yaml")
DEFAULT_MODEL = "yolo12n.pt"
DEFAULT_DEVICE: Optional[str | int] = "mps"

DEFAULT_IMGSZ = 640
DEFAULT_EPOCHS = 1
DEFAULT_BATCH = 16
DEFAULT_WORKERS = 8  # not used directly by this script, kept for reference
SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

VISUALIZE_ONE = False

SPACE = {
    # optimizer / schedule
    "lr0": (1e-4, 5e-3),
    "lrf": (0.1, 1.0),
    "weight_decay": (0.0, 5e-4),
    "momentum": (0.85, 0.98),
    "warmup_epochs": (0.0, 2.0),

    # loss balance
    "box": (5, 15),
    "cls": (0.2, 2.0),

    # light augmentations (fundus-safe)
    "degrees": (0.0, 7.5),
    "translate": (0.0, 0.10),
    "scale": (0.0, 0.20),
    "shear": (0.0, 3.0),
    "hsv_h": (0.0, 0.02),
    "hsv_s": (0.0, 0.30),
    "hsv_v": (0.0, 0.30),
    "flipud": (0.0, 0.10),
    "fliplr": (0.0, 0.10),

    # avoid aggressive mixing for medical images
    "mosaic": (0.0, 0.10),
    "mixup": (0.0, 0.05),
    "copy_paste": (0.0, 0.0),
}

# =========================
# Helpers
# =========================
def set_global_seed(seed: int = 42) -> None:
    """Best-effort reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
    except Exception:
        pass  # torch may not be installed

def infer_data_yaml_path(out_dir: Path) -> Path:
    """Conventional location for the dataset YAML produced by create_yolo_dataset()."""
    return out_dir / "data.yaml"

def locate_best_weights(run_dir: Path) -> Optional[Path]:
    weights_dir = run_dir / "weights"
    if weights_dir.exists():
        best = next(weights_dir.glob("best*.pt"), None)
        if best:
            print(f"[OK] Best weights: {best}")
            return best
    return None

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Train/Tune/Test YOLO on glaucoma datasets.")

    # Modes (all optional; Slurm script decides which to pass)
    p.add_argument("--train", action="store_true", help="Enable training mode.")
    p.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning mode.")
    p.add_argument("--test-weights", type=Path, dest="test_weights",
                   help="Path to weights for running evaluation on the TEST split.")

    # Paths/config
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                   help=f"Root directory with datasets (default: {DEFAULT_DATA_ROOT})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Output directory for YOLO dataset (default: {DEFAULT_OUT_DIR})")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR,
                   help=f"Output directory for runs (default: {DEFAULT_RUN_DIR})")
    p.add_argument("--cfg", type=Path, default=DEFAULT_CFG,
                   help=f"Ultralytics train config YAML (default: {DEFAULT_CFG})")
    p.add_argument("--yolo-ds", type=Path,
                   help="Path to preprocessed YOLO dataset directory (containing data.yaml).")

    # Training/tuning knobs
    p.add_argument("--resume", action="store_true", help="Resume training/tuning if possible.")
    p.add_argument("--run-name", type=str, help="Run name (overrides automatic names).")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help=f"Starting YOLO checkpoint (default: {DEFAULT_MODEL})")
    p.add_argument("--device", type=str, default=str(DEFAULT_DEVICE),
                   help=f"Device: '0', 'cpu', 'mps', or '0,1' (default: {DEFAULT_DEVICE})")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Epochs (default: {DEFAULT_EPOCHS})")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help=f"Batch size (default: {DEFAULT_BATCH})")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help=f"Image size (default: {DEFAULT_IMGSZ})")

    return p.parse_args()

def scan_filter(DATA_ROOT: Path, OUT_DIR: Path):
    """Scan datasets and filter to images with both masks; write summary."""
    print("[INFO] Scanning datasets…")
    image_factory = ImageFactory(root=DATA_ROOT, auto_scan=True)
    image_factory.filter_empty_masks()
    image_factory.filter_datasets(exclude=["PAPILA"])
    # image_factory.filter_random_subset(100)
    images = image_factory.make_images()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_factory.save_images(images, OUT_DIR / "saved_images.jsonl")
    if not images:
        raise RuntimeError("No images after filtering. Check dataset layout and mask availability.")
    print(f"[INFO] Found {len(images)} paired samples across datasets.")
    if VISUALIZE_ONE:
        try:
            images[0].visualize(show=True)
        except Exception as e:
            print(f"[WARN] Visualization failed: {e!r}")
    return images

def create_yolo_ds(images, OUT_DIR: Path) -> Path:
    """Create YOLO directory structure and return data.yaml path."""
    print("[INFO] Creating YOLO dataset structure…")
    create_yolo_dataset(images, train=TRAIN_RATIO, val=VAL_RATIO, test=TEST_RATIO, out_dir=OUT_DIR)
    data_yaml = infer_data_yaml_path(OUT_DIR)
    if not data_yaml.exists():
        raise FileNotFoundError(f"`data.yaml` not found at {data_yaml}.")
    return data_yaml

# =========================
# Main
# =========================
if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    set_global_seed(SEED)

    args = parse_args()

    DATA_ROOT: Path = args.data_root
    OUT_DIR: Path = args.out_dir
    RUN_DIR: Path = args.run_dir
    RUN_NAME: Optional[str] = args.run_name

    MODEL = args.model
    DEVICE = args.device
    CFG: Path = args.cfg
    YOLO_DS: Optional[Path] = args.yolo_ds
    EPOCHS = args.epochs
    BATCH = args.batch
    IMGSZ = args.imgsz
    TRAIN_MODE = args.train
    TUNE_MODE = args.tune
    RESUME = args.resume
    TEST_WEIGHTS: Optional[Path] = args.test_weights

    print(f"[INFO] DATA_ROOT = {DATA_ROOT}")
    print(f"[INFO] OUT_DIR   = {OUT_DIR}")
    print(f"[INFO] RUN_DIR   = {RUN_DIR}")
    print(f"[INFO] MODEL     = {MODEL}")
    print(f"[INFO] DEVICE    = {DEVICE}")
    print(f"[INFO] CFG       = {CFG}")
    print(f"[INFO] EPOCHS    = {EPOCHS}")
    print(f"[INFO] BATCH     = {BATCH}")
    print(f"[INFO] IMGSZ     = {IMGSZ}")
    print(f"[INFO] MODES     = train={TRAIN_MODE}, tune={TUNE_MODE}, test={'yes' if TEST_WEIGHTS else 'no'}")

    # Resolve dataset YAML
    if YOLO_DS:
        data_yaml = YOLO_DS / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"[ERR] --yolo-ds given but {data_yaml} does not exist.")
        print(f"[INFO] Using pre-existing YOLO dataset: {YOLO_DS}")
    else:
        # If testing only and no yolo-ds is provided, fail fast (avoid huge scan/build on cluster unexpectedly)
        if TEST_WEIGHTS and not (TRAIN_MODE or TUNE_MODE):
            raise RuntimeError("[ERR] Testing requested but --yolo-ds not provided. Supply a prebuilt dataset.")
        print("[INFO] Creating YOLO dataset (no --yolo-ds provided)…")
        images = scan_filter(DATA_ROOT, OUT_DIR)
        data_yaml = create_yolo_ds(images, OUT_DIR)

    print(f"[INFO] data.yaml = {data_yaml}")

    # Create model
    set_global_seed(SEED)
    model = YOLO(MODEL)

    # TRAIN
    if TRAIN_MODE:
        rn = RUN_NAME or "Train"
        print("[INFO] Starting training…")
        model.train(
            cfg=str(CFG),
            project=str(RUN_DIR),
            name=rn,
            data=str(data_yaml),
            device=DEVICE,
            epochs=EPOCHS,
            exist_ok=False,
            resume=RESUME,
            freeze=5,
            imgsz=IMGSZ,
            batch=BATCH,
        )

    # TUNE
    if TUNE_MODE:
        rn = RUN_NAME or "Tune"
        print("[INFO] Starting tuning…")
        best = model.tune(
            data=str(data_yaml),
            project=str(RUN_DIR),
            name=rn,
            device=DEVICE,
            imgsz=IMGSZ,
            batch=BATCH,
            iterations=15,
            resume=RESUME,
            epochs=EPOCHS,
            optimizer="AdamW",
            plots=True,
            save=True,
            val=True,
            space=SPACE,
        )
        print("[OK] Best hyperparameters saved at:", best)

    # TEST
    if TEST_WEIGHTS:
        if not TEST_WEIGHTS.exists():
            raise FileNotFoundError(f"[ERR] --test-weights not found: {TEST_WEIGHTS}")
        rn = RUN_NAME or "Test"
        print(f"[INFO] Running final evaluation on the TEST set… weights={TEST_WEIGHTS}")
        model = YOLO(str(TEST_WEIGHTS))
        metrics = model.val(
            data=str(data_yaml),
            split="test",
            device=DEVICE,
            imgsz=IMGSZ,
            batch=BATCH,
            save_json=True,
            plots=True,
        )
        out_json = OUT_DIR / "test_metrics.json"
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[OK] Saved test metrics to: {out_json}")
    elif not (TRAIN_MODE or TUNE_MODE):
        # If none of the modes were selected, warn
        print("[WARN] No mode selected (use --train, --tune, and/or --test-weights).")