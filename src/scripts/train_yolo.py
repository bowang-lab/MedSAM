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

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _parse_data_yaml(yaml_path: Path) -> Dict[str, str]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text())
    out = {}
    for k in ("train", "val", "test"):
        if k in data:
            out[k] = str(data[k])
    return out

def _read_lines_txt(txt_path: Path) -> List[Path]:
    with txt_path.open("r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return [Path(ln) for ln in lines]

def _list_images_under(p: Path) -> List[Path]:
    if not p.exists():
        return []
    if p.is_file():
        return [p] if p.suffix.lower() in _IMG_EXTS else []
    return sorted([q for q in p.rglob("*") if q.is_file() and q.suffix.lower() in _IMG_EXTS])

def _resolve_split_images(yolo_ds: Path, entry: str | None) -> List[Path]:
    if not entry:
        return []
    p = Path(entry)
    if not p.is_absolute():
        p = (yolo_ds / p).resolve()
    if p.suffix.lower() == ".txt":
        return _read_lines_txt(p)
    return _list_images_under(p)

def _image_to_label_path(img_path: Path, yolo_ds: Path) -> Optional[Path]:
    """
    Derive labels path from images path by replacing 'images' with 'labels' and extension with .txt.
    """
    try:
        parts = list(img_path.parts)
        # Replace the first occurrence of 'images' with 'labels'
        for i, seg in enumerate(parts):
            if seg == "images":
                parts[i] = "labels"
                break
        lbl_dir = Path(*parts[:-1])  # directory up to filename
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            return lbl_path
        # Fallback: search sibling labels dir under yolo_ds
        candidate = (yolo_ds / "labels" / img_path.parent.name / (img_path.stem + ".txt"))
        return candidate if candidate.exists() else None
    except Exception:
        return None

def _load_gt_boxes_norm(label_path: Path) -> Dict[int, NormalizedBox]:
    """
    Read YOLO label file and return {class_id: NormalizedBox} for classes present.
    If multiple boxes for a class exist, keep the largest area (rare for OD/OC).
    """
    by_cls: Dict[int, NormalizedBox] = {}
    if not label_path or not label_path.exists():
        return by_cls
    for ln in label_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        nb = NormalizedBox(xc, yc, w, h)
        if (cls not in by_cls) or (nb.area() > by_cls[cls].area()):
            by_cls[cls] = nb
    return by_cls

def _best_pred_box_norm_for_class(res, cls_id: int) -> Optional[NormalizedBox]:
    """
    From an Ultralytics Result (single image), return the top-confidence predicted box
    for class `cls_id` in normalized (xc,yc,w,h) space, or None if not found.
    """
    if res.boxes is None or len(res.boxes) == 0:
        return None
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()
    if hasattr(res.boxes, "xywhn") and res.boxes.xywhn is not None:
        xywhn = res.boxes.xywhn.cpu().numpy()
    else:
        # Normalize manually using image shape
        xywh = res.boxes.xywh.cpu().numpy()
        H, W = res.orig_shape
        xywhn = xywh.copy()
        xywhn[:, 0] /= W
        xywhn[:, 1] /= H
        xywhn[:, 2] /= W
        xywhn[:, 3] /= H

    idx = np.where(cls == cls_id)[0]
    if idx.size == 0:
        return None
    best = idx[np.argmax(conf[idx])]
    xc, yc, w, h = map(float, xywhn[best, :4])
    # clip to [0,1] for safety
    xc = float(np.clip(xc, 0.0, 1.0))
    yc = float(np.clip(yc, 0.0, 1.0))
    w  = float(np.clip(w,  0.0, 1.0))
    h  = float(np.clip(h,  0.0, 1.0))
    return NormalizedBox(xc, yc, w, h)

def evaluate_test_boxes(
    model: YOLO,
    yolo_ds: Path,
    out_dir: Path,
    device: str = "cpu",
    imgsz: int = 640,
    conf: float = 0.25,
) -> Tuple[Dict[str, float], Path, Path]:
    """
    Run inference on the test split and compute per-prediction Dice (box Dice) and box loss (1-CIoU).
    Returns (summary_dict, jsonl_path, summary_json_path).
    """
    yaml_path = yolo_ds / "data.yaml"
    entries = _parse_data_yaml(yaml_path)
    test_imgs = _resolve_split_images(yolo_ds, entries.get("test"))
    if not test_imgs:
        raise RuntimeError(f"No test images resolved from {yaml_path}")

    per_class_dice: Dict[int, List[float]] = {0: [], 1: []}
    per_class_loss: Dict[int, List[float]] = {0: [], 1: []}

    jsonl_path = out_dir / "test_box_metrics.jsonl"
    summary_path = out_dir / "test_box_summary.json"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w") as jf:
        for img_path in test_imgs:
            lbl_path = _image_to_label_path(img_path, yolo_ds)
            gt = _load_gt_boxes_norm(lbl_path) if lbl_path else {}

            # Run inference for this image
            res = model.predict(
                source=str(img_path),
                device=device,
                imgsz=imgsz,
                conf=conf,
                verbose=False
            )[0]

            # Evaluate for both classes (0=disc, 1=cup)
            for cls_id in (0, 1):
                rec = {
                    "image": str(img_path),
                    "class": int(cls_id),
                    "pred_exists": False,
                    "gt_exists": False,
                    "dice_box": None,
                    "box_loss": None,
                }

                gt_box = gt.get(cls_id, None)
                if gt_box is not None:
                    rec["gt_exists"] = True

                pred_box = _best_pred_box_norm_for_class(res, cls_id)
                if pred_box is not None:
                    rec["pred_exists"] = True

                if (pred_box is not None) and (gt_box is not None):
                    dice = float(pred_box.dice(gt_box))
                    loss = float(pred_box.box_loss(gt_box))
                    rec["dice_box"] = dice
                    rec["box_loss"] = loss
                    per_class_dice[cls_id].append(dice)
                    per_class_loss[cls_id].append(loss)
                else:
                    # If either side is missing, record zeros but do not include in mean
                    rec["dice_box"] = 0.0
                    rec["box_loss"] = 1.0

                jf.write(json.dumps(rec) + "\n")

    # Summary
    def _mean(lst: List[float]) -> float:
        return float(np.mean(lst)) if lst else float("nan")

    summary = {
        "disc": {
            "mean_dice_box": _mean(per_class_dice[0]),
            "mean_box_loss": _mean(per_class_loss[0]),
            "n_effective": len(per_class_dice[0]),
        },
        "cup": {
            "mean_dice_box": _mean(per_class_dice[1]),
            "mean_box_loss": _mean(per_class_loss[1]),
            "n_effective": len(per_class_dice[1]),
        },
        "notes": "Dice here is computed over boxes (not masks). Means exclude missing GT/pred pairs."
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("[OK] Wrote per-prediction metrics:", jsonl_path)
    print("[OK] Wrote summary:", summary_path)
    print("[SUMMARY]", summary)
    return summary, jsonl_path, summary_path

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
    # TEST
    if TEST_WEIGHTS:
        if not TEST_WEIGHTS.exists():
            raise FileNotFoundError(f"[ERR] --test-weights not found: {TEST_WEIGHTS}")
        rn = RUN_NAME or "Test"
        print(f"[INFO] Running final evaluation on the TEST set… weights={TEST_WEIGHTS}")
        model = YOLO(str(TEST_WEIGHTS))

        # 3.1 Ultralytics' built-in evaluation (kept as-is)
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

        # 3.2 Additional per-prediction Dice (boxes) and box-loss summary
        #     Pass the YOLO dataset directory so we can find data.yaml and label files.
        yolo_ds_dir = args.yolo_ds if args.yolo_ds else OUT_DIR  # if you created DS in this run
        summary, per_pred_path, summary_path = evaluate_test_boxes(
            model=model,
            yolo_ds=yolo_ds_dir,
            out_dir=OUT_DIR,
            device=DEVICE,
            imgsz=IMGSZ,
            conf=0.25,  # adjust to your detector's operating point
        )
        # Optional: also persist an aggregated file alongside Ultralytics metrics
        agg_json = OUT_DIR / "test_box_summary_combined.json"
        with open(agg_json, "w") as f:
            json.dump({
                "ultralytics": metrics,
                "box_metrics_summary": summary
            }, f, indent=2)
        print(f"[OK] Saved combined summary to: {agg_json}")

    elif not (TRAIN_MODE or TUNE_MODE):
        print("[WARN] No mode selected (use --train, --tune, and/or --test-weights).")