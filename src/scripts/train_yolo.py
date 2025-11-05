#!/usr/bin/env python3
# src.scripts.train_yolo.py

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou as uy_box_iou

from src.imgpipe.image_factory import ImageFactory
from src.imgpipe.yolo_splits import create_yolo_dataset

# =========================
# Default Configuration (overridden by CLI)
# =========================
DEFAULT_DATA_ROOT = Path("/Users/carlosperez/Library/CloudStorage/OneDrive-UBC/Ipek_Carlos/GlaucomaDatasets/All_Datasets_Organized")
DEFAULT_OUT_DIR = Path("/Users/carlosperez/PycharmProjects/MedSAM/TEST_DS_PAPILA_ONLY")
DEFAULT_RUN_DIR = Path("/Users/carlosperez/PycharmProjects/MedSAM/runs")
DEFAULT_CFG = Path("/Users/carlosperez/PycharmProjects/MedSAM/src/configs/train_custom.yaml")
DEFAULT_MODEL = "yolo12x.pt"
DEFAULT_DEVICE: Optional[str | int] = "mps"
DEFAULT_IMGSZ = 640
DEFAULT_EPOCHS = 1
DEFAULT_BATCH = 16
DEFAULT_WORKERS = 8  # not used directly here
DEFAULT_CONF = 0.01
DEFAULT_IOU = 0.7
SEED = 42

TRAIN_RATIO = 0.0
VAL_RATIO = 0.0
TEST_RATIO = 1.0

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
# Reproducibility
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

# =========================
# Dataset build helpers
# =========================
def infer_data_yaml_path(out_dir: Path) -> Path:
    return out_dir / "data.yaml"

def scan_filter(DATA_ROOT: Path, OUT_DIR: Path):
    print("[INFO] Scanning datasets…")
    image_factory = ImageFactory(root=DATA_ROOT, auto_scan=True)
    image_factory.filter_empty_masks()
    image_factory.filter_datasets(include=["PAPILA"])
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
    print("[INFO] Creating YOLO dataset structure…")
    create_yolo_dataset(images, train=TRAIN_RATIO, val=VAL_RATIO, test=TEST_RATIO, out_dir=OUT_DIR)
    data_yaml = infer_data_yaml_path(OUT_DIR)
    if not data_yaml.exists():
        raise FileNotFoundError(f"`data.yaml` not found at {data_yaml}.")
    return data_yaml

# =========================
# Ultralytics metrics → JSON-safe
# =========================
def _jsonify(o: Any) -> Any:
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if hasattr(o, "tolist"):
        try:
            return _jsonify(o.tolist())
        except Exception:
            pass
    if isinstance(o, dict):
        return {str(k): _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonify(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if "torch" in type(o).__module__:
        try:
            return str(o)
        except Exception:
            return None
    if hasattr(o, "__dict__"):
        return {k: _jsonify(v) for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)

def _extract_ultralytics_metrics(val_obj: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    box = getattr(val_obj, "box", None)
    if box is not None:
        out["box"] = {
            "mp": _jsonify(getattr(box, "mp", np.nan)),
            "mr": _jsonify(getattr(box, "mr", np.nan)),
            "map50": _jsonify(getattr(box, "map50", np.nan)),
            "map": _jsonify(getattr(box, "map", np.nan)),
            "maps": _jsonify(getattr(box, "maps", [])),
        }
    speed = getattr(val_obj, "speed", None)
    if speed is not None:
        out["speed"] = _jsonify(speed)
    cm = getattr(val_obj, "confusion_matrix", None)
    if cm is not None and hasattr(cm, "matrix"):
        out["confusion_matrix_shape"] = list(np.shape(getattr(cm, "matrix", [])))
    results_dict = getattr(val_obj, "results_dict", None)
    if results_dict:
        out["results_dict"] = _jsonify(results_dict)
    out["raw_summary"] = _jsonify(val_obj)
    return out

# =========================
# Test evaluator (Hungarian; unmatched → zero)
# =========================
def _resolve_test_images(yolo_ds: Path, data: Dict[str, Any]) -> List[Path]:
    entry = data.get("test")
    if not entry:
        return []
    p = Path(entry)
    if not p.is_absolute():
        p = (yolo_ds / p).resolve()
    if p.suffix.lower() == ".txt":
        return [Path(s.strip()) for s in p.read_text().splitlines() if s.strip()]
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return sorted(x for x in p.rglob("*") if x.suffix.lower() in exts)
    raise ValueError(f"Unsupported test entry in data.yaml: {entry}")

def _label_path_from_image(img: Path) -> Path:
    parts = list(img.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        lbl = Path(*parts).with_suffix(".txt")
        if lbl.exists():
            return lbl
    except ValueError:
        pass
    return img.with_suffix(".txt")

def _load_gt_xyxy(img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    lbl = _label_path_from_image(img_path)
    if not lbl.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    W, H = PILImage.open(img_path).size
    rows = [ln.strip() for ln in lbl.read_text().splitlines() if ln.strip()]
    if not rows:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    xyxy, cls = [], []
    for r in rows:
        parts = r.split()
        c = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        Xc, Yc = xc * W, yc * H
        Wb, Hb = w * W, h * H
        x1, y1 = Xc - Wb / 2.0, Yc - Hb / 2.0
        x2, y2 = Xc + Wb / 2.0, Yc + Hb / 2.0
        xyxy.append([x1, y1, x2, y2])
        cls.append(c)
    return np.asarray(xyxy, np.float32), np.asarray(cls, np.int64)

def _predict_xyxy_conf(
    model: YOLO,
    img_path: Path,
    *,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (xyxy, cls, conf) for all predictions passing thresholds.
    """
    res = model.predict(
        source=str(img_path),
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )[0]
    if res.boxes is None or len(res.boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    b = res.boxes
    xyxy = b.xyxy.cpu().numpy().astype(np.float32)
    cls = b.cls.cpu().numpy().astype(np.int64)
    cf = b.conf.cpu().numpy().astype(np.float32)
    return xyxy, cls, cf

def _keep_single_top_conf_per_class(
    xyxy: np.ndarray, cls: np.ndarray, conf: np.ndarray, nc: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduce predictions to at most one (the highest-confidence) per class.
    """
    if xyxy.size == 0:
        return xyxy, cls, conf
    keep_idx: List[int] = []
    for c in range(nc):
        inds = np.where(cls == c)[0]
        if inds.size == 0:
            continue
        best = inds[np.argmax(conf[inds])]
        keep_idx.append(int(best))
    if not keep_idx:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    keep_idx = np.array(sorted(keep_idx), dtype=int)
    return xyxy[keep_idx], cls[keep_idx], conf[keep_idx]

def evaluate_test_boxes_hungarian_strict(
    model: YOLO,
    yolo_ds: Path,
    *,
    device: str = "cuda:0",
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.70,
    save_jsonl: Path | None = None,
) -> Dict[str, Any]:
    """
    Evaluation with enforced single prediction per class:
      - For each image and class, keep only the **highest-confidence** predicted box (0 or 1).
      - Hungarian matching on IoU (effectively 1-to-1 given the reduction).
      - Dice = 2*IoU/(1+IoU) for matched pairs.
      - Denominator per class += max(#GT, #Pred) → unmatched count as zero.
    """
    data_yaml = Path(yolo_ds) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
    data = yaml.safe_load(data_yaml.read_text())
    test_imgs = _resolve_test_images(yolo_ds, data)
    if not test_imgs:
        raise RuntimeError("No test images resolved from data.yaml 'test' entry.")

    # class count from model
    names = getattr(model, "names", None) or {}
    if isinstance(names, dict):
        nc = len(names)
        class_names = [names[i] for i in range(nc)]
    elif isinstance(names, list):
        nc = len(names)
        class_names = names
    else:
        nc = int(max(getattr(model, "nc", 2), 2))
        class_names = [str(i) for i in range(nc)]

    dice_sum = np.zeros(nc, dtype=np.float64)
    match_cnt = np.zeros(nc, dtype=np.int64)

    jsonl_fp = None
    if save_jsonl is not None:
        Path(save_jsonl).parent.mkdir(parents=True, exist_ok=True)
        jsonl_fp = open(save_jsonl, "w")

    for img_path in test_imgs:
        gt_b, gt_c = _load_gt_xyxy(img_path)

        # All predictions for the image
        pr_b_all, pr_c_all, pr_cf_all = _predict_xyxy_conf(
            model, img_path, device=device, imgsz=imgsz, conf=conf, iou=iou
        )
        # Reduce to **one per class** (highest confidence)
        pr_b, pr_c, pr_cf = _keep_single_top_conf_per_class(pr_b_all, pr_c_all, pr_cf_all, nc)

        for c in range(nc):
            gi = np.where(gt_c == c)[0]
            pi = np.where(pr_c == c)[0]
            n_g, n_p = len(gi), len(pi)  # n_p is 0 or 1 due to reduction
            den = max(n_g, n_p)  # denominator counts unmatched as zero
            sum_dice = 0.0

            if n_g > 0 and n_p > 0:
                # With at most one pred per class, Hungarian degenerates to single IoU
                g = torch.from_numpy(gt_b[gi]).float()
                p = torch.from_numpy(pr_b[pi]).float()
                iou_mat = uy_box_iou(g, p).cpu().numpy()
                # Best one-to-one pairing (still use Hungarian/greedy for completeness)
                try:
                    import scipy.optimize as spo
                    row_ind, col_ind = spo.linear_sum_assignment(-iou_mat)
                    pairs = [(r, c2, iou_mat[r, c2]) for r, c2 in zip(row_ind, col_ind)]
                except Exception:
                    # Greedy fallback
                    pairs = []
                    used_g, used_p = set(), set()
                    flat = [(r, c2, iou_mat[r, c2]) for r in range(n_g) for c2 in range(n_p)]
                    flat.sort(key=lambda x: x[2], reverse=True)
                    for r, c2, v in flat:
                        if r not in used_g and c2 not in used_p:
                            used_g.add(r); used_p.add(c2); pairs.append((r, c2, v))
                for _, _, u in pairs:
                    d = (2.0 * float(u)) / (1.0 + float(u)) if u > 0.0 else 0.0
                    sum_dice += d

            dice_sum[c] += sum_dice
            match_cnt[c] += den

            if jsonl_fp is not None:
                rec = {
                    "image": str(img_path),
                    "class": int(c),
                    "n_gt": int(n_g),
                    "n_pred": int(n_p),
                    "sum_dice": float(sum_dice),
                    "den": int(den),
                }
                jsonl_fp.write(json.dumps(rec) + "\n")

    if jsonl_fp is not None:
        jsonl_fp.close()

    eps = 1e-12
    per_class = (dice_sum / np.maximum(match_cnt, eps)).tolist()
    macro = float(np.mean(per_class)) if nc else 0.0

    return {
        "per_class_dice": per_class,
        "macro_dice": macro,
        "match_count": match_cnt.tolist(),
        "class_names": class_names,
        "conf_used": conf,
        "iou_used": iou,
    }

# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/Tune/Test YOLO on glaucoma datasets.")
    # Modes
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
    # Eval thresholds (align test with validator)
    p.add_argument("--conf", type=float, default=DEFAULT_CONF, help=f"Confidence threshold (default: {DEFAULT_CONF})")
    p.add_argument("--iou", type=float, default=DEFAULT_IOU, help=f"IoU threshold (default: {DEFAULT_IOU})")
    return p.parse_args()

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
    CONF_TH = float(args.conf)
    IOU_TH = float(args.iou)

    # Ensure output dir exists for test-only runs as well
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] DATA_ROOT = {DATA_ROOT}")
    print(f"[INFO] OUT_DIR   = {OUT_DIR}")
    print(f"[INFO] RUN_DIR   = {RUN_DIR}")
    print(f"[INFO] MODEL     = {MODEL}")
    print(f"[INFO] DEVICE    = {DEVICE}")
    print(f"[INFO] CFG       = {CFG}")
    print(f"[INFO] EPOCHS    = {EPOCHS}")
    print(f"[INFO] BATCH     = {BATCH}")
    print(f"[INFO] IMGSZ     = {IMGSZ}")
    print(f"[INFO] CONF/IOU  = {CONF_TH} / {IOU_TH}")
    print(f"[INFO] MODES     = train={TRAIN_MODE}, tune={TUNE_MODE}, test={'yes' if TEST_WEIGHTS else 'no'}")

    # Resolve dataset YAML
    if YOLO_DS:
        data_yaml = YOLO_DS / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"[ERR] --yolo-ds given but {data_yaml} does not exist.")
        print(f"[INFO] Using pre-existing YOLO dataset: {YOLO_DS}")
    else:
        print("[INFO] Creating YOLO dataset (no --yolo-ds provided)…")
        images = scan_filter(DATA_ROOT, OUT_DIR)
        data_yaml = create_yolo_ds(images, OUT_DIR)
        YOLO_DS = OUT_DIR  # set for later use

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

        # 1) Ultralytics built-in evaluation (use same thresholds)
        val_obj = model.val(
            data=str(data_yaml),
            split="test",
            device=DEVICE,
            imgsz=IMGSZ,
            batch=BATCH,
            conf=CONF_TH,
            iou=IOU_TH,
            save_json=True,
            plots=True,
        )
        out_json = OUT_DIR / "test_metrics.json"
        with open(out_json, "w") as f:
            json.dump(_extract_ultralytics_metrics(val_obj), f, indent=2)
        print(f"[OK] Saved test metrics to: {out_json}")

        # 2) Strict Dice with unmatched counted as zero (Hungarian; single top-confidence pred per class)
        yolo_ds_dir = YOLO_DS if YOLO_DS is not None else OUT_DIR
        summary = evaluate_test_boxes_hungarian_strict(
            model=model,
            yolo_ds=yolo_ds_dir,
            device=DEVICE,
            imgsz=IMGSZ,
            conf=CONF_TH,
            iou=IOU_TH,
            save_jsonl=OUT_DIR / "test_dice_records.jsonl",
        )
        agg_json = OUT_DIR / "test_box_summary_combined.json"
        with open(agg_json, "w") as f:
            json.dump({
                "ultralytics": _extract_ultralytics_metrics(val_obj),
                "box_metrics_summary": summary
            }, f, indent=2)
        print(f"[OK] Saved combined summary to: {agg_json}")

    elif not (TRAIN_MODE or TUNE_MODE):
        print("[WARN] No mode selected (use --train, --tune, and/or --test-weights).")