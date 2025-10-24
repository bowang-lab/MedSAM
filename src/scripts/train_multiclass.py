#!/usr/bin/env python3
# src/model/train_multiclass_cfg.py
"""
Multiclass YOLO (disc=0, cup=1) trainer/evaluator with:
- Optional train-split filtering to only keep images that contain BOTH classes
- Robust resume logic (auto/explicit) and epoch extension
- Eval-only mode to compute *test* metrics vs. GT by ourselves (Box Error = 1 - IoU, Rect-Dice)
- When a filtered YAML is created, we keep val/test usable by writing ABSOLUTE paths
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from ultralytics import YOLO

from src.utils import ultralytics_device_arg  # only what we need


# ------------------ checkpoint helpers ------------------

def _load_ckpt_epochs(ckpt_path: Path) -> tuple[int, Optional[int]]:
    """
    Returns (trained_epochs_done, target_epochs_in_ckpt_or_None).
    PyTorch 2.6+: torch.load defaults to weights_only=True; we need the full object.
    """
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)  # PyTorch 2.6+
    except TypeError:  # older PyTorch
        ckpt = torch.load(str(ckpt_path), map_location="cpu")

    trained = int(ckpt.get("epoch", -1)) + 1  # convert 0-based to count
    targs = ckpt.get("train_args") or ckpt.get("args") or {}
    planned = targs.get("epochs")
    try:
        planned = int(planned) if planned is not None else None
    except Exception:
        planned = None
    return trained, planned


# ---------------- Rectangle-based Dice (proxy) ----------------

def _rect_to_mask(H: int, W: int, bbox_xywh) -> np.ndarray:
    x, y, w, h = bbox_xywh
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(W, int(round(x + w)))
    y2 = min(H, int(round(y + h)))
    m = np.zeros((H, W), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 1
    return m


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    s = a.sum() + b.sum()
    return (2.0 * inter / s) if s > 0 else float("nan")


# ---------------- helpers ----------------

def _resolve_weights(family: str | None, size: str | None, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    fam = (family or "auto").lower()
    sz = (size or "x").lower()
    if fam in ("auto", "yolo12"):
        return f"yolo12{sz}.pt"
    if fam == "yolo11":
        return f"yolo11{sz}.pt"
    return f"yolov8{sz}.pt"


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _count_images(path_key: str, data_yaml: Dict[str, Any], data_yaml_dir: Path) -> int:
    """
    Count images under the split referenced by 'train'/'val'/'test'.
    If data.yaml has a 'path:' base, interpret it RELATIVE TO THE YAML FILE.
    """
    split_entry = data_yaml.get(path_key)
    if not split_entry:
        return 0

    split_path = Path(str(split_entry))
    base_field = data_yaml.get("path")
    if base_field:
        base_path = Path(str(base_field))
        base = base_path if base_path.is_absolute() else (data_yaml_dir / base_path).resolve()
    else:
        base = data_yaml_dir

    if not split_path.is_absolute():
        split_path = (base / split_path).resolve()
    if not split_path.exists():
        return 0

    if split_path.is_dir():
        return sum(1 for p in split_path.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        try:
            return sum(
                1 for ln in split_path.read_text().splitlines()
                if ln.strip() and Path(ln.strip()).exists()
            )
        except Exception:
            return 0
    return 0


# ---------------- train-split "require both classes" filter ----------------

def _iter_images_from_dir(images_dir: Path) -> Iterable[Path]:
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def _label_path_candidates_for_image(img_path: Path) -> List[Path]:
    """Map an image path to likely label .txt paths (common YOLO layouts)."""
    cands: List[Path] = []

    # Case A: replace first 'images' with 'labels'
    parts = list(img_path.parts)
    if "images" in parts:
        i = parts.index("images")
        lbl_parts = parts[:]
        lbl_parts[i] = "labels"
        cands.append(Path(*lbl_parts).with_suffix(".txt"))

    # Case B: sibling labels/<split>/file.txt
    if img_path.parent.name.lower() in {"train", "val", "test"}:
        split = img_path.parent.name
        cands.append(img_path.parent.parent / "labels" / split / (img_path.stem + ".txt"))

    # Case C: labels folder next to current folder
    cands.append(img_path.parent.parent / "labels" / (img_path.stem + ".txt"))

    # Deduplicate
    seen = set()
    out: List[Path] = []
    for p in cands:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _label_has_both_classes(lbl_path: Path) -> bool:
    """True if label file exists and contains at least one class 0 and one class 1 line."""
    if not lbl_path.exists():
        return False
    try:
        cls_ids = set()
        for ln in lbl_path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            c0 = int(float(ln.split()[0]))
            cls_ids.add(c0)
        return (0 in cls_ids) and (1 in cls_ids)
    except Exception:
        return False


def _abs_split(entry: Any, base_dir: Path) -> str:
    """
    Return an absolute path for a split entry (dir or list.txt) given a base_dir.
    """
    if entry is None:
        return ""
    p = Path(str(entry))
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return str(p)


def _filter_train_to_both(data_yaml_path: Path, work_root: Path) -> Path:
    """
    Create a filtered data.yaml that keeps only training images with BOTH classes present.
    CRITICAL: write absolute paths for 'path', 'val', and 'test' so evaluation doesn't break.
    """
    y = yaml.safe_load(data_yaml_path.read_text()) or {}
    ydir = data_yaml_path.parent

    # Resolve base 'path' (relative to YAML file dir)
    base_field = y.get("path")
    if base_field:
        base_path = Path(str(base_field))
        base = base_path if base_path.is_absolute() else (ydir / base_path).resolve()
    else:
        base = ydir

    train_entry = y.get("train")
    if not train_entry:
        print("[FILTER] No 'train' entry found; skipping filter.")
        return data_yaml_path

    train_path = Path(str(train_entry))
    if not train_path.is_absolute():
        train_path = (base / train_path).resolve()

    # Collect candidate images
    images: List[Path] = []
    if train_path.is_dir():
        images = list(_iter_images_from_dir(train_path))
    elif train_path.is_file() and train_path.suffix.lower() == ".txt":
        try:
            for ln in train_path.read_text().splitlines():
                s = ln.strip()
                if not s:
                    continue
                p = Path(s)
                images.append(p if p.is_absolute() else (train_path.parent / p).resolve())
        except Exception as e:
            print(f"[FILTER] Could not read {train_path}: {e}")
    else:
        print(f"[FILTER] Unsupported 'train' entry: {train_path} (must be folder or .txt). Skipping filter.")
        return data_yaml_path

    # Keep only images whose labels contain BOTH classes
    keep: List[Path] = []
    drop = 0
    for img in images:
        ok = False
        for cand in _label_path_candidates_for_image(img):
            if _label_has_both_classes(cand):
                ok = True
                break
        if ok:
            keep.append(img.resolve())
        else:
            drop += 1

    if not keep:
        print("[FILTER] No training images with both classes were found. Using original 'train'.")
        return data_yaml_path

    # Output filtered YAML + list file
    out_dir = work_root / "_filtered_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_list = out_dir / "train_require_both.txt"
    train_list.write_text("\n".join(str(p) for p in keep) + "\n")

    new_y = dict(y)
    # Make base 'path' ABSOLUTE so val/test keep working even though YAML moved
    new_y["path"] = str(base.resolve())
    new_y["train"] = str(train_list.resolve())

    # Force absolute val/test (dirs or list files)
    if "val" in y and y["val"]:
        new_y["val"] = _abs_split(y["val"], base)
    if "test" in y and y["test"]:
        new_y["test"] = _abs_split(y["test"], base)

    new_yaml = out_dir / "data_filtered.yaml"
    new_yaml.write_text(yaml.safe_dump(new_y, sort_keys=False))

    print(f"[FILTER] Train images: kept={len(keep)} dropped={drop} → {train_list}")
    return new_yaml


# ---------------- config objects ----------------

@dataclass
class AugCfg:
    hsv_h: float = 0.015
    hsv_s: float = 0.70
    hsv_v: float = 0.40
    degrees: float = 10.0
    translate: float = 0.20
    scale: float = 0.60
    shear: float = 2.0
    perspective: float = 0.001
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 0.20
    mixup: float = 0.10
    copy_paste: float = 0.00
    erasing: float = 0.40


@dataclass
class TrainCfg:
    # paths
    data: Path
    runs_root: Path
    # model
    weights: Optional[str]
    family: str
    size: str
    # training
    resume: Optional[str]
    extend_epochs: Optional[int]
    name: str
    epochs: int
    imgsz: int
    batch: int
    workers: int
    seed: int
    device: Optional[str]
    amp: bool
    freeze: int
    optimizer: str
    cos_lr: bool
    patience: int
    multi_scale: bool
    close_mosaic: int
    require_both_train: bool
    # augmentations
    aug: AugCfg

    @staticmethod
    def from_yaml(path: Path) -> "TrainCfg":
        if not path.exists():
            raise SystemExit(f"[ERR] config YAML not found: {path}")
        cfg = yaml.safe_load(path.read_text()) or {}

        data_yaml = Path(str(cfg.get("data"))).expanduser().resolve() if cfg.get("data") else None
        if not data_yaml:
            raise SystemExit("[ERR] 'data' (path to data.yaml) is required in the config.")
        if not data_yaml.exists():
            raise SystemExit(f"[ERR] data.yaml not found: {data_yaml}")

        runs_root = Path(str(cfg.get("runs_root", "./runs/detect"))).expanduser().resolve()
        runs_root.mkdir(parents=True, exist_ok=True)

        aug = cfg.get("augment", {}) or {}
        aug_cfg = AugCfg(
            hsv_h=float(aug.get("hsv_h", 0.015)),
            hsv_s=float(aug.get("hsv_s", 0.70)),
            hsv_v=float(aug.get("hsv_v", 0.40)),
            degrees=float(aug.get("degrees", 10.0)),
            translate=float(aug.get("translate", 0.20)),
            scale=float(aug.get("scale", 0.60)),
            shear=float(aug.get("shear", 2.0)),
            perspective=float(aug.get("perspective", 0.001)),
            flipud=float(aug.get("flipud", 0.0)),
            fliplr=float(aug.get("fliplr", 0.5)),
            mosaic=float(aug.get("mosaic", 0.20)),
            mixup=float(aug.get("mixup", 0.10)),
            copy_paste=float(aug.get("copy_paste", 0.00)),
            erasing=float(aug.get("erasing", 0.40)),
        )

        dev = cfg.get("device", "auto")
        if dev in (None, "", "auto"):
            dev = ultralytics_device_arg()

        return TrainCfg(
            data=data_yaml,
            runs_root=runs_root,
            weights=cfg.get("weights"),
            family=str(cfg.get("family", "auto")),
            size=str(cfg.get("size", "x")),
            name=str(cfg.get("name", "multiclass_disc_cup")),
            epochs=int(cfg.get("epochs", 200)),
            imgsz=int(cfg.get("imgsz", 640)),
            batch=int(cfg.get("batch", 16)),
            workers=int(cfg.get("workers", 8)),
            seed=int(cfg.get("seed", 1337)),
            device=str(dev),
            amp=bool(cfg.get("amp", True)),
            freeze=int(cfg.get("freeze", 0)),
            optimizer=str(cfg.get("optimizer", "AdamW")),
            cos_lr=bool(cfg.get("cos_lr", True)),
            patience=int(cfg.get("patience", 50)),
            multi_scale=bool(cfg.get("multi_scale", True)),
            close_mosaic=int(cfg.get("close_mosaic", 10)),
            resume=(str(cfg.get("resume") or "") or None),
            extend_epochs=(
                int(cfg["extend_epochs"])
                if "extend_epochs" in cfg and cfg["extend_epochs"] not in ("", None)
                else None
            ),
            require_both_train=bool(cfg.get("require_both_train", True)),
            aug=aug_cfg,
        )


# ---------------- basic geometry ----------------

def _xywhn_to_xyxy_pixels(cx_n: float, cy_n: float, w_n: float, h_n: float, W: int, H: int) -> Tuple[int, int, int, int]:
    """YOLO-normalized center-based box → pixel xyxy (clamped)."""
    cx = cx_n * W
    cy = cy_n * H
    w = w_n * W
    h = h_n * H
    x1 = int(round(max(0.0, cx - w / 2.0)))
    y1 = int(round(max(0.0, cy - h / 2.0)))
    x2 = int(round(min(float(W), cx + w / 2.0)))
    y2 = int(round(min(float(H), cy + h / 2.0)))
    x1 = min(x1, W - 1)
    y1 = min(y1, H - 1)
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    return x1, y1, x2, y2


def _xyxy_to_xywh(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    if ua <= 0:
        return 0.0
    return inter / ua


# ---------------- gather split & labels ----------------

def _gather_split_images(data_yaml_path: Path, split: str) -> List[Path]:
    y = yaml.safe_load(data_yaml_path.read_text()) or {}
    ydir = data_yaml_path.parent
    base_field = y.get("path")
    if base_field:
        base = (ydir / str(base_field)).resolve() if not Path(str(base_field)).is_absolute() else Path(str(base_field))
    else:
        base = ydir

    entry = y.get(split)
    if not entry:
        return []

    p = Path(str(entry))
    if not p.is_absolute():
        p = (base / p).resolve()
    out: List[Path] = []
    if p.is_dir():
        for img in _iter_images_from_dir(p):
            out.append(img.resolve())
    elif p.is_file() and p.suffix.lower() == ".txt":
        for ln in p.read_text().splitlines():
            s = ln.strip()
            if not s:
                continue
            ii = Path(s)
            out.append(ii if ii.is_absolute() else (p.parent / ii).resolve())
    return out


def _read_gt_xyxy_for_image(img_path: Path, classes=(0, 1)) -> Dict[int, Tuple[int, int, int, int]]:
    """Read YOLO label file mapped from image → return {cls: xyxy_pixels} for requested classes."""
    # locate label file
    lbl: Optional[Path] = None
    for cand in _label_path_candidates_for_image(img_path):
        if cand.exists():
            lbl = cand
            break
    if lbl is None or not lbl.exists():
        return {}

    # image size
    with PILImage.open(str(img_path)) as im:
        W, H = im.size

    out: Dict[int, Tuple[int, int, int, int]] = {}
    for ln in lbl.read_text().splitlines():
        s = ln.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            if cls not in classes:
                continue
            cx, cy, w, h = map(float, parts[1:5])
        except Exception:
            continue
        xyxy = _xywhn_to_xyxy_pixels(cx, cy, w, h, W, H)
        # If multiple per class, keep the first; typical fundus has exactly one per class
        if cls not in out:
            out[cls] = xyxy
    return out


# ---------------- prediction & self metrics (TEST) ----------------

def _predict_best_per_class(model: YOLO, img_path: Path, imgsz: int, device: str, class_ids=(0, 1)) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Run inference on a single image and return the highest-confidence xyxy per requested class.
    Confidence threshold is effectively disabled (very low) to ensure we *always* pick one if present.
    """
    res_list = model.predict(
        source=str(img_path),
        imgsz=imgsz,
        device=device,
        conf=1e-4,
        iou=0.7,
        max_det=50,
        verbose=False
    )
    if not res_list:
        return {}

    res = res_list[0]
    if res.boxes is None or len(res.boxes) == 0:
        return {}

    cls_np = res.boxes.cls.detach().cpu().numpy().astype(int)
    conf_np = res.boxes.conf.detach().cpu().numpy()
    xyxy_np = res.boxes.xyxy.detach().cpu().numpy().astype(float)

    best: Dict[int, Tuple[int, int, int, int]] = {}
    for c in class_ids:
        idxs = np.where(cls_np == c)[0]
        if idxs.size == 0:
            continue
        # pick highest confidence
        j = int(idxs[np.argmax(conf_np[idxs])])
        x1, y1, x2, y2 = xyxy_np[j]
        best[c] = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
    return best


def _rect_dice_for_pair(H: int, W: int, gt_xyxy: Tuple[int, int, int, int], pr_xyxy: Tuple[int, int, int, int]) -> float:
    gx1, gy1, gx2, gy2 = gt_xyxy
    px1, py1, px2, py2 = pr_xyxy
    gxywh = _xyxy_to_xywh(gx1, gy1, gx2, gy2)
    pxywh = _xyxy_to_xywh(px1, py1, px2, py2)
    gm = _rect_to_mask(H, W, gxywh)
    pm = _rect_to_mask(H, W, pxywh)
    return _dice(gm, pm)


def _compute_test_metrics_self(ckpt: Path, data_yaml: Path, imgsz: int, device: str) -> Dict[str, Any]:
    """
    Compute per-class and overall TEST metrics by running inference ourselves and comparing
    to YOLO GT labels:
      - box_error = 1 - IoU
      - dice      = rectangle Dice between GT and predicted boxes
    Returns a dict with per-class means and counts.
    """
    # Gather test images
    test_images = _gather_split_images(data_yaml, "test")
    if not test_images:
        return {"error": "No 'test' split found or it is empty."}

    model = YOLO(str(ckpt))

    # Accumulators
    per_class_iou: Dict[int, List[float]] = {0: [], 1: []}
    per_class_dice: Dict[int, List[float]] = {0: [], 1: []}
    matched_counts: Dict[int, int] = {0: 0, 1: 0}

    for img_path in test_images:
        # GT
        gt = _read_gt_xyxy_for_image(img_path, classes=(0, 1))
        if not gt:
            continue

        # W,H for Dice masks
        with PILImage.open(str(img_path)) as im:
            W, H = im.size

        # Prediction (best box per class)
        pr = _predict_best_per_class(model, img_path, imgsz=imgsz, device=device, class_ids=(0, 1))

        for c in (0, 1):
            if c in gt and c in pr:
                iou = _iou_xyxy(gt[c], pr[c])
                dice = _rect_dice_for_pair(H, W, gt[c], pr[c])
                per_class_iou[c].append(float(iou))
                per_class_dice[c].append(float(dice))
                matched_counts[c] += 1

    # Aggregate
    def _mean(x: List[float]) -> Optional[float]:
        return float(np.mean(x)) if x else None

    out: Dict[str, Any] = {}
    for c, name in [(0, "disc"), (1, "cup")]:
        miou = _mean(per_class_iou[c])
        mdice = _mean(per_class_dice[c])
        out[f"{name}_n"] = matched_counts[c]
        out[f"{name}_box_error"] = (1.0 - miou) if (miou is not None) else None
        out[f"{name}_dice"] = mdice

    # Overall macro (average over classes that have data)
    box_errs = [out.get("disc_box_error"), out.get("cup_box_error")]
    dices = [out.get("disc_dice"), out.get("cup_dice")]
    box_errs = [v for v in box_errs if v is not None]
    dices = [v for v in dices if v is not None]
    out["macro_box_error"] = float(np.mean(box_errs)) if box_errs else None
    out["macro_dice"] = float(np.mean(dices)) if dices else None
    out["tested_images"] = len(test_images)
    return out


# ---------------- training / eval orchestration ----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train/evaluate Ultralytics multiclass (disc/cup) from a config YAML")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--no-filter-both", action="store_true",
                    help="Disable filtering of training images to those containing BOTH classes.")

    # Eval-only
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training and run evaluation only (self-computed TEST metrics).")
    ap.add_argument("--eval-ckpt", default="",
                    help="Path to a .pt checkpoint. If omitted, auto-picks runs/<name>/weights/{best,last}.pt")

    return ap.parse_args()


def _pick_ckpt_for_eval(runs_root: Path, name: str, explicit: str) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        return p if p.exists() else None
    best = runs_root / name / "weights" / "best.pt"
    last = runs_root / name / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        return last
    return None


def main() -> None:
    args = parse_args()
    cfg_path = Path(str(args.config)).expanduser().resolve()
    train_cfg = TrainCfg.from_yaml(cfg_path)

    # ---------- EVAL-ONLY ----------
    if args.eval_only:
        ckpt_path = _pick_ckpt_for_eval(train_cfg.runs_root, train_cfg.name, args.eval_ckpt)
        if ckpt_path is None:
            raise SystemExit("[ERR] --eval-only requested but no checkpoint found (and no --eval-ckpt provided).")

        # Report dataset split sizes
        try:
            dy = yaml.safe_load(train_cfg.data.read_text()) or {}
            dy_dir = train_cfg.data.parent
            n_train = _count_images("train", dy, dy_dir)
            n_val = _count_images("val", dy, dy_dir)
            n_test = _count_images("test", dy, dy_dir)
            print(f"[DATA] (eval-only) train={n_train} | val={n_val} | test={n_test}")
        except Exception as e:
            print(f"[WARN] Could not summarize splits: {e}")

        # Compute TEST metrics ourselves
        test_summary = _compute_test_metrics_self(
            ckpt=ckpt_path,
            data_yaml=train_cfg.data,
            imgsz=train_cfg.imgsz,
            device=train_cfg.device,
        )
        final = {
            "mode": "eval-only",
            "checkpoint": str(ckpt_path),
            "run_name": train_cfg.name,
            "test_metrics": test_summary,
        }
        print("===== FINAL SUMMARY =====")
        print(json.dumps(final, indent=2))
        return

    # ---------- TRAIN (with optional filtered train split) ----------
    data_yaml_for_training = train_cfg.data
    work_run_root = train_cfg.runs_root / train_cfg.name
    work_run_root.mkdir(parents=True, exist_ok=True)

    apply_filter = train_cfg.require_both_train and (not args.no_filter_both)
    if apply_filter:
        try:
            filtered_yaml = _filter_train_to_both(train_cfg.data, work_run_root)
            data_yaml_for_training = filtered_yaml
        except Exception as e:
            print(f"[FILTER] Failed to build filtered train list ({e}). Proceeding with original data.yaml.")

    # Resolve resume/init
    resume_flag = False
    resume_ckpt: Optional[Path] = None
    if train_cfg.resume:
        if str(train_cfg.resume).lower() == "auto":
            cand = train_cfg.runs_root / train_cfg.name / "weights" / "last.pt"
            if cand.exists():
                resume_ckpt = cand
                resume_flag = True
                print(f"[RESUME] auto → {resume_ckpt}")
            else:
                print(f"[RESUME] auto requested but not found: {cand} → starting fresh.")
        else:
            cand = Path(str(train_cfg.resume)).expanduser().resolve()
            if cand.exists():
                resume_ckpt = cand
                resume_flag = True
                print(f"[RESUME] explicit → {resume_ckpt}")
            else:
                print(f"[RESUME] explicit path not found: {cand} → starting fresh.")

    if resume_flag and resume_ckpt is not None:
        trained_done, planned_in_ckpt = _load_ckpt_epochs(resume_ckpt)
        print(f"[RESUME] checkpoint epochs: done={trained_done} planned={planned_in_ckpt}")

        if train_cfg.extend_epochs is not None:
            target_epochs = trained_done + int(train_cfg.extend_epochs)
            print(f"[RESUME] extend_epochs={train_cfg.extend_epochs} → target_epochs={target_epochs}")
        else:
            target_epochs = int(train_cfg.epochs)
            if target_epochs <= trained_done:
                raise SystemExit(
                    f"[ERR] YAML epochs={target_epochs} ≤ completed={trained_done}. "
                    f"Increase 'epochs' or set 'extend_epochs: N'."
                )

        model = YOLO(str(resume_ckpt))
        is_resume = True
        effective_epochs = target_epochs
    else:
        weights = _resolve_weights(train_cfg.family, train_cfg.size, train_cfg.weights)
        model = YOLO(weights)
        print(f"[INIT] starting from weights: {weights}")
        is_resume = False
        effective_epochs = int(train_cfg.epochs)

    print(f"[CFG] {cfg_path}")
    print(f"[INFO] data={data_yaml_for_training} | device={train_cfg.device} | "
          f"epochs={effective_epochs} | imgsz={train_cfg.imgsz} | batch={train_cfg.batch}")

    try:
        dy = yaml.safe_load(Path(data_yaml_for_training).read_text()) or {}
        dy_dir = Path(data_yaml_for_training).parent
        n_train = _count_images("train", dy, dy_dir)
        n_val = _count_images("val", dy, dy_dir)
        n_test = _count_images("test", dy, dy_dir)
        print(f"[DATA] train={n_train} | val={n_val} | test={n_test}")
    except Exception as e:
        print(f"[WARN] Could not summarize splits: {e}")

    overrides: Dict[str, Any] = dict(
        data=str(data_yaml_for_training),
        epochs=effective_epochs,
        imgsz=train_cfg.imgsz,
        batch=train_cfg.batch,
        device=train_cfg.device,
        project=str(train_cfg.runs_root),
        name=train_cfg.name,
        workers=train_cfg.workers,
        seed=train_cfg.seed,
        single_cls=False,
        pretrained=True,
        optimizer=train_cfg.optimizer,
        cos_lr=train_cfg.cos_lr,
        patience=train_cfg.patience,
        amp=train_cfg.amp,
        freeze=train_cfg.freeze,
        multi_scale=train_cfg.multi_scale,
        close_mosaic=train_cfg.close_mosaic,
        hsv_h=train_cfg.aug.hsv_h, hsv_s=train_cfg.aug.hsv_s, hsv_v=train_cfg.aug.hsv_v,
        degrees=train_cfg.aug.degrees, translate=train_cfg.aug.translate, scale=train_cfg.aug.scale,
        shear=train_cfg.aug.shear, perspective=train_cfg.aug.perspective,
        flipud=train_cfg.aug.flipud, fliplr=train_cfg.aug.fliplr,
        mosaic=train_cfg.aug.mosaic, mixup=train_cfg.aug.mixup,
        copy_paste=train_cfg.aug.copy_paste, erasing=train_cfg.aug.erasing,
        resume=is_resume,
    )

    model.train(**overrides)
    print("[OK] Training complete.")

    # ---------- After training: compute TEST metrics ourselves ----------
    # Choose best or last
    best_ckpt = train_cfg.runs_root / train_cfg.name / "weights" / "best.pt"
    last_ckpt = train_cfg.runs_root / train_cfg.name / "weights" / "last.pt"
    ckpt_path = best_ckpt if best_ckpt.exists() else (last_ckpt if last_ckpt.exists() else None)
    if ckpt_path is None:
        print("[WARN] No checkpoint found for evaluation.")
        return

    # Use the same YAML we trained with (it still points val/test to absolute paths if filtered)
    test_summary = _compute_test_metrics_self(
        ckpt=ckpt_path,
        data_yaml=Path(data_yaml_for_training),
        imgsz=train_cfg.imgsz,
        device=train_cfg.device,
    )
    final = {
        "mode": "train-then-eval",
        "checkpoint": str(ckpt_path),
        "run_name": train_cfg.name,
        "test_metrics": test_summary,
    }
    print("===== FINAL SUMMARY =====")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    sys.exit(main())