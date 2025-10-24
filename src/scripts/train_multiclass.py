#!/usr/bin/env python3
# src/model/train_multiclass_cfg.py
"""
Multiclass YOLO (disc=0, cup=1) trainer/evaluator with:
- Optional train-split filtering to only keep images that contain BOTH classes
- Robust resume logic (auto/explicit) and epoch extension
- Eval-only mode across chosen splits (val/test/train) with summary
- Rectangular Dice (proxy) computed from predictions/labels JSON
- FIX: when creating a filtered YAML, keep val/test usable by writing ABSOLUTE paths
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List

import numpy as np
import torch
import yaml
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


def compute_rect_dice_from_predictions(eval_dir: Path) -> dict:
    """
    Compute per-class Dice between GT rectangles and predicted rectangles.
    Requires Ultralytics 'predictions.json' and 'labels.json' (if available).
    Saves 'dice_rect_summary.json' in eval_dir if computed.
    """
    pred_json = eval_dir / "predictions.json"
    labels_json = eval_dir / "labels.json"  # not always present
    if not pred_json.exists() or not labels_json.exists():
        return {}

    preds = json.loads(pred_json.read_text())
    gts = json.loads(labels_json.read_text())

    by_img_pred, by_img_gt = {}, {}
    for p in preds:
        by_img_pred.setdefault(p["image_id"], []).append(p)
    for g in gts:
        by_img_gt.setdefault(g["image_id"], []).append(g)

    per_class: Dict[int, list[float]] = {}
    for img_id, gboxes in by_img_gt.items():
        H = gboxes[0].get("height")
        W = gboxes[0].get("width")
        if H is None or W is None:
            continue
        pboxes = by_img_pred.get(img_id, [])
        used = set()

        for g in gboxes:
            gc = g["category_id"]
            gb = g["bbox"]  # xywh
            gmask = _rect_to_mask(H, W, gb)
            best_dice = float("nan")
            best_j = -1
            for j, pr in enumerate(pboxes):
                if j in used or pr["category_id"] != gc:
                    continue
                pmask = _rect_to_mask(H, W, pr["bbox"])
                d = _dice(gmask, pmask)
                if not np.isnan(d) and (np.isnan(best_dice) or d > best_dice):
                    best_dice, best_j = d, j
            if best_j >= 0:
                used.add(best_j)
                per_class.setdefault(gc, []).append(float(best_dice))

    out = {str(k): float(np.nanmean(v)) for k, v in per_class.items() if v}
    (eval_dir / "dice_rect_summary.json").write_text(json.dumps(out, indent=2))
    return out


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
    CRITICAL FIX: write absolute paths for 'path', 'val', and 'test' so evaluation doesn't break.
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


# ---------------- metrics helpers ----------------

def _extract_metrics(results_obj) -> dict:
    """Return a float-only dict from Ultralytics results."""
    d = getattr(results_obj, "results_dict", {}) or {}
    out = {}
    for k, v in d.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    # Also alias common keys if present
    if "metrics/mAP50-95(B)" in out:
        out["map50-95"] = out["metrics/mAP50-95(B)"]
    if "metrics/mAP50(B)" in out:
        out["map50"] = out["metrics/mAP50(B)"]
    return out


def _min_train_box_loss_from_results_csv(run_dir: Path) -> tuple[Optional[float], Optional[int]]:
    """
    Parse Ultralytics results.csv in a run dir to get the min train box loss and its epoch.
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None, None
    try:
        rows = list(csv.DictReader(csv_path.open()))
        best_val = None
        best_ep = None
        # Try a few column name variants
        col_candidates = [
            "train/box_loss", "box_loss", "loss/box", "giou_loss", "loss/giou"
        ]
        col = None
        if rows:
            cols = rows[0].keys()
            for c in col_candidates:
                if c in cols:
                    col = c
                    break
        if not col:
            return None, None
        for r in rows:
            try:
                v = float(r[col])
                ep = int(r.get("epoch", r.get("Epoch", -1)))
            except Exception:
                continue
            if (best_val is None) or (v < best_val):
                best_val = v
                best_ep = ep
        return best_val, best_ep
    except Exception:
        return None, None


# ---------------- train & evaluate ----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train/evaluate Ultralytics multiclass (disc/cup) from a config YAML")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--no-filter-both", action="store_true",
                    help="Disable filtering of training images to those containing BOTH classes.")

    # Eval-only
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training and run evaluation only.")
    ap.add_argument("--eval-ckpt", default="",
                    help="Path to a .pt checkpoint. If omitted, auto-picks runs/<name>/weights/{best,last}.pt")
    ap.add_argument("--eval-splits", default="val,test",
                    help="Comma list from {train,val,test}. Default: val,test")

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


def _run_one_eval(ckpt: Path, data_yaml: Path, split: str, imgsz: int, device: str) -> tuple[dict, Path, dict]:
    """
    Returns (metrics_dict, save_dir, rect_dice_dict)
    """
    tester = YOLO(str(ckpt))
    res = tester.val(
        data=str(data_yaml),
        split=(split if split in ("train", "val", "test") else None),
        imgsz=imgsz,
        device=device,
        plots=True,
        save_json=True,
    )
    save_dir = Path(getattr(res, "save_dir", tester.model.yaml.get("project", ".")))
    metrics = _extract_metrics(res)
    rect_dice = {}
    try:
        rect_dice = compute_rect_dice_from_predictions(save_dir)
    except Exception:
        pass
    return metrics, save_dir, rect_dice


def main() -> None:
    args = parse_args()
    cfg_path = Path(str(args.config)).expanduser().resolve()
    train_cfg = TrainCfg.from_yaml(cfg_path)

    # ------------- Eval-only -------------
    if args.eval_only:
        ckpt_path = _pick_ckpt_for_eval(train_cfg.runs_root, train_cfg.name, args.eval_ckpt)
        if ckpt_path is None:
            raise SystemExit("[ERR] --eval-only requested but no checkpoint found (and no --eval-ckpt provided).")

        # Use the ORIGINAL dataset YAML for evaluation (no train filtering needed)
        base_yaml = train_cfg.data
        try:
            dy = yaml.safe_load(base_yaml.read_text()) or {}
            dy_dir = base_yaml.parent
            n_train = _count_images("train", dy, dy_dir)
            n_val = _count_images("val", dy, dy_dir)
            n_test = _count_images("test", dy, dy_dir)
            print(f"[DATA] (eval-only) train={n_train} | val={n_val} | test={n_test}")
        except Exception as e:
            print(f"[WARN] Could not summarize splits: {e}")

        splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
        summary = {
            "mode": "eval-only",
            "checkpoint": str(ckpt_path),
            "run_name": train_cfg.name,
        }

        # Best train box loss (if we can find the run dir next to the ckpt)
        run_dir = ckpt_path.parent.parent  # .../runs/<name>
        best_box_loss, best_epoch = _min_train_box_loss_from_results_csv(run_dir)
        if best_box_loss is not None:
            summary["best_train_box_loss"] = {"value": best_box_loss, "epoch": best_epoch}

        for split in splits:
            print(f"[EVAL] split='{split}' on ckpt: {ckpt_path}")
            metrics, save_dir, rect_dice = _run_one_eval(
                ckpt=ckpt_path,
                data_yaml=base_yaml,
                split=split,
                imgsz=train_cfg.imgsz,
                device=train_cfg.device,
            )
            summary[f"{split}_metrics"] = metrics
            if rect_dice:
                summary[f"{split}_rect_dice"] = rect_dice
            summary[f"{split}_artifacts_dir"] = str(save_dir)

        print("===== FINAL SUMMARY =====")
        print(json.dumps(summary, indent=2))
        return

    # ------------- Train (with optional filter) -------------
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

    # -------- Evaluate best/last on test (fallback val), plus optional train (for train Rect-Dice) --------
    dy = yaml.safe_load(Path(data_yaml_for_training).read_text()) or {}
    has_test = bool(dy.get("test"))

    best_ckpt = train_cfg.runs_root / train_cfg.name / "weights" / "best.pt"
    last_ckpt = train_cfg.runs_root / train_cfg.name / "weights" / "last.pt"
    ckpt_path = best_ckpt if best_ckpt.exists() else (last_ckpt if last_ckpt.exists() else None)
    if ckpt_path is None:
        print("[WARN] No checkpoint found for evaluation.")
        return

    summary = {
        "mode": "train-then-eval",
        "checkpoint": str(ckpt_path),
        "run_name": train_cfg.name,
    }

    # Best train box loss
    run_dir = ckpt_path.parent.parent
    best_box_loss, best_epoch = _min_train_box_loss_from_results_csv(run_dir)
    if best_box_loss is not None:
        summary["best_train_box_loss"] = {"value": best_box_loss, "epoch": best_epoch}

    # Evaluate on 'test' if present else 'val'
    split = "test" if has_test else "val"
    print(f"[EVAL] Evaluating on split='{split}' …")
    metrics, save_dir, rect_dice = _run_one_eval(
        ckpt=ckpt_path,
        data_yaml=Path(data_yaml_for_training),  # val/test still valid due to absolute paths fix
        split=split,
        imgsz=train_cfg.imgsz,
        device=train_cfg.device,
    )
    summary[f"{split}_metrics"] = metrics
    if rect_dice:
        summary[f"{split}_rect_dice"] = rect_dice
    summary[f"{split}_artifacts_dir"] = str(save_dir)

    # Also evaluate on 'val' if test already used (so you get both)
    if split == "test" and dy.get("val"):
        vm, vsd, vrd = _run_one_eval(
            ckpt=ckpt_path,
            data_yaml=Path(data_yaml_for_training),
            split="val",
            imgsz=train_cfg.imgsz,
            device=train_cfg.device,
        )
        summary["val_metrics"] = vm
        if vrd:
            summary["val_rect_dice"] = vrd
        summary["val_artifacts_dir"] = str(vsd)

    # Optional: evaluate on train to compute train Rect-Dice (best ckpt on train split)
    if dy.get("train"):
        tm, tsd, trd = _run_one_eval(
            ckpt=ckpt_path,
            data_yaml=Path(data_yaml_for_training),
            split="train",
            imgsz=train_cfg.imgsz,
            device=train_cfg.device,
        )
        summary["train_metrics"] = tm
        if trd:
            summary["train_rect_dice"] = trd
        summary["train_artifacts_dir"] = str(tsd)

    print("===== FINAL SUMMARY =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())