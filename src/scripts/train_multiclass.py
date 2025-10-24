#!/usr/bin/env python3
# src/model/train_multiclass_cfg.py
"""
Train a multiclass (disc=0, cup=1) YOLO detector from a YAML config, with:
- Robust resume logic (supports 'resume: auto' or explicit path, and 'extend_epochs')
- Automatic device selection ("auto" honors YOLO_DEVICES / CUDA_VISIBLE_DEVICES)
- Strong augment defaults (configurable)
- Always evaluates on test (falls back to val) and saves plots + predictions.json
- Optional rectangle-based Dice proxy computed from predictions/labels JSON

YAML keys (minimal):
  data: /abs/path/to/data.yaml
  runs_root: /abs/path/to/runs/detect
  name: "multiclass_disc_cup"
  epochs: 200
  batch: 16
  imgsz: 640
  resume: "auto" | "/abs/path/to/last.pt" | "" (fresh)
  extend_epochs: 100   # (optional) if resuming, adds N to completed epochs
  optimizer: "AdamW"
  patience: 50
  multi_scale: true
  close_mosaic: 10
  augment: { ... }     # see AugCfg below for full list

Notes:
- If you finished 200 epochs and want to continue to 300:
    either set epochs: 300 with resume: auto
    or set extend_epochs: 100 with resume: auto
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from src.utils import ultralytics_device_arg  # keep only what we use


# ------------------ checkpoint helpers ------------------

def _load_ckpt_epochs(ckpt_path: Path) -> tuple[int, Optional[int]]:
    """
    Returns (trained_epochs_done, target_epochs_in_ckpt_or_None).
    Ultralytics checkpoints usually store:
      - 'epoch' (0-based index of last completed epoch)
      - 'train_args' or 'args' with 'epochs' total planned
    """
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
    If 'labels.json' is absent, returns {} and does nothing.
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
    return 0


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
    resume: Optional[str]            # "", "auto", or path to last.pt
    extend_epochs: Optional[int]     # if set and resuming: total_epochs = trained + extend_epochs
    name: str
    epochs: int
    imgsz: int
    batch: int
    workers: int
    seed: int
    device: Optional[str]            # "auto" | None means auto; or "0", "0,1", "cpu"
    amp: bool
    freeze: int
    optimizer: str
    cos_lr: bool
    patience: int
    multi_scale: bool
    close_mosaic: int
    # augmentations
    aug: AugCfg

    @staticmethod
    def from_yaml(path: Path) -> "TrainCfg":
        # read & validate
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

        # device: "auto" → pick at runtime
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
            aug=aug_cfg,
        )


# ---------------- train & evaluate ----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train Ultralytics multiclass (disc/cup) from a config YAML")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(str(args.config)).expanduser().resolve()
    train_cfg = TrainCfg.from_yaml(cfg_path)

    # -------- Resolve resume / model init --------
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
                print(f"[RESUME] auto requested but not found: {cand}  → starting fresh.")
        else:
            cand = Path(str(train_cfg.resume)).expanduser().resolve()
            if cand.exists():
                resume_ckpt = cand
                resume_flag = True
                print(f"[RESUME] explicit → {resume_ckpt}")
            else:
                print(f"[RESUME] explicit path not found: {cand}  → starting fresh.")

    # If resuming, load model from the checkpoint; else from chosen base weights
    if resume_flag and resume_ckpt is not None:
        # Check how many epochs have completed
        trained_done, planned_in_ckpt = _load_ckpt_epochs(resume_ckpt)
        print(f"[RESUME] checkpoint epochs: done={trained_done} planned={planned_in_ckpt}")

        # Decide final target epochs
        if train_cfg.extend_epochs is not None:
            target_epochs = trained_done + int(train_cfg.extend_epochs)
            print(f"[RESUME] extend_epochs={train_cfg.extend_epochs} → target_epochs={target_epochs}")
        else:
            target_epochs = int(train_cfg.epochs)
            if target_epochs <= trained_done:
                raise SystemExit(
                    f"[ERR] Resume requested but YAML epochs={target_epochs} ≤ completed={trained_done}. "
                    f"Increase 'epochs' to > {trained_done} OR set 'extend_epochs: N' to continue."
                )

        model = YOLO(str(resume_ckpt))
        is_resume = True
        effective_epochs = target_epochs
    else:
        weights = _resolve_weights(train_cfg.family, train_cfg.size, train_cfg.weights)
        model = YOLO(weights)
        print(f"[INIT ] starting from weights: {weights}")
        is_resume = False
        effective_epochs = int(train_cfg.epochs)

    # -------- Info & split counts --------
    print(f"[CFG] {cfg_path}")
    print(f"[INFO] data={train_cfg.data} | device={train_cfg.device} | "
          f"epochs={effective_epochs} | imgsz={train_cfg.imgsz} | batch={train_cfg.batch}")
    print(f"[AUG ] scale={train_cfg.aug.scale} translate={train_cfg.aug.translate} "
          f"mosaic={train_cfg.aug.mosaic} mixup={train_cfg.aug.mixup} erasing={train_cfg.aug.erasing}")
    print(f"[RUN ] project={train_cfg.runs_root} name={train_cfg.name}")

    try:
        dy = yaml.safe_load(train_cfg.data.read_text()) or {}
        dy_dir = train_cfg.data.parent
        n_train = _count_images("train", dy, dy_dir)
        n_val = _count_images("val", dy, dy_dir)
        n_test = _count_images("test", dy, dy_dir)
        print(f"[DATA] train={n_train} | val={n_val} | test={n_test}")
    except Exception as e:
        print(f"[WARN] Could not summarize splits: {e}")

    # -------- Train --------
    overrides: Dict[str, Any] = dict(
        data=str(train_cfg.data),
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
        resume=is_resume,  # boolean only; model was constructed from ckpt if resuming
    )

    model.train(**overrides)
    print("[OK] Training complete.")

    # -------- Evaluate on test (fallback to val) --------
    dy = yaml.safe_load(train_cfg.data.read_text()) or {}
    has_test = bool(dy.get("test"))

    best_ckpt = train_cfg.runs_root / train_cfg.name / "weights" / "best.pt"
    last_ckpt = train_cfg.runs_root / train_cfg.name / "weights" / "last.pt"
    ckpt_path = best_ckpt if best_ckpt.exists() else (last_ckpt if last_ckpt.exists() else None)
    if ckpt_path is None:
        print("[WARN] No checkpoint found for test evaluation.")
        return

    tester = YOLO(str(ckpt_path))
    split = "test" if has_test else "val"
    print(f"[EVAL] Evaluating checkpoint on split='{split}' …")
    val_res = tester.val(
        data=str(train_cfg.data),
        split=split,
        imgsz=train_cfg.imgsz,
        device=train_cfg.device,
        plots=True,       # PR curves, confusion, etc.
        save_json=True,   # writes predictions.json (+ labels.json if available)
    )

    # Rectangle-Dice (proxy). Writes dice_rect_summary.json if GT JSON present.
    save_dir = Path(getattr(val_res, "save_dir", train_cfg.runs_root / train_cfg.name / "eval_tmp"))
    try:
        rect_dice = compute_rect_dice_from_predictions(save_dir)
        if rect_dice:
            print("[EVAL] Rectangle-Dice (per class):", rect_dice)
    except Exception as e:
        print(f"[WARN] Dice metric computation failed: {e}")

    # Print core losses/metrics robustly
    try:
        metrics = getattr(val_res, "results_dict", {}) or {}

        def pick(d: dict, keys: tuple[str, ...]):
            for k in keys:
                if k in d:
                    return d[k]
            return None

        box_loss = pick(metrics, ("loss/box", "box_loss", "box"))
        cls_loss = pick(metrics, ("loss/cls", "cls_loss", "cls"))
        dfl_loss = pick(metrics, ("loss/dfl", "dfl_loss", "dfl"))
        printable = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        print("[EVAL] key metrics:", printable)
        if box_loss is not None:
            print(f"[EVAL] box_loss={float(box_loss):.4f} | "
                  f"cls_loss={float(cls_loss or 0):.4f} | "
                  f"dfl_loss={float(dfl_loss or 0):.4f}")
    except Exception:
        print("[EVAL] Finished; see run directory for detailed metrics/plots.")


if __name__ == "__main__":
    sys.exit(main())