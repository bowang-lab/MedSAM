#!/usr/bin/env python3
# src/model/train_multiclass_cfg.py
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
from ultralytics import YOLO

from src.utils import ultralytics_device_arg, expand, need  # removed load_cfg import


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
    Given a key like 'train'/'val'/'test' from data.yaml, return #image files.
    Supports common YOLO layout with optional 'path:' base.
    """
    split_entry = data_yaml.get(path_key)
    if not split_entry:
        return 0

    split_path = Path(str(split_entry))
    # Only expand base if `path` exists in YAML; otherwise use the yaml's directory
    base = expand(data_yaml["path"]) if data_yaml.get("path") else data_yaml_dir

    # If split_path is relative, join to 'path' (or data_yaml_dir as fallback)
    if not split_path.is_absolute():
        split_path = (base / split_path).resolve()
    if not split_path.exists():
        return 0

    # Typical is images/<split>, but users may point directly at folder of images
    if split_path.is_dir():
        return sum(1 for p in split_path.rglob("*") if p.suffix.lower() in IMG_EXTS)

    # If it's a text file list (rare), could add logic here in future.
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
    name: str
    epochs: int
    imgsz: int
    batch: int
    workers: int
    seed: int
    device: Optional[str]  # "auto" | None means auto; or "0", "0,1", "cpu"
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
        need(path, "config YAML")
        cfg = yaml.safe_load(path.read_text()) or {}

        # minimal validation
        data_yaml = expand(cfg.get("data"))
        if not data_yaml:
            raise SystemExit("[ERR] 'data' (path to data.yaml) is required in the config.")
        need(data_yaml, "data.yaml")

        runs_root = expand(cfg.get("runs_root")) or Path("./runs/detect").resolve()
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

        # "auto" device in YAML → choose at runtime
        dev = cfg.get("device", "auto")
        if dev in (None, "", "auto"):
            dev = ultralytics_device_arg()

        return TrainCfg(
            data=data_yaml,
            runs_root=runs_root,
            weights=cfg.get("weights"),     # may be None → resolve by family/size
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
            aug=aug_cfg,
        )


# ---------------- train ----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train Ultralytics multiclass (disc/cup) from a config YAML"
    )
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = expand(args.config)  # <-- pass a Path, not a dict
    train_cfg = TrainCfg.from_yaml(cfg_path)

    weights = _resolve_weights(train_cfg.family, train_cfg.size, train_cfg.weights)

    print(f"[CFG] {cfg_path}")
    print(f"[INFO] data={train_cfg.data} | weights={weights} | device={train_cfg.device} | "
          f"epochs={train_cfg.epochs} | imgsz={train_cfg.imgsz} | batch={train_cfg.batch}")
    print(f"[AUG ] scale={train_cfg.aug.scale} translate={train_cfg.aug.translate} "
          f"mosaic={train_cfg.aug.mosaic} mixup={train_cfg.aug.mixup} erasing={train_cfg.aug.erasing}")
    print(f"[RUN ] project={train_cfg.runs_root} name={train_cfg.name}")

    # (Optional) show split counts for quick sanity
    try:
        dy = yaml.safe_load(train_cfg.data.read_text()) or {}
        dy_dir = train_cfg.data.parent
        n_train = _count_images("train", dy, dy_dir)
        n_val = _count_images("val", dy, dy_dir)
        n_test = _count_images("test", dy, dy_dir)
        print(f"[DATA] train={n_train} | val={n_val} | test={n_test}")
    except Exception as e:
        print(f"[WARN] Could not summarize splits: {e}")

    model = YOLO(weights)

    overrides: Dict[str, Any] = dict(
        # core
        data=str(train_cfg.data),
        epochs=train_cfg.epochs,
        imgsz=train_cfg.imgsz,
        batch=train_cfg.batch,
        device=train_cfg.device,                     # e.g. "0,1,2,3" → DDP
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

        # scheduling helpers
        multi_scale=train_cfg.multi_scale,
        close_mosaic=train_cfg.close_mosaic,

        # color
        hsv_h=train_cfg.aug.hsv_h,
        hsv_s=train_cfg.aug.hsv_s,
        hsv_v=train_cfg.aug.hsv_v,

        # geometric
        degrees=train_cfg.aug.degrees,
        translate=train_cfg.aug.translate,
        scale=train_cfg.aug.scale,                   # **key** for ROI crop/zoom robustness
        shear=train_cfg.aug.shear,
        perspective=train_cfg.aug.perspective,

        # flips
        flipud=train_cfg.aug.flipud,
        fliplr=train_cfg.aug.fliplr,

        # sample mixing / occlusion
        mosaic=train_cfg.aug.mosaic,
        mixup=train_cfg.aug.mixup,
        copy_paste=train_cfg.aug.copy_paste,
        erasing=train_cfg.aug.erasing,
    )

    model.train(**overrides)

    print("[OK] Training complete.")

    # --- Evaluate on test split (if present), else fall back to val ---
    # Load data.yaml to see if 'test' exists
    dy = yaml.safe_load(train_cfg.data.read_text()) or {}
    has_test = bool(dy.get("test"))

    # Locate best checkpoint
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
        plots=True  # saves PR curves, confusion matrix, etc. into the run dir
    )

    # Try to print a compact metrics summary (keys vary slightly by version/task)
    try:
        metrics = getattr(val_res, "results_dict", None) or {}
        if metrics:
            print("[EVAL] Metrics:", {k: float(v) for k, v in metrics.items()})
        else:
            print("[EVAL] Finished; see run directory for detailed metrics/plots.")
    except Exception:
        print("[EVAL] Finished; see run directory for detailed metrics/plots.")



if __name__ == "__main__":
    sys.exit(main())