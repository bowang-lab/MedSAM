#!/usr/bin/env python3
# File: src/scripts/train_yolo_semi.py
"""
Simplified Semi-Supervised YOLO Trainer.
Assumes input parquet is fully prepared (GTs valid, splits assigned).
1. Reads Parquet.
2. Materializes YOLO dataset (images/labels) based on 'split' column.
3. Fine-tunes YOLO model.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Optional

import yaml
from ultralytics import YOLO

from src.imgpipe.image import Image

SEED_DEFAULT = 42


def set_global_seed(seed: int) -> None:
    random.seed(seed)


def get_next_run_name(run_root: Path, base: str = "ss") -> str:
    i = 1
    while (run_root / f"{base}{i}").exists():
        i += 1
    return f"{base}{i}"


def ensure_dirs(ds_root: Path) -> None:
    for sub in (
            "images/train", "images/val", "images/test",
            "labels/train", "labels/val", "labels/test",
    ):
        (ds_root / sub).mkdir(parents=True, exist_ok=True)


def write_data_yaml(ds_root: Path) -> Path:
    data = {
        "path": str(ds_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "disc", 1: "cup"},
    }
    p = ds_root / "data.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


def fmt_yolo_line(cls_id: int, xc: float, yc: float, w: float, h: float) -> str:
    xc = float(min(1.0, max(0.0, xc)))
    yc = float(min(1.0, max(0.0, yc)))
    w = float(min(1.0, max(0.0, w)))
    h = float(min(1.0, max(0.0, h)))
    return f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def materialize_image_file(img: Image, dst_img_path: Path) -> None:
    dst_img_path.parent.mkdir(parents=True, exist_ok=True)
    ref = img.image_ref
    if ref is not None and getattr(ref, "packed", None) is not None:
        dst_img_path.write_bytes(ref.packed)
        return
    src = Path(img.image_path)
    if not src.exists():
        # Skip if source file missing (shouldn't happen if pipeline is robust)
        return
    shutil.copy2(src, dst_img_path)


def build_yolo_dataset_from_processed_parquet(
        images: list[Image],
        out_dir: Path,
) -> Path:
    """
    Materialize dataset assuming `img.split` and `gt_*_box` are already set.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_dirs(out_dir)

    counts = {"train": 0, "val": 0, "test": 0}

    print(f"[INFO] Materializing {len(images)} images to {out_dir}...")

    for img in images:
        split = img.split
        if split not in counts:
            continue  # Skip if split is None or invalid

        # Determine extension
        ext = (img.image_path.suffix or "").lower()
        if not ext:
            ext = getattr(img.image_ref, "ext", None) or ".png"
        if not ext.startswith("."):
            ext = "." + ext

        # Paths
        dst_img = out_dir / "images" / split / f"{img.uid}{ext}"
        dst_lbl = out_dir / "labels" / split / f"{img.uid}.txt"

        # Write Image
        materialize_image_file(img, dst_img)

        # Write Labels (Expects GT boxes to be present)
        lines = []
        if img.gt_disc_box:
            xc, yc, w, h = img.gt_disc_box.as_tuple()
            lines.append(fmt_yolo_line(0, xc, yc, w, h))
        if img.gt_cup_box:
            xc, yc, w, h = img.gt_cup_box.as_tuple()
            lines.append(fmt_yolo_line(1, xc, yc, w, h))

        if lines:
            dst_lbl.write_text("\n".join(lines), encoding="utf-8")
            counts[split] += 1

    data_yaml = write_data_yaml(out_dir)
    print(f"[INFO] Materialization complete. Counts: {json.dumps(counts)}")
    return data_yaml


def run_train(
        *,
        data_yaml: Path,
        runs_root: Path,
        init_weights: Path,
        device: str,
        epochs: int,
        batch: int,
        imgsz: int,
        workers: int,
        freeze: int = 0,
) -> Path:
    runs_root.mkdir(parents=True, exist_ok=True)
    run_name = get_next_run_name(runs_root, base="ss")
    run_root = runs_root / run_name
    run_root.mkdir(parents=True, exist_ok=False)

    print(f"[INFO] Loading weights for fine-tuning: {init_weights}")
    model = YOLO(str(init_weights))

    print(f"[INFO] Starting training... Output: {run_root}")
    model.train(
        data=str(data_yaml),
        project=str(runs_root),
        name=run_name,
        device=device,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        workers=workers,
        exist_ok=False,
        resume=False,
        freeze=freeze,
    )

    weights_dir = run_root / "weights"
    best_pt = weights_dir / "best.pt"
    return best_pt if best_pt.exists() else weights_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO from pre-processed semi-supervised parquet.")
    p.add_argument("--images-parquet", type=Path, required=True, help="Ready-to-go parquet (splits & GT set).")
    p.add_argument("--out-yolo-ds", type=Path, required=True, help="Where to materialize dataset.")
    p.add_argument("--runs-root", type=Path, required=True, help="Where to save training runs.")
    p.add_argument("--init-weights", type=Path, required=True, help="Pretrained weights to finetune.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    print(f"[INFO] Loading Parquet: {args.images_parquet}")
    images = Image.load_parquet(args.images_parquet)

    data_yaml = build_yolo_dataset_from_processed_parquet(
        images,
        out_dir=args.out_yolo_ds
    )

    best_weights = run_train(
        data_yaml=data_yaml,
        runs_root=args.runs_root,
        init_weights=args.init_weights,
        device=args.device,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers
    )
    print(f"[OK] Training finished. Best weights: {best_weights}")


if __name__ == "__main__":
    main()