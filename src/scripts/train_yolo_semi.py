#!/usr/bin/env python3
# File: src/scripts/train_yolo_semi.py
"""
Simplified Semi-Supervised YOLO Trainer.
Aligns with experiment.py by using the centralized YOLORunner class.

Workflow:
1. (Optional) Reads Parquet and Materializes YOLO dataset.
2. Instantiates YOLORunner (enforcing seeds, configs, and freeze settings).
3. Fine-tunes YOLO model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import yaml

from src.imgpipe.image import Image
from src.model.yolo import YOLORunner, set_global_seed
from src.utils import ultralytics_device_arg

SEED_DEFAULT = 42


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
            continue

        ext = (img.image_path.suffix or "").lower()
        if not ext:
            ext = getattr(img.image_ref, "ext", None) or ".png"
        if not ext.startswith("."):
            ext = "." + ext

        dst_img = out_dir / "images" / split / f"{img.uid}{ext}"
        dst_lbl = out_dir / "labels" / split / f"{img.uid}.txt"

        materialize_image_file(img, dst_img)

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO from pre-processed semi-supervised parquet (using YOLORunner).")

    # Dataset / IO
    p.add_argument("--images-parquet", type=Path, required=True, help="Ready-to-go parquet (splits & GT set).")
    p.add_argument("--out-yolo-ds", type=Path, required=True, help="Where to materialize dataset.")
    p.add_argument("--runs-root", type=Path, required=True, help="Where to save training runs.")

    # Training / Model
    p.add_argument("--init-weights", type=Path, required=True, help="Pretrained weights to fine-tune.")
    p.add_argument("--cfg", type=Path, default=Path("src/configs/train_custom.yaml"), help="YOLO training config file.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default=None, help='Device e.g. "0,1,2,3". Auto-detected if None.')
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)

    # Flags
    p.add_argument("--skip-materialization", action="store_true", help="Use existing YOLO dataset if found.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Set global seed (also done inside YOLORunner, but good to do early)
    set_global_seed(args.seed)

    target_device = args.device if args.device is not None else ultralytics_device_arg()

    # 1. Prepare Dataset
    data_yaml = args.out_yolo_ds / "data.yaml"

    if args.skip_materialization and data_yaml.exists():
        print(f"[INFO] Skipping materialization. Using existing dataset: {data_yaml}")
    else:
        if args.skip_materialization:
            print(f"[WARN] --skip-materialization set but {data_yaml} not found. Proceeding with creation.")

        print(f"[INFO] Loading Parquet: {args.images_parquet}")
        images = Image.load_parquet(args.images_parquet)

        data_yaml = build_yolo_dataset_from_processed_parquet(
            images,
            out_dir=args.out_yolo_ds
        )

    # 2. Determine Run Name
    # We calculate a unique name here (e.g., 'ss1', 'ss2') to keep runs organized
    args.runs_root.mkdir(parents=True, exist_ok=True)
    run_name = get_next_run_name(args.runs_root, base="ss")

    print(f"[INFO] Initializing YOLORunner for run: {run_name}")
    print(f"[INFO] Fine-tuning from: {args.init_weights}")

    # 3. Instantiate YOLORunner
    # This class ensures we use the exact same training logic (freeze=5, amp=False, etc.) as experiment.py
    runner = YOLORunner(
        data_root=args.out_yolo_ds,  # Required by init but unused when yolo_ds is passed
        out_dir=args.out_yolo_ds,  # Dataset location
        run_dir=args.runs_root,  # Base runs folder
        run_name=run_name,  # Unique run name (ss1, ss2...)
        cfg=args.cfg,  # Config file (e.g., src/configs/train_custom.yaml)
        model=str(args.init_weights),  # Weights to load/fine-tune
        device=target_device,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        # These are test-time defaults, irrelevant for training but required by __init__
        conf=0.001,
        iou=0.7,
        papila=0.0,
        seed=args.seed,
        yolo_ds=args.out_yolo_ds  # Explicitly point to the DS we just prepared
    )

    # 4. Run Training
    # YOLORunner.train() handles:
    # - Creating the run directory
    # - Calling model.train(..., freeze=5, amp=False, resume=False)
    best_weights = runner.train()

    print(f"[OK] Training finished.")
    if best_weights:
        print(f"[OK] Best weights saved at: {best_weights}")
    else:
        print("[WARN] Could not resolve path to best weights.")


if __name__ == "__main__":
    main()