#!/usr/bin/env python3
# train_stageA_disc_only.py
"""
Train a disc-only detector from an existing 2-class YOLO dataset (0=disc, 1=cup).

Adds resume support:
  --resume auto      -> uses runs/detect/<name>/weights/last.pt if present
  --resume /path/to/last.pt

Behavior:
- If resume is found, loads that checkpoint and calls model.train(resume=True).
- Otherwise starts from --model weights and trains from scratch.

The rest (building disc-only, validating, etc.) is unchanged.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import yaml
from ultralytics import YOLO

from device_utils import ultralytics_device_arg  # device string ("0" or "cpu")


# ----------------------------- Dataclass -----------------------------

@dataclass
class DiscOnlyConfig:
    # Core paths
    project_dir: Path
    data_root: Path             # Could be base YOLO root OR a precomputed disc-only root
    model_path: Path
    runs_root: Path

    # Augmentation support (for building disc-only train splits from an augmented set)
    aug_root: Optional[Path]    # Optional augmented YOLO dataset (for training splits only)
    train_splits: List[str]     # Which splits to source from aug_root (e.g., ["train"])

    # Data handling for building the derived dataset
    copy_images: bool           # Copy vs symlink
    drop_empty: bool            # Drop images with empty filtered labels

    # Training
    epochs: int
    imgsz: int
    batch: int
    exp_name: str               # Experiment name (Ultralytics subfolder)
    do_train: bool              # Whether to train or just validate

    # Resume
    resume: Optional[str]       # "", "auto", or path to last.pt


# ----------------------------- Small utils --------------------------

def _expand(p: str | Path) -> Path:
    """Expand ~ and resolve to an absolute Path."""
    return Path(os.path.expanduser(str(p))).resolve()

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _place(src: Path, dst: Path, copy_files: bool) -> None:
    """Copy or symlink src → dst."""
    _ensure_dir(dst.parent)
    if copy_files:
        shutil.copy2(src, dst)
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())

def _split_has_data(root: Path, split: str) -> bool:
    """Check if split has both images and labels present with at least one file each."""
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    if not (img_dir.exists() and lbl_dir.exists()):
        return False
    try:
        return any(img_dir.iterdir()) and any(lbl_dir.iterdir())
    except Exception:
        return False

def _disc_yaml_in(root: Path) -> Path:
    """Return the expected disc-only YAML path inside a dataset root."""
    return root / "data.yaml"

def _has_precomputed_disc_only(root: Path) -> bool:
    """True if root looks like a disc-only dataset (has od_only.yaml)."""
    return _disc_yaml_in(root).exists()

def _filter_label_lines_to_disc(lines: Iterable[str]) -> List[str]:
    """Keep only class-0 lines from a YOLO .txt file."""
    keep: List[str] = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        try:
            cls = int(float(parts[0]))
        except Exception:
            continue
        if cls == 0:
            keep.append(ln)
    return keep

def filter_labels_dir_to_disc(src_lbl_dir: Path, dst_lbl_dir: Path, drop_empty: bool) -> int:
    """
    Convert all .txt labels under src_lbl_dir → dst_lbl_dir, keeping only class 0.
    If drop_empty is True, skip writing files that end up with no lines.
    Returns the number of label files written.
    """
    _ensure_dir(dst_lbl_dir)
    written = 0
    for lbl in sorted(src_lbl_dir.glob("*.txt")):
        lines_in = [ln for ln in lbl.read_text().splitlines()]
        kept = _filter_label_lines_to_disc(lines_in)
        if not kept and drop_empty:
            continue
        (dst_lbl_dir / lbl.name).write_text("\n".join(kept) + ("\n" if kept else ""))
        written += 1
    return written

def build_disc_only_dataset(
    base_root: Path,
    copy_images: bool,
    drop_empty: bool,
    aug_root: Path | None = None,
    train_splits: List[str] = ("train",),
) -> Tuple[Path, Path]:
    """
    Create a disc-only dataset next to base_root:
      dst_root = base_root.parent / f"{base_root.name}_disc_only"
    For splits in train_splits and when aug_root is provided, source from aug_root;
    otherwise source from base_root. Images are linked/copied; labels filtered to disc-only.
    Returns (dst_root, od_yaml_path)
    """
    dst_root = base_root.parent / f"{base_root.name}_disc_only"
    _ensure_dir(dst_root)

    for split in ("train", "val", "test"):
        # choose source root for this split
        src_root = aug_root if (aug_root is not None and split in set(train_splits)) else base_root
        src_img_dir = src_root / "images" / split
        src_lbl_dir = src_root / "labels" / split
        if not src_img_dir.exists() or not src_lbl_dir.exists():
            continue

        # 1) images
        for img in sorted(src_img_dir.glob("*")):
            _place(img, dst_root / "images" / split / img.name, copy_images)

        # 2) labels (filtered to class-0)
        filter_labels_dir_to_disc(src_lbl_dir, dst_root / "labels" / split, drop_empty=drop_empty)

    # 3) dataset YAML
    data_yaml = {
        "path": str(dst_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": ["disc"],
    }
    if _split_has_data(dst_root, "test"):
        data_yaml["test"] = "images/test"

    od_yaml = _disc_yaml_in(dst_root)
    od_yaml.write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return dst_root, od_yaml


# ----------------------------- Training / Val -----------------------

def build_model(weights: Path) -> YOLO:
    """Instantiate a YOLO model from local weights (no net download)."""
    if not weights.exists():
        raise SystemExit(f"[ERR] Model weights not found at: {weights}")
    return YOLO(str(weights))

def find_last_ckpt(runs_root: Path, exp_name: str) -> Optional[Path]:
    p = runs_root / exp_name / "weights" / "last.pt"
    return p if p.exists() else None

def train_model(model: YOLO, data_yaml: Path, cfg: DiscOnlyConfig, device: str, resume: bool) -> None:
    """Train the model if cfg.do_train is True."""
    if not cfg.do_train:
        return
    model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=device,
        project=str(cfg.runs_root),
        name=cfg.exp_name,
        cos_lr=True,
        optimizer="AdamW",
        pretrained=not resume,   # don't re-init pretraining when resuming
        patience=50,
        single_cls=True,
        resume=resume,           # <— key flag
    )

def validate_model(model: YOLO, data_yaml: Path, imgsz: int, device: str) -> None:
    """Validate on 'test' if present in YAML else on 'val'."""
    y = yaml.safe_load(data_yaml.read_text())
    if "test" in y:
        model.val(data=str(data_yaml), split="test", imgsz=imgsz, device=device)
    else:
        model.val(data=str(data_yaml), imgsz=imgsz, device=device)


# ----------------------------- Arg parsing / Config -----------------

def _parse_train_splits(val: str | List[str]) -> List[str]:
    if isinstance(val, list):
        return [str(s).strip() for s in val if str(s).strip()]
    return [s.strip() for s in str(val).split(",") if s.strip()]

def build_config_from_cli() -> DiscOnlyConfig:
    ap = argparse.ArgumentParser(description="Train a disc-only detector from a 2-class YOLO dataset (0=disc,1=cup).")

    ap.add_argument("--project_dir", default=".", help="Project root")
    ap.add_argument("--data_root", default=None,
                    help="YOLO dataset root. If it already contains od_only.yaml, it is treated as a disc-only dataset; "
                         "otherwise this script will derive '<data_root>_disc_only'.")
    ap.add_argument("--aug_root", default=None, help="Augmented YOLO dataset root (use ONLY for training splits)")
    ap.add_argument("--train_splits", default="train",
                    help="Comma list OR YAML list of splits to source from aug_root (default: 'train')")
    ap.add_argument("--model", default=None,
                    help="Weights path (defaults to PROJECT_DIR/weights/yolov8n.pt)")
    ap.add_argument("--project", default=None,
                    help="Ultralytics runs root (defaults to PROJECT_DIR/bounding_box/runs/detect)")

    # Training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="stageA_disc_only",
                    help="Ultralytics experiment name (runs subfolder).")
    ap.add_argument("--train", type=int, default=1, help="1=train+val, 0=val-only")

    # Resume
    ap.add_argument("--resume", default="", help="Path to last.pt OR 'auto' to use runs/<name>/weights/last.pt")

    # Data curation (used if we need to build the disc-only dataset)
    ap.add_argument("--copy_images", action="store_true",
                    help="Copy images instead of symlinking (default: symlink)")
    ap.add_argument("--drop_empty", action="store_true",
                    help="Drop images whose disc-only label becomes empty (i.e., skip writing the label file)")

    args = ap.parse_args()

    # Resolve project
    project_dir = _expand(args.project_dir)

    data_root = _expand(args.data_root) if args.data_root else (project_dir / "data" / "yolo_split")
    aug_root = _expand(args.aug_root) if args.aug_root else None
    train_splits = _parse_train_splits(args.train_splits)

    model_path = _expand(args.model) if args.model else (project_dir / "weights" / "yolov8n.pt")
    runs_root = _expand(args.project) if args.project else (project_dir / "bounding_box" / "runs" / "detect")

    # Basic sanity for provided root (must look like a YOLO dataset root)
    if not (data_root / "images").exists() or not (data_root / "labels").exists():
        raise SystemExit(f"[ERR] Not a YOLO dataset root: {data_root} (missing images/ or labels/)")
    _ensure_dir(runs_root)

    # Optional sanity check for aug_root (only if provided)
    if aug_root is not None:
        if not (aug_root / "images").exists() or not (aug_root / "labels").exists():
            raise SystemExit(f"[ERR] --aug_root is not a YOLO dataset root: {aug_root}")

    return DiscOnlyConfig(
        project_dir=project_dir,
        data_root=data_root,
        model_path=model_path,
        runs_root=runs_root,
        aug_root=aug_root,
        train_splits=train_splits,
        copy_images=bool(args.copy_images),
        drop_empty=bool(args.drop_empty),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        exp_name=str(args.name),
        do_train=bool(args.train),
        resume=str(args.resume) if args.resume else "",
    )


# ----------------------------- Main --------------------------------

def main() -> None:
    cfg = build_config_from_cli()

    # 1) Determine disc-only dataset YAML to use (precomputed vs build)
    if _has_precomputed_disc_only(cfg.data_root):
        od_yaml = _disc_yaml_in(cfg.data_root)
        print(f"[INFO] Using precomputed disc-only dataset: {cfg.data_root}")
        print(f"[INFO] Dataset YAML: {od_yaml}")
    else:
        print(f"[INFO] Building disc-only dataset from: {cfg.data_root}")
        out_root, od_yaml = build_disc_only_dataset(
            cfg.data_root,
            copy_images=cfg.copy_images,
            drop_empty=cfg.drop_empty,
            aug_root=cfg.aug_root,
            train_splits=cfg.train_splits,
        )
        print(f"[OK] Disc-only dataset: {out_root}")
        print(f"[OK] Dataset YAML:      {od_yaml}")

    # 2) Build model & device (+ resume)
    device = ultralytics_device_arg()  # "0" or "cpu"

    resume_flag = False
    last_ckpt: Optional[Path] = None
    if cfg.resume:
        if str(cfg.resume).lower() == "auto":
            last_ckpt = find_last_ckpt(cfg.runs_root, cfg.exp_name)
        else:
            last_ckpt = _expand(cfg.resume)

    if last_ckpt and last_ckpt.exists():
        print(f"[INFO] Resuming from: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        resume_flag = True
    else:
        if cfg.resume:
            print(f"[WARN] --resume requested but checkpoint not found. Starting fresh. (Searched: {last_ckpt})")
        model = build_model(cfg.model_path)

    # 3) Train (optional) and Validate
    if cfg.do_train:
        print(f"[INFO] Training… (epochs={cfg.epochs}, imgsz={cfg.imgsz}, batch={cfg.batch}, resume={resume_flag})")
    train_model(model, od_yaml, cfg, device, resume=resume_flag)

    print("[INFO] Validating…")
    validate_model(model, od_yaml, cfg.imgsz, device)
    print("[OK] Done.")


if __name__ == "__main__":
    main()