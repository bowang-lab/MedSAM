# yolo_splits.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple, Optional, Iterable, Union
import json
import math
import shutil
from datetime import datetime

import numpy as np
from PIL import Image as PILImage

from src.imgpipe.image import Image  # expects Image objects produced by your ImageFactory

SplitName = Literal["train", "val", "test"]


# =============================================================================
# Split container
# =============================================================================


@dataclass(frozen=True)
class YoloSplit:
    train: List[Image]
    val: List[Image]
    test: List[Image]

    def as_mapping(self) -> Dict[SplitName, List[Image]]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def counts(self) -> Dict[str, int]:
        m = self.as_mapping()
        return {k: len(v) for k, v in m.items()}


# =============================================================================
# Utilities
# =============================================================================


def _validate_ratios(train: float, val: float, test: float) -> Tuple[float, float, float]:
    s = train + val + test
    if not math.isclose(s, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"train+val+test must sum to 1.0 (got {s})")
    if train < 0 or val < 0 or test < 0:
        raise ValueError("Ratios must be non-negative.")
    return train, val, test


def _split_indices(n: int, train: float, val: float, test: float):
    """Round-splitting that exactly covers n after shuffle."""
    n_train = int(math.floor(train * n))
    n_val = int(math.floor(val * n))
    n_test_floor = int(math.floor(test * n))

    rem = n - (n_train + n_val + n_test_floor)

    fracs = [
        ("train", train * n - n_train),
        ("val", val * n - n_val),
        ("test", test * n - n_test_floor),
    ]
    fracs.sort(key=lambda x: x[1], reverse=True)
    add = {"train": 0, "val": 0, "test": 0}
    for i in range(rem):
        add[fracs[i % 3][0]] += 1

    n_train += add["train"]
    n_val += add["val"]
    n_test = n - n_train - n_val

    def slicer(idx: List[int]) -> Tuple[List[int], List[int], List[int]]:
        a = idx[:n_train]
        b = idx[n_train:n_train + n_val]
        c = idx[n_train + n_val:]
        return a, b, c

    return slicer


def _ensure_dirs(root: Path) -> None:
    for sub in (
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ):
        (root / sub).mkdir(parents=True, exist_ok=True)
    # Mask dirs are created lazily in _save_gt_masks per split.


def _label_path_for(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def _write_label_file(img: Image, label_path: Path, *, use_gt: bool) -> None:
    """
    Write YOLO label lines. If no boxes exist, create an empty file (valid for YOLO).
    """
    lines = list(img.yolo_lines_2class(use_gt=use_gt))
    label_path.write_text("\n".join(lines), encoding="utf-8")


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_data_yaml(root: Path) -> None:
    """
    Write a minimal Ultralytics-style data.yaml pointing to this dataset root.
    """
    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "disc", 1: "cup"},
    }
    # Keep json so you don't need a yaml dependency; Ultralytics accepts YAML, but JSON is not guaranteed.
    # If you prefer strict YAML, swap to a tiny manual YAML writer.
    (root / "data.yaml").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _mask_to_image_size(img: Image, mref) -> Optional[np.ndarray]:
    """
    Load a GT mask as a bool array aligned to the image's (H, W).
    Mirrors Image._mask_to_image_size logic without importing BinaryMaskRef.
    """
    if mref is None:
        return None
    arr = mref.load().astype(bool)
    H, W = img.height, img.width
    if arr.shape == (H, W):
        return arr
    out = np.zeros((H, W), dtype=bool)
    h = min(H, arr.shape[0])
    w = min(W, arr.shape[1])
    out[:h, :w] = arr[:h, :w]
    return out


def _save_gt_masks(img: Image, split_name: str, out_dir: Path, stem: str) -> None:
    """
    Save ground-truth disc and cup masks (if present) as PNGs.

    Directory layout:
      out_dir/
        masks/
          train/
            disc/<stem>.png
            cup/<stem>.png
          val/
            disc/<stem>.png
            cup/<stem>.png
          test/
            disc/<stem>.png
            cup/<stem>.png
    """
    base = out_dir / "masks" / split_name
    disc_dir = base / "disc"
    cup_dir = base / "cup"
    disc_dir.mkdir(parents=True, exist_ok=True)
    cup_dir.mkdir(parents=True, exist_ok=True)

    if img.gt_disc_mask is not None:
        disc_mask = _mask_to_image_size(img, img.gt_disc_mask)
        if disc_mask is not None:
            disc_arr = (disc_mask.astype(np.uint8) * 255)
            PILImage.fromarray(disc_arr).save(disc_dir / f"{stem}.png")

    if img.gt_cup_mask is not None:
        cup_mask = _mask_to_image_size(img, img.gt_cup_mask)
        if cup_mask is not None:
            cup_arr = (cup_mask.astype(np.uint8) * 255)
            PILImage.fromarray(cup_arr).save(cup_dir / f"{stem}.png")


def _coerce_split(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = str(s).strip().lower()
    return s2 if s2 else None


def _split_from_existing(images: Sequence[Image]) -> YoloSplit:
    """
    Build YoloSplit from images that already have split assigned.
    Requires every image to have split in {"train","val","test"}.
    """
    tr: List[Image] = []
    va: List[Image] = []
    te: List[Image] = []
    missing = 0
    bad = 0

    for im in images:
        s = _coerce_split(getattr(im, "split", None))
        if s is None:
            missing += 1
            continue
        if s == "train":
            tr.append(im)
        elif s == "val":
            va.append(im)
        elif s == "test":
            te.append(im)
        else:
            bad += 1

    if missing or bad:
        raise ValueError(
            f"Existing-split mode requires split in {{train,val,test}} for every image; "
            f"missing={missing}, invalid={bad}."
        )
    return YoloSplit(train=tr, val=va, test=te)


# =============================================================================
# Main entrypoints
# =============================================================================


def create_yolo_dataset(
    images: Sequence[Image],
    *,
    train: float,
    val: float,
    test: float,
    out_dir: Path,
    seed: int = 42,
    use_gt: bool = True,
) -> YoloSplit:
    """
    Build a YOLO-ready directory (images/ & labels/) and return the split (in memory).

    This mode ASSIGNS splits by ratios and seed.

    Effects
    -------
    Creates:
      out_dir/
        images/{train,val,test}/<stem>.<ext>
        labels/{train,val,test}/<stem>.txt
        masks/{train,val,test}/disc/<stem>.png     (if GT disc mask exists)
        masks/{train,val,test}/cup/<stem>.png      (if GT cup mask exists)
        data.yaml  (Ultralytics format)
        split_meta.json
        saved_images.parquet   (Image metadata only, no pixels)
    """
    _validate_ratios(train, val, test)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_dirs(out_dir)

    # Shuffle indices deterministically
    rng = np.random.default_rng(int(seed))
    idxs = np.arange(len(images))
    rng.shuffle(idxs)

    slicer = _split_indices(len(images), train, val, test)
    train_idx, val_idx, test_idx = slicer(idxs.tolist())

    split = YoloSplit(
        train=[images[i] for i in train_idx],
        val=[images[i] for i in val_idx],
        test=[images[i] for i in test_idx],
    )

    _materialize_yolo_layout(split, out_dir=out_dir, use_gt=use_gt)

    parquet_path = out_dir / "saved_images.parquet"
    Image.save_parquet(images, parquet_path, drop_none=False)

    meta = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seed": seed,
        "mode": "assign_splits",
        "ratios": {"train": train, "val": val, "test": test},
        "counts": split.counts(),
        "total": len(images),
        "use_gt": use_gt,
        "has_gt_masks": True,
        "saved_images_parquet": str(parquet_path),
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _write_data_yaml(out_dir)
    return split


def create_yolo_dataset_from_parquet(
        in_parquet: Path,
        *,
        out_dir: Path,
        batch_size: int = 2048,
        use_gt: bool = True,
        save_images_parquet: bool = True,
        exclude_datasets: Optional[List[str]] = None,  # <--- NEW ARG
) -> YoloSplit:
    """
    Build the same YOLO directory structure, but using a Parquet file.
    Supports excluding datasets by substring match (case-insensitive).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_dirs(out_dir)

    # Stream in -> bucket by split
    imgs: List[Image] = []

    # Pre-compute lower-case exclusions for speed
    excludes_lower = [e.lower() for e in exclude_datasets] if exclude_datasets else []

    for im in Image.iter_parquet(in_parquet, batch_size=int(batch_size)):
        # NEW: Check exclusion
        if excludes_lower:
            ds_name = (im.dataset or "").lower()
            if any(ex in ds_name for ex in excludes_lower):
                continue

        imgs.append(im)

    split = _split_from_existing(imgs)
    _materialize_yolo_layout(split, out_dir=out_dir, use_gt=use_gt)

    parquet_path = out_dir / "saved_images.parquet"
    if save_images_parquet:
        Image.save_parquet(imgs, parquet_path, drop_none=False)

    meta = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "mode": "existing_splits",
        "counts": split.counts(),
        "total": len(imgs),
        "use_gt": use_gt,
        "has_gt_masks": True,
        "input_parquet": str(Path(in_parquet).resolve()),
        "saved_images_parquet": str(parquet_path) if save_images_parquet else None,
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _write_data_yaml(out_dir)
    return split


# =============================================================================
# Internal: write directory layout
# =============================================================================


def _materialize_yolo_layout(split: YoloSplit, *, out_dir: Path, use_gt: bool) -> None:
    for split_name, imgs in split.as_mapping().items():
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name

        for im in imgs:
            # Ensure split is consistent (important if downstream re-saves metadata)
            im.set_split(split_name)

            src = im.image_path
            if not src.exists():
                continue

            dst = img_dir / src.name
            _copy_image(src, dst)

            label_path = _label_path_for(dst, lbl_dir)
            _write_label_file(im, label_path, use_gt=use_gt)

            _save_gt_masks(im, split_name, out_dir, stem=dst.stem)