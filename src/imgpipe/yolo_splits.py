# yolo_splits.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple, Optional
import json
import math
import os
import random
import shutil
from datetime import datetime

import numpy as np
from PIL import Image as PILImage

from src.imgpipe.image import Image  # expects Image objects produced by your ImageFactory
from src.utils import save_images_jsonl

SplitName = Literal["train", "val", "test"]


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


def _validate_ratios(train: float, val: float, test: float) -> Tuple[float, float, float]:
    s = train + val + test
    if not math.isclose(s, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"train+val+test must sum to 1.0 (got {s})")
    if train < 0 or val < 0 or test < 0:
        raise ValueError("Ratios must be non-negative.")
    return train, val, test


def _split_indices(n: int, train: float, val: float, test: float) -> Tuple[List[int], List[int], List[int]]:
    """Round-splitting that exactly covers n after shuffle."""
    # Base floors
    n_train = int(math.floor(train * n))
    n_val = int(math.floor(val * n))
    # Assign remainder to the split with the largest fractional part
    rem = n - (n_train + n_val + int(math.floor(test * n)))
    # Distribute remainder using fractional parts
    fracs = [
        ("train", train * n - n_train),
        ("val", val * n - n_val),
        ("test", test * n - int(math.floor(test * n))),
    ]
    fracs.sort(key=lambda x: x[1], reverse=True)
    add = {"train": 0, "val": 0, "test": 0}
    for i in range(rem):
        add[fracs[i % 3][0]] += 1

    n_train += add["train"]
    n_val += add["val"]
    n_test = n - n_train - n_val

    # Produce slices after caller shuffles an index list
    def slicer(idx: List[int]) -> Tuple[List[int], List[int], List[int]]:
        a = idx[:n_train]
        b = idx[n_train:n_train + n_val]
        c = idx[n_train + n_val:]
        return a, b, c

    return slicer  # returns a function; caller applies to shuffled index list


def _ensure_dirs(root: Path) -> None:
    # Images / labels
    for sub in ("images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    # Mask dirs are created lazily in _save_gt_masks per split,
    # so no need to pre-create them here.


def _label_path_for(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def _write_label_file(img: Image, label_path: Path, *, use_gt: bool) -> None:
    """
    Write YOLO label lines. If no boxes exist, create an empty file (valid for YOLO).
    """
    lines = list(img.yolo_lines_2class(use_gt=use_gt))
    label_path.write_text("\n".join(lines), encoding="utf-8")


def _copy_image(src: Path, dst: Path) -> None:
    """Copy image bytes; ensures dst parent exists."""
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

    # Disc mask
    if img.gt_disc_mask is not None:
        disc_mask = _mask_to_image_size(img, img.gt_disc_mask)
        if disc_mask is not None:
            disc_arr = (disc_mask.astype(np.uint8) * 255)
            PILImage.fromarray(disc_arr).save(disc_dir / f"{stem}.png")

    # Cup mask
    if img.gt_cup_mask is not None:
        cup_mask = _mask_to_image_size(img, img.gt_cup_mask)
        if cup_mask is not None:
            cup_arr = (cup_mask.astype(np.uint8) * 255)
            PILImage.fromarray(cup_arr).save(cup_dir / f"{stem}.png")


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

    Inputs
    ------
    images : Sequence[Image]
        Objects produced by ImageFactory.make_images(...).
    train, val, test : float
        Fractions summing to 1.0.
    out_dir : Path
        Directory to create the YOLO dataset in.
    seed : int
        RNG seed for reproducibility.
    use_gt : bool
        If True, label files use ground-truth boxes (disc=0, cup=1).
        If False, uses intermediate predicted boxes (inter_pred_*).

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

    Returns
    -------
    YoloSplit
        In-memory split object with train/val/test lists of Images.
    """
    _validate_ratios(train, val, test)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_dirs(out_dir)

    # Deterministic shuffle
    rng = random.Random(seed)
    idxs = list(range(len(images)))
    rng.shuffle(idxs)

    slicer = _split_indices(len(images), train, val, test)
    train_idx, val_idx, test_idx = slicer(idxs)

    split = YoloSplit(
        train=[images[i] for i in train_idx],
        val=[images[i] for i in val_idx],
        test=[images[i] for i in test_idx],
    )

    # Materialize images, labels, and GT masks
    for split_name, imgs in split.as_mapping().items():
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name

        for im in imgs:
            im.set_split(split_name)
            src = im.image_path
            if not src.exists():
                # Skip missing sources defensively
                continue

            # Copy image
            dst = img_dir / src.name
            _copy_image(src, dst)

            # Write label (empty file if no boxes)
            label_path = _label_path_for(dst, lbl_dir)
            _write_label_file(im, label_path, use_gt=use_gt)

            # Save GT masks (if present)
            _save_gt_masks(im, split_name, out_dir, stem=dst.stem)

    save_images_jsonl(images, out_dir / "saved_images.jsonl")
    # Write metadata for reproducibility
    meta = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seed": seed,
        "ratios": {"train": train, "val": val, "test": test},
        "counts": split.counts(),
        "total": len(images),
        "use_gt": use_gt,
        "has_gt_masks": True,  # indicates this script is capable of exporting GT masks
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # YOLO data.yaml
    _write_data_yaml(out_dir)

    return split