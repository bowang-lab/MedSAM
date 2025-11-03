# yolo_splits.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple
import json
import math
import os
import random
import shutil
from datetime import datetime

from src.imgpipe.image import Image  # expects Image objects produced by your ImageFactory

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
    for sub in ("images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"):
        (root / sub).mkdir(parents=True, exist_ok=True)


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
        data.yaml  (Ultralytics format)

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

    # Materialize images and labels
    for split_name, imgs in split.as_mapping().items():
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name

        for im in imgs:
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

    # Write metadata for reproducibility
    meta = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seed": seed,
        "ratios": {"train": train, "val": val, "test": test},
        "counts": split.counts(),
        "total": len(images),
        "use_gt": use_gt,
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # YOLO data.yaml
    _write_data_yaml(out_dir)

    return split