#!/usr/bin/env python3
"""
Process REFUGE2-style data into flat folders of PNGs.

Given an input directory with structure:

    ROOT/
      train/
        images/*.jpg
        mask/*.(bmp|png)
      val/
        images/*.jpg
        mask/*.(bmp|png)
      test/
        images/*.jpg
        mask/*.(bmp|png)

Each mask file contains both optic disc (OD) and optic cup (OC) as
different non-zero values. The OD is the *outer* structure, and the
cup is fully contained within the disc.

This script creates:

    ROOT/REGUGE2_processed/
        fundus/   -> fundus PNGs (RGB)
        oc_mask/  -> binary PNG masks (cup = 255, background = 0)
        od_mask/  -> binary PNG masks (disc = 255, background = 0)

Both OD and OC masks are **filled** (no hollow rings).
Output filenames: refuge2_{stem}.png
"""

import argparse
from pathlib import Path
from collections import deque

import numpy as np
from PIL import Image


SPLITS = ["train", "val", "test"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_mask_path(masks_dir: Path, stem: str) -> Path | None:
    """
    Find a mask file for a given stem, trying .bmp then .png.
    Returns Path or None if not found.
    """
    for ext in (".bmp", ".png"):
        cand = masks_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand

    # Case-insensitive / mixed extension fallback
    for p in masks_dir.iterdir():
        if p.is_file() and p.stem == stem and p.suffix.lower() in {".bmp", ".png"}:
            return p

    return None


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Binary hole-filling using flood-fill from the image border.

    mask: boolean array, True = foreground.
    Returns boolean array with interior holes filled.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)

    h, w = mask.shape
    if h == 0 or w == 0:
        return mask

    inv = ~mask  # True where background
    visited = np.zeros_like(inv, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    # Initialize queue with all border pixels that are background
    for x in range(w):
        if inv[0, x] and not visited[0, x]:
            visited[0, x] = True
            q.append((0, x))
        if inv[h - 1, x] and not visited[h - 1, x]:
            visited[h - 1, x] = True
            q.append((h - 1, x))

    for y in range(h):
        if inv[y, 0] and not visited[y, 0]:
            visited[y, 0] = True
            q.append((y, 0))
        if inv[y, w - 1] and not visited[y, w - 1]:
            visited[y, w - 1] = True
            q.append((y, w - 1))

    # 4-connected flood-fill
    while q:
        cy, cx = q.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w:
                if inv[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))

    # Holes are background pixels not connected to border
    holes = inv & ~visited
    filled = mask | holes
    return filled


def extract_mask_cup_and_disc(mask_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract cup and disc masks from a REFUGE2-style mask.

    For REFUGE2, the expected mask is of the following format:
      - Background is all white (highest intensity).
      - Disc mask is all non-white pixels (includes the cup region).
      - Cup mask is all black pixels (lowest intensity).

    Returns:
        cup_mask_bool, disc_mask_bool
        Both are boolean arrays with filled regions (no hollow rings).
    """
    arr = mask_arr

    # Convert RGB/RGBA to single channel by taking the first channel
    if arr.ndim == 3:
        arr = arr[..., 0]

    arr = arr.astype(np.uint8)
    unique_vals = np.unique(arr)

    # Degenerate case: everything the same value -> no structures
    if unique_vals.size == 1:
        h, w = arr.shape
        empty = np.zeros((h, w), dtype=bool)
        return empty, empty

    # Assume lowest intensity is cup (black), highest is background (white)
    cup_val = int(unique_vals[0])
    bg_val = int(unique_vals[-1])

    background = arr == bg_val
    cup_mask_bool = arr == cup_val

    # Disc is everything that is not background (includes cup)
    disc_mask_bool = ~background

    # Fill holes to ensure solid regions
    disc_mask_bool = _fill_holes(disc_mask_bool)

    if cup_mask_bool.any():
        cup_mask_bool = _fill_holes(cup_mask_bool)

    return cup_mask_bool, disc_mask_bool


def process_split(root: Path, split: str, out_root: Path) -> None:
    images_dir = root / split / "images"
    masks_dir = root / split / "mask"

    if not images_dir.is_dir() or not masks_dir.is_dir():
        print(f"[WARN] Missing images/mask directories for split '{split}', skipping.")
        return

    out_fundus = out_root / "fundus"
    out_oc = out_root / "oc_mask"
    out_od = out_root / "od_mask"

    ensure_dir(out_fundus)
    ensure_dir(out_oc)
    ensure_dir(out_od)

    for img_path in sorted(images_dir.glob("*.jpg")):
        stem = img_path.stem
        mask_path = find_mask_path(masks_dir, stem)

        if mask_path is None:
            print(f"[WARN] No mask for image {img_path}, expected {stem}.bmp or {stem}.png")
            continue

        # Load and save fundus as PNG (RGB)
        img = Image.open(img_path).convert("RGB")
        out_name = f"refuge2_{stem}.png"
        img.save(out_fundus / out_name)

        # Load mask as grayscale array
        mask_img = Image.open(mask_path)
        mask_arr = np.array(mask_img)

        cup_mask_bool, disc_mask_bool = extract_mask_cup_and_disc(mask_arr)

        if not disc_mask_bool.any():
            print(f"[WARN] No disc region found in mask {mask_path}, skipping.")
            continue

        if not cup_mask_bool.any():
            print(f"[INFO] No cup region detected in {mask_path}; OC mask set to all zeros.")

        # Binary OD mask: disc = 255, background = 0 (black)
        od_arr = np.where(disc_mask_bool, 255, 0).astype(np.uint8)

        # Binary OC mask: cup = 255, background = 0 (black)
        oc_arr = np.where(cup_mask_bool, 255, 0).astype(np.uint8)

        # Save as single-channel grayscale PNGs with black background and white mask
        Image.fromarray(od_arr, mode="L").save(out_od / out_name)
        Image.fromarray(oc_arr, mode="L").save(out_oc / out_name)


def main():
    parser = argparse.ArgumentParser(
        description="Convert REFUGE2-style images + BMP/PNG masks into flat PNG folders with filled OD/OC masks."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/Users/carlosperez/Library/CloudStorage/OneDrive-UBC/Ipek_Carlos/GlaucomaDatasets/REFUGE2",
        help="Path to directory containing train/val/test subdirectories.",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="REGUGE2_processed",
        help="Name of the output directory created inside ROOT (default: REGUGE2_processed).",
    )

    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_root = root / args.out_name

    ensure_dir(out_root)

    print(f"[INFO] Input root : {root}")
    print(f"[INFO] Output root: {out_root}")

    for split in SPLITS:
        print(f"[INFO] Processing split: {split}")
        process_split(root, split, out_root)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()