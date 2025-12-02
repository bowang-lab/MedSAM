#!/usr/bin/env python3
"""
Notebook-style script for Chaksu .tif annotations where:
- background is black (0),
- optic disc contour is gray,
- optic cup contour is WHITE (brightest),
- cup is inside disc.

Creates:
OUT_DIR/od_mask/{stem}.png
OUT_DIR/oc_mask/{stem}.png
"""

# ======================================
# CONFIG
# ======================================
from pathlib import Path

TIF_DIR  = Path("/Volumes/T7/NovaLab/Datasets/Raw_Datasets/Chaksu/Train/segmentation_fusion/Remidio/Mean")
OUT_DIR  = Path("/Volumes/T7/NovaLab/Datasets/Processed/Chaksu/")

LINE_CLOSING_RAD = 2
LINE_MIN_SIZE    = 150

DISC_CLOSING_RAD = 5
DISC_OPENING_RAD = 2
DISC_MIN_SIZE    = 500

CUP_CLOSING_RAD  = 3
CUP_OPENING_RAD  = 1
CUP_MIN_SIZE     = 200

# ======================================
# IMPORTS
# ======================================
import warnings
import numpy as np
import tifffile as tiff
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import (
    remove_small_objects,
    binary_closing,
    binary_opening,
    disk,
)
from imageio.v2 import imwrite

# ======================================
# HELPERS
# ======================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_tif_as_2d(tif_path: Path) -> np.ndarray | None:
    """Robust reader for .tif files."""
    try:
        arr = tiff.imread(str(tif_path))
    except Exception as e:
        try:
            with Image.open(tif_path) as im:
                arr = np.array(im)
            warnings.warn(
                f"[WARN] {tif_path.name}: not a TIFF for tifffile ({e}); loaded via PIL."
            )
        except Exception as e2:
            # warnings.warn(
            #     f"[SKIP] {tif_path.name}: unreadable as TIFF or image. "
            #     f"tifffile err={e}; PIL err={e2}"
            # )
            return None

    if arr.ndim == 3:
        arr = arr[..., 0]

    if np.issubdtype(arr.dtype, np.floating):
        amax = float(arr.max()) if arr.max() != 0 else 1.0
        arr = (arr / amax * 255.0).astype(np.uint8)
    else:
        if arr.dtype != np.uint8:
            arr = arr.astype(np.int32)

    return arr


def seal_and_clean_lines(lines: np.ndarray) -> np.ndarray:
    m = lines.astype(bool)
    if LINE_CLOSING_RAD > 0:
        m = binary_closing(m, footprint=disk(LINE_CLOSING_RAD))
    if LINE_MIN_SIZE > 0:
        m = remove_small_objects(m, min_size=LINE_MIN_SIZE)
    return m.astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab = label(mask, connectivity=2)
    if lab.max() == 0:
        return np.zeros_like(mask, dtype=bool)

    props = regionprops(lab)
    props.sort(key=lambda r: r.area, reverse=True)
    return (lab == props[0].label)


def smooth_filled(mask: np.ndarray, closing_rad: int, opening_rad: int, min_size: int) -> np.ndarray:
    m = mask.astype(bool)
    if closing_rad > 0:
        m = binary_closing(m, footprint=disk(closing_rad))
    if opening_rad > 0:
        m = binary_opening(m, footprint=disk(opening_rad))
    if min_size > 0:
        m = remove_small_objects(m, min_size=min_size)
    m = binary_fill_holes(m)
    return m.astype(bool)


def extract_disc_and_cup(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nonzero = arr > 0
    if not nonzero.any():
        empty = np.zeros_like(arr, dtype=bool)
        return empty, empty

    # ----- DISC -----
    disc_lines = seal_and_clean_lines(nonzero)
    disc_lines = largest_component(disc_lines)
    disc_filled = smooth_filled(disc_lines, DISC_CLOSING_RAD, DISC_OPENING_RAD, DISC_MIN_SIZE)

    # ----- CUP -----
    nz_vals = arr[nonzero].astype(np.float32)
    thr_high = threshold_otsu(nz_vals)

    cup_lines = (arr >= thr_high) & nonzero
    cup_lines = seal_and_clean_lines(cup_lines)
    cup_lines = largest_component(cup_lines)

    cup_filled = binary_fill_holes(cup_lines)
    if cup_filled.any():
        cup_filled = smooth_filled(cup_filled, CUP_CLOSING_RAD, CUP_OPENING_RAD, CUP_MIN_SIZE)

    # enforce containment
    if cup_filled.any():
        cup_filled = cup_filled & disc_filled

    return cup_filled.astype(bool), disc_filled.astype(bool)


# ======================================
# MAIN PROCESSING
# ======================================
def process_dir(tif_dir: Path, out_dir: Path) -> None:
    oc_dir = out_dir / "oc_mask"
    od_dir = out_dir / "od_mask"
    ensure_dir(oc_dir)
    ensure_dir(od_dir)

    tif_paths = sorted(tif_dir.glob("*.tif"))
    if not tif_paths:
        raise FileNotFoundError(f"No .tif files found in {tif_dir}")

    count = 0  # <-- NEW COUNTER

    for tif_path in tif_paths:
        arr = read_tif_as_2d(tif_path)
        if arr is None:
            continue

        cup_bool, disc_bool = extract_disc_and_cup(arr)

        stem = tif_path.stem
        out_name = f"{stem}.png"

        imwrite((od_dir / out_name), (disc_bool.astype(np.uint8) * 255))
        imwrite((oc_dir / out_name), (cup_bool.astype(np.uint8) * 255))

        print(f"[OK] {tif_path.name} -> od_mask/{out_name}, oc_mask/{out_name}")
        count += 1  # <-- increment

    # FINAL SUMMARY PRINT
    print(f"\n[SUMMARY] Processed {count} .tif annotation images.")


# ======================================
# RUN
# ======================================
print(f"[INFO] tif_dir: {TIF_DIR}")
print(f"[INFO] out_dir: {OUT_DIR}")
ensure_dir(OUT_DIR)

process_dir(TIF_DIR, OUT_DIR)

print("[INFO] Done.")