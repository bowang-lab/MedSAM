#!/usr/bin/env python3
"""
Batch script to build OD/OC masks from multiple annotated fundus images.

You provide:
  --in-dir   : directory containing .jpeg/.jpg/.tif images
               (both originals and annotated versions)
  --out-dir  : output root
  --stem     : prefix to prepend to all output filenames

Assumptions:
- Original (non-annotated) images have "prime" at the end of the stem:
    e.g., "image1prime.tif"
- Annotated images share the same base but with a "-n" suffix:
    e.g., "image1-1.tif", "image1-2.jpeg", "Image1-3.tif", ...
- Matching is capitalization-insensitive.
- There can be multiple annotations per original.

For each original "base" image:
  - Save the original fundus (no annotations) to:
        out_dir/fundus/{stem}_{base}.png
  - For each annotation k (sorted by filename), treat it as Expert k:
        out_dir/expert{k}/oc_mask/{stem}_{base}.png
        out_dir/expert{k}/od_mask/{stem}_{base}.png
  - After all experts for that base:
        compute **binary consensus masks** (majority vote per pixel),
        then apply extra smoothing:
            - disc: moderate smoothing
            - cup : stronger smoothing (to avoid non-convex artifacts)
        save to:
            out_dir/oc_mask_mean/{stem}_{base}.png
            out_dir/od_mask_mean/{stem}_{base}.png

All outputs are PNG; no change to dimensions.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops
from skimage.morphology import (
    remove_small_objects,
    binary_closing,
    binary_opening,
    disk,
)
from imageio.v2 import imwrite

# =========================
# Global configuration
# =========================

# Background (outer black) masking
BLACK_THRESH = 10  # mean intensity (0–255)

# Difference thresholding (orig vs annotated)
DIFF_METHOD = "otsu"        # "otsu" or "percentile"
DIFF_PERCENTILE = 99.0      # used only if DIFF_METHOD="percentile"
DIFF_MIN_SIZE = 80          # remove tiny diff specks before stroke cleanup

# Seal/clean stroke lines
STROKE_CLOSING_RAD = 2      # close gaps in rings
STROKE_MIN_SIZE    = 120    # remove tiny stroke components

# Filter filled candidates by size relative to retina area
MIN_AREA_FRAC = 0.0003
MAX_AREA_FRAC = 0.25

# Smooth final per-expert filled masks (morphology)
DISC_CLOSING_RAD = 5
DISC_OPENING_RAD = 2
DISC_MIN_SIZE    = 500

CUP_CLOSING_RAD  = 3
CUP_OPENING_RAD  = 1
CUP_MIN_SIZE     = 150

# VERY SMALL contour smoothing (Gaussian on mask)
GAUSS_SMOOTH_SIGMA = 0.6   # 0.4–0.8 is small
GAUSS_THRESH       = 0.5   # re-binarize threshold

# Majority-vote threshold for consensus mean masks
CONSENSUS_THRESH = 0.5

# EXTRA smoothing for consensus mean masks
# Disc: moderate smoothing
MEAN_DISC_CLOSING_RAD = 9
MEAN_DISC_OPENING_RAD = 3
MEAN_DISC_MIN_SIZE    = DISC_MIN_SIZE

# Cup: stronger smoothing (as requested)
MEAN_CUP_CLOSING_RAD  = 11
MEAN_CUP_OPENING_RAD  = 4
MEAN_CUP_MIN_SIZE     = CUP_MIN_SIZE


# =========================
# Low-level helpers
# =========================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_rgb(path: Path) -> np.ndarray:
    """Read an image (jpg/tif/whatever PIL supports) and return RGB uint8 (H, W, 3)."""
    im = Image.open(path).convert("RGB")
    return np.array(im)


def retina_mask(rgb: np.ndarray) -> np.ndarray:
    """Mask out black outer background based on mean intensity."""
    intensity = rgb.mean(axis=2)
    return intensity > BLACK_THRESH


def diff_strokes(orig_rgb: np.ndarray, ann_rgb: np.ndarray) -> np.ndarray:
    """
    Compute a robust difference mask highlighting annotation strokes,
    using Lab color difference between original and annotated.
    """
    o_lab = rgb2lab(orig_rgb / 255.0)
    a_lab = rgb2lab(ann_rgb / 255.0)

    dist = np.linalg.norm(a_lab - o_lab, axis=2)

    rm = retina_mask(orig_rgb) | retina_mask(ann_rgb)
    if not rm.any():
        return np.zeros(dist.shape, dtype=bool)

    if DIFF_METHOD == "otsu":
        thr = threshold_otsu(dist[rm])
    elif DIFF_METHOD == "percentile":
        thr = np.percentile(dist[rm], DIFF_PERCENTILE)
    else:
        raise ValueError(f"Unknown DIFF_METHOD={DIFF_METHOD}")

    strokes = (dist >= thr) & rm

    if DIFF_MIN_SIZE > 0:
        strokes = remove_small_objects(strokes, min_size=DIFF_MIN_SIZE)

    if STROKE_CLOSING_RAD > 0:
        strokes = binary_closing(strokes, footprint=disk(STROKE_CLOSING_RAD))
    if STROKE_MIN_SIZE > 0:
        strokes = remove_small_objects(strokes, min_size=STROKE_MIN_SIZE)

    return strokes.astype(bool)


def tiny_contour_smooth(mask: np.ndarray) -> np.ndarray:
    """
    Very small smoothing: blur the binary mask slightly then threshold back.
    This rounds pixel stair-steps without significantly shrinking/growing shapes.
    """
    if GAUSS_SMOOTH_SIGMA <= 0:
        return mask.astype(bool)

    m = mask.astype(np.float32)
    m_blur = gaussian(m, sigma=GAUSS_SMOOTH_SIGMA, preserve_range=True)
    m_bin = m_blur >= GAUSS_THRESH
    return m_bin.astype(bool)


def smooth_filled(mask: np.ndarray, closing_rad: int, opening_rad: int, min_size: int) -> np.ndarray:
    """Morphology smoothing + hole fill + tiny Gaussian smoothing."""
    m = mask.astype(bool)
    if closing_rad > 0:
        m = binary_closing(m, footprint=disk(closing_rad))
    if opening_rad > 0:
        m = binary_opening(m, footprint=disk(opening_rad))
    if min_size > 0:
        m = remove_small_objects(m, min_size=min_size)
    m = binary_fill_holes(m)
    m = tiny_contour_smooth(m)
    return m.astype(bool)


def smooth_mean_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    """
    Extra smoothing for consensus mean masks.
    `kind` is "disc" or "cup".
    """
    if kind == "disc":
        closing_rad = MEAN_DISC_CLOSING_RAD
        opening_rad = MEAN_DISC_OPENING_RAD
        min_size = MEAN_DISC_MIN_SIZE
    else:  # "cup"
        closing_rad = MEAN_CUP_CLOSING_RAD
        opening_rad = MEAN_CUP_OPENING_RAD
        min_size = MEAN_CUP_MIN_SIZE

    m = mask.astype(bool)
    if closing_rad > 0:
        m = binary_closing(m, footprint=disk(closing_rad))
    if opening_rad > 0:
        m = binary_opening(m, footprint=disk(opening_rad))
    if min_size > 0:
        m = remove_small_objects(m, min_size=min_size)
    m = binary_fill_holes(m)
    m = tiny_contour_smooth(m)
    return m.astype(bool)


def extract_disc_and_cup_from_strokes(strokes: np.ndarray, rm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a stroke mask (annotations only) and retina mask,
    fill each connected component, filter by size, and pick disc/cup.

    Returns:
        cup_bool, disc_bool
    """
    retinal_area = float(rm.sum()) if rm.any() else float(strokes.size)

    lab_img = label(strokes, connectivity=2)
    props = regionprops(lab_img)
    if not props:
        empty = np.zeros_like(strokes, dtype=bool)
        return empty, empty

    candidates: List[Tuple[float, np.ndarray]] = []

    for p in props:
        comp = (lab_img == p.label)
        filled = binary_fill_holes(comp)
        area = float(filled.sum())
        frac = area / retinal_area if retinal_area > 0 else 0.0

        if MIN_AREA_FRAC <= frac <= MAX_AREA_FRAC:
            candidates.append((area, filled))

    if not candidates:
        for p in props:
            comp = (lab_img == p.label)
            filled = binary_fill_holes(comp)
            candidates.append((float(filled.sum()), filled))

    candidates.sort(key=lambda x: x[0], reverse=True)

    disc_filled = candidates[0][1]
    cup_filled = candidates[1][1] if len(candidates) > 1 else np.zeros_like(disc_filled)

    disc_filled = smooth_filled(disc_filled, DISC_CLOSING_RAD, DISC_OPENING_RAD, DISC_MIN_SIZE)
    if cup_filled.any():
        cup_filled = smooth_filled(cup_filled, CUP_CLOSING_RAD, CUP_OPENING_RAD, CUP_MIN_SIZE)

    if cup_filled.sum() > disc_filled.sum():
        disc_filled, cup_filled = cup_filled, disc_filled

    cup_filled = cup_filled & disc_filled

    return cup_filled.astype(bool), disc_filled.astype(bool)


def extract_masks_for_pair(orig_path: Path, ann_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load original + annotated images, compute cup and disc masks.
    """
    orig_rgb = read_rgb(orig_path)
    ann_rgb = read_rgb(ann_path)

    if orig_rgb.shape != ann_rgb.shape:
        ann_rgb = np.array(
            Image.fromarray(ann_rgb).resize((orig_rgb.shape[1], orig_rgb.shape[0]))
        )

    strokes = diff_strokes(orig_rgb, ann_rgb)
    rm = retina_mask(orig_rgb)
    cup_bool, disc_bool = extract_disc_and_cup_from_strokes(strokes, rm)
    return cup_bool, disc_bool


# =========================
# Dataset grouping helpers
# =========================

def parse_name(path: Path) -> Tuple[str, str, int | None]:
    """
    Parse filename into (kind, base, idx):

      - kind = "prime", base=<root>, idx=None     for originals (endswith "prime")
      - kind = "annot", base=<root>, idx=<int>   for annotations (endswith "-n")
      - kind = "other", base=<stem>, idx=None    otherwise

    Matching is case-insensitive.
    """
    stem = path.stem
    lower = stem.lower()

    if lower.endswith("prime"):
        base = stem[:-5]  # strip "prime"
        return "prime", base, None

    m = re.match(r"^(.*?)-(\d+)$", stem, flags=re.IGNORECASE)
    if m:
        base = m.group(1)
        idx = int(m.group(2))
        return "annot", base, idx

    return "other", stem, None


def collect_groups(in_dir: Path) -> Dict[str, Dict[str, object]]:
    """
    Scan in_dir, group originals + annotations by base stem (case-insensitive).

    Returns dict:
      key (str) -> {
        "base": original base name (preserve first seen),
        "prime": Path | None,
        "annots": List[Tuple[int | None, Path]],
      }
    """
    exts = {".jpg", ".jpeg", ".tif", ".tiff"}
    groups: Dict[str, Dict[str, object]] = {}

    for p in in_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue

        kind, base, idx = parse_name(p)
        key = base.lower()

        if key not in groups:
            groups[key] = {"base": base, "prime": None, "annots": []}

        if kind == "prime":
            groups[key]["prime"] = p
        elif kind == "annot":
            groups[key]["annots"].append((idx, p))

    return groups


# =========================
# Main dataset processing
# =========================

def process_dataset(in_dir: Path, out_dir: Path, stem: str) -> None:
    ensure_dir(out_dir)

    fundus_dir = out_dir / "fundus"
    oc_mean_dir = out_dir / "oc_mask_mean"
    od_mean_dir = out_dir / "od_mask_mean"
    ensure_dir(fundus_dir)
    ensure_dir(oc_mean_dir)
    ensure_dir(od_mean_dir)

    groups = collect_groups(in_dir)

    total_roots = 0
    total_annots = 0

    for key, info in sorted(groups.items()):
        prime_path: Path | None = info["prime"]  # type: ignore[assignment]
        annots: List[Tuple[int | None, Path]] = info["annots"]  # type: ignore[assignment]

        if prime_path is None or not annots:
            print(f"[WARN] Skipping base '{info['base']}' (prime or annotations missing).")
            continue

        annots_sorted = sorted(
            annots,
            key=lambda t: (t[0] if t[0] is not None else 9999, t[1].name.lower()),
        )

        base_raw = info["base"]
        out_name = f"{stem}_{base_raw}.png"

        print(f"[INFO] Processing base '{base_raw}' with {len(annots_sorted)} annotations.")

        orig_rgb = read_rgb(prime_path)
        imwrite(fundus_dir / out_name, orig_rgb)

        h, w, _ = orig_rgb.shape
        cup_accum = np.zeros((h, w), dtype=np.float32)
        disc_accum = np.zeros((h, w), dtype=np.float32)
        n_experts = 0

        for j, (_, ann_path) in enumerate(annots_sorted, start=1):
            expert_dir = out_dir / f"expert{j}"
            oc_dir = expert_dir / "oc_mask"
            od_dir = expert_dir / "od_mask"
            ensure_dir(oc_dir)
            ensure_dir(od_dir)

            cup_bool, disc_bool = extract_masks_for_pair(prime_path, ann_path)

            cup_accum += cup_bool.astype(np.float32)
            disc_accum += disc_bool.astype(np.float32)
            n_experts += 1

            oc_png = (cup_bool.astype(np.uint8) * 255)
            od_png = (disc_bool.astype(np.uint8) * 255)

            imwrite(oc_dir / out_name, oc_png)
            imwrite(od_dir / out_name, od_png)

            print(f"  [OK] Expert {j}: {ann_path.name} -> masks saved as {out_name}")

        # ---- Binary consensus + extra smoothing for mean masks ----
        if n_experts > 0:
            cup_prob = cup_accum / float(n_experts)
            disc_prob = disc_accum / float(n_experts)

            cup_cons = cup_prob >= CONSENSUS_THRESH
            disc_cons = disc_prob >= CONSENSUS_THRESH

            # Extra smoothing (cup further smoothed than disc)
            cup_mean_bool = smooth_mean_mask(cup_cons, kind="cup")
            disc_mean_bool = smooth_mean_mask(disc_cons, kind="disc")

            cup_mean_png = (cup_mean_bool.astype(np.uint8) * 255)
            disc_mean_png = (disc_mean_bool.astype(np.uint8) * 255)

            imwrite(oc_mean_dir / out_name, cup_mean_png)
            imwrite(od_mean_dir / out_name, disc_mean_png)

            print(f"  [OK] Consensus mean masks saved as {out_name} (N_experts = {n_experts})")

        total_roots += 1
        total_annots += len(annots_sorted)

    # ===== Summary =====
    print("\n[SUMMARY]")
    print(f"  Fundus images processed : {total_roots}")
    print(f"  Total annotations used  : {total_annots}")
    print(f"  Output root             : {out_dir}")


# =========================
# CLI entry point
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch extract OD/OC masks from annotated fundus images "
                    "using original vs annotated image pairs, with smoothed "
                    "binary consensus mean masks."
    )
    parser.add_argument(
        "--in-dir",
        type=str,
        required=True,
        help="Directory containing original and annotated images (.jpeg/.jpg/.tif).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output root directory (will be created if it does not exist).",
    )
    parser.add_argument(
        "--stem",
        type=str,
        required=True,
        help="Prefix to prepend to all output filenames (e.g., 'Chaksu').",
    )

    args = parser.parse_args()

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    stem = args.stem

    if not in_dir.is_dir():
        raise NotADirectoryError(f"in-dir is not a directory: {in_dir}")

    print(f"[INFO] Input dir : {in_dir}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Stem      : {stem}")
    print(f"[INFO] Consensus threshold: {CONSENSUS_THRESH}")
    print(f"[INFO] Mean disc smoothing: closing={MEAN_DISC_CLOSING_RAD}, opening={MEAN_DISC_OPENING_RAD}")
    print(f"[INFO] Mean cup smoothing : closing={MEAN_CUP_CLOSING_RAD}, opening={MEAN_CUP_OPENING_RAD}")

    process_dataset(in_dir, out_dir, stem)


if __name__ == "__main__":
    main()