#!/usr/bin/env python3
"""
Batch script to build OD/OC masks from LabelMe-style JSON annotations
for the GRAPE dataset, placing final outputs directly into:

  out_dir/fundus
  out_dir/oc_mask
  out_dir/od_mask

You provide:
  --orig-dir : directory containing original fundus images (any common type),
               possibly nested. Used only as fallback if JSON lacks imageData.
  --json-dir : directory containing JSON annotation files, possibly nested.
               Each JSON file has the SAME STEM as its corresponding image.
               Example:
                   orig-dir/.../1_OD_1.jpg
                   json-dir/.../1_OD_1.json
  --out-dir  : output root
  --stem     : prefix to prepend to all output filenames

JSON assumptions:
- JSON has "shapes": list of polygon objects
- Each polygon has:
    - "label": "OD" or "OC" (case-insensitive)
    - "points": [[x1,y1], [x2,y2], ...]
    - "shape_type": "polygon"
- imageHeight / imageWidth are present (fallback to image size if not).
- "imageData" may be present and base64-encoded; if present, we use it.
- No smoothing, no morphology: masks are rasterized exactly from polygons.

Outputs:
- Fundus saved as PNG:
      out_dir/fundus/{stem}_{base}.png
- Cup mask (OC) as PNG:
      out_dir/oc_mask/{stem}_{base}.png
- Disc mask (OD) as PNG:
      out_dir/od_mask/{stem}_{base}.png
"""

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
from imageio.v2 import imwrite


# =========================
# Low-level helpers
# =========================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_rgb(path: Path) -> np.ndarray:
    """Read an image from disk and return RGB uint8 (H, W, 3)."""
    im = Image.open(path).convert("RGB")
    return np.array(im)


def decode_image_data(image_data_b64: str) -> np.ndarray:
    """Decode LabelMe 'imageData' base64 string into RGB uint8 array."""
    raw = base64.b64decode(image_data_b64)
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(im)


def rasterize_polygons(
    polygons: List[List[Tuple[float, float]]],
    height: int,
    width: int,
) -> np.ndarray:
    """
    Rasterize a list of polygons into a binary mask.
    Polygons are lists of (x, y) points.
    Returns (H, W) bool mask.
    """
    if not polygons:
        return np.zeros((height, width), dtype=bool)

    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    for pts in polygons:
        if len(pts) >= 3:
            draw.polygon(pts, outline=1, fill=1)

    return np.array(canvas, dtype=bool)


def load_json_and_masks(
    json_path: Path,
    orig_fallback_path: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load JSON. Return (fundus_rgb, cup_mask_bool, disc_mask_bool)
    with NO smoothing.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Fundus image: prefer imageData if present
    image_data_b64 = data.get("imageData", None)
    if isinstance(image_data_b64, str) and len(image_data_b64) > 0:
        fundus_rgb = decode_image_data(image_data_b64)
    else:
        if orig_fallback_path is None or not orig_fallback_path.exists():
            raise FileNotFoundError(
                f"No imageData in JSON and no original file found for {json_path.stem}"
            )
        fundus_rgb = read_rgb(orig_fallback_path)

    h, w, _ = fundus_rgb.shape
    height = int(data.get("imageHeight", h))
    width = int(data.get("imageWidth", w))

    # If JSON dimensions differ, resize fundus to match JSON
    if (height, width) != (h, w):
        fundus_rgb = np.array(
            Image.fromarray(fundus_rgb).resize((width, height))
        )
        h, w = height, width

    shapes = data.get("shapes", [])
    od_polys: List[List[Tuple[float, float]]] = []
    oc_polys: List[List[Tuple[float, float]]] = []

    for sh in shapes:
        label = str(sh.get("label", "")).strip().lower()
        shape_type = str(sh.get("shape_type", "")).strip().lower()
        if shape_type != "polygon":
            continue

        pts_raw = sh.get("points", [])
        pts: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in pts_raw]

        if label == "od":
            od_polys.append(pts)
        elif label == "oc":
            oc_polys.append(pts)

    disc_bool = rasterize_polygons(od_polys, height, width)
    cup_bool  = rasterize_polygons(oc_polys, height, width)

    # Basic sanity: enforce cup inside disc if both exist
    if cup_bool.any() and disc_bool.any():
        if cup_bool.sum() > disc_bool.sum():
            disc_bool, cup_bool = cup_bool, disc_bool
        cup_bool = cup_bool & disc_bool

    return fundus_rgb, cup_bool.astype(bool), disc_bool.astype(bool)


# =========================
# Dataset helpers
# =========================

def list_images_recursive(root: Path, exts: set[str]) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def build_image_index(orig_dir: Path) -> Dict[str, Path]:
    """
    Index original images by lowercase stem (first match wins).
    Used only as fallback if JSON lacks imageData.
    """
    exts = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".bmp"}
    orig_files = list_images_recursive(orig_dir, exts)
    idx: Dict[str, Path] = {}
    for p in sorted(orig_files, key=lambda x: x.as_posix().lower()):
        key = p.stem.lower()
        if key not in idx:
            idx[key] = p
    return idx


def list_jsons_recursive(json_dir: Path) -> List[Path]:
    return sorted(
        [p for p in json_dir.rglob("*.json") if p.is_file()],
        key=lambda x: x.as_posix().lower()
    )


# =========================
# Main processing
# =========================

def process_dataset(orig_dir: Path, json_dir: Path, out_dir: Path, stem: str) -> None:
    ensure_dir(out_dir)

    fundus_dir = out_dir / "fundus"
    oc_dir = out_dir / "oc_mask"
    od_dir = out_dir / "od_mask"
    ensure_dir(fundus_dir)
    ensure_dir(oc_dir)
    ensure_dir(od_dir)

    img_index = build_image_index(orig_dir)
    json_files = list_jsons_recursive(json_dir)

    processed = 0

    for json_path in json_files:
        key = json_path.stem.lower()
        orig_fallback = img_index.get(key)

        try:
            fundus_rgb, cup_bool, disc_bool = load_json_and_masks(
                json_path, orig_fallback
            )
        except Exception as e:
            print(f"[WARN] Skipping '{json_path.name}': {e}")
            continue

        out_name = f"{stem}_{json_path.stem}.png"

        # Save fundus PNG
        imwrite(fundus_dir / out_name, fundus_rgb)

        # Save masks
        oc_png = (cup_bool.astype(np.uint8) * 255)
        od_png = (disc_bool.astype(np.uint8) * 255)
        imwrite(oc_dir / out_name, oc_png)
        imwrite(od_dir / out_name, od_png)

        print(f"[OK] {json_path.name} -> {out_name}")
        processed += 1

    print("\n[SUMMARY]")
    print(f"  JSON files scanned      : {len(json_files)}")
    print(f"  Images processed        : {processed}")
    print(f"  Output root             : {out_dir}")


# =========================
# CLI entry point
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build OD/OC masks from GRAPE LabelMe-style JSON polygons, "
                    "using JSON imageData when present, with NO smoothing. "
                    "Outputs go directly into fundus/oc_mask/od_mask."
    )
    parser.add_argument(
        "--orig-dir",
        type=str,
        required=True,
        help="Directory containing original fundus images (fallback if JSON lacks imageData).",
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        required=True,
        help="Directory containing JSON annotation files. Each JSON stem matches image stem.",
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
        help="Prefix to prepend to all output filenames (e.g., 'Grape').",
    )

    args = parser.parse_args()

    orig_dir = Path(args.orig_dir).resolve()
    json_dir = Path(args.json_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    stem = args.stem

    if not orig_dir.is_dir():
        raise NotADirectoryError(f"orig-dir is not a directory: {orig_dir}")
    if not json_dir.is_dir():
        raise NotADirectoryError(f"json-dir is not a directory: {json_dir}")

    print(f"[INFO] Orig dir : {orig_dir}")
    print(f"[INFO] JSON dir : {json_dir}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Stem      : {stem}")

    process_dataset(orig_dir, json_dir, out_dir, stem)


if __name__ == "__main__":
    main()