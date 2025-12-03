#!/usr/bin/env python3
"""
Build a Parquet file of Image metadata from an EYEPACS CSV and images.
Uses Image.save_parquet() for canonical schema consistency and streaming IO.

Key Features:
- Pre-indexes image files for O(1) lookup.
- Uses Multiprocessing to process rows and read image headers.
- Uses Pillow for fast header-only size reading.
- Handles metadata parsing (Eye, Glaucoma, Extras) robustly.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
from PIL import Image as PILImage, ImageFile

# Ensure these match your project structure
from src.imgpipe.enums import Eye
from src.imgpipe.image import Image

# Allow Pillow to parse headers even if the file is slightly truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ------------------------- Helpers -------------------------

def infer_eye(image_type: object) -> Optional[Eye]:
    if not isinstance(image_type, str):
        return None
    s = image_type.strip().lower()
    if "right" in s or " od" in s or s.startswith("od"):
        return Eye.OD
    if "left" in s or " os" in s or s.startswith("os"):
        return Eye.OS
    return None


def safe_int(x: object) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def parse_glaucoma_flag(glaucoma_hx: object) -> Optional[bool]:
    try:
        if pd.isna(glaucoma_hx):
            return None
    except Exception:
        pass
    if glaucoma_hx is None:
        return None

    s = str(glaucoma_hx).strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return True
    if s in {"no", "n", "false", "0"}:
        return False
    return None


def clean_extras_value(v: object) -> object:
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def normalize_image_id(image_id: object) -> Optional[str]:
    try:
        if pd.isna(image_id):
            return None
    except Exception:
        pass
    if image_id is None:
        return None
    if isinstance(image_id, str):
        s = image_id.strip()
        return s or None
    try:
        return str(int(image_id))
    except Exception:
        s = str(image_id).strip()
        return s or None


def fast_image_size_pil(p: Path) -> Tuple[int, int]:
    """Fast header-only size read using Pillow."""
    with PILImage.open(p) as im:
        w, h = im.size
    return int(w), int(h)


def build_image_index(img_dir: Path) -> Dict[str, Path]:
    """
    Map image_id -> path by scanning directory once.
    Expects filenames like: EYEPACS_<image_id>.png
    """
    idx: Dict[str, Path] = {}
    # Scan for common extensions
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for p in img_dir.glob(ext):
            stem = p.stem  # e.g. "EYEPACS_123_h"
            # Logic: assume <dataset>_<id> or just <id> depending on your naming convention
            # Here we follow the logic: EYEPACS_<id>
            parts = stem.split("_", 1)
            if len(parts) == 2:
                image_id = parts[1].strip()
                if image_id:
                    idx[image_id] = p
            else:
                # Fallback: if filename is just the ID
                idx[stem] = p
    return idx


# ------------------------- Multiprocessing Work Unit -------------------------

@dataclass(frozen=True)
class WorkItem:
    image_id: str
    img_path: Path
    dataset_name: str
    subject_id: str
    laterality: Optional[str]  # "OD" or "OS"
    age: Optional[int]
    glaucoma: Optional[bool]
    extras: Dict[str, Any]


def _worker_make_image(w: WorkItem) -> Tuple[str, Optional[Image], Optional[str], Optional[str]]:
    """
    Worker function to construct an Image object.
    Returns: ("ok", image, None, None) or ("bad", None, path, error)
    """
    try:
        if not w.img_path.is_file() or w.img_path.stat().st_size <= 0:
            return ("bad", None, str(w.img_path), "Missing or empty file")

        width, height = fast_image_size_pil(w.img_path)

        # Create Image using the factory method from Image class
        img = Image.from_path(
            image_path=w.img_path,
            dataset=w.dataset_name,
            subject_id=w.subject_id,
            width=width,
            height=height,
            uid=f"{w.dataset_name}:{w.img_path.stem}",
        )

        # Set Metadata
        img.laterality = None
        if w.laterality:
            try:
                img.laterality = Eye[w.laterality]
            except Exception:
                pass

        img.age = w.age
        img.glaucoma = w.glaucoma
        img.extras = w.extras

        return ("ok", img, None, None)
    except Exception as e:
        return ("bad", None, str(w.img_path), f"{type(e).__name__}: {e}")


# ------------------------- Streaming Builder -------------------------

def iter_images_from_csv(
        csv_path: Path,
        img_dir: Path,
        dataset_name: str,
        *,
        mp_workers: int,
        mp_chunksize: int,
        log_every: int,
) -> Iterator[Image]:
    logging.info("Reading CSV: %s", csv_path)
    df = pd.read_csv(csv_path, engine="python")
    n_rows = len(df)
    logging.info("CSV rows: %d", n_rows)

    logging.info("Indexing images in: %s", img_dir)
    idx = build_image_index(img_dir)
    logging.info("Indexed %d images.", len(idx))

    work: List[WorkItem] = []
    missing_count = 0

    # Columns we want to capture in 'extras'
    extras_cols = [
        "site_id", "case_id", "gender", "ethnicity", "years_with_diabetes",
        "hba1c", "cholesterol", "triglycerides", "insulin_dependent",
        "dr_level", "dr_icd10", "image_quality", "image_quality_factor",
        "assessment_and_recommendation"
    ]

    # Optimization: pre-check column existence
    cols = set(df.columns)
    valid_extras_cols = [c for c in extras_cols if c in cols]

    def _get(row, name):
        return getattr(row, name) if name in cols else None

    for row in df.itertuples(index=False):
        raw_id = _get(row, "image_id")
        image_id_str = normalize_image_id(raw_id)
        if not image_id_str:
            continue

        p = idx.get(image_id_str)
        if p is None:
            missing_count += 1
            continue

        # Subject ID
        pat_id = _get(row, "patient_id")
        subject_id = str(pat_id) if (pat_id is not None and not pd.isna(pat_id)) else ""

        # Metadata
        lat_enum = infer_eye(_get(row, "image_type"))
        lat_str = lat_enum.name if lat_enum else None

        age = safe_int(_get(row, "age_at_encounter"))
        glaucoma = parse_glaucoma_flag(_get(row, "glaucoma_hx"))

        # Extras
        extras = {}
        for c in valid_extras_cols:
            val = clean_extras_value(_get(row, c))
            if val is not None:
                extras[c] = val

        work.append(WorkItem(
            image_id=image_id_str,
            img_path=p,
            dataset_name=dataset_name,
            subject_id=subject_id,
            laterality=lat_str,
            age=age,
            glaucoma=glaucoma,
            extras=extras
        ))

    logging.info("Work items prepared: %d (Missing images: %d)", len(work), missing_count)

    # Serial Execution
    if mp_workers <= 0 or mp_workers == 1:
        bad_count = 0
        for i, w in enumerate(work, 1):
            status, img, _, _ = _worker_make_image(w)
            if status == "ok" and img:
                if log_every and i % log_every == 0:
                    logging.info("Processed %d/%d", i, len(work))
                yield img
            else:
                bad_count += 1
        return

    # Parallel Execution
    ctx = mp.get_context("fork")
    bad_count = 0
    processed_count = 0
    bad_paths = []

    logging.info("Starting pool with %d workers...", mp_workers)
    with ctx.Pool(processes=mp_workers, maxtasksperchild=2000) as pool:
        for status, img, bad_path, err in pool.imap_unordered(_worker_make_image, work, chunksize=mp_chunksize):
            processed_count += 1
            if status == "ok" and img:
                if log_every and processed_count % log_every == 0:
                    logging.info("Processed %d/%d (bad=%d)", processed_count, len(work), bad_count)
                yield img
            else:
                bad_count += 1
                if bad_path:
                    bad_paths.append(f"{bad_path} | {err}")

    if bad_paths:
        log_file = img_dir / "bad_images_log.txt"
        try:
            log_file.write_text("\n".join(bad_paths))
            logging.warning("Wrote %d bad image errors to %s", len(bad_paths), log_file)
        except Exception:
            pass

    logging.info("Finished. Total: %d, Bad/Skipped: %d", processed_count, bad_count)


# ------------------------- Main -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build EYEPACS Parquet using Image class writer.")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--root-dir", type=Path, required=True)
    p.add_argument("--fundus-subdir", type=str, default=None)
    p.add_argument("--dataset-name", type=str, default="EYEPACS")
    p.add_argument("--out-parquet", type=Path, required=True)

    p.add_argument("--mp-workers", type=int, default=16)
    p.add_argument("--mp-chunksize", type=int, default=256)
    p.add_argument("--write-batch", type=int, default=8192)
    p.add_argument("--compression", type=str, default="zstd")

    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--log-every", type=int, default=10000)

    # Consistency args matching Image.save_parquet signature
    p.add_argument("--include-image-bytes", action="store_true", help="Embed raw image bytes in Parquet.")
    p.add_argument("--include-mask-bytes", action="store_true", help="Embed mask bytes (if any).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    img_dir = args.root_dir
    if args.fundus_subdir:
        img_dir = img_dir / args.fundus_subdir

    logging.info("Configuration:")
    logging.info("  CSV: %s", args.csv)
    logging.info("  Root: %s", img_dir)
    logging.info("  Out: %s", args.out_parquet)
    logging.info("  Workers: %d", args.mp_workers)

    images_generator = iter_images_from_csv(
        csv_path=args.csv,
        img_dir=img_dir,
        dataset_name=args.dataset_name,
        mp_workers=args.mp_workers,
        mp_chunksize=args.mp_chunksize,
        log_every=args.log_every,
    )

    # Use the consistent class method for writing
    Image.save_parquet(
        images_generator,
        path=args.out_parquet,
        drop_none=False,  # Keep schema consistent with full definition
        include_image_bytes=args.include_image_bytes,
        include_mask_bytes=args.include_mask_bytes,
        compression=args.compression,
        write_batch=args.write_batch,
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()