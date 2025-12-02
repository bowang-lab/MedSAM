#!/usr/bin/env python3
"""
Build a Parquet file of Image metadata from an EYEPACS CSV and images, using Image.save_parquet()
(stable schema, streaming, nested-safe).

Key speed/robustness features:
- Pre-index images in IMG_DIR once (avoid 90k Path.is_file() stats).
- Multiprocessing to construct Image objects.
- Uses Pillow to read (W,H) from headers (avoids imageio/SimpleITK failures).
- Bad/corrupt images are skipped (logged), won't crash the whole job.

Images expected in:
  --root-dir/
    EYEPACS_<image_id>.png
or if --fundus-subdir is provided:
  --root-dir/<fundus-subdir>/EYEPACS_<image_id>.png
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
from PIL import Image as PILImage, ImageFile

from src.imgpipe.enums import Eye
from src.imgpipe.image import Image

# Allow Pillow to parse headers even if the file is slightly truncated.
# (Still may fail on truly corrupt/non-image files; those are skipped.)
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ------------------------- helpers -------------------------

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
    except Exception:
        pass
    try:
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
    """
    Fast header-only size read using Pillow.
    Does not decode full image.
    """
    with PILImage.open(p) as im:
        w, h = im.size
    return int(w), int(h)


def build_image_index(img_dir: Path) -> Dict[str, Path]:
    """
    Map image_id -> path by scanning directory once.
    Expects filenames: EYEPACS_<image_id>.png
    """
    idx: Dict[str, Path] = {}
    for p in img_dir.glob("EYEPACS_*.png"):
        stem = p.stem  # EYEPACS_<id>
        # split once from left
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        image_id = parts[1].strip()
        if image_id:
            idx[image_id] = p
    return idx


# ------------------------- multiprocessing work unit -------------------------

@dataclass(frozen=True)
class WorkItem:
    image_id: str
    img_path: Path
    dataset_name: str
    subject_id: str
    laterality: Optional[str]   # Eye name or None ("OD"/"OS")
    age: Optional[int]
    glaucoma: Optional[bool]
    extras: Dict[str, Any]


def _worker_make_image(w: WorkItem) -> Tuple[str, Optional[Image], Optional[str], Optional[str]]:
    """
    Returns:
      ("ok", image, None, None)
      ("bad", None, image_path, error_string)
    """
    try:
        if not w.img_path.is_file() or w.img_path.stat().st_size <= 0:
            return ("bad", None, str(w.img_path), "Missing or empty file")

        width, height = fast_image_size_pil(w.img_path)

        img = Image.from_path(
            image_path=w.img_path,
            dataset=w.dataset_name,
            subject_id=w.subject_id,
            width=width,
            height=height,
        )

        # Assign metadata
        img.laterality = None
        if w.laterality:
            try:
                img.laterality = Eye[w.laterality]
            except Exception:
                img.laterality = None

        img.age = w.age
        img.glaucoma = w.glaucoma
        img.extras = w.extras

        return ("ok", img, None, None)
    except Exception as e:
        return ("bad", None, str(w.img_path), f"{type(e).__name__}: {e}")


# ------------------------- streaming builder -------------------------

def iter_images_from_csv(
    csv_path: Path,
    img_dir: Path,
    dataset_name: str,
    *,
    mp_workers: int,
    mp_chunksize: int,
    log_every: int,
) -> Iterator[Image]:
    df = pd.read_csv(csv_path, engine="python")
    n_rows = len(df)
    logging.info("CSV rows scanned: %d", n_rows)

    # One directory scan instead of N filesystem stats
    idx = build_image_index(img_dir)
    logging.info("Indexed %d images under: %s", len(idx), img_dir)

    work: List[WorkItem] = []
    missing = 0

    # itertuples is faster than iterrows
    cols = set(df.columns)

    def _get(row, name: str) -> Any:
        return getattr(row, name) if name in cols else None

    # Keep only extras you care about (avoid huge dicts)
    extras_cols = [
        "site_id",
        "case_id",
        "gender",
        "ethnicity",
        "years_with_diabetes",
        "hba1c",
        "cholesterol",
        "triglycerides",
        "insulin_dependent",
        "dr_level",
        "dr_icd10",
        "image_quality",
        "image_quality_factor",
        "assessment_and_recommendation",
        "referral_time_assessment",
    ]

    for row in df.itertuples(index=False):
        image_id_str = normalize_image_id(_get(row, "image_id"))
        if not image_id_str:
            continue

        p = idx.get(image_id_str)
        if p is None:
            missing += 1
            continue

        patient_id = _get(row, "patient_id")
        subject_id = ""
        if patient_id is not None:
            try:
                if not pd.isna(patient_id):
                    subject_id = str(patient_id)
            except Exception:
                subject_id = str(patient_id)

        laterality = infer_eye(_get(row, "image_type"))
        laterality_s = laterality.name if laterality is not None else None

        age = safe_int(_get(row, "age_at_encounter"))
        glaucoma = parse_glaucoma_flag(_get(row, "glaucoma_hx"))

        ethnicity_raw = clean_extras_value(_get(row, "ethnicity"))
        extras: Dict[str, Any] = {}
        for c in extras_cols:
            if c not in cols:
                continue
            v = clean_extras_value(_get(row, c))
            if c == "ethnicity":
                extras["ethnicity_raw"] = ethnicity_raw
            else:
                extras[c] = v

        work.append(
            WorkItem(
                image_id=image_id_str,
                img_path=p,
                dataset_name=dataset_name,
                subject_id=subject_id,
                laterality=laterality_s,
                age=age,
                glaucoma=glaucoma,
                extras=extras,
            )
        )

    logging.info("Work items: %d (missing images by id: %d)", len(work), missing)

    if mp_workers <= 0 or mp_workers == 1:
        bad = 0
        for j, w in enumerate(work, start=1):
            status, img, bad_path, err = _worker_make_image(w)
            if status == "ok" and img is not None:
                if log_every and (j % log_every == 0):
                    logging.info("Processed %d/%d (bad=%d)", j, len(work), bad)
                yield img
            else:
                bad += 1
                if bad_path:
                    logging.warning("Bad image skipped: %s | %s", bad_path, err)
        return

    # Multiprocessing path
    ctx = mp.get_context("fork")  # linux HPC; fastest startup
    bad_paths: List[str] = []
    bad = 0
    processed = 0

    logging.info("Multiprocessing enabled: workers=%d, tasks=%d", mp_workers, len(work))

    with ctx.Pool(processes=mp_workers, maxtasksperchild=2000) as pool:
        for status, img, bad_path, err in pool.imap_unordered(_worker_make_image, work, chunksize=mp_chunksize):
            processed += 1
            if status == "ok" and img is not None:
                if log_every and (processed % log_every == 0):
                    logging.info("Processed %d/%d (bad=%d)", processed, len(work), bad)
                yield img
            else:
                bad += 1
                if bad_path:
                    bad_paths.append(bad_path)
                    logging.warning("Bad image skipped: %s | %s", bad_path, err)

    if bad_paths:
        sidecar = img_dir / "bad_images_skipped.txt"
        try:
            sidecar.write_text("\n".join(bad_paths) + "\n")
            logging.info("Wrote bad image list: %s (n=%d)", sidecar, len(bad_paths))
        except Exception:
            logging.info("Failed to write bad image list (n=%d)", len(bad_paths))

    logging.info("Finished: processed=%d bad_skipped=%d", processed, bad)


# ------------------------- main -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build EYEPACS Image parquet from CSV + images (fast, CPU, robust).")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--root-dir", type=Path, required=True)
    p.add_argument("--fundus-subdir", type=str, default=None)
    p.add_argument("--dataset-name", type=str, default="EYEPACS")
    p.add_argument("--out-parquet", type=Path, required=True)

    # speed knobs
    p.add_argument("--mp-workers", type=int, default=0, help="0 disables multiprocessing; else worker count.")
    p.add_argument("--mp-chunksize", type=int, default=256)
    p.add_argument("--write-batch", type=int, default=8192)
    p.add_argument("--compression", type=str, default="zstd")
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--log-every", type=int, default=10000)

    # typically OFF
    p.add_argument("--include-image-bytes", action="store_true")
    p.add_argument("--include-mask-bytes", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    img_dir = args.root_dir if not args.fundus_subdir else (args.root_dir / args.fundus_subdir)
    if not img_dir.is_dir():
        raise FileNotFoundError(f"IMG_DIR not found: {img_dir}")

    logging.info("CSV:      %s", args.csv)
    logging.info("IMG_DIR:  %s", img_dir)
    logging.info("OUT:      %s", args.out_parquet)
    logging.info("DATASET:  %s", args.dataset_name)
    logging.info("mp_workers=%d write_batch=%d", int(args.mp_workers), int(args.write_batch))

    images_iter = iter_images_from_csv(
        csv_path=args.csv,
        img_dir=img_dir,
        dataset_name=str(args.dataset_name),
        mp_workers=int(args.mp_workers),
        mp_chunksize=int(args.mp_chunksize),
        log_every=int(args.log_every),
    )

    Image.save_parquet(
        images_iter,
        path=args.out_parquet,
        drop_none=False,
        include_image_bytes=bool(args.include_image_bytes),
        include_mask_bytes=bool(args.include_mask_bytes),
        compression=str(args.compression),
        write_batch=int(args.write_batch),
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()