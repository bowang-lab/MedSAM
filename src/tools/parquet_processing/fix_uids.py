#!/usr/bin/env python3
# File: src/tools/parquet_processing/fix_uids.py
"""
Fix UIDs in a Parquet file to match the canonical format: "{dataset}:{stem}".

This ensures consistency between datasets created via CSV (which might lack prefixes)
and datasets created via directory scanning (which use Dataset:Stem).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterator, Optional, Sequence

from src.imgpipe.image import Image


def stream_fixed_images(in_path: Path) -> Iterator[Image]:
    """
    Stream images from input, updating the UID to f"{dataset}:{stem}" if it differs.
    """
    total_count = 0
    fixed_count = 0

    for img in Image.iter_parquet(in_path):
        total_count += 1

        # We need both a dataset name and a reliable stem to form the UID
        if not img.dataset:
            # Cannot standardize without a dataset name; yield as-is
            yield img
            continue

        # Determine canonical stem
        # 1. Prefer actual filename stem
        if img.image_path and str(img.image_path) != "." and img.image_path.name:
            stem = img.image_path.stem
        # 2. Fallback to patient_id if image_path is missing/placeholder
        elif img.patient_id:
            stem = img.patient_id
        # 3. Fallback to existing UID's suffix if possible (risky, but valid fallback)
        else:
            stem = img.uid

        canonical_uid = f"{img.dataset}:{stem}"

        # Update if necessary
        if img.uid != canonical_uid:
            img.uid = canonical_uid
            fixed_count += 1

        yield img

    logging.info(f"Processed {total_count} images. Updated UIDs for {fixed_count} images.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enforce 'Dataset:Stem' UID format in a Parquet file.")
    p.add_argument("--in-parquet", type=Path, required=True, help="Input Parquet file or directory.")
    p.add_argument("--out-parquet", type=Path, required=True, help="Output Parquet file.")
    p.add_argument("--compression", type=str, default="zstd")
    p.add_argument("--write-batch", type=int, default=1024)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    logging.info(f"Fixing UIDs from: {args.in_parquet}")
    logging.info(f"Saving to:        {args.out_parquet}")

    if args.in_parquet.resolve() == args.out_parquet.resolve():
        raise ValueError("Input and output paths must be different.")

    Image.save_parquet(
        stream_fixed_images(args.in_parquet),
        path=args.out_parquet,
        drop_none=False,
        include_image_bytes=False,
        # Assuming this is a metadata fix; keep false to save time/space unless embedding is needed
        include_mask_bytes=True,  # Preserve masks if they exist
        compression=args.compression,
        write_batch=args.write_batch,
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()