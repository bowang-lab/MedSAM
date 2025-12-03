#!/usr/bin/env python3
# File: src/tools/parquet_processing/fix_uids_canonical.py
"""
Rewrite UIDs in a Parquet file to match the canonical format: "{Dataset}:{FileStem}".

This repairs datasets where UIDs were randomly generated (UUIDs) instead of being
derived deterministically from the filename. This is critical for merging and splitting.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterator, Optional, Sequence

from src.imgpipe.image import Image


def stream_corrected_images(in_path: Path, target_dataset: Optional[str] = None) -> Iterator[Image]:
    """
    Stream images and fix UIDs.
    If target_dataset is provided, only fixes UIDs for that dataset.
    Otherwise, fixes all images.
    """
    total = 0
    fixed = 0

    for img in Image.iter_parquet(in_path):
        total += 1

        # Filter if user only wants to fix specific dataset (e.g. EYEPACS)
        if target_dataset and img.dataset != target_dataset:
            yield img
            continue

        # We need a valid path to derive the stem
        if not img.image_path:
            # Fallback: if no path, we can't safely derive the canonical ID
            yield img
            continue

        # Canonical logic: Dataset:FilenameStem
        # e.g. "EYEPACS:12345_left"
        current_stem = Path(img.image_path).stem
        canonical_uid = f"{img.dataset}:{current_stem}"

        if img.uid != canonical_uid:
            img.uid = canonical_uid
            fixed += 1

        yield img

    logging.info(f"Scanned {total} images.")
    logging.info(f"Fixed UIDs for {fixed} images.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fix random UIDs to canonical 'Dataset:Stem' format.")
    p.add_argument("--in-parquet", type=Path, required=True, help="Input Parquet file.")
    p.add_argument("--out-parquet", type=Path, required=True, help="Output Parquet file.")
    p.add_argument("--dataset", type=str, default=None, help="Optional: Only fix this dataset (e.g. EYEPACS).")
    p.add_argument("--compression", type=str, default="zstd")

    # Flags to control data preservation
    p.add_argument("--include-image-bytes", action="store_true", help="Preserve embedded image bytes (slower).")
    p.add_argument("--no-mask-bytes", action="store_true", help="Drop embedded mask bytes (saves space).")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    if args.in_parquet.resolve() == args.out_parquet.resolve():
        raise ValueError("Input and output paths must be different to avoid corruption.")

    logging.info(f"Reading: {args.in_parquet}")
    logging.info(f"Target : {args.dataset if args.dataset else 'ALL datasets'}")

    # Use the robust Image class writer
    Image.save_parquet(
        stream_corrected_images(args.in_parquet, target_dataset=args.dataset),
        path=args.out_parquet,
        drop_none=False,
        include_image_bytes=args.include_image_bytes,  # False by default (metadata operation)
        include_mask_bytes=(not args.no_mask_bytes),  # True by default
        compression=args.compression,
        write_batch=1024,
    )

    logging.info(f"Saved to: {args.out_parquet}")


if __name__ == "__main__":
    main()