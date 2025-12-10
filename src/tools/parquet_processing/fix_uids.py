#!/usr/bin/env python3
# File: src/tools/parquet_processing/fix_uids.py
"""
Fix UIDs in a Parquet file to match the canonical format: "{Dataset}:{FileStem}".

This ensures consistency between datasets created via CSV (which might lack prefixes)
and datasets created via directory scanning (which use Dataset:Stem).

Usage:
    python -m src.tools.parquet_processing.fix_uids \
        --in-parquet input.parquet \
        --out-parquet output.parquet \
        [--dataset EYEPACS]  # Optional: only fix this dataset
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterator, Optional, Sequence

from src.imgpipe.image import Image


def stream_fixed_images(
    in_path: Path,
    target_dataset: Optional[str] = None,
) -> Iterator[Image]:
    """
    Stream images from input, updating the UID to f"{dataset}:{stem}" if it differs.
    
    Args:
        in_path: Input Parquet file or directory.
        target_dataset: If provided, only fix UIDs for this dataset. Otherwise, fix all.
    """
    total_count = 0
    fixed_count = 0
    skipped_count = 0

    for img in Image.iter_parquet(in_path):
        total_count += 1

        # Filter by dataset if specified
        if target_dataset and img.dataset != target_dataset:
            yield img
            continue

        # We need a dataset name to form the UID
        if not img.dataset:
            skipped_count += 1
            yield img
            continue

        # Determine canonical stem
        # 1. Prefer actual filename stem
        if img.image_path and str(img.image_path) != "." and img.image_path.name:
            stem = img.image_path.stem
        # 2. Fallback to patient_id if image_path is missing/placeholder
        elif img.patient_id:
            stem = img.patient_id
        # 3. Fallback to existing UID's suffix if possible
        else:
            stem = img.uid
            skipped_count += 1

        canonical_uid = f"{img.dataset}:{stem}"

        # Update if necessary
        if img.uid != canonical_uid:
            img.uid = canonical_uid
            fixed_count += 1

        yield img

    logging.info(f"Processed {total_count} images.")
    logging.info(f"Updated UIDs for {fixed_count} images.")
    if skipped_count > 0:
        logging.info(f"Skipped {skipped_count} images (missing dataset or path).")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enforce 'Dataset:Stem' UID format in a Parquet file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Fix all UIDs in a parquet file
    python -m src.tools.parquet_processing.fix_uids \\
        --in-parquet data/images.parquet \\
        --out-parquet data/images_fixed.parquet

    # Only fix UIDs for a specific dataset
    python -m src.tools.parquet_processing.fix_uids \\
        --in-parquet data/images.parquet \\
        --out-parquet data/images_fixed.parquet \\
        --dataset EYEPACS
        """,
    )
    p.add_argument("--in-parquet", type=Path, required=True, help="Input Parquet file or directory.")
    p.add_argument("--out-parquet", type=Path, required=True, help="Output Parquet file.")
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional: Only fix UIDs for this dataset (e.g., EYEPACS). If not provided, fixes all.",
    )
    p.add_argument("--compression", type=str, default="zstd", help="Parquet compression codec.")
    p.add_argument("--write-batch", type=int, default=1024, help="Batch size for writing.")
    
    # Flags to control data preservation
    p.add_argument(
        "--include-image-bytes",
        action="store_true",
        help="Preserve embedded image bytes (slower, larger output).",
    )
    p.add_argument(
        "--no-mask-bytes",
        action="store_true",
        help="Drop embedded mask bytes (saves space).",
    )
    
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    logging.info(f"Fixing UIDs from: {args.in_parquet}")
    logging.info(f"Target dataset: {args.dataset if args.dataset else 'ALL'}")
    logging.info(f"Saving to: {args.out_parquet}")

    if args.in_parquet.resolve() == args.out_parquet.resolve():
        raise ValueError("Input and output paths must be different to avoid corruption.")

    Image.save_parquet(
        stream_fixed_images(args.in_parquet, target_dataset=args.dataset),
        path=args.out_parquet,
        drop_none=False,
        include_image_bytes=args.include_image_bytes,
        include_mask_bytes=(not args.no_mask_bytes),
        compression=args.compression,
        write_batch=args.write_batch,
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()
