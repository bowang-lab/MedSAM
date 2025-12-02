#!/usr/bin/env python3
# File: src/tools/parquet_processing/rename_dataset_refuge.py
"""
Rename dataset names inside a Parquet file or directory using the canonical Image class.

Logic:
  If dataset matches regex r"^REFUGE\d+$" (e.g. REFUGE1, REFUGE2) -> Rename to "REFUGE-1"
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterator, Optional, Sequence

from src.imgpipe.image import Image

# Regex to catch REFUGE1, REFUGE2, ... REFUGE99
REFUGE_PATTERN = re.compile(r"^REFUGE\d+$", re.IGNORECASE)


def stream_renamed_images(in_path: Path) -> Iterator[Image]:
    """
    Stream images, modifying the dataset field if it matches the pattern.
    """
    for img in Image.iter_parquet(in_path):
        ds = getattr(img, "dataset", "")
        if ds and REFUGE_PATTERN.match(ds):
            img.dataset = "REFUGE-1"
        yield img


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rename REFUGE<N> -> REFUGE-1 consistently.")
    p.add_argument("--in-parquet", type=Path, required=True, help="Input file or directory.")
    p.add_argument("--out-parquet", type=Path, required=True, help="Output parquet file.")
    p.add_argument("--compression", type=str, default="zstd")
    p.add_argument("--write-batch", type=int, default=1024)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    logging.info(f"Processing: {args.in_parquet}")
    logging.info(f"Output:     {args.out_parquet}")

    # Use the robust class method for writing
    # This ensures consistency with create_images, merge_smart, etc.
    Image.save_parquet(
        stream_renamed_images(args.in_parquet),
        path=args.out_parquet,
        drop_none=False,
        include_image_bytes=False, # Usually explicit renaming implies metadata op
        include_mask_bytes=True,   # Keep masks if they exist
        compression=args.compression,
        write_batch=args.write_batch,
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()