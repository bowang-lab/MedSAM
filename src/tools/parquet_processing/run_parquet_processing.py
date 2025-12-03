#!/usr/bin/env python3
# File: src/tools/parquet_processing/run_parquet_processing.py
"""
Script to merge development and eyepacs datasets using the ParquetProcessor API.

Logic:
1. Loads 'images_dev_yolo_split.parquet' (Primary for Dev).
2. Merges 'images_dev.parquet' into it (Secondary for Dev - fills missing fields).
   Result: A complete Dev dataset with splits and metadata.
3. Merges 'images_eyepacs.parquet' into the result.
   Result: A combined dataset containing both Dev and Eyepacs records.
4. Saves the final merged dataset.
"""

import argparse
import logging
from pathlib import Path
from src.tools.parquet_processing.parquet_processor import ParquetProcessor


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Merge Dev and Eyepacs datasets.")
    parser.add_argument("--dev-split", type=Path, required=True, help="Path to images_dev_yolo_split.parquet")
    parser.add_argument("--dev-base", type=Path, required=True, help="Path to images_dev.parquet")
    parser.add_argument("--eyepacs", type=Path, required=True, help="Path to images_eyepacs.parquet")
    parser.add_argument("--out", type=Path, required=True, help="Output path for the merged parquet.")

    args = parser.parse_args()

    # 1. Initialize with the split file (Primary)
    processor = ParquetProcessor()
    processor.load(args.dev_split)

    # 2. Merge the base dev file (Secondary - fills nulls)
    # Priority: dev_split > dev_base
    processor.merge(args.dev_base)

    # 3. Merge Eyepacs
    # Since UIDs should be distinct between Dev and Eyepacs, this acts as an append.
    # If there were overlaps, Dev would take priority.
    processor.merge(args.eyepacs)

    # 4. Summarize
    processor.summarize()

    # 5. Save
    processor.save(
        args.out,
        include_mask_bytes=True,  # Preserve masks
        include_image_bytes=False  # Keep file size small (metadata only)
    )


if __name__ == "__main__":
    main()