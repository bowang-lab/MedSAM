#!/usr/bin/env python3
# File: src/tools/process_parquet/merge_smart.py
"""
Intelligently merge two parquet files of Image records.

Logic:
1. Loads 'Input B' (secondary) into a memory map keyed by UID.
2. Streams 'Input A' (primary).
3. If an image from A exists in B (by UID):
   - Merges fields: A's value takes priority if present; B's value fills A's nulls.
   - Merges 'extras' dictionaries: B.extras + A.extras (A overwrites B keys).
   - Removes the entry from B's map (marking it as handled).
4. Writes the merged/original A record.
5. After A is exhausted, writes any remaining records from B (those not found in A).

Usage:
  python -m src.tools.process_parquet.merge_smart \
    --in-a data/primary.parquet \
    --in-b data/secondary.parquet \
    --out data/merged.parquet
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

from src.imgpipe.image import Image


def intelligent_merge(primary: Image, secondary: Image) -> Image:
    """
    Merge secondary fields into primary in-place.
    Rule: Primary wins if not None. If Primary is None, take Secondary.
    Special: 'extras' are dict-merged (Primary updates Secondary).
    """
    for field in dataclasses.fields(Image):
        name = field.name
        val_a = getattr(primary, name)
        val_b = getattr(secondary, name)

        # 1. Handle Dictionary 'extras' specifically for smarter merging
        if name == "extras":
            # Start with B's extras (if any), then update with A's
            merged_extras = val_b.copy() if val_b else {}
            if val_a:
                merged_extras.update(val_a)
            setattr(primary, name, merged_extras)
            continue

        # 2. General Field Merging
        # If A is None and B is not None, fill A with B.
        # If A has value, it keeps it (Priority A).
        if val_a is None and val_b is not None:
            setattr(primary, name, val_b)

    return primary


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Intelligently merge two Parquet datasets.")

    parser.add_argument("--in-a", type=Path, required=True,
                        help="Primary input (Parquet file or dir). Priority on conflict.")
    parser.add_argument("--in-b", type=Path, required=True, help="Secondary input (Parquet file or dir). Fills nulls.")
    parser.add_argument("--out", type=Path, required=True, help="Output Parquet file.")
    parser.add_argument("--key", type=str, default="uid", help="Unique field to match images (default: uid).")
    parser.add_argument("--batch-size", type=int, default=4096, help="Read batch size.")
    parser.add_argument("--write-batch", type=int, default=1024, help="Write batch size.")
    parser.add_argument("--compression", type=str, default="zstd", help="Output compression.")

    args = parser.parse_args(argv)

    match_key = args.key

    # ---------------------------------------------------------
    # 1. Load Input B into Memory Map (O(1) Access)
    # ---------------------------------------------------------
    logging.info(f"Loading secondary dataset (B) from {args.in_b} ...")
    b_map: Dict[str, Image] = {}
    b_count = 0

    for img in Image.iter_parquet(args.in_b, batch_size=args.batch_size):
        k = getattr(img, match_key, None)
        if k:
            b_map[str(k)] = img
        b_count += 1

    logging.info(f"Loaded {len(b_map)} unique records from B (scanned {b_count}).")

    # ---------------------------------------------------------
    # 2. Stream Input A, Merge, and Yield
    # ---------------------------------------------------------
    def merged_generator() -> Iterator[Image]:
        logging.info(f"Streaming primary dataset (A) from {args.in_a} and merging...")
        a_count = 0
        merged_count = 0

        # Stream A
        for img_a in Image.iter_parquet(args.in_a, batch_size=args.batch_size):
            a_count += 1
            k = getattr(img_a, match_key, None)

            # If exists in B, merge and pop from B (so we don't write it again later)
            if k and str(k) in b_map:
                img_b = b_map.pop(str(k))
                intelligent_merge(img_a, img_b)
                merged_count += 1

            yield img_a

        logging.info(f"Processed {a_count} records from A. Merged {merged_count} overlaps.")

        # ---------------------------------------------------------
        # 3. Yield Remaining B (disjoint set)
        # ---------------------------------------------------------
        remaining_b = len(b_map)
        if remaining_b > 0:
            logging.info(f"Writing {remaining_b} remaining records from B (not found in A)...")
            for img_b in b_map.values():
                yield img_b

    # ---------------------------------------------------------
    # 4. Write Output
    # ---------------------------------------------------------
    logging.info(f"Writing merged dataset to {args.out}...")
    Image.save_parquet(
        merged_generator(),
        args.out,
        drop_none=False,  # Essential for schema stability
        include_image_bytes=False,  # Assuming metadata merge; adjust if image bytes are needed
        include_mask_bytes=True,  # Preserve masks if present
        compression=args.compression,
        write_batch=args.write_batch
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()