#!/usr/bin/env python3
# File: src/tools/combine_parquet.py
"""
Combine (concatenate) two parquet inputs into a single output parquet with streaming IO.

Behavior:
- Rows from --in-a are duplicated (repeated) N times via --repeat-a (default: 2),
  optionally only for rows in split=="train".
- Rows from --in-b are written once.

Robustness:
- Reads via ParquetFile.iter_batches() (safe for nested columns).
- Writes via Image.save_parquet() (stable canonical schema, nested-safe, handles extras).

Options:
- --dedup on|off with --dedup-key:
    - If on, skips duplicates by key.
    - Note: dedup ON will effectively prevent repeated A rows from being duplicated.
- --repeat-a-only-train:
    - If set, only repeats rows from A whose split is "train".
      Non-train rows from A are emitted once.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set

import pyarrow.parquet as pq

from src.imgpipe.image import Image


# -------------------------
# Streaming read (rows -> Image)
# -------------------------


def iter_parquet_files(in_path: Path) -> List[Path]:
    in_path = in_path.resolve()
    if in_path.is_file():
        if in_path.suffix.lower() != ".parquet":
            raise ValueError(f"Expected a .parquet file, got: {in_path}")
        return [in_path]
    if not in_path.is_dir():
        raise FileNotFoundError(f"Input not found: {in_path}")
    files = sorted(in_path.rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"No .parquet files found under: {in_path}")
    return files


def iter_rows_streaming(in_path: Path, *, batch_size: int) -> Iterator[Dict[str, Any]]:
    for f in iter_parquet_files(in_path):
        pf = pq.ParquetFile(str(f))
        for rb in pf.iter_batches(batch_size=int(batch_size)):
            for rec in rb.to_pylist():
                yield rec


def iter_images_streaming(in_path: Path, *, batch_size: int) -> Iterator[Image]:
    for rec in iter_rows_streaming(in_path, batch_size=batch_size):
        yield Image.from_dict(rec)


# -------------------------
# Dedup
# -------------------------


def detect_default_dedup_key() -> str:
    # Works with Image.to_dict() schema. Preferred order for stability.
    return "uid"


def get_dedup_value_from_image(img: Image, key: str) -> Optional[str]:
    v = getattr(img, key, None)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


# -------------------------
# Split helper
# -------------------------


def is_train_split(img: Image) -> bool:
    s = getattr(img, "split", None)
    return (s is not None) and (str(s).strip().lower() == "train")


# -------------------------
# CLI + main
# -------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine two parquet inputs (streaming, nested-safe).")

    p.add_argument("--in-a", type=Path, required=True, help="Parquet A (file or directory of parts).")
    p.add_argument("--in-b", type=Path, required=True, help="Parquet B (file or directory of parts).")
    p.add_argument("--out", type=Path, required=True, help="Output parquet file path.")

    p.add_argument("--repeat-a", type=int, default=2, help="Repeat each row from --in-a this many times (default: 2).")
    p.add_argument(
        "--repeat-a-only-train",
        action="store_true",
        help="If set, only repeat rows from A where split=='train' (others emitted once).",
    )

    p.add_argument("--batch-size", type=int, default=1024, help="Streaming batch size for reading.")
    p.add_argument("--write-batch", type=int, default=1024, help="Batch size for Parquet writing.")

    p.add_argument("--dedup", choices=("off", "on"), default="off")
    p.add_argument("--dedup-key", type=str, default=None)

    p.add_argument("--compression", type=str, default="zstd")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    repeat_a = int(args.repeat_a)
    if repeat_a < 1:
        raise ValueError("--repeat-a must be >= 1")

    repeat_only_train = bool(args.repeat_a_only_train)

    do_dedup = (args.dedup == "on")
    dedup_key = (args.dedup_key or detect_default_dedup_key()).strip()

    if do_dedup and not dedup_key:
        raise ValueError("Dedup is enabled but no dedup key was provided or inferred.")

    logging.info("Repeat A: %d", repeat_a)
    logging.info("Repeat A only train: %s", repeat_only_train)
    logging.info("Dedup: %s", f"ON (key={dedup_key})" if do_dedup else "OFF")
    logging.info("Output: %s", args.out.resolve())

    seen: Set[str] = set()
    n_in_a = 0
    n_in_b = 0
    n_out = 0
    n_dedup_skipped = 0

    def maybe_yield(img: Image) -> bool:
        nonlocal n_dedup_skipped
        if not do_dedup:
            return True
        k = get_dedup_value_from_image(img, dedup_key)
        if k is not None and k in seen:
            n_dedup_skipped += 1
            return False
        if k is not None:
            seen.add(k)
        return True

    def combined_images() -> Iterator[Image]:
        nonlocal n_in_a, n_in_b, n_out

        # A: repeat or dedup
        for img in iter_images_streaming(args.in_a, batch_size=int(args.batch_size)):
            n_in_a += 1

            if do_dedup:
                # Dedup makes repeating meaningless; keep first occurrence only
                if maybe_yield(img):
                    n_out += 1
                    yield img
                continue

            # Not dedup: repeat logic (possibly only for train)
            reps = repeat_a if (not repeat_only_train or is_train_split(img)) else 1
            for _ in range(int(reps)):
                n_out += 1
                yield img

        # B: once, subject to dedup
        for img in iter_images_streaming(args.in_b, batch_size=int(args.batch_size)):
            n_in_b += 1
            if not maybe_yield(img):
                continue
            n_out += 1
            yield img

    # Write using Image's canonical Parquet writer (stable schema, nested-safe, extras handled).
    # Keep bytes OFF by default; combine_parquet is usually metadata-level.
    Image.save_parquet(
        combined_images(),
        args.out,
        drop_none=False,
        include_image_bytes=False,
        include_mask_bytes=False,
        compression=str(args.compression),
        write_batch=int(args.write_batch),
    )

    logging.info("Input A rows scanned: %d", n_in_a)
    logging.info("Input B rows scanned: %d", n_in_b)
    logging.info("Output rows written (logical): %d", n_out)
    if do_dedup:
        logging.info("Dedup rows skipped: %d", n_dedup_skipped)


if __name__ == "__main__":
    main()