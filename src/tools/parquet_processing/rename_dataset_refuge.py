#!/usr/bin/env python3
# File: src/scripts/rename_dataset_names_in_parquet.py
"""
Rename dataset names inside a Parquet file or Parquet dataset directory.

Specifically:
  REFUGE<digit>  ->  REFUGE-1
Examples:
  REFUGE1, REFUGE2, REFUGE9 -> REFUGE-1

This version avoids pyarrow.dataset.Scanner.to_reader() because it can fail with:
  ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs

Instead:
- Streams via pyarrow.parquet.ParquetFile.iter_batches()
- Preserves input Arrow schema exactly
- Writes via ParquetWriter incrementally
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_REFUGE_DIGIT_RE = re.compile(r"^REFUGE\d+$")


def _iter_parquet_files(in_path: Path) -> List[Path]:
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


def _rename_dataset_array(arr: pa.Array) -> pa.Array:
    """
    Vectorized rename:
      dataset == REFUGE<digit>  -> REFUGE-1
      else unchanged

    Handles nulls safely.
    """
    # Ensure string array
    if not pa.types.is_string(arr.type) and not pa.types.is_large_string(arr.type):
        # Best-effort cast (if somehow not string)
        arr = pc.cast(arr, pa.string(), safe=False)

    # Boolean mask for REFUGE<digit> using regex
    # match_substring_regex returns null for null inputs; coalesce to False.
    m = pc.match_substring_regex(arr, r"^REFUGE\d+$")
    m = pc.coalesce(m, pa.scalar(False))

    replaced = pc.if_else(m, pa.scalar("REFUGE-1"), arr)
    return replaced


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rename dataset values in parquet: REFUGE<digit> -> REFUGE-1")
    p.add_argument("--in-parquet", type=Path, required=True, help="Input parquet file OR parquet dataset directory.")
    p.add_argument("--out-parquet", type=Path, required=True, help="Output parquet file path.")
    p.add_argument("--batch-size", type=int, default=2048, help="Batch size for streaming reads.")
    p.add_argument("--dry-run", action="store_true", help="Do not write output; only report counts.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    in_files = _iter_parquet_files(args.in_parquet)
    out_path = args.out_parquet.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine schema from first file; enforce identical schema across all files
    first_pf = pq.ParquetFile(str(in_files[0]))
    schema = first_pf.schema_arrow

    if "dataset" not in schema.names:
        raise ValueError(f"Input schema has no 'dataset' column. Columns: {schema.names}")

    per_ds_before = Counter()
    per_ds_after = Counter()

    writer: Optional[pq.ParquetWriter] = None
    n_rows = 0
    n_changed = 0

    try:
        for f in in_files:
            pf = pq.ParquetFile(str(f))
            this_schema = pf.schema_arrow
            if this_schema != schema:
                raise ValueError(
                    "Schema mismatch across input files.\n"
                    f"First file: {in_files[0]}\nThis file: {f}\n"
                    "Tip: ensure all part files were written with the same writer schema."
                )

            for rb in pf.iter_batches(batch_size=int(args.batch_size)):
                tbl = pa.Table.from_batches([rb], schema=schema)

                ds_arr = tbl["dataset"]
                # Count BEFORE
                for v in pc.drop_null(ds_arr).to_pylist():
                    per_ds_before[str(v)] += 1

                new_ds = _rename_dataset_array(ds_arr)

                # Count how many changed in this batch
                changed_mask = pc.and_(
                    pc.is_valid(ds_arr),
                    pc.not_equal(ds_arr, new_ds),
                )
                # sum(boolean) works after cast to int8
                n_changed += int(pc.sum(pc.cast(changed_mask, pa.int32())).as_py())

                # Apply replacement
                new_tbl = tbl.set_column(tbl.schema.get_field_index("dataset"), "dataset", new_ds)

                # Count AFTER
                for v in pc.drop_null(new_tbl["dataset"]).to_pylist():
                    per_ds_after[str(v)] += 1

                n_rows += new_tbl.num_rows

                if args.dry_run:
                    continue

                if writer is None:
                    writer = pq.ParquetWriter(where=str(out_path), schema=schema, compression="zstd")
                writer.write_table(new_tbl)

    finally:
        if writer is not None:
            writer.close()

    logging.info("Read rows:    %d", n_rows)
    logging.info("Changed rows: %d", n_changed)
    logging.info("Dry run:      %s", bool(args.dry_run))

    def _log_ctr(title: str, ctr: Counter) -> None:
        total = sum(int(v) for v in ctr.values())
        logging.info("%s (total=%d)", title, total)
        for k in sorted(ctr.keys()):
            logging.info("  %-24s %d", k, int(ctr[k]))

    _log_ctr("Per-dataset BEFORE", per_ds_before)
    _log_ctr("Per-dataset AFTER", per_ds_after)

    if not args.dry_run:
        logging.info("Wrote: %s", out_path)


if __name__ == "__main__":
    main()