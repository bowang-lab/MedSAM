#!/usr/bin/env python3
# File: src/tools/parquet_processing/run_process_predictions.py
"""
Pipeline for Semi-Supervised Dataset Creation (Memory Optimized).

1. PREDICTIONS: Stream Dev & Eyepacs -> Filter by Conf ON-THE-FLY -> Promote to GT -> Force Split='train'.
2. REAL GT: Load YOLO Split -> Filter Datasets -> Lazy Duplicate Train set.
3. MERGE: Combine Real GT (Priority) with Pseudo-labels.
"""

import argparse
import logging
from pathlib import Path
from typing import Iterator

# Import Image directly for streaming
from src.imgpipe.image import Image
from src.tools.parquet_processing.parquet_processor import ParquetProcessor


def make_conf_filter(threshold: float):
    """Returns a filter function for ParquetProcessor."""

    def _filter(img: Image) -> bool:
        d = img.yolo_disc_conf
        c = img.yolo_cup_conf
        if d is None or c is None:
            return False
        try:
            return float(d) >= threshold and float(c) >= threshold
        except (ValueError, TypeError):
            return False

    return _filter


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Merge and process predictions and splits (Memory Optimized).")

    # Input files
    parser.add_argument("--pred-dev", type=Path, required=True, help="Path to dev predictions parquet.")
    parser.add_argument("--pred-eyepacs", type=Path, required=True, help="Path to eyepacs predictions parquet.")
    parser.add_argument("--yolo-split", type=Path, required=True, help="Path to existing YOLO split parquet.")

    # Output
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to save output files.")

    # Parameters
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for predictions.")
    parser.add_argument("--duplication-factor", type=int, default=2,
                        help="Factor to duplicate training data (e.g., 2).")

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Process Predictions (The Pseudo-Label Branch)
    # ---------------------------------------------------------
    logging.info("--- Step 1: Processing Predictions (Streaming Mode) ---")

    conf_filter = make_conf_filter(args.conf)
    proc_preds = ParquetProcessor()

    # A. Merge Dev (Priority) - Filter on load
    proc_preds.merge_images(Image.iter_parquet(args.pred_dev), pre_filter=conf_filter)

    # B. Merge Eyepacs (Append) - Filter on load
    proc_preds.merge_images(Image.iter_parquet(args.pred_eyepacs), pre_filter=conf_filter)

    logging.info(f"High-confidence predictions kept in memory: {len(proc_preds.images)}")

    # Promote to GT and force split to 'train'
    proc_preds.promote_predictions_to_gt()
    proc_preds.set_split_for_all("train")  # <--- UPDATED: Ensures pseudo-labels are train set

    # ---------------------------------------------------------
    # 2. Process YOLO Split (The Real GT Branch)
    # ---------------------------------------------------------
    logging.info("--- Step 2: Processing Real GT Splits ---")
    proc_split = ParquetProcessor()
    proc_split.load(args.yolo_split)

    # Filter specific datasets
    proc_split.filter_by_dataset(["PAPILA", "GRAPE"], mode="exclude")

    # Duplicate ONLY the 'train' split
    if args.duplication_factor > 1:
        proc_split.duplicate(factor=args.duplication_factor, splits=["train"])

    # ---------------------------------------------------------
    # 3. Final Merge
    # ---------------------------------------------------------
    logging.info("--- Step 3: Final Merge ---")

    # Merge pseudo-labels INTO the real GT processor.
    # Real GT (proc_split) takes priority in case of ID collisions.
    proc_split.merge_images(proc_preds.images)

    # Clear prediction memory
    del proc_preds
    import gc
    gc.collect()

    # Summarize final state
    proc_split.summarize()

    # Save
    out_path = args.out_dir / f"semi_supervised_dataset_{args.conf}.parquet"
    proc_split.save(
        out_path,
        include_mask_bytes=True,
        include_image_bytes=False
    )


if __name__ == "__main__":
    main()