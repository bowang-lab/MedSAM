#!/usr/bin/env python3
# File: src/tools/parquet_processing/run_process_predictions.py
"""
Pipeline for Semi-Supervised Dataset Creation (Memory Optimized).

1. PREDICTIONS: Stream Dev & Eyepacs -> Filter by Conf ON-THE-FLY -> Promote to GT.
2. REAL GT: Load YOLO Split -> Filter Datasets -> Duplicate Train set.
3. MERGE: Combine Real GT (Priority) with Pseudo-labels.
"""

import argparse
import logging
from pathlib import Path
from typing import Iterator

# Import Image directly for streaming
from src.imgpipe.image import Image
from src.tools.parquet_processing.parquet_processor import ParquetProcessor


def iter_filtered_preds(path: Path, conf_thresh: float) -> Iterator[Image]:
    """
    Generator that streams images and yields only those meeting the confidence threshold.
    This prevents loading millions of low-confidence rows into RAM.
    """
    logging.info(f"Streaming and filtering {path} (conf >= {conf_thresh})...")
    for img in Image.iter_parquet(path):
        # Check valid confidence (float and non-None)
        d_conf = img.yolo_disc_conf
        c_conf = img.yolo_cup_conf

        # Skip if missing or below threshold
        if d_conf is None or c_conf is None:
            continue

        try:
            if float(d_conf) >= conf_thresh and float(c_conf) >= conf_thresh:
                yield img
        except (ValueError, TypeError):
            continue


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

    # Initialize processor without loading data yet
    proc_preds = ParquetProcessor()

    # A. Merge Dev (Priority) - Filter on load
    # Using merge_images with a generator avoids loading the whole file first
    proc_preds.merge_images(iter_filtered_preds(args.pred_dev, args.conf))

    # B. Merge Eyepacs (Append) - Filter on load
    # This is the critical memory fix: garbage rows never touch the list
    proc_preds.merge_images(iter_filtered_preds(args.pred_eyepacs, args.conf))

    logging.info(f"High-confidence predictions kept in memory: {len(proc_preds.images)}")

    # Promote to GT
    proc_preds.promote_predictions_to_gt()

    # Explicitly set split to 'train' for these pseudo-labels
    for img in proc_preds.images:
        img.split = "train"

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

    # Clear the prediction processor to free memory immediately
    del proc_preds
    import gc
    gc.collect()

    # Summarize final state
    proc_split.summarize()

    # Save
    out_path = args.out_dir / "final_semi_supervised_dataset.parquet"
    proc_split.save(
        out_path,
        include_mask_bytes=True,  # Need masks for training
        include_image_bytes=False
    )


if __name__ == "__main__":
    main()