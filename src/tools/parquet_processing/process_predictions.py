#!/usr/bin/env python3
# File: src/tools/parquet_processing/process_predictions.py
"""
Pipeline for Semi-Supervised Dataset Creation:
1. PREDICTIONS: Merge Dev & Eyepacs preds -> Filter by Conf -> Promote to GT (Pseudo-labels).
2. REAL GT: Load YOLO Split -> Filter Datasets -> Duplicate Train set.
3. MERGE: Combine Real GT (Priority) with Pseudo-labels (Append).
"""

import argparse
import logging
from pathlib import Path
from src.tools.parquet_processing.parquet_processor import ParquetProcessor


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Merge and process predictions and splits.")

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
    logging.info("--- Step 1: Processing Predictions ---")
    proc_preds = ParquetProcessor()

    # Load and Merge Predictions
    proc_preds.load(args.pred_dev)
    proc_preds.merge(args.pred_eyepacs)

    # Filter by Confidence
    proc_preds.filter_by_confidence(threshold=args.conf)

    # Promote Predictions to Ground Truth (Pseudo-labeling)
    proc_preds.promote_predictions_to_gt()

    # Explicitly set these to 'train' so they can be used for training
    # (Optional but good practice for semi-supervised logic)
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