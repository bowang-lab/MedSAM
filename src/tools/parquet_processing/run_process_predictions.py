#!/usr/bin/env python3
# File: src/tools/parquet_processing/run_process_predictions.py
"""
Pipeline for Semi-Supervised Dataset Creation (Memory Optimized).

1. PREDICTIONS: Merge Dev & Eyepacs -> Filter by Conf ON-THE-FLY -> Promote to GT.
2. REAL GT: Load YOLO Split -> Filter Datasets -> Lazy Duplicate Train set.
3. MERGE: Combine Real GT (Priority) with Pseudo-labels.
"""

import argparse
import logging
from pathlib import Path
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
    parser.add_argument("--pred-dev", type=Path, required=True, help="Path to dev predictions.")
    parser.add_argument("--pred-eyepacs", type=Path, required=True, help="Path to eyepacs predictions.")
    parser.add_argument("--yolo-split", type=Path, required=True, help="Path to existing YOLO split.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--duplication-factor", type=int, default=2, help="Factor to duplicate training data.")

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Process Predictions (The Pseudo-Label Branch)
    # ---------------------------------------------------------
    logging.info("--- Step 1: Processing Predictions ---")

    # Create the filter function
    conf_filter = make_conf_filter(args.conf)

    proc_preds = ParquetProcessor()

    # Load & Filter ON INGRESS (Saves Memory)
    proc_preds.load(args.pred_dev, pre_filter=conf_filter)
    proc_preds.merge(args.pred_eyepacs, pre_filter=conf_filter)

    # Promote to GT
    proc_preds.promote_predictions_to_gt()

    # Set pseudo-labels to 'train'
    for img in proc_preds.images:
        img.split = "train"

    # ---------------------------------------------------------
    # 2. Process YOLO Split (The Real GT Branch)
    # ---------------------------------------------------------
    logging.info("--- Step 2: Processing Real GT Splits ---")
    proc_split = ParquetProcessor()
    proc_split.load(args.yolo_split)

    proc_split.filter_by_dataset(["PAPILA", "GRAPE"], mode="exclude")

    # Lazy Duplication (Saves Memory - copies generated at save time)
    if args.duplication_factor > 1:
        proc_split.duplicate(factor=args.duplication_factor, splits=["train"])

    # ---------------------------------------------------------
    # 3. Final Merge & Save
    # ---------------------------------------------------------
    logging.info("--- Step 3: Final Merge ---")

    # Merge pseudo-labels (Secondary) into Real GT (Primary)
    # Real GT takes priority on ID collision.
    proc_split.merge_images(proc_preds.images)

    # Free up prediction memory
    del proc_preds

    proc_split.summarize()

    out_path = args.out_dir / f"final_semi_supervised_dataset{args.conf}.parquet"
    proc_split.save(
        out_path,
        include_mask_bytes=True,
        include_image_bytes=False
    )


if __name__ == "__main__":
    main()