#!/usr/bin/env python3
# File: src/tools/parquet_processing/parquet_processor.py

from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import numpy as np

from src.imgpipe.image import Image


class ParquetProcessor:
    def __init__(self, initial_images: Optional[List[Image]] = None):
        """
        Initialize the processor.
        :param initial_images: Optional list of Image objects to start with.
        """
        self.images: List[Image] = initial_images if initial_images is not None else []
        self._uid_map: Dict[str, Image] = {img.uid: img for img in self.images}
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def load(self, path: Union[str, Path]) -> 'ParquetProcessor':
        """
        Load images from a Parquet file, replacing current state.
        """
        self.logger.info(f"Loading parquet from {path}...")
        self.images = list(Image.iter_parquet(path))
        self._rebuild_map()
        self.logger.info(f"Loaded {len(self.images)} images.")
        return self

    def _rebuild_map(self):
        self._uid_map = {img.uid: img for img in self.images}

    # =========================================================================
    # MERGE LOGIC (Replaces merge_smart.py)
    # =========================================================================

    def merge(self, *paths: Union[str, Path]) -> 'ParquetProcessor':
        """
        Merge multiple parquet files into the current state.
        Priority is given to the current state, then the first path, then the second, etc.

        Logic:
          - If UID exists: The existing record is 'Primary'. The new record is 'Secondary'.
            Primary retains its values. Secondary fills nulls in Primary. Extras are merged.
          - If UID is new: The record is added.
        """
        for p in paths:
            self.logger.info(f"Merging data from {p}...")
            incoming_count = 0
            merged_count = 0
            new_count = 0

            for incoming_img in Image.iter_parquet(p):
                incoming_count += 1
                uid = incoming_img.uid

                if uid in self._uid_map:
                    # Intelligent merge into existing (priority) record
                    self._intelligent_merge_record(self._uid_map[uid], incoming_img)
                    merged_count += 1
                else:
                    # Add new record
                    self.images.append(incoming_img)
                    self._uid_map[uid] = incoming_img
                    new_count += 1

            self.logger.info(f"  - Scanned: {incoming_count}")
            self.logger.info(f"  - Merged (filled gaps): {merged_count}")
            self.logger.info(f"  - New records added: {new_count}")

        return self

    @staticmethod
    def _intelligent_merge_record(primary: Image, secondary: Image) -> None:
        """
        Merge secondary fields into primary in-place.
        Rule: Primary wins if not None. If Primary is None, take Secondary.
        Special: 'extras' are dict-merged (Primary overwrites Secondary keys).
        """
        for field in dataclasses.fields(Image):
            name = field.name
            val_a = getattr(primary, name)
            val_b = getattr(secondary, name)

            # 1. Handle Dictionary 'extras' specifically
            if name == "extras":
                # Start with B's extras (if any), then update with A's
                merged_extras = val_b.copy() if val_b else {}
                if val_a:
                    merged_extras.update(val_a)
                setattr(primary, name, merged_extras)
                continue

            # 2. General Field Merging
            # If A is None and B is not None, fill A with B.
            if val_a is None and val_b is not None:
                setattr(primary, name, val_b)

    # =========================================================================
    # FILTERING & MODIFICATION LOGIC
    # =========================================================================

    def filter_by_split(self, splits: Sequence[str]) -> 'ParquetProcessor':
        """
        Keep only images belonging to the specified splits (e.g., ['train', 'val']).
        """
        original_len = len(self.images)
        valid_splits = set(splits)
        self.images = [img for img in self.images if img.split in valid_splits]
        self._rebuild_map()
        self.logger.info(f"Filter Splits {splits}: {original_len} -> {len(self.images)} images.")
        return self

    def filter_by_dataset(self, datasets: Sequence[str], mode: str = 'include') -> 'ParquetProcessor':
        """
        Filter images based on their dataset name.
        :param datasets: List of dataset names to filter.
        :param mode: 'include' to keep only these datasets, 'exclude' to drop them.
        """
        original_len = len(self.images)
        target_sets = set(datasets)

        if mode == 'include':
            self.images = [img for img in self.images if img.dataset in target_sets]
        elif mode == 'exclude':
            self.images = [img for img in self.images if img.dataset not in target_sets]
        else:
            raise ValueError(f"Invalid mode '{mode}'. Use 'include' or 'exclude'.")

        self._rebuild_map()
        self.logger.info(f"Filter Dataset ({mode} {datasets}): {original_len} -> {len(self.images)} images.")
        return self

    def filter_non_null_fields(self, fields: Sequence[str], mode: str = 'any') -> 'ParquetProcessor':
        """
        Keep images that have non-null values for the specified fields.
        :param fields: List of attribute names (e.g., ['gt_disc_mask', 'pred_disc_mask'])
        :param mode: 'any' (keep if at least one field is present) or 'all' (keep only if all are present).
        """
        original_len = len(self.images)

        def check(img):
            present = [getattr(img, f, None) is not None for f in fields]
            if mode == 'all':
                return all(present)
            return any(present)

        self.images = [img for img in self.images if check(img)]
        self._rebuild_map()
        self.logger.info(f"Filter Non-Null {fields} ({mode}): {original_len} -> {len(self.images)} images.")
        return self

    def filter_by_confidence(self, threshold: float, allow_missing: bool = False) -> 'ParquetProcessor':
        """
        Keep images where prediction confidence is >= threshold.
        Checks 'yolo_disc_conf' and 'yolo_cup_conf'. Both must pass if present.
        """
        original_len = len(self.images)

        def is_conf_ok(val) -> bool:
            if val is None:
                return allow_missing
            try:
                return float(val) >= threshold
            except (ValueError, TypeError):
                return allow_missing

        new_images = []
        for img in self.images:
            disc_ok = is_conf_ok(img.yolo_disc_conf)
            cup_ok = is_conf_ok(img.yolo_cup_conf)
            if disc_ok and cup_ok:
                new_images.append(img)

        self.images = new_images
        self._rebuild_map()
        self.logger.info(f"Filter Confidence >={threshold}: {original_len} -> {len(self.images)} images.")
        return self

    def duplicate(self, factor: int) -> 'ParquetProcessor':
        """
        Duplicate the entire dataset 'factor' times.
        e.g., factor=2 means result size is 2x original (Original + 1 copy).

        UIDs of copies are modified:
        - Original: {uid}
        - Copy 1:   {uid}_copy_1
        - ...
        - Copy N:   {uid}_copy_{factor-1}
        """
        if factor < 1:
            raise ValueError("Duplication factor must be >= 1.")
        if factor == 1:
            return self

        original_len = len(self.images)
        self.logger.info(f"Duplicating dataset by factor {factor}...")

        new_images = []

        for img in self.images:
            # 1. Keep Original
            new_images.append(img)

            # 2. Create copies
            for i in range(1, factor):
                # Shallow copy via replace is fine for immutable Image fields,
                # but we copy mutable extras to be safe.
                new_uid = f"{img.uid}_copy_{i}"
                new_extras = img.extras.copy() if img.extras else {}

                copy_img = dataclasses.replace(img, uid=new_uid, extras=new_extras)
                new_images.append(copy_img)

        self.images = new_images
        self._rebuild_map()
        self.logger.info(f"Duplication complete: {original_len} -> {len(self.images)} images.")
        return self

    # =========================================================================
    # SUMMARIZATION (Replaces open_parquet_test.py)
    # =========================================================================

    def summarize(self, check_embedded_masks: bool = False) -> None:
        """
        Print a comprehensive summary of the current dataset state.
        Counts by dataset, split, mask presence, box presence, and optionally validates embedded mask bytes.
        """
        print("\n=== Dataset Summary ===")
        print(f"Total Images: {len(self.images)}")

        per_dataset = Counter()
        per_split = Counter()

        # Annotations
        n_gt = 0
        n_pred = 0

        # Specific fields
        mask_counts = Counter()
        box_counts = Counter()

        # Mask validation
        mask_issues = Counter()

        mask_fields = ["gt_disc_mask", "gt_cup_mask", "pred_disc_mask", "pred_cup_mask"]
        box_fields = ["gt_disc_box", "gt_cup_box", "pred_disc_box", "pred_cup_box"]

        for img in self.images:
            ds = img.dataset or "UNKNOWN"
            sp = img.split or "NONE"
            per_dataset[ds] += 1
            per_split[sp] += 1

            # Check generic presence
            has_gt = bool(img.gt_disc_box or img.gt_cup_box or img.gt_disc_mask or img.gt_cup_mask)
            has_pred = bool(img.pred_disc_box or img.pred_cup_box or img.pred_disc_mask or img.pred_cup_mask)
            if has_gt: n_gt += 1
            if has_pred: n_pred += 1

            # Detailed field counts
            for f in mask_fields:
                if getattr(img, f, None) is not None:
                    mask_counts[f] += 1
            for f in box_fields:
                if getattr(img, f, None) is not None:
                    box_counts[f] += 1

            # Optional embedded mask check (CPU intensive)
            if check_embedded_masks:
                for f in mask_fields:
                    mref = getattr(img, f, None)
                    if mref and hasattr(mref, 'packed') and mref.packed is not None:
                        try:
                            # Just try loading the bytes to ensure validity
                            _ = mref.load()
                        except Exception as e:
                            mask_issues[f"{f}_load_error"] += 1

        # Print Report
        print("\n--- Distribution ---")
        self._print_counter("By Dataset", per_dataset)
        self._print_counter("By Split", per_split)

        print("\n--- Annotation Coverage ---")
        print(f"  With any GT:   {n_gt} ({n_gt / len(self.images):.1%})")
        print(f"  With any Pred: {n_pred} ({n_pred / len(self.images):.1%})")

        print("\n--- Field Availability ---")
        self._print_counter("Masks Present", mask_counts)
        self._print_counter("Boxes Present", box_counts)

        if check_embedded_masks:
            print("\n--- Mask Integrity (Embedded) ---")
            if len(mask_issues) == 0:
                print("  All embedded masks loaded successfully.")
            else:
                self._print_counter("Mask Load Issues", mask_issues)
        print("=======================\n")

    @staticmethod
    def _print_counter(title: str, ctr: Counter):
        print(f"{title}:")
        for k, v in sorted(ctr.items()):
            print(f"  {k:<20} {v}")

    # =========================================================================
    # IO
    # =========================================================================

    def save(self, path: Union[str, Path], include_mask_bytes: bool = True, include_image_bytes: bool = False) -> None:
        """
        Save the current state to a Parquet file.
        """
        self.logger.info(f"Saving {len(self.images)} images to {path}...")
        Image.save_parquet(
            self.images,
            path=path,
            drop_none=False,
            include_image_bytes=include_image_bytes,
            include_mask_bytes=include_mask_bytes,
            compression="zstd"
        )
        self.logger.info("Save complete.")


if __name__ == "__main__":
    # Simple CLI wrapper for testing
    import sys

    if len(sys.argv) > 1:
        proc = ParquetProcessor()
        proc.load(sys.argv[1])
        proc.summarize()