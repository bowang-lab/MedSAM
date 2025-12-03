#!/usr/bin/env python3
# File: src/tools/parquet_processing/parquet_processor.py
"""
A unified class for processing, merging, filtering, and summarizing Image Parquet datasets.
Includes capabilities for dataset-based filtering, split-based duplication, and smart merging.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from src.imgpipe.image import Image


class ParquetProcessor:
    def __init__(self, initial_images: Optional[List[Image]] = None):
        self.images: List[Image] = initial_images if initial_images is not None else []
        self._uid_map: Dict[str, Image] = {img.uid: img for img in self.images}
        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def load(self, path: Union[str, Path]) -> 'ParquetProcessor':
        self.logger.info(f"Loading parquet from {path}...")
        self.images = list(Image.iter_parquet(path))
        self._rebuild_map()
        self.logger.info(f"Loaded {len(self.images)} images.")
        return self

    def _rebuild_map(self):
        self._uid_map = {img.uid: img for img in self.images}

    # =========================================================================
    # MERGE LOGIC
    # =========================================================================

    def merge(self, *paths: Union[str, Path]) -> 'ParquetProcessor':
        """Merge images from files into current state."""
        for p in paths:
            self.logger.info(f"Merging data from file {p}...")
            self.merge_images(Image.iter_parquet(p))
        return self

    def merge_images(self, incoming_images: Union[List[Image], Any]) -> 'ParquetProcessor':
        """
        Merge a list/iterator of Image objects into the current state.
        Existing UIDs (Primary) take precedence over Incoming (Secondary).
        """
        incoming_count = 0
        merged_count = 0
        new_count = 0

        for incoming_img in incoming_images:
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
        for field in dataclasses.fields(Image):
            name = field.name
            val_a = getattr(primary, name)
            val_b = getattr(secondary, name)

            if name == "extras":
                merged_extras = val_b.copy() if val_b else {}
                if val_a:
                    merged_extras.update(val_a)
                setattr(primary, name, merged_extras)
                continue

            if val_a is None and val_b is not None:
                setattr(primary, name, val_b)

    # =========================================================================
    # FILTERING & MODIFICATION
    # =========================================================================

    def filter_by_split(self, splits: Sequence[str]) -> 'ParquetProcessor':
        """Keep only images in specified splits."""
        original_len = len(self.images)
        valid_splits = set(splits)
        self.images = [img for img in self.images if img.split in valid_splits]
        self._rebuild_map()
        self.logger.info(f"Filter Splits {splits}: {original_len} -> {len(self.images)} images.")
        return self

    def filter_by_dataset(self, datasets: Sequence[str], mode: str = 'include') -> 'ParquetProcessor':
        """Filter by dataset name (include/exclude)."""
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
        original_len = len(self.images)

        def check(img):
            present = [getattr(img, f, None) is not None for f in fields]
            if mode == 'all': return all(present)
            return any(present)

        self.images = [img for img in self.images if check(img)]
        self._rebuild_map()
        self.logger.info(f"Filter Non-Null {fields} ({mode}): {original_len} -> {len(self.images)} images.")
        return self

    def filter_by_confidence(self, threshold: float, allow_missing: bool = False) -> 'ParquetProcessor':
        original_len = len(self.images)

        def is_conf_ok(val) -> bool:
            if val is None: return allow_missing
            try:
                return float(val) >= threshold
            except (ValueError, TypeError):
                return allow_missing

        new_images = []
        for img in self.images:
            if is_conf_ok(img.yolo_disc_conf) and is_conf_ok(img.yolo_cup_conf):
                new_images.append(img)

        self.images = new_images
        self._rebuild_map()
        self.logger.info(f"Filter Confidence >={threshold}: {original_len} -> {len(self.images)} images.")
        return self

    def duplicate(self, factor: int, splits: Optional[Sequence[str]] = None) -> 'ParquetProcessor':
        """
        Duplicate images 'factor' times.
        If 'splits' is provided, only images in those splits are duplicated.
        """
        if factor < 1:
            raise ValueError("Duplication factor must be >= 1.")
        if factor == 1:
            return self

        original_len = len(self.images)
        self.logger.info(f"Duplicating dataset by factor {factor} (splits={splits})...")

        target_splits = set(splits) if splits else None
        new_images = []

        for img in self.images:
            new_images.append(img)  # Always keep original

            # Check if this specific image should be duplicated
            if target_splits is None or img.split in target_splits:
                for i in range(1, factor):
                    new_uid = f"{img.uid}_copy_{i}"
                    new_extras = img.extras.copy() if img.extras else {}
                    copy_img = dataclasses.replace(img, uid=new_uid, extras=new_extras)
                    new_images.append(copy_img)

        self.images = new_images
        self._rebuild_map()
        self.logger.info(f"Duplication complete: {original_len} -> {len(self.images)} images.")
        return self

    def promote_predictions_to_gt(self) -> 'ParquetProcessor':
        """Overwrite GT fields with Prediction fields."""
        count = 0
        for img in self.images:
            if img.pred_disc_mask: img.gt_disc_mask = img.pred_disc_mask
            if img.pred_cup_mask: img.gt_cup_mask = img.pred_cup_mask

            dbox = img.inter_pred_disc_box or img.pred_disc_box
            if dbox: img.gt_disc_box = dbox

            cbox = img.inter_pred_cup_box or img.pred_cup_box
            if cbox: img.gt_cup_box = cbox

            if img.pred_cdr is not None: img.gt_cdr = img.pred_cdr
            count += 1
        self.logger.info(f"Promoted predictions to GT for {count} images.")
        return self

    # =========================================================================
    # SUMMARIZATION
    # =========================================================================

    def summarize(self, check_embedded_masks: bool = False) -> None:
        print("\n=== Dataset Summary ===")
        print(f"Total Images: {len(self.images)}")

        per_dataset = Counter()
        per_split = Counter()
        n_gt = 0

        for img in self.images:
            per_dataset[img.dataset or "UNKNOWN"] += 1
            per_split[img.split or "NONE"] += 1
            if img.gt_disc_box or img.gt_cup_box or img.gt_disc_mask or img.gt_cup_mask:
                n_gt += 1

        self._print_counter("By Dataset", per_dataset)
        self._print_counter("By Split", per_split)
        print(f"\nImages with GT: {n_gt} ({n_gt / len(self.images):.1%})")
        print("=======================\n")

    @staticmethod
    def _print_counter(title: str, ctr: Counter):
        print(f"\n{title}:")
        for k, v in sorted(ctr.items()):
            print(f"  {k:<20} {v}")

    # =========================================================================
    # IO
    # =========================================================================

    def save(self, path: Union[str, Path], include_mask_bytes: bool = True, include_image_bytes: bool = False) -> None:
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