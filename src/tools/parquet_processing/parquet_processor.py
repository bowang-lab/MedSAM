#!/usr/bin/env python3
# File: src/tools/parquet_processing/parquet_processor.py
"""
A unified class for processing, merging, filtering, and summarizing Image Parquet datasets.
Includes memory optimizations (Lazy Duplication, Ingress Filtering).
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Union

from src.imgpipe.image import Image


class ParquetProcessor:
    def __init__(self, initial_images: Optional[List[Image]] = None):
        self.images: List[Image] = initial_images if initial_images is not None else []
        self._uid_map: Dict[str, Image] = {img.uid: img for img in self.images}

        # Lazy Duplication State
        self._duplication_factor: int = 1
        self._duplication_splits: Optional[Set[str]] = None

        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def load(self, path: Union[str, Path], pre_filter: Optional[Callable[[Image], bool]] = None) -> 'ParquetProcessor':
        """
        Load images from a Parquet file.
        :param pre_filter: Optional function(Image) -> bool. If False, image is discarded immediately (saves RAM).
        """
        self.logger.info(f"Loading parquet from {path}...")
        count = 0
        kept = 0

        new_images = []
        for img in Image.iter_parquet(path):
            count += 1
            if pre_filter and not pre_filter(img):
                continue
            new_images.append(img)
            kept += 1

        self.images = new_images
        self._rebuild_map()
        self.logger.info(f"Loaded {len(self.images)} images (scanned {count}).")
        return self

    def _rebuild_map(self):
        self._uid_map = {img.uid: img for img in self.images}

    # =========================================================================
    # MERGE LOGIC
    # =========================================================================

    def merge(self, *paths: Union[str, Path],
              pre_filter: Optional[Callable[[Image], bool]] = None) -> 'ParquetProcessor':
        """
        Merge images from files into current state.
        :param pre_filter: Filter applied on-the-fly during read to reduce memory usage.
        """
        for p in paths:
            self.logger.info(f"Merging data from file {p}...")
            self.merge_images(Image.iter_parquet(p), pre_filter=pre_filter)
        return self

    def merge_images(self, incoming_images: Iterator[Image],
                     pre_filter: Optional[Callable[[Image], bool]] = None) -> 'ParquetProcessor':
        """
        Merge an iterator of Image objects into the current state.
        """
        incoming_count = 0
        merged_count = 0
        new_count = 0

        for incoming_img in incoming_images:
            incoming_count += 1

            # Ingress Filter (Memory Optimization)
            if pre_filter and not pre_filter(incoming_img):
                continue

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
        self.logger.info(f"  - Merged:  {merged_count}")
        self.logger.info(f"  - Added:   {new_count}")
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
        original_len = len(self.images)
        valid_splits = set(splits)
        self.images = [img for img in self.images if img.split in valid_splits]
        self._rebuild_map()
        self.logger.info(f"Filter Splits {splits}: {original_len} -> {len(self.images)} images.")
        return self

    def filter_by_dataset(self, datasets: Sequence[str], mode: str = 'include') -> 'ParquetProcessor':
        original_len = len(self.images)
        target_sets = set(datasets)

        if mode == 'include':
            self.images = [img for img in self.images if img.dataset in target_sets]
        elif mode == 'exclude':
            self.images = [img for img in self.images if img.dataset not in target_sets]
        else:
            raise ValueError(f"Invalid mode '{mode}'.")

        self._rebuild_map()
        self.logger.info(f"Filter Dataset ({mode} {datasets}): {original_len} -> {len(self.images)} images.")
        return self

    def filter_by_confidence(self, threshold: float, allow_missing: bool = False) -> 'ParquetProcessor':
        """In-memory filter. For memory savings, prefer using pre_filter in load/merge."""
        original_len = len(self.images)

        def is_conf_ok(val) -> bool:
            if val is None: return allow_missing
            try:
                return float(val) >= threshold
            except (ValueError, TypeError):
                return allow_missing

        self.images = [
            img for img in self.images
            if is_conf_ok(img.yolo_disc_conf) and is_conf_ok(img.yolo_cup_conf)
        ]
        self._rebuild_map()
        self.logger.info(f"Filter Confidence >={threshold}: {original_len} -> {len(self.images)} images.")
        return self

    def duplicate(self, factor: int, splits: Optional[Sequence[str]] = None) -> 'ParquetProcessor':
        """
        Configure Lazy Duplication.
        Data is NOT duplicated in memory. It is expanded only during save().
        """
        if factor < 1:
            raise ValueError("Duplication factor must be >= 1.")

        self._duplication_factor = factor
        self._duplication_splits = set(splits) if splits else None

        self.logger.info(f"Lazy Duplication configured: factor={factor}, splits={splits}")
        return self

    def promote_predictions_to_gt(self) -> 'ParquetProcessor':
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

    def summarize(self) -> None:
        print("\n=== Dataset Summary ===")
        print(f"Stored Images (In-Memory): {len(self.images)}")
        if self._duplication_factor > 1:
            print(f"Lazy Duplication Factor:   {self._duplication_factor} (Effective size will be larger on save)")

        per_dataset = Counter()
        per_split = Counter()
        n_gt = 0

        for img in self.images:
            per_dataset[img.dataset or "UNKNOWN"] += 1
            per_split[img.split or "NONE"] += 1
            if img.gt_disc_box or img.gt_cup_box:
                n_gt += 1

        self._print_counter("By Dataset", per_dataset)
        self._print_counter("By Split", per_split)
        print(f"Images with GT Boxes: {n_gt}")
        print("=======================\n")

    @staticmethod
    def _print_counter(title: str, ctr: Counter):
        print(f"{title}:")
        for k, v in sorted(ctr.items()):
            print(f"  {k:<20} {v}")

    # =========================================================================
    # IO (With Lazy Expansion)
    # =========================================================================

    def _generate_output_stream(self) -> Iterator[Image]:
        """Yields images, handling lazy duplication logic on the fly."""
        factor = self._duplication_factor
        target_splits = self._duplication_splits

        for img in self.images:
            # Always yield original
            yield img

            # Yield copies if needed
            if factor > 1:
                if target_splits is None or img.split in target_splits:
                    for i in range(1, factor):
                        new_uid = f"{img.uid}_copy_{i}"
                        # Shallow copy with modified fields
                        copy_img = dataclasses.replace(img, uid=new_uid)
                        # Note: extras/nested mutable objects are shared by default in replace
                        # deep copy them only if you plan to mutate them per-copy
                        if img.extras:
                            copy_img.extras = img.extras.copy()
                        yield copy_img

    def save(self, path: Union[str, Path], include_mask_bytes: bool = True, include_image_bytes: bool = False) -> None:
        self.logger.info(f"Saving to {path} (Lazy Factor: {self._duplication_factor})...")

        Image.save_parquet(
            self._generate_output_stream(),
            path=path,
            drop_none=False,
            include_image_bytes=include_image_bytes,
            include_mask_bytes=include_mask_bytes,
            compression="zstd"
        )
        self.logger.info("Save complete.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        proc = ParquetProcessor()
        proc.load(sys.argv[1])
        proc.summarize()