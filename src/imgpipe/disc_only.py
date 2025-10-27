from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from .yolo_io import YoloDatasetIO


class DiscOnlyDatasetBuilder:
    """
    Build a disc-only dataset from a 2-class YOLO split.
    Keeps only class-0 lines; remaps to single-class '0'; preserves splits.
    """

    def __init__(self, drop_empty: bool = False, prefer_copy: bool = False) -> None:
        self.drop_empty = drop_empty
        self.prefer_copy = prefer_copy

    def build(self, src_root: Path, out_root: Path) -> None:
        # class-0 == disc
        YoloDatasetIO.filter_to_single_class(
            base_root=src_root, out_root=out_root, keep_class_id=0, drop_empty=self.drop_empty, prefer_copy=self.prefer_copy
        )