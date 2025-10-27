# MedSAM/src/imgpipe/image_factory.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Dict

import numpy as np
from PIL import Image as PILImage

from src.utils import stem_map_by_first_match, list_files_with_ext
from .enums import LabelType, Structure
from .bounding_box import BoundingBox
from .image import Image


def _name_matches_filters(name: str, include: Optional[List[str]], exclude: Optional[List[str]]) -> bool:
    n = name.lower()
    if include and not any(s.lower() in n for s in include):
        return False
    if exclude and any(s.lower() in n for s in exclude):
        return False
    return True


class ImageFactory:
    """
    Creates Image objects from an images root and optional disc/cup mask roots.
    If masks are present for a stem, we compute bounding boxes and set GT boxes.
    """

    def __init__(
        self,
        dataset_name: str,
        images_root: Path,
        disc_masks_root: Optional[Path] = None,
        cup_masks_root: Optional[Path] = None,
        include_name_contains: Optional[List[str]] = None,
        exclude_name_contains: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> None:
        self.dataset = dataset_name
        self.images_root = images_root
        self.disc_masks_root = disc_masks_root
        self.cup_masks_root = cup_masks_root
        self.include = include_name_contains
        self.exclude = exclude_name_contains
        self.recursive = recursive

    def _load_mask(self, p: Optional[Path]) -> Optional[np.ndarray]:
        if p is None:
            return None
        arr = np.array(PILImage.open(p).convert("L"))
        return (arr > 0).astype(np.uint8)

    def _image_size(self, p: Path) -> tuple[int, int]:
        with PILImage.open(p) as im:
            w, h = im.size
        return w, h

    def collect(self) -> List[Image]:
        img_paths = list_files_with_ext(self.images_root, recursive=self.recursive)
        disc_map: Dict[str, Path] = stem_map_by_first_match(self.disc_masks_root) if self.disc_masks_root else {}
        cup_map: Dict[str, Path] = stem_map_by_first_match(self.cup_masks_root) if self.cup_masks_root else {}

        items: List[Image] = []
        for ip in img_paths:
            if not _name_matches_filters(ip.name, self.include, self.exclude):
                continue
            stem = ip.stem
            w, h = self._image_size(ip)

            img = Image.from_path(
                image_path=ip,
                dataset=self.dataset,
                subject_id=stem,   # you can later normalize to a patient id at split time via regex if needed
                uid=stem,
                width=w,
                height=h,
                split=None,
            )

            # If masks exist, compute boxes and set GT
            disc_mask = self._load_mask(disc_map.get(stem))
            if disc_mask is not None:
                disc_box = BoundingBox.from_mask(disc_mask)
                if disc_box:
                    img.set_box(Structure.DISC, LabelType.GT, disc_box)

            cup_mask = self._load_mask(cup_map.get(stem))
            if cup_mask is not None:
                cup_box = BoundingBox.from_mask(cup_mask)
                if cup_box:
                    img.set_box(Structure.CUP, LabelType.GT, cup_box)

            items.append(img)
        return items