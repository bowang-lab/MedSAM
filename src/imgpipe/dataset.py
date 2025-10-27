from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .image import Image
from src.utils import ensure_dir
from .yolo_io import YoloDatasetIO


@dataclass
class Dataset:
    """Container for Image objects, with split/save helpers."""
    images: List[Image] = field(default_factory=list)
    # split is stored per-Image (Image.split)

    def __len__(self) -> int:
        return len(self.images)

    def filter(self, pred) -> "Dataset":
        return Dataset([im for im in self.images if pred(im)])

    def set_all_split(self, split: Optional[str]) -> None:
        for im in self.images:
            im.set_split(split)

    # ---------- splitting by subject/patient ----------

    def _patient_id(self, im: Image, rx: Optional[re.Pattern]) -> str:
        if rx:
            m = rx.search(im.subject_id)
            if m:
                return m.group(1)
        # default: use subject_id as-is
        return im.subject_id

    def split_by_patient(self, val_frac: float, test_frac: float, seed: int, patient_regex: Optional[str] = None) -> Tuple["Dataset", "Dataset", "Dataset"]:
        """
        Splits by patient so the same patient does not appear in multiple splits.
        """
        rx = re.compile(patient_regex, re.I) if patient_regex else None
        by_pid: Dict[str, List[Image]] = {}
        for im in self.images:
            pid = self._patient_id(im, rx)
            by_pid.setdefault(pid, []).append(im)

        pids = list(by_pid.keys())
        rng = random.Random(seed)
        rng.shuffle(pids)

        n = len(pids)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        test_pids = set(pids[:n_test])
        val_pids = set(pids[n_test:n_test + n_val])
        train_pids = set(pids[n_test + n_val:])

        tr, va, te = [], [], []
        for pid, ims in by_pid.items():
            tgt = tr
            if pid in val_pids:
                tgt = va
            elif pid in test_pids:
                tgt = te
            for im in ims:
                im.set_split("train" if tgt is tr else ("val" if tgt is va else "test"))
                tgt.append(im)

        return Dataset(tr), Dataset(va), Dataset(te)

    # ---------- YOLO writing ----------

    def save_as_yolo(self, out_root: Path, write_yaml: bool = True, prefer_copy: bool = False) -> None:
        """
        Writes a YOLO dataset under out_root using Image.split to route images/labels.
        """
        YoloDatasetIO.write_split(self.images, out_root, write_yaml=write_yaml, prefer_copy=prefer_copy)

    # ---------- augmentation (simple pluggable hook) ----------

    def augment(self, out_root: Path, *, splits: Iterable[str], multiplier: int, out_ext: str, include_images_without_labels: bool, prefer_copy: bool, augmentor) -> None:
        """
        Delegates to an Augmentor for specific transforms. Keeps structure images/{split}, labels/{split}.
        """
        augmentor.run(self.images, out_root, splits=splits, multiplier=multiplier, out_ext=out_ext, include_images_without_labels=include_images_without_labels, prefer_copy=prefer_copy)