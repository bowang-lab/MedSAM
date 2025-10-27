# src/imgpipe/collector.py
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

from .config import PipelineConfig
from .image import Image
from .image_factory import ImageFactory
from .dataset import Dataset
from src.utils import (
    ensure_dir,
)

def normalize_steps(steps_arg: str | None) -> List[str]:
    if not steps_arg:
        return ["collect", "split", "save", "augment", "disc_only", "roi"]
    return [s.strip() for s in steps_arg.split(",") if s.strip()]

def has_multi_datasets(cfg: PipelineConfig) -> bool:
    raw = getattr(cfg, "raw", None)
    return isinstance(raw, dict) and isinstance(raw.get("datasets"), list) and len(raw["datasets"]) > 0

def group_by_subject(images: List[Image]) -> dict[str, List[Image]]:
    by_subj: dict[str, List[Image]] = defaultdict(list)
    for im in images:
        sid = im.subject_id or im.uid  # fallback to uid if subject_id missing
        by_subj[sid].append(im)
    return by_subj

def subset_roots(base_project_dir: Path, n: int, seed: int) -> dict[str, Path]:
    root = base_project_dir / "data" / "_subset" / f"N{n}_seed{seed}"
    return {
        "root": root,
        "yolo_split":      root / "yolo_split",
        "yolo_aug":        root / "yolo_split_aug",
        "yolo_roi":        root / "yolo_split_cupROI",
        "yolo_disc_only":  root / "yolo_split_disc_only",
        "viz_out":         root / "viz_labels",
    }


class DatasetCollector:
    """
    Encapsulates dataset collection and (optional) patient-wise subsetting.

    Responsibilities
    ----------------
    - Collect Image objects from one or multiple dataset blocks in cfg.
    - Support include/exclude filename filters and recursive directory search.
    - Optionally downselect to a patient-wise subset and redirect output roots.

    Public API
    ----------
    collect() -> Dataset
        Collect all Image objects (multi- or single-dataset based on cfg).

    subset_if_enabled(ds: Dataset) -> tuple[Dataset, dict[str, Path]]
        If cfg.subset_n > 0, returns a patient-wise subset and a dict of
        redirected output paths (yolo_split, yolo_aug, yolo_roi, yolo_disc_only, viz_out).
        Otherwise returns (ds, original_out_roots).
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    # -------------------- single dataset block --------------------

    @staticmethod
    def _collect_single_block(
        dataset_name: str,
        images_root: Path,
        disc_masks_root: Optional[Path],
        cup_masks_root: Optional[Path],
        include_name_contains: Optional[Iterable[str]],
        exclude_name_contains: Optional[Iterable[str]],
        recursive: bool,
    ) -> List[Image]:
        factory = ImageFactory(
            dataset_name=dataset_name,
            images_root=images_root,
            disc_masks_root=disc_masks_root,
            cup_masks_root=cup_masks_root,
            include_name_contains=list(include_name_contains) if include_name_contains else None,
            exclude_name_contains=list(exclude_name_contains) if exclude_name_contains else None,
            recursive=recursive,
        )
        return factory.collect()

    # ------------------------ main collection ------------------------

    def collect(self) -> Dataset:
        """
        Multi-dataset mode (cfg.raw['datasets']) or single-dataset mode (cfg.* roots).
        Prints a brief per-block summary.
        """
        if has_multi_datasets(self.cfg):
            images: List[Image] = []
            blocks = self.cfg.raw["datasets"]  # type: ignore[index]
            for block in blocks:
                tag = str(block.get("tag", "DS"))
                imgs_root = Path(block["images_root"]).expanduser().resolve()
                disc_root = Path(block["disc_masks"]).expanduser().resolve() if block.get("disc_masks") else None
                cup_root  = Path(block["cup_masks"]).expanduser().resolve()  if block.get("cup_masks")  else None
                include   = block.get("include_name_contains")
                exclude   = block.get("exclude_name_contains")
                recursive = bool(block.get("recursive", False))

                part = self._collect_single_block(
                    dataset_name=tag,
                    images_root=imgs_root,
                    disc_masks_root=disc_root,
                    cup_masks_root=cup_root,
                    include_name_contains=include,
                    exclude_name_contains=exclude,
                    recursive=recursive,
                )
                print(f"[COLLECT] {tag}: {len(part)} images")
                images.extend(part)
            return Dataset(images)

        # single-dataset mode
        part = self._collect_single_block(
            dataset_name="MDSET",
            images_root=self.cfg.images_root,
            disc_masks_root=self.cfg.disc_masks,
            cup_masks_root=self.cfg.cup_masks,
            include_name_contains=self.cfg.include_name_contains,
            exclude_name_contains=self.cfg.exclude_name_contains,
            recursive=self.cfg.recursive,
        )
        print(f"[COLLECT] MDSET: {len(part)} images")
        return Dataset(part)

    # ------------------ optional subset & outputs -------------------

    def subset_if_enabled(self, ds: Dataset) -> tuple[Dataset, dict[str, Path]]:
        """
        If cfg.subset_n > 0:
          - Build a patient-wise subset (using group_by_subject).
          - Redirect output roots to a subset sandbox.
        Else:
          - Return the original dataset and original outputs.

        Returns
        -------
        (Dataset, dict[str, Path]):
            Dataset (possibly subset) and output roots mapping:
            { 'yolo_split', 'yolo_aug', 'yolo_roi', 'yolo_disc_only', 'viz_out' }
        """
        # Default outputs (no subset)
        outs = {
            "yolo_split":     self.cfg.yolo_split,
            "yolo_aug":       self.cfg.yolo_aug,
            "yolo_roi":       self.cfg.yolo_roi,
            "yolo_disc_only": self.cfg.yolo_disc_only,
            "viz_out":        self.cfg.viz_out,
        }

        n_items = getattr(self.cfg, "subset_n", 0) or 0
        if n_items <= 0:
            return ds, outs

        # Patient-wise subset
        rng_seed = getattr(self.cfg, "subset_seed", None)
        by_subj = group_by_subject(ds.images)
        subjects = list(by_subj.keys())

        # Deterministic shuffle if seed provided
        import random
        rng = random.Random(rng_seed)
        rng.shuffle(subjects)

        selected: List[Image] = []
        used_subjects = 0
        for sid in subjects:
            if len(selected) >= n_items:
                break
            selected.extend(by_subj[sid])
            used_subjects += 1

        if len(selected) > n_items:
            selected = selected[:n_items]

        ds_sub = Dataset(selected)

        # Redirect outputs to a subset sandbox
        subset = subset_roots(self.cfg.project_dir, n_items, rng_seed)
        for p in subset.values():
            ensure_dir(p)

        print(f"[SUBSET] patient-wise → {len(selected)} items from {used_subjects} subjects → {subset['root']}")

        return ds_sub, {
            "yolo_split":     subset["yolo_split"],
            "yolo_aug":       subset["yolo_aug"],
            "yolo_roi":       subset["yolo_roi"],
            "yolo_disc_only": subset["yolo_disc_only"],
            "viz_out":        subset["viz_out"],
        }