# src/imgpipe/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PipelineConfig:
    # Core
    project_dir: Path

    # Single-dataset (still supported even if you also use multi-dataset blocks in YAML)
    images_root: Path
    disc_masks: Optional[Path]
    cup_masks: Optional[Path]

    # Outputs
    yolo_split: Path
    yolo_aug: Path
    yolo_roi: Path
    yolo_disc_only: Path
    viz_out: Path

    # Optional/legacy (not used by pipeline_oop but allowed in YAML)
    labels_dir: Optional[Path] = None

    # Name filters and search behavior
    include_name_contains: Optional[List[str]] = None
    exclude_name_contains: Optional[List[str]] = None
    recursive: bool = False

    # Subset mode
    subset_n: int = 0
    subset_seed: int = 1337
    subset_copy: bool = False
    prefer_copy: bool = False

    # Split
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 1337
    patient_regex: str = ""

    # Augment
    augment_enabled: bool = True
    augment_splits: Optional[List[str]] = None  # e.g. ["train"] or ["train","val"]
    augment_multiplier: int = 2
    augment_out_ext: str = ".jpg"
    include_images_without_labels: bool = False

    # Disc-only
    disc_only_from_aug_train: bool = False
    disc_only_train_splits: Optional[List[str]] = None
    disc_only_copy_images: bool = False
    disc_only_drop_empty: bool = False

    # ROI
    roi_pad_pct: float = 0.0
    keep_roi_negatives: bool = False
    roi_from_aug: bool = False

    # Keep raw YAML for multi-dataset blocks etc.
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        import os
        import yaml

        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        def _p(v: Any) -> Optional[Path]:
            if v in (None, "", False):
                return None
            return Path(os.path.expanduser(str(v))).resolve()

        # Project dir (default to config file's parent if not supplied)
        proj = _p(cfg.get("project_dir")) or path.parent.resolve()

        # Default output locations under project_dir/data/*
        def _default_data_dir(name: str) -> Path:
            # You can change this layout to your preferred project structure
            return (proj / "data" / name).resolve()

        # Required/commonly-used roots
        images_root = _p(cfg.get("images_root")) or (proj / "data" / "images").resolve()
        disc_masks  = _p(cfg.get("disc_masks"))
        cup_masks   = _p(cfg.get("cup_masks"))

        yolo_split     = _p(cfg.get("yolo_split"))     or _default_data_dir("yolo_split")
        yolo_aug       = _p(cfg.get("yolo_aug"))       or _default_data_dir("yolo_split_aug")
        yolo_roi       = _p(cfg.get("yolo_roi"))       or _default_data_dir("yolo_split_cupROI")
        yolo_disc_only = _p(cfg.get("yolo_disc_only")) or _default_data_dir("yolo_split_disc_only")
        viz_out        = _p(cfg.get("viz_out"))        or _default_data_dir("viz_labels")

        labels_dir = _p(cfg.get("labels_dir")) if "labels_dir" in cfg else None

        # Normalize include/exclude filters
        inc = cfg.get("include_name_contains")
        exc = cfg.get("exclude_name_contains")
        if isinstance(inc, str):
            inc = [x.strip() for x in inc.split(",") if x.strip()]
        if isinstance(exc, str):
            exc = [x.strip() for x in exc.split(",") if x.strip()]

        return cls(
            project_dir=proj,
            images_root=images_root,
            disc_masks=disc_masks,
            cup_masks=cup_masks,
            yolo_split=yolo_split,
            yolo_aug=yolo_aug,
            yolo_roi=yolo_roi,
            yolo_disc_only=yolo_disc_only,
            viz_out=viz_out,
            labels_dir=labels_dir,
            include_name_contains=inc,
            exclude_name_contains=exc,
            recursive=bool(cfg.get("recursive", False)),

            subset_n=int(cfg.get("subset_n", 0)),
            subset_seed=int(cfg.get("subset_seed", 1337)),
            subset_copy=bool(cfg.get("subset_copy", False)),
            prefer_copy=bool(cfg.get("prefer_copy", False)),

            val_frac=float(cfg.get("val_frac", 0.15)),
            test_frac=float(cfg.get("test_frac", 0.15)),
            seed=int(cfg.get("seed", 1337)),
            patient_regex=str(cfg.get("patient_regex", "") or ""),

            augment_enabled=bool(cfg.get("augment_enabled", True)),
            augment_splits=cfg.get("augment_splits"),
            augment_multiplier=int(cfg.get("augment_multiplier", 2)),
            augment_out_ext=str(cfg.get("augment_out_ext", ".jpg")),
            include_images_without_labels=bool(cfg.get("include_images_without_labels", False)),

            disc_only_from_aug_train=bool(cfg.get("disc_only_from_aug_train", False)),
            disc_only_train_splits=cfg.get("disc_only_train_splits"),
            disc_only_copy_images=bool(cfg.get("disc_only_copy_images", False)),
            disc_only_drop_empty=bool(cfg.get("disc_only_drop_empty", False)),

            roi_pad_pct=float(cfg.get("roi_pad_pct", 0.0)),
            keep_roi_negatives=bool(cfg.get("keep_roi_negatives", False)),
            roi_from_aug=bool(cfg.get("roi_from_aug", False)),

            raw=cfg,
        )