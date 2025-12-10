#!/usr/bin/env python3
# File: src/scripts/predict_summarize.py
"""
YOLO+MedSAM Prediction Pipeline with Multi-GPU Support.

This module provides both a CLI interface and a programmatic API for running
predictions using YOLO detection + MedSAM segmentation.

API Usage:
    from src.scripts.predict_summarize import PredictConfig, run_predictions
    
    config = PredictConfig(
        images_parquet=Path("images.parquet"),
        yolo_weights=Path("best.pt"),
        medsam_checkpoint=Path("medsam_vit_b.pth"),
        out_dir=Path("predictions/"),
        device="cuda",
    )
    summary = run_predictions(config)

CLI Usage:
    python -m src.scripts.predict_summarize \\
        --images-parquet images.parquet \\
        --yolo-weights best.pt \\
        --medsam-checkpoint medsam_vit_b.pth \\
        --out-dir predictions/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image as PILImage, ImageFile
from tqdm import tqdm

# Use your unified Image class
from src.imgpipe.image import Image
from src.model.predictor import (
    Predictor,
    PredictorConfig as _PredictorConfig,
    YoloPredictorConfig,
    MedSamPredictorConfig,
)

# Import centralized utilities
from src.utils import (
    ensure_dir as _util_ensure_dir,
    is_finite as _is_finite,
    mean_std as _mean_std,
    list_files_with_ext,
    DistInfo,
    get_dist_info as _get_dist_info,
    maybe_init_torch_distributed as _maybe_init_torch_distributed,
    barrier_if_possible as _barrier_if_possible,
    device_for_rank as _device_for_rank,
    pin_torch_cuda_device as _pin_torch_cuda_device,
    ultralytics_device_for_current_process as _ultralytics_device_str_for_current_process,
    iter_sharded as _iter_sharded_items,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# Configuration Dataclass
# =========================

@dataclass
class PredictConfig:
    """
    Configuration for running YOLO+MedSAM predictions.
    
    Required fields:
        out_dir: Output directory for predictions
        yolo_weights: Path to YOLO weights file
        medsam_checkpoint: Path to MedSAM checkpoint
        
    Input source (one required):
        images_parquet: Path to parquet file with images
        images_dir: Path to directory with images
    """
    # Output
    out_dir: Path
    
    # Model weights (required)
    yolo_weights: Path
    medsam_checkpoint: Path
    
    # Input source (one required)
    images_parquet: Optional[Path] = None
    images_dir: Optional[Path] = None
    image_pattern: str = "*"
    inference_dataset: str = "inference"
    inference_split: Optional[str] = None
    
    # Device / Hardware
    device: str = "cuda"
    
    # YOLO settings
    conf: float = 0.001
    iou: float = 0.70
    imgsz: int = 640
    yolo_batch: Optional[int] = None
    
    # MedSAM settings
    sam_amp: bool = True
    sam_resize_backend: str = "cv2"
    box_pad_frac: float = 0.05
    
    # Output options
    save_overlays: bool = False
    mask_store: str = "parquet"  # "png", "parquet", "both", "none"
    image_store: str = "none"  # "parquet", "none"
    
    # Filtering
    splits: Optional[str] = None  # comma-separated
    datasets: Optional[str] = None  # comma-separated
    
    # Batch / Performance
    parquet_read_batch: int = 1024
    predict_batch_size: int = 128
    
    # Resume / Recompute
    resume: bool = False
    force_recompute: bool = False
    
    # Output naming
    final_parquet_name: str = "predictions.parquet"
    final_parquet_compression: str = "zstd"
    final_parquet_batch: int = 2048
    summary_csv: Optional[Path] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.images_parquet and not self.images_dir:
            raise ValueError("Either images_parquet or images_dir must be provided")
        # Convert to Path if needed
        if isinstance(self.out_dir, str):
            self.out_dir = Path(self.out_dir)
        if isinstance(self.yolo_weights, str):
            self.yolo_weights = Path(self.yolo_weights)
        if isinstance(self.medsam_checkpoint, str):
            self.medsam_checkpoint = Path(self.medsam_checkpoint)
        if self.images_parquet and isinstance(self.images_parquet, str):
            self.images_parquet = Path(self.images_parquet)
        if self.images_dir and isinstance(self.images_dir, str):
            self.images_dir = Path(self.images_dir)


# =========================
# Helpers
# =========================

def _ensure_dir(p: Path) -> Path:
    """Wrapper that returns the path for chaining."""
    _util_ensure_dir(p)
    return p


def _relative_subdir(image_path: Path, images_root: Optional[Path]) -> Path:
    p = Path(image_path)
    if images_root is not None:
        try:
            return p.parent.relative_to(images_root)
        except Exception:
            pass
    for parent in p.parents:
        if parent.name.lower() == "fundus":
            try:
                return p.parent.relative_to(parent)
            except Exception:
                return Path()
    return Path()


def _ensure_scalar_metrics(img: Image) -> None:
    # CDR
    try:
        if not _is_finite(getattr(img, "gt_cdr", None)):
            gt_cdr = img.cdr(use_pred=False, axis="vertical")
            if _is_finite(gt_cdr):
                img.gt_cdr = float(gt_cdr)
        if not _is_finite(getattr(img, "pred_cdr", None)):
            pr_cdr = img.cdr(use_pred=True, axis="vertical")
            if _is_finite(pr_cdr):
                img.pred_cdr = float(pr_cdr)
    except Exception:
        pass

    # RDR
    try:
        if not _is_finite(getattr(img, "gt_rdr", None)):
            gt_r = img.rim_metrics(use_pred=False) or {}
            v = gt_r.get("rim_over_disc")
            if _is_finite(v):
                img.gt_rdr = float(v)
        if not _is_finite(getattr(img, "pred_rdr", None)):
            pr_r = img.rim_metrics(use_pred=True) or {}
            v = pr_r.get("rim_over_disc")
            if _is_finite(v):
                img.pred_rdr = float(v)
    except Exception:
        pass


def _pack_pred_masks_for_parquet(img: Image) -> None:
    """
    Align predicted masks to image size, then pack them into BinaryMaskRef so
    Image.to_dict(include_mask_bytes=True) can serialize bytes into Parquet.
    """
    for attr in ("pred_disc_mask", "pred_cup_mask"):
        mref = getattr(img, attr, None)
        if mref is None:
            continue
        try:
            arr = img._mask_to_image_size(mref)
            if arr is None:
                continue
            mref.array = arr
            if hasattr(mref, "pack_inplace"):
                mref.pack_inplace()
            # Optional: keep parquet self-contained
            mref.path = None
        except Exception:
            pass


def _save_pred_masks_for_image(
        img: Image,
        *,
        oc_mask_root: Path,
        od_mask_root: Path,
        rel_subdir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Write predicted masks as PNGs and update mref.path. Returns (disc_path, cup_path)."""
    disc_path: Optional[Path] = None
    cup_path: Optional[Path] = None

    for attr_name, out_root, is_disc in (
            ("pred_disc_mask", od_mask_root, True),
            ("pred_cup_mask", oc_mask_root, False),
    ):
        mref = getattr(img, attr_name, None)
        if mref is None:
            continue

        try:
            arr = img._mask_to_image_size(mref)
        except Exception:
            arr = None
        if arr is None:
            continue

        out_path = out_root / rel_subdir / img.image_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        m_uint8 = (arr.astype(bool)).astype("uint8") * 255
        PILImage.fromarray(m_uint8).save(str(out_path))

        try:
            mref.path = out_path
            if hasattr(mref, "array"):
                mref.array = None
        except Exception:
            pass

        if is_disc:
            disc_path = out_path
        else:
            cup_path = out_path

    return disc_path, cup_path


def _gather_image_files(images_dir: Path, pattern: str) -> List[Path]:
    """Gather image files from directory. Uses centralized list_files_with_ext."""
    # Filter by pattern if not "*"
    all_images = list_files_with_ext(images_dir, recursive=True)
    if pattern == "*":
        return all_images
    import fnmatch
    return sorted(p for p in all_images if fnmatch.fnmatch(p.name, pattern))


def _make_images_from_dir(
        images_dir: Path,
        pattern: str = "*",
        dataset: str = "inference",
        split: Optional[str] = None,
) -> List[Image]:
    """Scans directory and creates lightweight Image objects."""
    img_paths = _gather_image_files(images_dir, pattern)
    if not img_paths:
        raise RuntimeError(f"No images found under {images_dir} with pattern '{pattern}'.")

    images: List[Image] = []
    for p in img_paths:
        try:
            # Fast header read
            with PILImage.open(p) as im:
                W, H = im.size
        except Exception as e:
            logging.warning("Skipping unreadable image %s: %r", p, e)
            continue

        # Construct Image directly
        im_obj = Image.from_path(
            image_path=p,
            dataset=dataset,
            subject_id=p.stem,
            uid=p.stem,
            split=split,
            width=W,
            height=H,
        )
        images.append(im_obj)
    return images


# =========================
# Schema helpers (for summarization only)
# =========================

def append_summary_row_to_csv(csv_path: Path, pred_ref: Path, summary: Dict[str, Any]) -> None:
    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    md = summary["mask_dice_stats"]
    bd = summary["box_dice_stats"]
    me = summary["metric_error"]
    se = summary.get("scalar_error", {})

    header = [
        "pred_path",
        "n_images",
        "mask_dice_disc_mean", "mask_dice_disc_std",
        "mask_dice_cup_mean", "mask_dice_cup_std",
        "box_dice_disc_mean", "box_dice_disc_std",
        "box_dice_cup_mean", "box_dice_cup_std",
        "cdr_v_mae_mean", "cdr_v_mae_std",
        "rim_over_disc_mae_mean", "rim_over_disc_mae_std",
        "I_over_S_mae_mean", "I_over_S_mae_std",
        "cdr_mae_mean", "cdr_mae_std",
        "rdr_mae_mean", "rdr_mae_std",
    ]

    row = [
        str(pred_ref),
        summary["counts"]["images"],
        md["disc"]["mean"], md["disc"]["std"],
        md["cup"]["mean"], md["cup"]["std"],
        bd["disc"]["mean"], bd["disc"]["std"],
        bd["cup"]["mean"], bd["cup"]["std"],
        me["cdr_v"]["mae_mean"], me["cdr_v"]["mae_std"],
        me["rim_over_disc"]["mae_mean"], me["rim_over_disc"]["mae_std"],
        me["I_over_S"]["mae_mean"], me["I_over_S"]["mae_std"],
        (se.get("cdr_mae") or {}).get("mae_mean"),
        (se.get("cdr_mae") or {}).get("mae_std"),
        (se.get("rdr_mae") or {}).get("mae_mean"),
        (se.get("rdr_mae") or {}).get("mae_std"),
    ]

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)


# =========================
# Distributed / Multi-GPU helpers
# =========================
# NOTE: Core distributed helpers (DistInfo, get_dist_info, maybe_init_torch_distributed,
#       barrier_if_possible, device_for_rank, pin_torch_cuda_device, 
#       ultralytics_device_for_current_process) are now imported from src.utils

def _dataset_dir(out_dir: Path) -> Path:
    return out_dir / "predictions.dataset"


def _rank_dataset_dir(dataset_dir: Path, rank: int) -> Path:
    return dataset_dir / f"gpu={rank:03d}"


def _rank_progress_path(out_dir: Path, rank: int) -> Path:
    return out_dir / "progress" / f"gpu_{rank:03d}.jsonl"


def _rank_part_path(rank_dir: Path, part_idx: int) -> Path:
    return rank_dir / f"part-{part_idx:06d}.parquet"


def _load_done_set(progress_file: Path) -> Set[str]:
    done: Set[str] = set()
    if not progress_file.exists():
        return done
    with open(progress_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = rec.get("image_path") or rec.get("uid")
                if key:
                    done.add(str(key))
            except Exception:
                continue
    return done


def _append_done(progress_file: Path, *, image_path: str, uid: str) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"image_path": image_path, "uid": uid}) + "\n")


def _existing_part_count(rank_dir: Path) -> int:
    if not rank_dir.exists():
        return 0
    return len(list(rank_dir.glob("part-*.parquet")))


def _iter_sharded_images(it: Iterable[Image], *, rank: int, world_size: int) -> Iterable[Image]:
    """Wrapper around iter_sharded for Image objects."""
    return _iter_sharded_items(it, rank=rank, world_size=world_size)


# =========================
# Summarization Logic (Refactored to use Image.iter_parquet)
# =========================

def summarize_predictions_dataset(
        dataset_dir: Path,
        splits: Optional[Set[str]] = None,
        datasets: Optional[Set[str]] = None,
        batch_size: int = 2048,
) -> Dict[str, Any]:
    """
    Stream the entire dataset_dir using Image.iter_parquet and compute metrics.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Predictions dataset directory not found: {dataset_dir}")

    n_images = 0
    det_disc = det_cup = seg_disc = seg_cup = 0

    # Metrics storage
    mask_disc_vals: List[float] = []
    mask_cup_vals: List[float] = []
    box_disc_vals: List[float] = []
    box_cup_vals: List[float] = []

    # Absolute errors for scalar metrics
    err_abs_adv: Dict[str, List[float]] = {
        "cdr_v": [], "cdr_h": [], "rim_over_disc": [], "I_over_S": []
    }
    cdr_abs_err: List[float] = []
    rdr_abs_err: List[float] = []

    # Stream efficiently from directory
    for img in Image.iter_parquet(dataset_dir, batch_size=batch_size):
        # Filtering
        if datasets and getattr(img, "dataset", None) not in datasets:
            continue
        if splits and getattr(img, "split", None) not in splits:
            continue

        n_images += 1

        # Detection counts
        if getattr(img, "inter_pred_disc_box", None): det_disc += 1
        if getattr(img, "inter_pred_cup_box", None): det_cup += 1
        if getattr(img, "pred_disc_mask", None): seg_disc += 1
        if getattr(img, "pred_cup_mask", None): seg_cup += 1

        # Mask Dice (use cached or compute from mask bytes)
        try:
            img.update_mask_dice(fallback_to_boxes=True)
        except Exception:
            pass

        if _is_finite(img.mask_dice_disc): mask_disc_vals.append(float(img.mask_dice_disc))
        if _is_finite(img.mask_dice_cup): mask_cup_vals.append(float(img.mask_dice_cup))

        # Box Dice
        if img.pred_disc_box and img.gt_disc_box:
            box_disc_vals.append(float(img.pred_disc_box.dice(img.gt_disc_box)))
        if img.pred_cup_box and img.gt_cup_box:
            box_cup_vals.append(float(img.pred_cup_box.dice(img.gt_cup_box)))

        # CDR Error
        try:
            gt_cdr = img.gt_cdr if _is_finite(img.gt_cdr) else img.cdr(use_pred=False, axis="vertical")
            pr_cdr = img.pred_cdr if _is_finite(img.pred_cdr) else img.cdr(use_pred=True, axis="vertical")
            if _is_finite(gt_cdr) and _is_finite(pr_cdr):
                cdr_abs_err.append(abs(float(pr_cdr) - float(gt_cdr)))
        except Exception:
            pass

        # RDR Error
        try:
            gt_rdr = img.gt_rdr if _is_finite(img.gt_rdr) else (img.rim_metrics(use_pred=False) or {}).get(
                "rim_over_disc")
            pr_rdr = img.pred_rdr if _is_finite(img.pred_rdr) else (img.rim_metrics(use_pred=True) or {}).get(
                "rim_over_disc")
            if _is_finite(gt_rdr) and _is_finite(pr_rdr):
                rdr_abs_err.append(abs(float(pr_rdr) - float(gt_rdr)))
        except Exception:
            pass

        # Advanced Metrics Error (Rim, I/S, etc)
        try:
            metrics = img.metrics_summary()
            gt_m = metrics.get("gt", {})
            pr_m = metrics.get("pred", {})
            for key in ("cdr_v", "cdr_h", "rim_over_disc", "I_over_S"):
                gv, pv = gt_m.get(key), pr_m.get(key)
                if _is_finite(gv) and _is_finite(pv):
                    err_abs_adv[key].append(abs(float(pv) - float(gv)))
        except Exception:
            pass

    # Aggregation
    md_disc_mean, md_disc_std = _mean_std(mask_disc_vals)
    md_cup_mean, md_cup_std = _mean_std(mask_cup_vals)
    bd_disc_mean, bd_disc_std = _mean_std(box_disc_vals)
    bd_cup_mean, bd_cup_std = _mean_std(box_cup_vals)

    metric_error_adv = {
        k: {"mae_mean": m, "mae_std": s, "n": len(err_abs_adv[k])}
        for k, (m, s) in {name: _mean_std(vals) for name, vals in err_abs_adv.items()}.items()
    }

    return {
        "counts": {
            "images": n_images,
            "det_rate_disc": det_disc / max(1, n_images),
            "det_rate_cup": det_cup / max(1, n_images),
            "seg_rate_disc": seg_disc / max(1, n_images),
            "seg_rate_cup": seg_cup / max(1, n_images),
        },
        "mask_dice_stats": {
            "disc": {"mean": md_disc_mean, "std": md_disc_std, "n": len(mask_disc_vals)},
            "cup": {"mean": md_cup_mean, "std": md_cup_std, "n": len(mask_cup_vals)},
        },
        "box_dice_stats": {
            "disc": {"mean": bd_disc_mean, "std": bd_disc_std, "n": len(box_disc_vals)},
            "cup": {"mean": bd_cup_mean, "std": bd_cup_std, "n": len(box_cup_vals)},
        },
        "metric_error": metric_error_adv,
        "scalar_error": {
            "cdr_mae": {"mae_mean": _mean_std(cdr_abs_err)[0], "mae_std": _mean_std(cdr_abs_err)[1],
                        "n": len(cdr_abs_err)},
            "rdr_mae": {"mae_mean": _mean_std(rdr_abs_err)[0], "mae_std": _mean_std(rdr_abs_err)[1],
                        "n": len(rdr_abs_err)},
        },
    }


# =========================
# Core Prediction API
# =========================

def run_predictions(config: PredictConfig) -> Optional[Dict[str, Any]]:
    """
    Run YOLO+MedSAM predictions using the provided configuration.
    
    This is the main programmatic API for running predictions. It handles:
    - Multi-GPU distribution (via environment variables)
    - Resumable processing
    - Output to parquet and/or PNG masks
    - Summary generation
    
    Args:
        config: PredictConfig instance with all settings
        
    Returns:
        Summary dictionary on rank 0, None on other ranks
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    out_dir: Path = _ensure_dir(config.out_dir)
    dist = _get_dist_info()

    # Best-effort distributed init so we can barrier for final concatenation.
    _maybe_init_torch_distributed(dist)

    # Filters
    split_set: Optional[Set[str]] = None
    if config.splits:
        split_set = {s.strip() for s in config.splits.split(",") if s.strip()}

    dataset_set: Optional[Set[str]] = None
    if config.datasets:
        dataset_set = {d.strip() for d in config.datasets.split(",") if d.strip()}

    # Outputs
    dataset_dir = _dataset_dir(out_dir)
    rank_dir = _rank_dataset_dir(dataset_dir, dist.rank)

    save_png_masks = config.mask_store in ("png", "both")
    save_parquet_masks = config.mask_store in ("parquet", "both")
    save_overlays = config.save_overlays
    save_parquet_images = config.image_store == "parquet"

    oc_mask_root = (out_dir / "oc_mask" / f"gpu={dist.rank:03d}") if save_png_masks else None
    od_mask_root = (out_dir / "od_mask" / f"gpu={dist.rank:03d}") if save_png_masks else None
    overlay_root = (out_dir / "overlay" / f"gpu={dist.rank:03d}") if save_overlays else None

    if oc_mask_root is not None: _ensure_dir(oc_mask_root)
    if od_mask_root is not None: _ensure_dir(od_mask_root)
    if overlay_root is not None: _ensure_dir(overlay_root)

    progress_file = _rank_progress_path(out_dir, dist.rank)

    # Resume bookkeeping
    done: Set[str] = set()
    if config.force_recompute:
        logging.info("force-recompute enabled: ignoring any resume state.")
    elif config.resume:
        done = _load_done_set(progress_file)
        logging.info("resume enabled (rank=%d): loaded %d done items", dist.rank, len(done))

    def input_image_iter() -> Iterable[Image]:
        if config.images_parquet is not None:
            logging.info("Reading images from Parquet: %s", config.images_parquet)
            for img in Image.iter_parquet(config.images_parquet, batch_size=config.parquet_read_batch):
                if dataset_set and getattr(img, "dataset", None) not in dataset_set: continue
                if split_set and getattr(img, "split", None) not in split_set: continue
                yield img
        else:
            assert config.images_dir is not None
            logging.info("Creating images from directory: %s", config.images_dir)
            images = _make_images_from_dir(
                images_dir=config.images_dir,
                pattern=config.image_pattern,
                dataset=config.inference_dataset,
                split=config.inference_split,
            )
            for im in images:
                if dataset_set and getattr(im, "dataset", None) not in dataset_set: continue
                if split_set and getattr(im, "split", None) not in split_set: continue
                yield im

    # Device for this rank
    device = _device_for_rank(config.device, dist.local_rank)
    _pin_torch_cuda_device(device)
    yolo_device = _ultralytics_device_str_for_current_process(device)

    # Predictor Setup
    yolo_kwargs: Dict[str, Any] = dict(
        weights=config.yolo_weights, device=yolo_device, imgsz=config.imgsz, 
        conf=config.conf, iou=config.iou,
    )
    if config.yolo_batch is not None:
        yolo_kwargs["batch_size"] = int(config.yolo_batch)

    try:
        yolo_cfg = YoloPredictorConfig(**yolo_kwargs)
    except TypeError:
        yolo_kwargs.pop("batch_size", None)
        yolo_cfg = YoloPredictorConfig(**yolo_kwargs)

    sam_kwargs: Dict[str, Any] = {
        "checkpoint": config.medsam_checkpoint, 
        "device": device, 
        "use_amp": config.sam_amp,
        "resize_backend": config.sam_resize_backend
    }
    try:
        sam_cfg = MedSamPredictorConfig(**sam_kwargs)
    except TypeError:
        sam_kwargs.pop("use_amp", None)
        sam_kwargs.pop("resize_backend", None)
        sam_cfg = MedSamPredictorConfig(**sam_kwargs)

    pred_cfg = _PredictorConfig(box_pad_frac=config.box_pad_frac)
    predictor = Predictor(yolo_cfg, sam_cfg, pred_cfg)

    images_root = config.images_dir.resolve() if config.images_dir else None
    part_idx = _existing_part_count(rank_dir)

    def process_batch(batch: List[Image]) -> None:
        nonlocal part_idx
        if not batch:
            return

        try:
            preds = predictor.predict(batch)
        except Exception as e:
            logging.error("Batch failed on rank=%d: %r. Trying per-image fallback.", dist.rank, e)
            preds = []
            for img in batch:
                try:
                    preds.extend(predictor.predict([img]))
                except Exception:
                    if config.resume:
                        ip = str(img.image_path)
                        done.add(ip)
                        _append_done(progress_file, image_path=ip, uid=img.uid)
            if not preds: return

        batch_done_records: List[Tuple[str, str]] = []

        for img in preds:
            try:
                img.update_mask_dice(fallback_to_boxes=True)
            except Exception:
                pass

            _ensure_scalar_metrics(img)
            rel_subdir = _relative_subdir(img.image_path, images_root=images_root)

            if save_png_masks and oc_mask_root and od_mask_root:
                _save_pred_masks_for_image(img=img, oc_mask_root=oc_mask_root, od_mask_root=od_mask_root,
                                           rel_subdir=rel_subdir)

            if save_parquet_masks:
                _pack_pred_masks_for_parquet(img)

            if save_overlays and overlay_root:
                overlay_path = overlay_root / rel_subdir / img.image_path.name
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    img.visualize(show=False, save_path=overlay_path, dpi=140, mask_alpha=0.7)
                    if img.extras is None: img.extras = {}
                    img.extras["overlay_path"] = str(overlay_path)
                except Exception:
                    pass

            batch_done_records.append((str(img.image_path), str(img.uid)))

        # Write part using robust Image class writer
        part_path = _rank_part_path(rank_dir, part_idx)
        Image.save_parquet(
            images=preds,
            path=part_path,
            drop_none=False,
            include_image_bytes=save_parquet_images,
            include_mask_bytes=save_parquet_masks,
            compression="zstd",
            write_batch=1024,
        )
        part_idx += 1

        if config.resume and not config.force_recompute:
            for ip, uid in batch_done_records:
                done.add(ip)
                _append_done(progress_file, image_path=ip, uid=uid)

    # Main Processing Loop
    batch: List[Image] = []
    n_seen = 0

    sharded_iter = _iter_sharded_images(input_image_iter(), rank=dist.rank, world_size=dist.world_size)

    for img in tqdm(sharded_iter, desc=f"Predict (rank={dist.rank})", unit="img"):
        n_seen += 1
        if not config.force_recompute and config.resume and str(img.image_path) in done:
            continue

        batch.append(img)
        if len(batch) >= config.predict_batch_size:
            process_batch(batch)
            batch = []

    if batch:
        process_batch(batch)

    logging.info("rank=%d finished. seen=%d parts=%d", dist.rank, n_seen, part_idx)
    _barrier_if_possible()

    # Rank 0: Summarize and Concatenate
    summary = None
    if dist.rank == 0:
        logging.info("Summarizing predictions from dataset %s", dataset_dir)
        summary = summarize_predictions_dataset(
            dataset_dir, splits=split_set, datasets=dataset_set, batch_size=config.parquet_read_batch
        )

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if config.summary_csv:
            append_summary_row_to_csv(config.summary_csv, dataset_dir, summary)

        final_path = out_dir / config.final_parquet_name
        logging.info("Concatenating into single file: %s", final_path)

        Image.save_parquet(
            Image.iter_parquet(dataset_dir),
            path=final_path,
            include_image_bytes=save_parquet_images,
            include_mask_bytes=save_parquet_masks,
            compression=config.final_parquet_compression,
            write_batch=config.final_parquet_batch
        )
        logging.info("Done.")
    
    return summary


# =========================
# CLI
# =========================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run YOLO+MedSAM predictions (resumable + multi-GPU) and summarize results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run predictions from parquet
  python -m src.scripts.predict_summarize \\
      --images-parquet images.parquet \\
      --yolo-weights best.pt \\
      --medsam-checkpoint medsam_vit_b.pth \\
      --out-dir predictions/

  # Run predictions from directory
  python -m src.scripts.predict_summarize \\
      --images-dir /path/to/images \\
      --yolo-weights best.pt \\
      --medsam-checkpoint medsam_vit_b.pth \\
      --out-dir predictions/
        """,
    )

    p.add_argument("--images-parquet", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--image-pattern", type=str, default="*")
    p.add_argument("--inference-dataset", type=str, default="inference")
    p.add_argument("--inference-split", type=str, default=None)

    p.add_argument("--yolo-weights", type=Path, required=True)
    p.add_argument("--medsam-checkpoint", type=Path, required=True)

    p.add_argument("--device", type=str, default="cuda", help="cuda | cuda:<id> | cpu | mps")

    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--imgsz", type=int, default=640)

    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--box-pad-frac", type=float, default=0.05)

    p.add_argument("--sam-amp", action="store_true", help="Enable CUDA AMP for MedSAM.")
    p.add_argument("--sam-no-amp", action="store_true", help="Disable CUDA AMP for MedSAM.")
    p.add_argument("--sam-resize-backend", type=str, default="cv2", choices=("cv2", "pil"))

    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--summary-csv", type=Path, default=None)

    p.add_argument("--splits", type=str, default=None)
    p.add_argument("--datasets", type=str, default=None)

    p.add_argument("--parquet-read-batch", type=int, default=1024)
    p.add_argument("--predict-batch-size", type=int, default=128)

    p.add_argument("--yolo-batch", type=int, default=None, help="Ultralytics YOLO batch size for inference.")

    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-recompute", action="store_true")

    p.add_argument(
        "--mask-store",
        type=str,
        choices=("png", "parquet", "both", "none"),
        default="parquet",
        help="Where to store predicted masks.",
    )

    p.add_argument(
        "--image-store",
        type=str,
        choices=("parquet", "none"),
        default="none",
        help="Whether to embed raw image bytes in the output parquets.",
    )

    p.add_argument("--final-parquet-name", type=str, default="predictions.parquet")
    p.add_argument("--final-parquet-compression", type=str, default="zstd")
    p.add_argument("--final-parquet-batch", type=int, default=2048)

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    
    # Validate input source
    if not args.images_parquet and not args.images_dir:
        raise ValueError("You must pass --images-parquet or --images-dir to run predictions.")
    
    # Determine AMP setting
    sam_amp = True  # default
    if args.sam_no_amp:
        sam_amp = False
    elif args.sam_amp:
        sam_amp = True
    
    # Build config from CLI args
    config = PredictConfig(
        out_dir=args.out_dir,
        yolo_weights=args.yolo_weights,
        medsam_checkpoint=args.medsam_checkpoint,
        images_parquet=args.images_parquet,
        images_dir=args.images_dir,
        image_pattern=args.image_pattern,
        inference_dataset=args.inference_dataset,
        inference_split=args.inference_split,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        yolo_batch=args.yolo_batch,
        sam_amp=sam_amp,
        sam_resize_backend=args.sam_resize_backend,
        box_pad_frac=args.box_pad_frac,
        save_overlays=args.save_overlays,
        mask_store=args.mask_store,
        image_store=args.image_store,
        splits=args.splits,
        datasets=args.datasets,
        parquet_read_batch=args.parquet_read_batch,
        predict_batch_size=args.predict_batch_size,
        resume=args.resume,
        force_recompute=args.force_recompute,
        final_parquet_name=args.final_parquet_name,
        final_parquet_compression=args.final_parquet_compression,
        final_parquet_batch=args.final_parquet_batch,
        summary_csv=args.summary_csv,
    )
    
    run_predictions(config)


if __name__ == "__main__":
    main()