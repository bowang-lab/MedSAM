#!/usr/bin/env python3
# File: src/tools/filter_parquet.py
"""
Filter a Parquet file of Image predictions (NO resampling) using Image.save_parquet
for robust STREAMING WRITE.

Key properties:
- Robust streaming read: uses ParquetFile.iter_batches() (avoids dataset Scanner nested/chunked issues).
- Robust streaming write: delegated to Image.save_parquet(), which:
    - streams rows in batches
    - infers schema from first non-empty batch
    - makes schema fully nullable
    - casts each batch to the writer schema (safe=False)

Filtering / labeling behavior:
- Dataset gating:
    - Optional exclude list:      --exclude-datasets  (applies first)
    - Optional include whitelist: --datasets
- Optional "strip GT" for specified datasets BEFORE filtering: --strip-gt-datasets
- Optional GT-box presence filter: --require-gt (any/all/none/either)
- YOLO confidence filtering (disc AND cup), with scope: --yolo-filter-scope = no_gt|with_gt|all
- Pseudo-labeling:
    - For rows without GT: copy predictions -> GT, CLEAR predictions, and force split=train
    - With --prepare-for-yolo (default ON): drop real-GT rows
    - With --no-prepare-for-yolo: keep real-GT rows and split them train/val/test with seed
"""

from __future__ import annotations
import argparse
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pyarrow.parquet as pq

from src.imgpipe.image import Image


# -------------------------
# Small general helpers
# -------------------------


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def csv_set(s: Optional[str]) -> Optional[Set[str]]:
    if not s:
        return None
    parts = [x.strip() for x in s.split(",")]
    out = {x for x in parts if x}
    return out or None


def is_finite(x: Any) -> bool:
    try:
        return (x is not None) and bool(np.isfinite(float(x)))
    except Exception:
        return False


def dataset_name(img: Image) -> str:
    v = getattr(img, "dataset", None)
    return str(v) if (v is not None and str(v).strip()) else "UNKNOWN"


def split_name(img: Image) -> str:
    v = getattr(img, "split", None)
    return str(v) if (v is not None and str(v).strip()) else "NONE"


def log_counter(title: str, ctr: Counter) -> None:
    total = sum(int(v) for v in ctr.values())
    logging.info("%s (total=%d)", title, total)
    for k in sorted(ctr.keys()):
        logging.info("  %-24s %d", k, int(ctr[k]))



# -------------------------
# GT + pseudo-label utilities
# -------------------------


def has_any_gt(img: Image) -> bool:
    return bool(
        getattr(img, "gt_disc_box", None) is not None
        or getattr(img, "gt_cup_box", None) is not None
        or getattr(img, "gt_disc_mask", None) is not None
        or getattr(img, "gt_cup_mask", None) is not None
    )


def has_gt_box(img: Image, mode: str) -> bool:
    disc = getattr(img, "gt_disc_box", None) is not None
    cup = getattr(img, "gt_cup_box", None) is not None
    if mode == "any":
        return disc or cup
    if mode == "all":
        return disc and cup
    if mode == "none":
        return (not disc) and (not cup)
    raise ValueError(f"Unknown require-gt mode: {mode}")


def set_split_inplace(img: Image, split: str) -> None:
    if hasattr(img, "set_split"):
        try:
            img.set_split(split)
            return
        except Exception:
            pass
    try:
        img.split = split  # type: ignore[attr-defined]
    except Exception:
        pass


def strip_gt_annotations_inplace(img: Image) -> bool:
    changed = False

    def _set_none(attr: str) -> None:
        nonlocal changed
        if hasattr(img, attr):
            try:
                if getattr(img, attr, None) is not None:
                    setattr(img, attr, None)
                    changed = True
            except Exception:
                pass

    for a in (
        "gt_disc_box",
        "gt_cup_box",
        "gt_disc_mask",
        "gt_cup_mask",
        "gt_cd_ratio",
        "gt_cdr",
        "gt_rdr",
    ):
        _set_none(a)

    return changed


def copy_predictions_to_gt_inplace(img: Image) -> None:
    # masks
    if getattr(img, "gt_disc_mask", None) is None and getattr(img, "pred_disc_mask", None) is not None:
        try:
            img.gt_disc_mask = img.pred_disc_mask  # type: ignore[attr-defined]
        except Exception:
            pass

    if getattr(img, "gt_cup_mask", None) is None and getattr(img, "pred_cup_mask", None) is not None:
        try:
            img.gt_cup_mask = img.pred_cup_mask  # type: ignore[attr-defined]
        except Exception:
            pass

    # boxes
    if getattr(img, "gt_disc_box", None) is None:
        src = getattr(img, "inter_pred_disc_box", None) or getattr(img, "pred_disc_box", None)
        if src is not None:
            try:
                img.gt_disc_box = src  # type: ignore[attr-defined]
            except Exception:
                pass

    if getattr(img, "gt_cup_box", None) is None:
        src = getattr(img, "inter_pred_cup_box", None) or getattr(img, "pred_cup_box", None)
        if src is not None:
            try:
                img.gt_cup_box = src  # type: ignore[attr-defined]
            except Exception:
                pass

    # scalar
    if getattr(img, "gt_cd_ratio", None) is None and getattr(img, "pred_cd_ratio", None) is not None:
        try:
            img.gt_cd_ratio = img.pred_cd_ratio  # type: ignore[attr-defined]
        except Exception:
            pass

    try:
        img.ensure_boxes_from_masks()
    except Exception:
        pass


def remove_predictions_inplace(img: Image) -> None:
    """
    After pseudo-labeling (copy pred -> gt), clear prediction fields so the row no longer
    carries predictions (only GT remains).
    Safe: only clears attributes that exist.
    """
    pred_attrs = [
        # masks
        "pred_disc_mask",
        "pred_cup_mask",
        # boxes
        "inter_pred_disc_box",
        "inter_pred_cup_box",
        "pred_disc_box",
        "pred_cup_box",
        # scalar preds
        "pred_cd_ratio",
        "pred_cdr",
        "pred_rdr",
        # confidences
        "yolo_disc_conf",
        "yolo_cup_conf",
        "sam_disc_conf",
        "sam_cup_conf",
        # derived metrics
        "mask_dice_disc",
        "mask_dice_cup",
    ]

    for a in pred_attrs:
        if hasattr(img, a):
            try:
                setattr(img, a, None)
            except Exception:
                pass

    # Optional file/path artifacts from prediction runs
    for a in ("yolo_label_path", "overlay_path"):
        if hasattr(img, a):
            try:
                setattr(img, a, None)
            except Exception:
                pass


# -------------------------
# YOLO confidence filtering
# -------------------------


def passes_yolo_conf_both(
    img: Image,
    *,
    thresh_disc: float,
    thresh_cup: float,
    allow_missing: bool,
) -> bool:
    v_disc = getattr(img, "yolo_disc_conf", None)
    v_cup = getattr(img, "yolo_cup_conf", None)

    if not is_finite(v_disc):
        disc_ok = bool(allow_missing)
    else:
        disc_ok = float(v_disc) >= float(thresh_disc)

    if not is_finite(v_cup):
        cup_ok = bool(allow_missing)
    else:
        cup_ok = float(v_cup) >= float(thresh_cup)

    return bool(disc_ok and cup_ok)


def should_apply_conf_filter(*, has_gt: bool, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "with_gt":
        return has_gt
    return (not has_gt)  # "no_gt"


# -------------------------
# Split assignment (index-based)
# -------------------------


def split_counts(n: int) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def make_gt_split_map(gt_row_ids: List[int], *, seed: int) -> Dict[int, str]:
    n = len(gt_row_ids)
    if n == 0:
        return {}

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_train, n_val, n_test = split_counts(n)
    assert n_train + n_val + n_test == n

    split_map: Dict[int, str] = {}
    for j, pi in enumerate(perm.tolist()):
        rid = gt_row_ids[pi]
        if j < n_train:
            split_map[rid] = "train"
        elif j < n_train + n_val:
            split_map[rid] = "val"
        else:
            split_map[rid] = "test"
    return split_map


# -------------------------
# Configuration + filtering logic
# -------------------------


@dataclass(frozen=True)
class FilterConfig:
    include_datasets: Optional[Set[str]]
    exclude_datasets: Optional[Set[str]]
    strip_gt_datasets: Optional[Set[str]]
    require_gt_mode: str  # "" means no filtering
    yolo_thresh_disc: float
    yolo_thresh_cup: float
    yolo_allow_missing: bool
    yolo_filter_scope: str  # no_gt|with_gt|all
    prepare_for_yolo: bool
    seed: int
    read_batch: int
    write_batch: int


def resolve_thresholds(args: argparse.Namespace) -> Tuple[float, float]:
    tdisc = args.yolo_thresh_disc
    tcup = args.yolo_thresh_cup
    if args.yolo_thresh is not None:
        if tdisc is None:
            tdisc = args.yolo_thresh
        if tcup is None:
            tcup = args.yolo_thresh
    if tdisc is None or tcup is None:
        raise ValueError("Need BOTH YOLO thresholds: --yolo-thresh OR both --yolo-thresh-disc/--yolo-thresh-cup")
    return float(tdisc), float(tcup)


def dataset_allowed(ds: Optional[str], *, include: Optional[Set[str]], exclude: Optional[Set[str]]) -> bool:
    ds_s = str(ds).strip() if ds is not None else ""
    ds_norm = ds_s if ds_s else "UNKNOWN"
    if exclude is not None and ds_norm in exclude:
        return False
    if include is not None and ds_norm not in include:
        return False
    return True


def apply_dataset_gating(img: Image, cfg: FilterConfig) -> bool:
    ds = getattr(img, "dataset", None)
    return dataset_allowed(ds, include=cfg.include_datasets, exclude=cfg.exclude_datasets)


def maybe_strip_gt(img: Image, cfg: FilterConfig) -> bool:
    """
    Strip GT if dataset is in cfg.strip_gt_datasets.
    Returns True if changed.
    """
    ds = getattr(img, "dataset", None)
    ds_s = str(ds).strip() if ds is not None else ""
    if cfg.strip_gt_datasets is None:
        return False
    if ds_s in cfg.strip_gt_datasets:
        return strip_gt_annotations_inplace(img)
    return False


def passes_require_gt_filter(img: Image, cfg: FilterConfig) -> bool:
    if not cfg.require_gt_mode:
        return True
    return has_gt_box(img, cfg.require_gt_mode)


def passes_yolo_conf_filter(img: Image, cfg: FilterConfig, *, has_gt: bool) -> bool:
    if not should_apply_conf_filter(has_gt=has_gt, scope=cfg.yolo_filter_scope):
        return True
    return passes_yolo_conf_both(
        img,
        thresh_disc=cfg.yolo_thresh_disc,
        thresh_cup=cfg.yolo_thresh_cup,
        allow_missing=cfg.yolo_allow_missing,
    )


def passes_all_filters_and_get_has_gt(img: Image, cfg: FilterConfig) -> Tuple[bool, bool]:
    """
    Apply all filtering steps in order.
    Returns (keep, has_gt_after_optional_strip).
    """
    if not apply_dataset_gating(img, cfg):
        return False, False

    maybe_strip_gt(img, cfg)
    has_gt = has_any_gt(img)

    if not passes_require_gt_filter(img, cfg):
        return False, has_gt

    if not passes_yolo_conf_filter(img, cfg, has_gt=has_gt):
        return False, has_gt

    return True, has_gt


def postprocess_kept_row_for_output(
    img: Image,
    *,
    has_gt: bool,
    cfg: FilterConfig,
    kept_row_id: int,
    gt_split_map: Dict[int, str],
) -> bool:
    """
    Apply pseudo-labeling/splitting rules in-place.

    Returns True if this row should be written; False if it should be dropped
    (e.g., prepare_for_yolo + has_gt).
    """
    if has_gt:
        if cfg.prepare_for_yolo:
            return False
        split = gt_split_map.get(kept_row_id, "train")
        set_split_inplace(img, split)
        return True

    # No-GT rows become pseudo-labeled GT, then predictions are cleared.
    copy_predictions_to_gt_inplace(img)
    remove_predictions_inplace(img)
    set_split_inplace(img, "train")
    return True


# -------------------------
# CLI
# -------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream-filter a predictions parquet and write output safely (nested OK).")

    p.add_argument("--in-parquet", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--out-name", type=str, default="filtered_predictions.parquet")

    p.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset whitelist to keep.")
    p.add_argument("--exclude-datasets", type=str, default=None, help="Comma-separated dataset names to exclude first.")
    p.add_argument("--require-gt", type=str, choices=("any", "all", "none", "either"), default="either")
    p.add_argument("--strip-gt-datasets", type=str, default=None)

    p.add_argument("--yolo-thresh", type=float, default=None)
    p.add_argument("--yolo-thresh-disc", type=float, default=None)
    p.add_argument("--yolo-thresh-cup", type=float, default=None)
    p.add_argument("--yolo-allow-missing", action="store_true")
    p.add_argument("--yolo-filter-scope", type=str, choices=("no_gt", "all", "with_gt"), default="no_gt")

    p.add_argument("--seed", type=int, default=42)

    g = p.add_mutually_exclusive_group()
    g.add_argument("--prepare-for-yolo", dest="prepare_for_yolo", action="store_true")
    g.add_argument("--no-prepare-for-yolo", dest="prepare_for_yolo", action="store_false")
    p.set_defaults(prepare_for_yolo=True)

    p.add_argument("--read-batch", type=int, default=2048)
    p.add_argument("--write-batch", type=int, default=1024)

    return p.parse_args(argv)


# -------------------------
# Main
# -------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    in_path: Path = args.in_parquet
    out_dir: Path = ensure_dir(args.out_dir)
    out_path: Path = (out_dir / args.out_name).resolve()

    include_set = csv_set(args.datasets)
    exclude_set = csv_set(args.exclude_datasets)
    strip_gt_set = csv_set(args.strip_gt_datasets)

    tdisc, tcup = resolve_thresholds(args)
    require_gt_mode = "" if args.require_gt == "either" else str(args.require_gt)

    cfg = FilterConfig(
        include_datasets=include_set,
        exclude_datasets=exclude_set,
        strip_gt_datasets=strip_gt_set,
        require_gt_mode=require_gt_mode,
        yolo_thresh_disc=tdisc,
        yolo_thresh_cup=tcup,
        yolo_allow_missing=bool(args.yolo_allow_missing),
        yolo_filter_scope=str(args.yolo_filter_scope),
        prepare_for_yolo=bool(args.prepare_for_yolo),
        seed=int(args.seed),
        read_batch=int(args.read_batch),
        write_batch=int(args.write_batch),
    )

    if cfg.prepare_for_yolo and cfg.require_gt_mode in ("any", "all"):
        raise RuntimeError(
            "You used --prepare-for-yolo with --require-gt any/all -> output will be empty. "
            "Use --require-gt either/none or --no-prepare-for-yolo."
        )

    # -------------------------
    # PASS 1: count + discover GT row ids among KEPT rows (for split assignment)
    # -------------------------
    n_in = 0
    n_kept = 0

    n_conf_checked = 0
    n_conf_failed = 0

    n_strip_gt_seen = 0
    n_strip_gt_changed = 0
    per_ds_strip_seen = Counter()
    per_ds_strip_changed = Counter()

    per_ds_in = Counter()
    per_ds_kept = Counter()

    kept_gt_row_ids: List[int] = []
    kept_row_id = 0

    for img in Image.iter_parquet(in_path, batch_size=cfg.read_batch):
        n_in += 1
        ds_display = dataset_name(img)
        per_ds_in[ds_display] += 1

        if not apply_dataset_gating(img, cfg):
            continue

        # strip-GT stats (only for dataset-gated rows)
        ds_raw = getattr(img, "dataset", None)
        ds_raw_s = str(ds_raw).strip() if ds_raw is not None else ""
        if cfg.strip_gt_datasets is not None and ds_raw_s in cfg.strip_gt_datasets:
            n_strip_gt_seen += 1
            per_ds_strip_seen[ds_display] += 1
            if strip_gt_annotations_inplace(img):
                n_strip_gt_changed += 1
                per_ds_strip_changed[ds_display] += 1

        has_gt = has_any_gt(img)

        if cfg.require_gt_mode and not has_gt_box(img, cfg.require_gt_mode):
            continue

        if should_apply_conf_filter(has_gt=has_gt, scope=cfg.yolo_filter_scope):
            n_conf_checked += 1
            if not passes_yolo_conf_both(
                img,
                thresh_disc=cfg.yolo_thresh_disc,
                thresh_cup=cfg.yolo_thresh_cup,
                allow_missing=cfg.yolo_allow_missing,
            ):
                n_conf_failed += 1
                continue

        n_kept += 1
        per_ds_kept[dataset_name(img)] += 1

        if has_gt:
            kept_gt_row_ids.append(kept_row_id)

        kept_row_id += 1

    if n_kept == 0:
        raise RuntimeError("No rows passed your filters; nothing to write.")

    gt_split_map = {} if cfg.prepare_for_yolo else make_gt_split_map(kept_gt_row_ids, seed=cfg.seed)

    # -------------------------
    # PASS 2: stream again, apply same filters, post-process,
    #         track output stats, and delegate writing to Image.save_parquet
    # -------------------------
    per_out_ds = Counter()
    per_out_split = Counter()
    n_out = 0

    def filtered_images_generator() -> Iterable[Image]:
        nonlocal n_out
        kept_row_id_local = 0

        for img in Image.iter_parquet(in_path, batch_size=cfg.read_batch):
            keep, has_gt = passes_all_filters_and_get_has_gt(img, cfg)
            if not keep:
                continue

            write_this = postprocess_kept_row_for_output(
                img,
                has_gt=has_gt,
                cfg=cfg,
                kept_row_id=kept_row_id_local,
                gt_split_map=gt_split_map,
            )

            kept_row_id_local += 1  # counts only "kept" rows (same convention as pass 1)

            if not write_this:
                continue

            # Stats for output
            n_out += 1
            per_out_ds[dataset_name(img)] += 1
            per_out_split[split_name(img)] += 1

            yield img

    # Use the Image class's robust Parquet writer (streaming + schema handling)
    Image.save_parquet(
        filtered_images_generator(),
        path=out_path,
        drop_none=False,
        include_image_bytes=False,
        include_mask_bytes=True,
        compression="zstd",
        write_batch=cfg.write_batch,
    )

    if n_out == 0:
        raise RuntimeError("Output is empty after post-processing. Check your flags/filters.")

    # -------------------------
    # Logging
    # -------------------------
    logging.info("Input rows:                   %d", n_in)
    logging.info("Rows kept (post-filter):      %d", n_kept)
    logging.info("YOLO conf filter scope:       %s", cfg.yolo_filter_scope)
    logging.info("YOLO conf checks performed:   %d", n_conf_checked)
    logging.info("YOLO conf failed:             %d", n_conf_failed)
    logging.info("Prepare for YOLO:             %s", cfg.prepare_for_yolo)
    logging.info("Kept rows w/ GT (post-strip): %d", len(kept_gt_row_ids))
    logging.info("Kept rows w/o GT:             %d", (n_kept - len(kept_gt_row_ids)))
    logging.info("Output rows written:          %d", n_out)
    logging.info("Wrote: %s", out_path)

    if cfg.exclude_datasets:
        logging.info("Excluded datasets:            %s", ",".join(sorted(cfg.exclude_datasets)))
    if cfg.include_datasets:
        logging.info("Included datasets (whitelist): %s", ",".join(sorted(cfg.include_datasets)))

    if cfg.strip_gt_datasets:
        logging.info("GT stripping enabled for datasets: %s", ",".join(sorted(cfg.strip_gt_datasets)))
        logging.info("GT strip rows seen:           %d", n_strip_gt_seen)
        logging.info("GT strip rows changed:        %d", n_strip_gt_changed)
        log_counter("Per-dataset GT strip seen (post dataset gating)", per_ds_strip_seen)
        log_counter("Per-dataset GT strip changed (post dataset gating)", per_ds_strip_changed)

    log_counter("Per-dataset BEFORE filtering (raw input)", per_ds_in)
    log_counter("Per-dataset AFTER filtering (kept)", per_ds_kept)
    log_counter("Per-dataset OUTPUT", per_out_ds)
    log_counter("Per-split OUTPUT", per_out_split)


if __name__ == "__main__":
    main()