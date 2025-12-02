#!/usr/bin/env python3
"""
Filter a predictions parquet -> write FILTERED parquet (NOT resampled) ->
(optionally) combine with an EXTRA parquet (which *is* resampled via repetition)
-> write FINAL parquet.

CLI args:
  --pred-in            (required)
  --extra-in           (optional)
  --out                (required)
  --conf-thresh        (required)
  --repeat-resample    (used only if --extra-in is provided)
  --exclude-datasets
  --strip-gt-datasets

Behavior:
- If --extra-in is PROVIDED:
    - Identical to original behavior:
        * Filtered set is NEVER resampled (written to <out_stem>__filtered.parquet).
        * Extra set IS resampled (repeated) by --repeat-resample.
        * Combined (extra_resampled + filtered_once) is written to --out.
- If --extra-in is NOT provided:
    - Only filtering is performed.
    - The filtered set is written directly to --out (no combining, no resampling).

Implementation notes:
- Uses ParquetFile.iter_batches() for nested-safe streaming read.
- Writes via Image.save_parquet() for robust nested-safe streaming write.
- Filtering logic (hardcoded defaults):
    - Dataset gating: exclude list applied first.
    - Optional GT stripping for specified datasets BEFORE filtering.
    - Require-GT mode: "either" (no GT-box gating).
    - YOLO conf filter applies ONLY to no-GT rows (scope="no_gt").
    - Requires BOTH disc and cup conf >= threshold; missing conf is NOT allowed (allow_missing=False).
    - Pseudo-labeling:
        * Rows without GT: copy predictions -> GT, clear predictions, force split="train"
        * Rows with real GT are DROPPED (prepare_for_yolo=True)
- Output bytes:
    - Filtered output includes mask bytes, not image bytes.
    - Final output includes mask bytes, not image bytes.
- Compression: zstd
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, Set, Tuple

import numpy as np

from src.imgpipe.image import Image


# ============================================================
# Hardcoded defaults (edit here if needed)
# ============================================================

SEED = 42

# Streaming IO
READ_BATCH = 2048
WRITE_BATCH = 1024

# Filtering logic
YOLO_FILTER_SCOPE = "no_gt"   # only apply conf filter to rows WITHOUT GT
YOLO_ALLOW_MISSING = False    # if yolo conf missing -> fail
PREPARE_FOR_YOLO = True       # drop true-GT rows; only keep pseudo-labeled rows

# Bytes & compression
FILTER_INCLUDE_MASK_BYTES = True
FILTER_INCLUDE_IMAGE_BYTES = False
FINAL_INCLUDE_MASK_BYTES = True
FINAL_INCLUDE_IMAGE_BYTES = False
COMPRESSION = "zstd"


# ============================================================
# Helpers
# ============================================================

def ensure_parent_dir(p: Path) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def csv_set(s: Optional[str]) -> Optional[Set[str]]:
    if not s:
        return None
    out = {x.strip() for x in s.split(",") if x.strip()}
    return out or None


def is_finite(x: Any) -> bool:
    try:
        return (x is not None) and bool(np.isfinite(float(x)))
    except Exception:
        return False


def dataset_name(img: Image) -> str:
    v = getattr(img, "dataset", None)
    s = str(v).strip() if v is not None else ""
    return s if s else "UNKNOWN"


def split_name(img: Image) -> str:
    v = getattr(img, "split", None)
    s = str(v).strip() if v is not None else ""
    return s if s else "NONE"


def log_counter(title: str, ctr: Counter) -> None:
    total = sum(int(v) for v in ctr.values())
    logging.info("%s (total=%d)", title, total)
    for k in sorted(ctr.keys()):
        logging.info("  %-24s %d", k, int(ctr[k]))

# ============================================================
# GT / pseudo-label utilities
# ============================================================

def has_any_gt(img: Image) -> bool:
    return bool(
        getattr(img, "gt_disc_box", None) is not None
        or getattr(img, "gt_cup_box", None) is not None
        or getattr(img, "gt_disc_mask", None) is not None
        or getattr(img, "gt_cup_mask", None) is not None
    )


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

    # boxes (prefer intermediate preds if present)
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

    # scalars
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
    pred_attrs = [
        "pred_disc_mask",
        "pred_cup_mask",
        "inter_pred_disc_box",
        "inter_pred_cup_box",
        "pred_disc_box",
        "pred_cup_box",
        "pred_cd_ratio",
        "pred_cdr",
        "pred_rdr",
        "yolo_disc_conf",
        "yolo_cup_conf",
        "sam_disc_conf",
        "sam_cup_conf",
        "mask_dice_disc",
        "mask_dice_cup",
    ]
    for a in pred_attrs:
        if hasattr(img, a):
            try:
                setattr(img, a, None)
            except Exception:
                pass
    for a in ("yolo_label_path", "overlay_path"):
        if hasattr(img, a):
            try:
                setattr(img, a, None)
            except Exception:
                pass


# ============================================================
# Filtering
# ============================================================

def dataset_allowed(ds: Optional[str], *, exclude: Optional[Set[str]]) -> bool:
    ds_s = str(ds).strip() if ds is not None else ""
    ds_norm = ds_s if ds_s else "UNKNOWN"
    if exclude is not None and ds_norm in exclude:
        return False
    return True


def should_apply_conf_filter(*, has_gt: bool) -> bool:
    if YOLO_FILTER_SCOPE == "all":
        return True
    if YOLO_FILTER_SCOPE == "with_gt":
        return has_gt
    return (not has_gt)  # "no_gt"


def passes_yolo_conf_both(img: Image, *, thresh: float) -> bool:
    v_disc = getattr(img, "yolo_disc_conf", None)
    v_cup = getattr(img, "yolo_cup_conf", None)

    if not is_finite(v_disc):
        disc_ok = bool(YOLO_ALLOW_MISSING)
    else:
        disc_ok = float(v_disc) >= float(thresh)

    if not is_finite(v_cup):
        cup_ok = bool(YOLO_ALLOW_MISSING)
    else:
        cup_ok = float(v_cup) >= float(thresh)

    return bool(disc_ok and cup_ok)


def filter_and_postprocess_stream(
    pred_in: Path,
    *,
    out_filtered: Path,
    conf_thresh: float,
    exclude_datasets: Optional[Set[str]],
    strip_gt_datasets: Optional[Set[str]],
) -> Tuple[int, Counter, Counter]:
    """
    Streams pred_in twice:
      - pass 1: count + basic stats
      - pass 2: write filtered_out
    Returns:
      (n_written, per_ds_out, per_split_out)
    """
    per_ds_in = Counter()
    per_ds_out = Counter()
    per_split_out = Counter()

    # pass 1: stats
    n_in = 0
    n_after_ds_gate = 0
    n_strip_seen = 0
    n_strip_changed = 0
    n_conf_checked = 0
    n_conf_failed = 0
    n_true_gt_dropped = 0
    n_no_gt_kept = 0

    for img in Image.iter_parquet(pred_in, batch_size=READ_BATCH):
        n_in += 1
        per_ds_in[dataset_name(img)] += 1

        if not dataset_allowed(getattr(img, "dataset", None), exclude=exclude_datasets):
            continue
        n_after_ds_gate += 1

        # optional strip GT by dataset
        ds_raw = getattr(img, "dataset", None)
        ds_raw_s = str(ds_raw).strip() if ds_raw is not None else ""
        if strip_gt_datasets is not None and ds_raw_s in strip_gt_datasets:
            n_strip_seen += 1
            if strip_gt_annotations_inplace(img):
                n_strip_changed += 1

        has_gt = has_any_gt(img)

        if should_apply_conf_filter(has_gt=has_gt):
            n_conf_checked += 1
            if not passes_yolo_conf_both(img, thresh=conf_thresh):
                n_conf_failed += 1
                continue

        if PREPARE_FOR_YOLO and has_gt:
            n_true_gt_dropped += 1
            continue

        if not has_gt:
            n_no_gt_kept += 1

    logging.info("Filter stats:")
    logging.info("  pred_in rows scanned:           %d", n_in)
    logging.info("  after exclude-datasets gating:  %d", n_after_ds_gate)
    if strip_gt_datasets:
        logging.info("  strip-gt rows seen/changed:     %d / %d", n_strip_seen, n_strip_changed)
    logging.info("  YOLO conf checks / failed:      %d / %d", n_conf_checked, n_conf_failed)
    logging.info("  true-GT rows dropped:           %d", n_true_gt_dropped)
    logging.info("  no-GT rows kept (pseudo-label): %d", n_no_gt_kept)
    log_counter("Per-dataset pred_in (raw)", per_ds_in)

    # pass 2: stream + write
    n_written = 0

    def gen() -> Iterable[Image]:
        nonlocal n_written
        for img in Image.iter_parquet(pred_in, batch_size=READ_BATCH):
            if not dataset_allowed(getattr(img, "dataset", None), exclude=exclude_datasets):
                continue

            ds_raw = getattr(img, "dataset", None)
            ds_raw_s = str(ds_raw).strip() if ds_raw is not None else ""
            if strip_gt_datasets is not None and ds_raw_s in strip_gt_datasets:
                strip_gt_annotations_inplace(img)

            has_gt = has_any_gt(img)

            if should_apply_conf_filter(has_gt=has_gt):
                if not passes_yolo_conf_both(img, thresh=conf_thresh):
                    continue

            if PREPARE_FOR_YOLO and has_gt:
                continue

            if not has_gt:
                copy_predictions_to_gt_inplace(img)
                remove_predictions_inplace(img)
                set_split_inplace(img, "train")

            n_written += 1
            per_ds_out[dataset_name(img)] += 1
            per_split_out[split_name(img)] += 1
            yield img

    Image.save_parquet(
        gen(),
        path=out_filtered,
        drop_none=False,
        include_image_bytes=FILTER_INCLUDE_IMAGE_BYTES,
        include_mask_bytes=FILTER_INCLUDE_MASK_BYTES,
        compression=COMPRESSION,
        write_batch=WRITE_BATCH,
    )

    return n_written, per_ds_out, per_split_out


# ============================================================
# Combine (extra is resampled; filtered is not)
# ============================================================

def combine_streams(
    *,
    extra_in: Path,
    filtered_in: Path,
    out: Path,
    repeat_resample: int,
) -> None:
    n_in_extra = 0
    n_in_filtered = 0
    n_out = 0

    def gen() -> Iterator[Image]:
        nonlocal n_in_extra, n_in_filtered, n_out

        # EXTRA first (resampled)
        for img in Image.iter_parquet(extra_in, batch_size=READ_BATCH):
            n_in_extra += 1
            for _ in range(int(repeat_resample)):
                n_out += 1
                yield img

        # FILTERED once (never resampled)
        for img in Image.iter_parquet(filtered_in, batch_size=READ_BATCH):
            n_in_filtered += 1
            n_out += 1
            yield img

    Image.save_parquet(
        gen(),
        path=out,
        drop_none=False,
        include_image_bytes=FINAL_INCLUDE_IMAGE_BYTES,
        include_mask_bytes=FINAL_INCLUDE_MASK_BYTES,
        compression=COMPRESSION,
        write_batch=WRITE_BATCH,
    )

    logging.info("Combine stats:")
    logging.info("  extra_in rows scanned:          %d", n_in_extra)
    logging.info("  filtered_in rows scanned:       %d", n_in_filtered)
    logging.info("  final out rows written (logic): %d", n_out)


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Filter predictions parquet (not resampled). "
            "If --extra-in is given, resample extra parquet and combine -> final output. "
            "If --extra-in is omitted, only the filtered output is written to --out."
        )
    )
    p.add_argument("--pred-in", type=Path, required=True)
    p.add_argument("--extra-in", type=Path, default=None, help="Optional extra parquet to resample and combine.")
    p.add_argument("--out", type=Path, required=True)

    p.add_argument(
        "--conf-thresh",
        type=float,
        required=True,
        help="YOLO confidence threshold applied to BOTH disc and cup (scope hardcoded to no-GT rows).",
    )
    p.add_argument(
        "--repeat-resample",
        type=int,
        default=2,
        help="Repeat factor for rows from --extra-in (ignored if --extra-in is not provided).",
    )

    p.add_argument("--exclude-datasets", type=str, default=None, help="Comma-separated datasets to exclude.")
    p.add_argument("--strip-gt-datasets", type=str, default=None, help="Comma-separated datasets to strip GT before filtering.")

    return p.parse_args(argv)


# ============================================================
# Main
# ============================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    pred_in = Path(args.pred_in)
    extra_in: Optional[Path] = Path(args.extra_in) if args.extra_in is not None else None
    out = ensure_parent_dir(Path(args.out)).resolve()

    if int(args.repeat_resample) < 1:
        raise ValueError("--repeat-resample must be >= 1")

    exclude = csv_set(args.exclude_datasets)
    strip_gt = csv_set(args.strip_gt_datasets)

    # If extra_in is provided, keep original behavior:
    #   filtered_out = <out_stem>__filtered.parquet
    #   final combined written to out
    #
    # If extra_in is NOT provided:
    #   filtered_out = out (filtered file is the final output; no combining/resampling).
    if extra_in is not None:
        filtered_out = out.with_name(f"{out.stem}__filtered.parquet")
    else:
        filtered_out = out

    logging.info("=== CONFIG ===")
    logging.info("pred_in:          %s", pred_in.resolve())
    if extra_in is not None:
        logging.info("extra_in:         %s", extra_in.resolve())
    else:
        logging.info("extra_in:         (none; skipping combine/resample)")
    logging.info("out:              %s", out)
    logging.info("filtered_out:     %s", filtered_out)
    logging.info("conf_thresh:      %.6f", float(args.conf_thresh))
    logging.info("repeat_resample:  %d (used only if extra_in is provided)", int(args.repeat_resample))
    logging.info("exclude_datasets: %s", ",".join(sorted(exclude)) if exclude else "(none)")
    logging.info("strip_gt_datasets:%s", ",".join(sorted(strip_gt)) if strip_gt else "(none)")
    logging.info(
        "Hardcoded: scope=%s allow_missing=%s prepare_for_yolo=%s read_batch=%d write_batch=%d compression=%s",
        YOLO_FILTER_SCOPE,
        YOLO_ALLOW_MISSING,
        PREPARE_FOR_YOLO,
        READ_BATCH,
        WRITE_BATCH,
        COMPRESSION,
    )

    # 1) Filter predictions -> filtered_out (never resampled)
    n_written, per_ds_out, per_split_out = filter_and_postprocess_stream(
        pred_in,
        out_filtered=filtered_out,
        conf_thresh=float(args.conf_thresh),
        exclude_datasets=exclude,
        strip_gt_datasets=strip_gt,
    )

    if n_written == 0:
        raise RuntimeError("Filtered output is empty. Check exclude/strip/conf settings.")

    log_counter("Per-dataset FILTERED output", per_ds_out)
    log_counter("Per-split  FILTERED output", per_split_out)

    # 2) If extra_in is provided, combine: extra_in resampled + filtered_out once -> out
    if extra_in is not None:
        combine_streams(
            extra_in=extra_in,
            filtered_in=filtered_out,
            out=out,
            repeat_resample=int(args.repeat_resample),
        )
        logging.info("Done. Final out (combined): %s", out)
    else:
        # No extra_in: filtered_out is already the final output.
        logging.info("Done. Final out (filtered only, no combine): %s", filtered_out)


if __name__ == "__main__":
    main()