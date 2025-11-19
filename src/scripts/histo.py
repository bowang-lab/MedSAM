#!/usr/bin/env python3
"""
Plot histograms of prediction quality from a saved_images JSONL file.

This script expects a JSONL file produced by `Predictor.predict`, where each
line is a JSON record with (at least) the following structure:

- "dice": {
      "disc": <float or null>,
      "cup":  <float or null>,
  }

- "metrics": {
      "gt": {
          "cdr_v": ...,
          "cdr_h": ...,
          "rim_over_disc": ...,
          "I_over_S": ...,
          ... (optionally more)
      },
      "pred": {
          "cdr_v": ...,
          "cdr_h": ...,
          "rim_over_disc": ...,
          "I_over_S": ...,
          ... (same keys as gt)
      }
  }

The script:
- Reads all records from a JSONL file;
- Optionally filters by split (e.g. train/val/test);
- Collects:
    * disc Dice values
    * cup Dice values
    * absolute errors (|pred - gt|) for each metric key
      present in both metrics["gt"] and metrics["pred"];
- Writes one histogram PNG per metric to the output directory.

Examples
--------
Basic usage:

    python plot_error_histograms.py \
        --pred-jsonl /path/to/saved_images.jsonl \
        --out-dir /path/to/histograms

With filtering and custom bins:

    python plot_error_histograms.py \
        --pred-jsonl /path/to/saved_images.jsonl \
        --out-dir /path/to/histograms \
        --splits test \
        --bins 40
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import matplotlib.pyplot as plt
import numpy as np


# =========================
# Helpers
# =========================

def _is_finite(x: Optional[float]) -> bool:
    return (x is not None) and np.isfinite(x)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create histograms of Dice scores and metric absolute errors "
                    "from a saved_images JSONL file."
    )
    p.add_argument(
        "--pred-jsonl",
        type=Path,
        required=True,
        help="Path to predictions JSONL (saved_images.jsonl from Predictor).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where histogram PNGs will be saved.",
    )
    p.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Optional comma-separated list of splits to include (e.g. 'train,val,test'). "
             "If omitted, all splits are used.",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for each histogram (default: 30).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="DPI for saved histogram PNGs (default: 120).",
    )
    p.add_argument(
        "--filename-prefix",
        type=str,
        default="",
        help="Optional prefix for output filenames (e.g. 'papila_').",
    )
    return p.parse_args(argv)


def load_records(
    jsonl_path: Path,
    splits: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Load prediction records from JSONL, optionally filtering by split."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Predictions JSONL not found: {jsonl_path}")

    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            if splits is not None:
                rec_split = rec.get("split")
                if rec_split not in splits:
                    continue

            records.append(rec)
    return records


def collect_metrics(records: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    From a list of prediction records, collect:

      - "dice_disc" -> list of disc Dice values
      - "dice_cup"  -> list of cup Dice values
      - "<metric>_abs_err" -> |pred - gt| for each metric key present in both
        metrics["gt"] and metrics["pred"]

    Returns:
        dict: metric_name -> list of values
    """
    metric_values: Dict[str, List[float]] = defaultdict(list)

    for rec in records:
        # Dice scores
        dice = rec.get("dice", {})
        d_disc = dice.get("disc")
        d_cup = dice.get("cup")

        if _is_finite(d_disc):
            metric_values["dice_disc"].append(float(d_disc))
        if _is_finite(d_cup):
            metric_values["dice_cup"].append(float(d_cup))

        # Metric absolute errors
        metrics = rec.get("metrics", {})
        gt_m = metrics.get("gt", {}) or {}
        pr_m = metrics.get("pred", {}) or {}

        # Only consider keys present in both gt and pred
        for key in set(gt_m.keys()).intersection(pr_m.keys()):
            gt_val = gt_m.get(key)
            pr_val = pr_m.get(key)
            if _is_finite(gt_val) and _is_finite(pr_val):
                err = abs(float(pr_val) - float(gt_val))
                metric_values[f"{key}_abs_err"].append(err)

    return metric_values


def plot_histogram(
    values: List[float],
    name: str,
    out_path: Path,
    bins: int,
    dpi: int,
) -> None:
    """Plot and save a single histogram."""
    if not values:
        print(f"[WARN] No values for metric '{name}', skipping histogram.")
        return

    arr = np.asarray(values, dtype=float)

    plt.figure()
    plt.hist(arr, bins=bins)
    plt.title(f"Histogram of {name}")
    plt.xlabel(name)
    plt.ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()

    print(f"[INFO] Saved histogram for '{name}' to {out_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split filter
    split_set: Optional[Set[str]] = None
    if args.splits:
        split_set = {s.strip() for s in args.splits.split(",") if s.strip()}

    print(f"[INFO] Loading records from: {args.pred_jsonl}")
    if split_set is not None:
        print(f"[INFO] Filtering to splits: {sorted(split_set)}")

    records = load_records(args.pred_jsonl, splits=split_set)
    print(f"[INFO] Loaded {len(records)} records after filtering.")

    if not records:
        print("[WARN] No records found. Exiting without plotting.")
        return

    metric_values = collect_metrics(records)

    # Explicitly mention key histograms of interest if available
    preferred_order = [
        "dice_disc",
        "dice_cup",
        "cdr_v_abs_err",
        "cdr_h_abs_err",
        "rim_over_disc_abs_err",
        "I_over_S_abs_err",
    ]
    # Then any remaining metrics in sorted order
    remaining = sorted(
        set(metric_values.keys()) - set(preferred_order)
    )
    all_metrics = [m for m in preferred_order if m in metric_values] + remaining

    print("[INFO] Metrics found:")
    for m in all_metrics:
        print(f"  - {m} (n={len(metric_values[m])})")

    # Plot histograms
    for metric_name in all_metrics:
        values = metric_values[metric_name]
        fname = f"{args.filename_prefix}{metric_name}_hist.png"
        out_path = out_dir / fname
        plot_histogram(values, metric_name, out_path, bins=args.bins, dpi=args.dpi)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()