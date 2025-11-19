#!/usr/bin/env python3
"""
Make histograms of prediction quality from a saved_images.jsonl file.

This script:
- Reads per-image prediction records (JSONL) produced by Predictor.
- Optionally filters images by YOLO confidence (disc/cup).
- Computes distributions for:
    * Disc mask Dice
    * Cup mask Dice
    * |pred - gt| for vertical CDR (cdr_v)
    * |pred - gt| for horizontal CDR (cdr_h), if present
    * |pred - gt| for rim_over_disc
    * |pred - gt| for I_over_S
- Saves one histogram PNG per metric, with:
    * Descriptive axis labels
    * A vertical dashed line marking the mean (after filtering).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Helpers
# =========================

def _is_finite(x: Any) -> bool:
    """Return True if x is a finite scalar float-like value."""
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _load_metric_arrays(
    jsonl_path: Path,
    min_conf: float,
) -> Dict[str, List[float]]:
    """
    Load metrics and errors from a predictions JSONL file.

    Filtering rules:
    - A record is included for a metric only if:
        * The corresponding Dice or metric values are finite.
        * The relevant YOLO confidences are >= min_conf:
            - Disc Dice uses disc confidence.
            - Cup Dice uses cup confidence.
            - Metrics depending on both disc and cup (cdr, rim_over_disc, I_over_S)
              require BOTH disc_conf >= min_conf and cup_conf >= min_conf.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Predictions JSONL not found: {jsonl_path}")

    disc_dice: List[float] = []
    cup_dice: List[float] = []
    cdr_v_err: List[float] = []
    cdr_h_err: List[float] = []
    rim_over_disc_err: List[float] = []
    I_over_S_err: List[float] = []

    n_total = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_total += 1

            yolo = rec.get("yolo", {})
            disc_yolo = yolo.get("disc", {})
            cup_yolo = yolo.get("cup", {})

            disc_conf = disc_yolo.get("conf")
            cup_conf = cup_yolo.get("conf")

            # Guard: require finite conf if we have a threshold > 0
            disc_conf_ok = (
                _is_finite(disc_conf) and float(disc_conf) >= min_conf
            )
            cup_conf_ok = (
                _is_finite(cup_conf) and float(cup_conf) >= min_conf
            )

            # ----- Dice -----
            dice = rec.get("dice", {})
            d_disc = dice.get("disc")
            d_cup = dice.get("cup")

            if _is_finite(d_disc) and disc_conf_ok:
                disc_dice.append(float(d_disc))

            if _is_finite(d_cup) and cup_conf_ok:
                cup_dice.append(float(d_cup))

            # ----- metric errors (require both structures) -----
            metrics = rec.get("metrics", {})
            gt_m = metrics.get("gt", {})
            pr_m = metrics.get("pred", {})

            both_conf_ok = disc_conf_ok and cup_conf_ok

            def _append_abs_err(
                key_src: str,
                dest: List[float],
            ) -> None:
                if not both_conf_ok:
                    return
                gt_val = gt_m.get(key_src)
                pr_val = pr_m.get(key_src)
                if _is_finite(gt_val) and _is_finite(pr_val):
                    dest.append(abs(float(pr_val) - float(gt_val)))

            _append_abs_err("cdr_v", cdr_v_err)
            _append_abs_err("cdr_h", cdr_h_err)
            _append_abs_err("rim_over_disc", rim_over_disc_err)
            _append_abs_err("I_over_S", I_over_S_err)

    print(f"[INFO] Read {n_total} records from {jsonl_path}")
    print(f"[INFO] After confidence filter (min_conf={min_conf}):")
    print(f"       disc_dice           : {len(disc_dice)} values")
    print(f"       cup_dice            : {len(cup_dice)} values")
    print(f"       |cdr_v| errors      : {len(cdr_v_err)} values")
    print(f"       |cdr_h| errors      : {len(cdr_h_err)} values")
    print(f"       |rim_over_disc| err : {len(rim_over_disc_err)} values")
    print(f"       |I_over_S| errors   : {len(I_over_S_err)} values")

    return {
        "disc_dice": disc_dice,
        "cup_dice": cup_dice,
        "cdr_v_err": cdr_v_err,
        "cdr_h_err": cdr_h_err,
        "rim_over_disc_err": rim_over_disc_err,
        "I_over_S_err": I_over_S_err,
    }


def _plot_histogram(
    values: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    bins: int = 40,
) -> None:
    """
    Plot and save a histogram with a vertical line at the mean.
    """
    if not values:
        print(f"[WARN] No values available for {title}; skipping plot.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(values, dtype=float)
    mean_val = float(np.mean(arr))

    plt.figure()
    plt.hist(arr, bins=bins, edgecolor="black", alpha=0.75)
    plt.axvline(mean_val, linestyle="--", linewidth=2, label=f"Mean = {mean_val:.4f}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"[INFO] Saved histogram: {out_path}")


def make_histograms(
    jsonl_path: Path,
    out_dir: Path,
    min_conf: float,
) -> None:
    """
    High-level function to load metrics, then produce histograms
    for all supported metrics/errors.
    """
    metrics = _load_metric_arrays(jsonl_path, min_conf=min_conf)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Dice distributions
    _plot_histogram(
        metrics["disc_dice"],
        title="Disc Mask Dice Coefficient Distribution",
        xlabel="Dice coefficient (disc mask)",
        ylabel="Number of images",
        out_path=out_dir / "disc_dice_hist.png",
    )

    _plot_histogram(
        metrics["cup_dice"],
        title="Cup Mask Dice Coefficient Distribution",
        xlabel="Dice coefficient (cup mask)",
        ylabel="Number of images",
        out_path=out_dir / "cup_dice_hist.png",
    )

    # Absolute error distributions
    _plot_histogram(
        metrics["cdr_v_err"],
        title="Vertical CDR Absolute Error Distribution",
        xlabel="|predicted CDR_v - ground truth CDR_v|",
        ylabel="Number of images",
        out_path=out_dir / "cdr_v_mae_hist.png",
    )

    _plot_histogram(
        metrics["cdr_h_err"],
        title="Horizontal CDR Absolute Error Distribution",
        xlabel="|predicted CDR_h - ground truth CDR_h|",
        ylabel="Number of images",
        out_path=out_dir / "cdr_h_mae_hist.png",
    )

    _plot_histogram(
        metrics["rim_over_disc_err"],
        title="Rim-over-Disc Absolute Error Distribution",
        xlabel="|predicted rim_over_disc - ground truth rim_over_disc|",
        ylabel="Number of images",
        out_path=out_dir / "rim_over_disc_mae_hist.png",
    )

    _plot_histogram(
        metrics["I_over_S_err"],
        title="Inferior/Superior Rim Ratio Absolute Error Distribution",
        xlabel="|predicted I/S - ground truth I/S|",
        ylabel="Number of images",
        out_path=out_dir / "I_over_S_mae_hist.png",
    )


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create histograms of prediction quality from a saved_images.jsonl file.\n"
            "Filters records by YOLO confidence and plots distributions of Dice and metric errors."
        )
    )
    parser.add_argument(
        "--pred-jsonl",
        type=Path,
        required=True,
        help="Path to saved_images.jsonl (predictions JSONL created by Predictor).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Directory to save histogram PNG files. "
            "If omitted, defaults to <pred-jsonl-parent>/histograms."
        ),
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.0,
        help=(
            "Minimum YOLO confidence required to include a prediction. "
            "For Dice, uses the corresponding disc/cup confidence; "
            "for CDR and rim metrics, requires BOTH disc and cup confidences "
            "to be >= this threshold."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    jsonl_path: Path = args.pred_jsonl
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = jsonl_path.resolve().parent / "histograms"

    print(f"[INFO] Predictions JSONL: {jsonl_path}")
    print(f"[INFO] Output directory  : {out_dir}")
    print(f"[INFO] Min confidence    : {args.min_conf}")

    make_histograms(jsonl_path=jsonl_path, out_dir=out_dir, min_conf=args.min_conf)


if __name__ == "__main__":
    main()