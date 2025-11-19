# File: scripts/run_predict_and_summarize.py

#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image as PILImage

from src.imgpipe.image import Image
from src.imgpipe.normalized_box import NormalizedBox
from src.model.predictor import (
    Predictor,
    PredictorConfig,
    YoloPredictorConfig,
    MedSamPredictorConfig,
)


# =========================
# Helpers
# =========================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _is_finite(x: Optional[float]) -> bool:
    return (x is not None) and np.isfinite(x)


def _mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not xs:
        return None, None
    arr = np.asarray(xs, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def _to_uint8_mask(m: np.ndarray) -> np.ndarray:
    """Convert a boolean / 0-1 mask to uint8 (0 or 255)."""
    return (m > 0).astype(np.uint8) * 255


def _load_images_from_jsonl(path: Path, splits: Optional[Set[str]]) -> List[Image]:
    """
    Load Image objects from a JSONL file where each line is a JSON dump
    compatible with Image.from_json (i.e., produced by Image.to_json()).
    Optionally filter by image.split membership in `splits`.
    """
    images: List[Image] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            im = Image.from_json(line)
            if splits is not None and im.split not in splits:
                continue
            images.append(im)
    return images


def summarize_predictions(
    jsonl_path: Path,
    splits: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Read a per-image predictions JSONL (as written by this script) and compute
    aggregate statistics, optionally restricted to selected splits.

    Output schema (high level):
    {
      "counts": {...},
      "mask_dice_stats": {...},
      "box_dice_stats": {...},
      "metric_error": {
        "cdr_v": {"mae_mean", "mae_std", "n"},
        "cdr_h": {...},
        "rim_over_disc": {...},
        "I_over_S": {...},
      }
    }
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Predictions JSONL not found: {jsonl_path}")

    n_images = 0
    det_disc = det_cup = seg_disc = seg_cup = 0

    mask_disc_vals: List[float] = []
    mask_cup_vals: List[float] = []
    box_disc_vals: List[float] = []
    box_cup_vals: List[float] = []

    # Absolute errors for metrics
    err_abs: Dict[str, List[float]] = {
        "cdr_v": [],
        "cdr_h": [],
        "rim_over_disc": [],
        "I_over_S": [],
    }

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

            n_images += 1

            # Detection/segmentation availability
            yolo = rec.get("yolo", {})
            disc_yolo = yolo.get("disc", {})
            cup_yolo = yolo.get("cup", {})
            if disc_yolo.get("box_norm") is not None:
                det_disc += 1
            if cup_yolo.get("box_norm") is not None:
                det_cup += 1

            pm = rec.get("pred_masks", {})
            if pm.get("disc_path") is not None:
                seg_disc += 1
            if pm.get("cup_path") is not None:
                seg_cup += 1

            # Mask Dice
            dice = rec.get("dice", {})
            d_disc = dice.get("disc")
            d_cup = dice.get("cup")
            if _is_finite(d_disc):
                mask_disc_vals.append(float(d_disc))  # only defined cases
            if _is_finite(d_cup):
                mask_cup_vals.append(float(d_cup))

            # Box Dice (pred vs GT)
            pred_boxes = rec.get("pred_boxes", {})
            gt_boxes = rec.get("gt_boxes", {})

            pb_disc = pred_boxes.get("disc")
            gb_disc = gt_boxes.get("disc")
            pb_cup = pred_boxes.get("cup")
            gb_cup = gt_boxes.get("cup")

            if pb_disc is not None and gb_disc is not None:
                pb_disc_nb = NormalizedBox(*map(float, pb_disc))
                gb_disc_nb = NormalizedBox(*map(float, gb_disc))
                box_disc_vals.append(float(pb_disc_nb.dice(gb_disc_nb)))

            if pb_cup is not None and gb_cup is not None:
                pb_cup_nb = NormalizedBox(*map(float, pb_cup))
                gb_cup_nb = NormalizedBox(*map(float, gb_cup))
                box_cup_vals.append(float(pb_cup_nb.dice(gb_cup_nb)))

            # Metric errors
            metrics = rec.get("metrics", {})
            gt_m = metrics.get("gt", {})
            pr_m = metrics.get("pred", {})

            for key_src, key_out in [
                ("cdr_v", "cdr_v"),
                ("cdr_h", "cdr_h"),
                ("rim_over_disc", "rim_over_disc"),
                ("I_over_S", "I_over_S"),
            ]:
                gt_val = gt_m.get(key_src)
                pr_val = pr_m.get(key_src)
                if _is_finite(gt_val) and _is_finite(pr_val):
                    err_abs[key_out].append(abs(float(pr_val - gt_val)))

    # Aggregations
    md_disc_mean, md_disc_std = _mean_std(mask_disc_vals)
    md_cup_mean, md_cup_std = _mean_std(mask_cup_vals)
    bd_disc_mean, bd_disc_std = _mean_std(box_disc_vals)
    bd_cup_mean, bd_cup_std = _mean_std(box_cup_vals)

    metric_error: Dict[str, Dict[str, Optional[float]]] = {}
    for name, vals in err_abs.items():
        mae_mean, mae_std = _mean_std(vals)
        metric_error[name] = {
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "n": len(vals),
        }

    det_rate_disc = det_disc / max(1, n_images)
    det_rate_cup = det_cup / max(1, n_images)
    seg_rate_disc = seg_disc / max(1, n_images)
    seg_rate_cup = seg_cup / max(1, n_images)

    summary: Dict[str, Any] = {
        "counts": {
            "images": n_images,
            "detected_disc": det_disc,
            "detected_cup": det_cup,
            "segmented_disc": seg_disc,
            "segmented_cup": seg_cup,
        },
        "rates": {
            "det_rate_disc": det_rate_disc,
            "det_rate_cup": det_rate_cup,
            "seg_rate_disc": seg_rate_disc,
            "seg_rate_cup": seg_rate_cup,
        },
        "mask_dice_stats": {
            "disc": {"mean": md_disc_mean, "std": md_disc_std, "n": len(mask_disc_vals)},
            "cup": {"mean": md_cup_mean, "std": md_cup_std, "n": len(mask_cup_vals)},
        },
        "box_dice_stats": {
            "disc": {"mean": bd_disc_mean, "std": bd_disc_std, "n": len(box_disc_vals)},
            "cup": {"mean": bd_cup_mean, "std": bd_cup_std, "n": len(box_cup_vals)},
        },
        "metric_error": metric_error,
    }
    return summary


def append_summary_row_to_csv(
    csv_path: Path,
    jsonl_path: Path,
    summary: Dict[str, Any],
) -> None:
    """
    Append a single row with key run-level statistics to a CSV file.
    """
    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    md = summary["mask_dice_stats"]
    bd = summary["box_dice_stats"]
    me = summary["metric_error"]

    header = [
        "jsonl_path",
        "n_images",
        "mask_dice_disc_mean",
        "mask_dice_disc_std",
        "mask_dice_cup_mean",
        "mask_dice_cup_std",
        "box_dice_disc_mean",
        "box_dice_disc_std",
        "box_dice_cup_mean",
        "box_dice_cup_std",
        "cdr_v_mae_mean",
        "cdr_v_mae_std",
        "rim_over_disc_mae_mean",
        "rim_over_disc_mae_std",
        "I_over_S_mae_mean",
        "I_over_S_mae_std",
    ]

    row = [
        str(jsonl_path),
        summary["counts"]["images"],
        md["disc"]["mean"],
        md["disc"]["std"],
        md["cup"]["mean"],
        md["cup"]["std"],
        bd["disc"]["mean"],
        bd["disc"]["std"],
        bd["cup"]["mean"],
        bd["cup"]["std"],
        me["cdr_v"]["mae_mean"],
        me["cdr_v"]["mae_std"],
        me["rim_over_disc"]["mae_mean"],
        me["rim_over_disc"]["mae_std"],
        me["I_over_S"]["mae_mean"],
        me["I_over_S"]["mae_std"],
    ]

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


# =========================
# CLI
# =========================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run YOLO+MedSAM predictions and/or summarize a predictions JSONL."
    )

    # Prediction inputs
    p.add_argument(
        "--images-jsonl",
        type=Path,
        default=None,
        help="JSONL of Image objects (Image.to_json per line). If omitted, no new predictions are run.",
    )
    p.add_argument(
        "--yolo-weights",
        type=Path,
        default=None,
        help="Path to YOLO weights (.pt). Required if --images-jsonl is provided.",
    )
    p.add_argument(
        "--medsam-checkpoint",
        type=Path,
        default=None,
        help="Path to MedSAM ViT-B checkpoint. Required if --images-jsonl is provided.",
    )

    # Predictor configuration
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for YOLO and MedSAM (e.g., 'cuda:0', 'cpu', 'mps').",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="YOLO confidence threshold.",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="YOLO IoU threshold.",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )
    p.add_argument(
        "--save-overlays",
        action="store_true",
        help="If set, save overlay visualizations.",
    )
    p.add_argument(
        "--no-save-masks",
        action="store_true",
        help="If set, do not save predicted masks.",
    )
    p.add_argument(
        "--box-pad-frac",
        type=float,
        default=0.05,
        help="Fraction to pad YOLO boxes before MedSAM prompting.",
    )

    # I/O
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for masks, overlays, predictions JSONL, and summary.json.",
    )
    p.add_argument(
        "--pred-jsonl",
        type=Path,
        default=None,
        help="Path to predictions JSONL. "
             "If not provided, defaults to <out-dir>/saved_images.jsonl.",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional CSV to append a summary row to.",
    )

    # Filtering
    p.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated list of splits to include (e.g., 'train,val,test').",
    )

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    out_dir: Path = _ensure_dir(args.out_dir)

    # Split filter
    split_set: Optional[Set[str]] = None
    if args.splits:
        split_set = {s.strip() for s in args.splits.split(",") if s.strip()}

    # Predictions JSONL path
    pred_jsonl: Path = args.pred_jsonl or (out_dir / "saved_images.jsonl")

    # Directories for masks and overlays
    mask_dir = _ensure_dir(out_dir / "masks")
    viz_dir = _ensure_dir(out_dir / "viz")

    # 1) Optionally run prediction
    if args.images_jsonl is not None:
        if args.yolo_weights is None or args.medsam_checkpoint is None:
            raise ValueError(
                "--yolo-weights and --medsam-checkpoint are required when --images-jsonl is provided."
            )

        images = _load_images_from_jsonl(args.images_jsonl, splits=split_set)

        # Build predictor configs
        yolo_cfg = YoloPredictorConfig(
            weights=args.yolo_weights,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
        )
        sam_cfg = MedSamPredictorConfig(
            checkpoint=args.medsam_checkpoint,
            device=args.device,
        )
        pred_cfg = PredictorConfig(
            box_pad_frac=args.box_pad_frac,
        )

        predictor = Predictor(yolo_cfg, sam_cfg, pred_cfg)
        images = predictor.predict(images)

        # After prediction, this script takes over:
        # - compute metrics per image
        # - save masks and overlays
        # - write per-image records JSONL
        with open(pred_jsonl, "w", encoding="utf-8") as jf:
            for img in images:
                # Compute / update Dice and metrics on the Image
                img.update_mask_dice(fallback_to_boxes=True)
                metrics_summary = img.metrics_summary()

                # Predicted and GT boxes
                pred_disc_box = img.pred_disc_box.as_tuple() if img.pred_disc_box is not None else None
                pred_cup_box = img.pred_cup_box.as_tuple() if img.pred_cup_box is not None else None
                gt_disc_box = img.gt_disc_box.as_tuple() if img.gt_disc_box is not None else None
                gt_cup_box = img.gt_cup_box.as_tuple() if img.gt_cup_box is not None else None

                # Save predicted masks (if requested)
                disc_pred_mask_path: Optional[Path] = None
                cup_pred_mask_path: Optional[Path] = None
                if not args.no_save_masks:
                    disc_arr = img._mask_to_image_size(img.pred_disc_mask)  # type: ignore[attr-defined]
                    cup_arr = img._mask_to_image_size(img.pred_cup_mask)   # type: ignore[attr-defined]
                    if disc_arr is not None:
                        disc_pred_mask_path = mask_dir / f"{img.uid}_disc.png"
                        PILImage.fromarray(_to_uint8_mask(disc_arr)).save(str(disc_pred_mask_path))
                    if cup_arr is not None:
                        cup_pred_mask_path = mask_dir / f"{img.uid}_cup.png"
                        PILImage.fromarray(_to_uint8_mask(cup_arr)).save(str(cup_pred_mask_path))

                # Save overlay visualization (if requested)
                overlay_path: Optional[Path] = None
                if args.save_overlays:
                    overlay_path = viz_dir / f"{img.uid}_overlay.png"
                    try:
                        img.visualize(
                            show=False,
                            save_path=overlay_path,
                            dpi=140,
                            mask_alpha=0.7,
                        )
                    except Exception:
                        overlay_path = None

                # GT mask paths (if BinaryMaskRef has path)
                disc_gt_path = getattr(img.gt_disc_mask, "path", None)
                cup_gt_path = getattr(img.gt_cup_mask, "path", None)

                # YOLO confidences, if attached to the Image (optional)
                disc_conf = getattr(img, "pred_disc_conf", None)
                cup_conf = getattr(img, "pred_cup_conf", None)

                rec: Dict[str, Any] = {
                    "uid": img.uid,
                    "dataset": img.dataset,
                    "subject_id": img.subject_id,
                    "image_path": str(img.image_path),
                    "split": img.split,
                    "yolo": {
                        "disc": {
                            "conf": float(disc_conf) if _is_finite(disc_conf) else None,
                            "box_norm": (
                                img.inter_pred_disc_box.as_tuple()
                                if img.inter_pred_disc_box is not None
                                else None
                            ),
                        },
                        "cup": {
                            "conf": float(cup_conf) if _is_finite(cup_conf) else None,
                            "box_norm": (
                                img.inter_pred_cup_box.as_tuple()
                                if img.inter_pred_cup_box is not None
                                else None
                            ),
                        },
                        "conf_th": args.conf,
                        "iou_th": args.iou,
                    },
                    "pred_masks": {
                        "disc_path": str(disc_pred_mask_path) if disc_pred_mask_path else None,
                        "cup_path": str(cup_pred_mask_path) if cup_pred_mask_path else None,
                    },
                    "gt_masks": {
                        "disc_path": str(disc_gt_path) if disc_gt_path else None,
                        "cup_path": str(cup_gt_path) if cup_gt_path else None,
                    },
                    "pred_boxes": {
                        "disc": pred_disc_box,
                        "cup": pred_cup_box,
                    },
                    "gt_boxes": {
                        "disc": gt_disc_box,
                        "cup": gt_cup_box,
                    },
                    "dice": {
                        "disc": img.mask_dice_disc,
                        "cup": img.mask_dice_cup,
                    },
                    "metrics": metrics_summary,
                    "overlay_path": str(overlay_path) if overlay_path else None,
                }
                jf.write(json.dumps(rec) + "\n")

    # 2) Summarize predictions JSONL (existing or just written)
    summary = summarize_predictions(pred_jsonl, splits=split_set)

    # Always write summary.json
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Optionally append row to CSV
    if args.summary_csv is not None:
        append_summary_row_to_csv(args.summary_csv, pred_jsonl, summary)


if __name__ == "__main__":
    main()