#!/usr/bin/env python3
# src/scripts/eval_medsam.py
"""
Evaluate MedSAM using *ground-truth* disc/cup bounding boxes as prompts,
optionally padded by a fraction of each box size, and compare performance
across padding levels.

Outputs (under --out-dir):
  pad_000/, pad_010/, ... each with:
    pred_disc/, pred_cup/, [viz/ if --save-viz], details.csv, stats.json, [top10/, bottom10/ if --save-viz]
  details_all.csv, summary_by_pad.csv, metrics_vs_pad.png

Notes
-----
- Visualizations (overlay JPEGs) are now optional and DISABLED by default.
  Enable with:  --save-viz
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Project classes & utils ---
from src.imgpipe.image_factory import ImageFactory
from src.imgpipe.image import Image
from src.imgpipe.enums import Structure, LabelType
from src.model.MedSAM_infer import (
    MedSAMModel,
    load_medsam,
    embed_image_1024,
    medsam_infer,
    pick_device,
)
from src.utils import (
    ensure_dir,
    load_image_bgr,
    save_mask_png,
    dice,
    overlay_masks_and_boxes,  # overlay helper for viz (used only if --save-viz)
)

# =============================
# Defaults (retain these)
# =============================

PAD_DEFAULTS = [0.00, 0.05, 0.10, 0.20, 0.30]

DEFAULT_DATA_ROOT = Path("/Users/carlosperez/Library/CloudStorage/OneDrive-UBC/Ipek_Carlos/GlaucomaDatasets/All_Datasets_Organized")
DEFAULT_OUT_DIR   = Path("/Users/carlosperez/PycharmProjects/MedSAM/EVAL")
DEFAULT_MEDSAM_CKPT = "/Users/carlosperez/PycharmProjects/MedSAM/work_dir/MedSAM/medsam_updated.pth"

# =============================
# Small helpers
# =============================

def set_global_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
    except Exception:
        pass

def _pad_xyxy(
    xyxy: Tuple[int, int, int, int], pad_frac: float, W: int, H: int
) -> Tuple[int, int, int, int]:
    """Pad a pixel xyxy box by a fraction of its size, clamp to image bounds."""
    x1, y1, x2, y2 = map(float, xyxy)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    dx = w * pad_frac
    dy = h * pad_frac
    nx1 = max(0.0, x1 - dx)
    ny1 = max(0.0, y1 - dy)
    nx2 = min(float(W), x2 + dx)
    ny2 = min(float(H), y2 + dy)
    if nx2 <= nx1:
        nx2 = min(float(W), nx1 + 1.0)
    if ny2 <= ny1:
        ny2 = min(float(H), ny1 + 1.0)
    return int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))

def _xyxy_from_nbox(nbox, W: int, H: int) -> Tuple[int, int, int, int]:
    """Convert NormalizedBox → pixel xyxy (ints)."""
    x1, y1, x2, y2 = nbox.to_pixel_xyxy(W, H)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def _summarize(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "p50": float("nan")}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if a.size >= 2 else float("nan"),
        "p50": float(np.percentile(a, 50)),
    }

def _cdr_summary(pred: List[float], gt: List[float]) -> Dict[str, float]:
    if not pred or not gt or len(pred) != len(gt):
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "r": float("nan")}
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt,   dtype=np.float64)
    d = p - g
    mae  = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    bias = float(np.mean(d))
    r = float(np.corrcoef(p, g)[0, 1]) if p.size >= 2 else float("nan")
    return {"n": int(p.size), "mae": mae, "rmse": rmse, "bias": bias, "r": r}

def _write_csv(rows: List[dict], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def _plot_vs_pad(pads: List[float], metrics: Dict[str, List[float]], out_path: Path) -> None:
    """Simple multi-line plot for Dice/MAE vs pad fraction; saved unconditionally."""
    ensure_dir(out_path.parent)
    plt.figure(figsize=(8, 5))
    if "disc_dice_mean" in metrics:
        plt.plot(pads, metrics["disc_dice_mean"], marker="o", label="Disc Dice (mean)")
    if "cup_dice_mean" in metrics:
        plt.plot(pads, metrics["cup_dice_mean"], marker="o", label="Cup Dice (mean)")
    if "cdr_mae" in metrics:
        plt.plot(pads, metrics["cdr_mae"], marker="o", label="CDR MAE")
    plt.xlabel("Padding fraction")
    plt.ylabel("Metric value")
    plt.title("MedSAM performance vs padding")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()

def _rank_best_worst(rows: List[dict], k: int = 10) -> Tuple[List[dict], List[dict]]:
    with_err = [r for r in rows if r.get("cdr_abs_error") is not None]
    if with_err:
        srt = sorted(with_err, key=lambda r: r["cdr_abs_error"])
        return srt[:k], list(reversed(srt[-k:]))
    with_dice = [r for r in rows if r.get("disc_dice") is not None and r.get("cup_dice") is not None]
    srt = sorted(with_dice, key=lambda r: 0.5 * (r["disc_dice"] + r["cup_dice"]), reverse=True)
    return srt[:k], list(reversed(srt[-k:]))

def _copy_best_worst(rows: List[dict], out_pad_dir: Path, k: int = 10) -> None:
    best, worst = _rank_best_worst(rows, k=k)
    for tag, subset in (("top10", best), ("bottom10", worst)):
        tgt = out_pad_dir / tag
        ensure_dir(tgt)
        for r in subset:
            src = Path(r.get("viz_path") or "")
            if src.exists():
                img = cv2.imread(str(src))
                if img is not None:
                    cv2.imwrite(str(tgt / src.name), img)

# =============================
# CLI
# =============================

@dataclass
class CLI:
    data_root: Path
    out_dir: Path
    medsam_ckpt: Path
    pad_fracs: List[float]
    device: Optional[str]
    subset_n: int
    subset_seed: int
    exclude: List[str]
    save_viz: bool
    topk: int

def _parse_args() -> CLI:
    p = argparse.ArgumentParser(description="Evaluate MedSAM with GT disc/cup boxes across padding levels.")
    p.add_argument("--data-root",   type=Path, default=DEFAULT_DATA_ROOT,   help="Root directory containing datasets.")
    p.add_argument("--out-dir",     type=Path, default=DEFAULT_OUT_DIR,     help="Output directory.")
    p.add_argument("--medsam-ckpt", type=Path, default=DEFAULT_MEDSAM_CKPT, help="Path to MedSAM checkpoint (.pth/.pt).")
    p.add_argument("--pad-fracs", nargs="+", type=float, default=PAD_DEFAULTS, help="Padding fractions (e.g. 0.0 0.1 0.2).")
    p.add_argument("--device", type=str, default=None, help="CUDA device (e.g. '0') or 'cpu'.")
    p.add_argument("--subset-n", type=int, default=0, help="Optionally evaluate on N images.")
    p.add_argument("--subset-seed", type=int, default=43, help="Subset RNG seed.")
    p.add_argument("--exclude", nargs="*", default=["PAPILA"], help="Dataset names to exclude.")
    # NEW: control visualization saving (default: False)
    p.add_argument("--save-viz", action="store_true", help="Save overlay visualizations (default: off).")
    p.add_argument("--topk", type=int, default=10, help="If saving viz, copy top/bottom K examples (default: 10).")
    a = p.parse_args()
    return CLI(
        data_root=a.data_root,
        out_dir=a.out_dir,
        medsam_ckpt=a.medsam_ckpt,
        pad_fracs=list(map(float, a.pad_fracs)),
        device=a.device,
        subset_n=int(a.subset_n),
        subset_seed=int(a.subset_seed),
        exclude=list(a.exclude),
        save_viz=bool(a.save_viz),
        topk=int(a.topk),
    )

# =============================
# Core
# =============================

def _gather_images(data_root: Path, exclude: List[str], subset_n: int, subset_seed: int) -> List[Image]:
    """Use ImageFactory to get Image objects with fundus + disc/cup masks."""
    print("[INFO] Scanning datasets…")
    fac = ImageFactory(root=data_root, auto_scan=True)
    fac.filter_empty_masks()
    # if exclude:
    #     fac.filter_datasets(exclude=exclude)
    fac.filter_datasets(include=["PAPILA"])
    images: List[Image] = fac.make_images()
    if not images:
        raise RuntimeError("No images with both disc/cup masks found.")
    if subset_n and 0 < subset_n < len(images):
        rng = np.random.RandomState(subset_seed)
        idx = rng.choice(len(images), size=subset_n, replace=False)
        images = [images[i] for i in idx]
    print(f"[INFO] Using N={len(images)} images.")
    return images

def _eval_one_image_for_pad(
    msam: MedSAMModel,
    img: Image,
    pad_frac: float,
    out_pad_dir: Path,
    save_viz: bool,
) -> Optional[dict]:
    """
    - Ensure GT boxes from masks (normalized) → pixel xyxy.
    - Pad boxes, embed once, infer disc/cup.
    - Attach predicted masks to `img`, compute metrics.
    - If save_viz: write overlay JPEG and include its path; else skip heavy viz work.
    """
    # Load RGB/BGR and geometry
    bgr = load_image_bgr(img.image_path)
    if bgr is None:
        return None
    H, W = img.height, img.width

    # Ensure GT boxes exist from masks
    img.ensure_boxes_from_masks()
    gt_disc_n = img.get_box_norm(Structure.DISC, LabelType.GT)
    gt_cup_n  = img.get_box_norm(Structure.CUP,  LabelType.GT)
    if gt_disc_n is None or gt_cup_n is None:
        return None

    # Normalized → pixel xyxy, then pad
    disc_xyxy = _xyxy_from_nbox(gt_disc_n, W, H)
    cup_xyxy  = _xyxy_from_nbox(gt_cup_n,  W, H)
    disc_xyxy_p = _pad_xyxy(disc_xyxy, pad_frac, W, H)
    cup_xyxy_p  = _pad_xyxy(cup_xyxy,  pad_frac, W, H)

    # Embed once; infer twice
    emb, Hx, Wx, _ = embed_image_1024(msam, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if Hx != H or Wx != W:
        raise RuntimeError("Unexpected size change in embedding path.")
    pred_disc = medsam_infer(msam, emb, disc_xyxy_p, H, W)
    pred_cup  = medsam_infer(msam, emb, cup_xyxy_p,  H, W)

    # Attach predictions to the Image
    img.set_mask(Structure.DISC, LabelType.PRED, pred_disc.astype(np.uint8))
    img.set_mask(Structure.CUP,  LabelType.PRED, pred_cup.astype(np.uint8))
    img.ensure_boxes_from_masks()

    # Aligned masks for metrics
    gt_disc = img._mask_to_image_size(img.gt_disc_mask)
    gt_cup  = img._mask_to_image_size(img.gt_cup_mask)
    pr_disc = img._mask_to_image_size(img.pred_disc_mask)
    pr_cup  = img._mask_to_image_size(img.pred_cup_mask)
    if gt_disc is None or gt_cup is None or pr_disc is None or pr_cup is None:
        return None

    # Metrics
    d_dice = float(dice(pr_disc.astype(np.uint8), gt_disc.astype(np.uint8)))
    c_dice = float(dice(pr_cup.astype(np.uint8),  gt_cup.astype(np.uint8)))
    pred_cdr = img.cdr(use_pred=True,  axis="vertical")
    gt_cdr   = img.cdr(use_pred=False, axis="vertical")
    cdr_err = cdr_abs = None
    if (pred_cdr is not None) and (gt_cdr is not None):
        cdr_err = float(pred_cdr - gt_cdr)
        cdr_abs = abs(cdr_err)

    # Prepare output dirs for masks (always saved) and viz (optional)
    disc_dir = out_pad_dir / "pred_disc"
    cup_dir  = out_pad_dir / "pred_cup"
    ensure_dir(disc_dir); ensure_dir(cup_dir)

    # Save predicted masks
    disc_png = disc_dir / f"{img.image_path.stem}.png"
    cup_png  = cup_dir  / f"{img.image_path.stem}.png"
    save_mask_png(disc_png, pr_disc.astype(np.uint8))
    save_mask_png(cup_png,  pr_cup.astype(np.uint8))

    viz_path_str = ""  # default when not saving viz

    if save_viz:
        viz_dir = out_pad_dir / "viz"
        ensure_dir(viz_dir)
        # Compose compact text including Dice and CDR terms
        cdr_txt = (
            f"Disc Dice={d_dice:.3f} | Cup Dice={c_dice:.3f} | "
            f"CDR pred={pred_cdr:.3f}, GT={gt_cdr:.3f}, |err|={(cdr_abs if cdr_abs is not None else math.nan):.3f}"
            if (pred_cdr is not None and gt_cdr is not None)
            else f"Disc Dice={d_dice:.3f} | Cup Dice={c_dice:.3f} | CDR: N/A"
        )
        viz = overlay_masks_and_boxes(
            bgr, pr_disc.astype(np.uint8), pr_cup.astype(np.uint8),
            disc_xyxy_p, cup_xyxy_p, cdr_text=cdr_txt
        )
        viz_path = viz_dir / f"{img.image_path.stem}_viz.jpg"
        cv2.imwrite(str(viz_path), viz)
        viz_path_str = str(viz_path)

    # Row
    row = {
        "stem": img.image_path.stem,
        "image_path": str(img.image_path),
        "pred_disc_path": str(disc_png),
        "pred_cup_path":  str(cup_png),
        "viz_path":       viz_path_str,  # empty if not saved
        "pad_frac": float(pad_frac),
        "disc_box": list(map(int, disc_xyxy)),
        "cup_box":  list(map(int, cup_xyxy)),
        "disc_box_padded": list(map(int, disc_xyxy_p)),
        "cup_box_padded":  list(map(int, cup_xyxy_p)),
        "disc_dice": d_dice,
        "cup_dice":  c_dice,
        "pred_cdr": (float(pred_cdr) if pred_cdr is not None else None),
        "gt_cdr":   (float(gt_cdr)   if gt_cdr   is not None else None),
        "cdr_error": (cdr_err if cdr_err is not None else None),
        "cdr_abs_error": (cdr_abs if cdr_abs is not None else None),
    }
    return row

# =============================
# Main
# =============================

def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    set_global_seed(42)

    args = _parse_args()
    ensure_dir(args.out_dir)

    # Collect images via ImageFactory (consistent with train.py)
    images = _gather_images(args.data_root, args.exclude, args.subset_n, args.subset_seed)

    # Device & model
    dev = pick_device(args.device)
    msam = load_medsam(args.medsam_ckpt, dev, variant="vit_b")

    all_rows: List[dict] = []
    summary_by_pad: List[dict] = []

    for pad in args.pad_fracs:
        tag = f"pad_{int(round(pad * 100)):03d}"
        out_pad_dir = args.out_dir / tag
        ensure_dir(out_pad_dir)

        pad_rows: List[dict] = []
        print(f"[INFO] Evaluating pad={pad:.2f} on N={len(images)} images…")
        for img in images:
            try:
                row = _eval_one_image_for_pad(msam, img, pad, out_pad_dir, save_viz=args.save_viz)
                if row is not None:
                    pad_rows.append(row)
            except Exception as e:
                print(f"[WARN] Skipped {img.image_path.name}: {e}")

        _write_csv(pad_rows, out_pad_dir / "details.csv")

        # Best/Worst thumbnails only make sense if viz were saved
        if args.save_viz:
            _copy_best_worst(pad_rows, out_pad_dir, k=args.topk)

        # Per-pad stats
        disc_dice_vals = [r["disc_dice"] for r in pad_rows if r.get("disc_dice") is not None]
        cup_dice_vals  = [r["cup_dice"]  for r in pad_rows if r.get("cup_dice")  is not None]
        pred_cdr_vals  = [r["pred_cdr"]  for r in pad_rows if r.get("pred_cdr")  is not None]
        gt_cdr_vals    = [r["gt_cdr"]    for r in pad_rows if r.get("gt_cdr")    is not None]

        disc_stat = _summarize(disc_dice_vals)
        cup_stat  = _summarize(cup_dice_vals)
        cdr_stat  = _cdr_summary(pred_cdr_vals, gt_cdr_vals)

        stats = {
            "pad_frac": float(pad),
            "n_images": len(pad_rows),
            "disc_dice": disc_stat,
            "cup_dice":  cup_stat,
            "cdr":       cdr_stat,
        }
        (out_pad_dir / "stats.json").write_text(json.dumps(stats, indent=2))

        summary_by_pad.append({
            "pad_frac": float(pad),
            "n_images": len(pad_rows),
            "disc_dice_mean": disc_stat["mean"],
            "disc_dice_std":  disc_stat["std"],
            "cup_dice_mean":  cup_stat["mean"],
            "cup_dice_std":   cup_stat["std"],
            "cdr_mae":  cdr_stat["mae"],
            "cdr_rmse": cdr_stat["rmse"],
            "cdr_bias": cdr_stat["bias"],
            "cdr_r":    cdr_stat["r"],
        })

        all_rows.extend(pad_rows)
        print(f"[PAD {pad:.2f}] N={len(pad_rows)} | Disc Dice μ={disc_stat['mean']:.4f} | Cup Dice μ={cup_stat['mean']:.4f} | CDR MAE={cdr_stat['mae']:.4f}")

    # Write combined tables
    _write_csv(all_rows, args.out_dir / "details_all.csv")
    _write_csv(summary_by_pad, args.out_dir / "summary_by_pad.csv")

    # Plot summary (always saved; separate from per-image viz)
    pads = [row["pad_frac"] for row in summary_by_pad]
    metrics = {
        "disc_dice_mean": [row["disc_dice_mean"] for row in summary_by_pad],
        "cup_dice_mean":  [row["cup_dice_mean"]  for row in summary_by_pad],
        "cdr_mae":        [row["cdr_mae"]        for row in summary_by_pad],
    }
    _plot_vs_pad(pads, metrics, args.out_dir / "metrics_vs_pad.png")

    print("\n[OK] Complete.")
    print(f"  details_all.csv     → {args.out_dir / 'details_all.csv'}")
    print(f"  summary_by_pad.csv  → {args.out_dir / 'summary_by_pad.csv'}")
    print(f"  metrics_vs_pad.png  → {args.out_dir / 'metrics_vs_pad.png'}")
    for pad in args.pad_fracs:
        tag = f"pad_{int(round(pad * 100)):03d}"
        if args.save_viz:
            print(f"  {tag}/details.csv, {tag}/viz/, {tag}/top10/, {tag}/bottom10/")
        else:
            print(f"  {tag}/details.csv")

if __name__ == "__main__":
    main()