#!/usr/bin/env python3
# eval_medsam_from_gt_boxes_with_padding.py
"""
Evaluate MedSAM using *ground-truth* disc and cup bounding boxes as prompts,
optionally padded by a fraction of each box size, and compare performance
across padding levels.

For each image:
  1) Load GT disc/cup masks.
  2) Build GT boxes from masks; pad by a chosen fraction; clamp to image bounds.
  3) Run MedSAM with disc/cup boxes → predicted masks.
  4) Compute Dice for disc/cup and CDR metrics (pred vs GT).
  5) Save per-image rows and visualizations.
  6) Select 10 best / 10 worst by |CDR error| for each pad fraction.

Outputs:
  out_dir/
    pad_000/ (for pad=0.00)
      pred_disc/         # predicted disc masks (PNG)
      pred_cup/          # predicted cup masks (PNG)
      viz/               # overlay viz (pred vs GT)
      top10/ bottom10/   # copies of top/bottom visualization frames
      details.csv        # per-image rows for this padding
      stats.json         # summary stats for this padding
    pad_010/ ...
    ...
    summary_by_pad.csv   # per-padding table (means, stds, MAE, RMSE, etc)
    details_all.csv      # all rows combined (with 'pad_frac' column)
    metrics_vs_pad.png   # plot comparing Dice & CDR MAE across paddings

Requirements:
  - Your project utilities (src/...) as used in other scripts.
  - MedSAM checkpoint (pass --medsam-ckpt).
  - GT masks directories for disc and cup (pass --gt-disc-masks, --gt-cup-masks).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils import (
    ensure_dir, stem_map_by_first_match, load_image_bgr, save_mask_png,
    overlay_masks_and_boxes, cdr_from_masks, tight_bbox_from_mask, dice, expand
)
from src.model.MedSAM_infer import (
    MedSAMModel, medsam_infer, embed_image_1024, load_medsam, pick_device
)
# add this to your imports at the top (if you have it)
from src.utils import make_side_by_side

# ----------------------- Helpers -----------------------

def _xyxy_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    return tight_bbox_from_mask(mask)

def _pad_box_xyxy(
    xyxy: Tuple[int, int, int, int],
    pad_frac: float,
    W: int,
    H: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, xyxy)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    dx = w * pad_frac
    dy = h * pad_frac
    nx1 = max(0.0, x1 - dx)
    ny1 = max(0.0, y1 - dy)
    nx2 = min(float(W), x2 + dx)
    ny2 = min(float(H), y2 + dy)
    # ensure valid ordering after clamping
    if nx2 <= nx1: nx2 = min(float(W), nx1 + 1.0)
    if ny2 <= ny1: ny2 = min(float(H), ny1 + 1.0)
    return int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2))

def _summarize(arr: List[float]) -> Dict[str, float]:
    if not arr:
        return dict(n=0, mean=float("nan"), std=float("nan"),
                    p50=float("nan"), p25=float("nan"), p75=float("nan"))
    a = np.asarray(arr, dtype=np.float64)
    return dict(
        n=int(a.size),
        mean=float(np.mean(a)),
        std=float(np.std(a, ddof=1)) if a.size >= 2 else float("nan"),
        p50=float(np.percentile(a, 50)),
        p25=float(np.percentile(a, 25)),
        p75=float(np.percentile(a, 75)),
    )

def _cdr_summary(pred: List[float], gt: List[float]) -> Dict[str, float]:
    if not pred or not gt or len(pred) != len(gt):
        return dict(n=0, mae=float("nan"), rmse=float("nan"),
                    bias=float("nan"), mape=float("nan"), r=float("nan"), r2=float("nan"))
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt,   dtype=np.float64)
    d = p - g
    mae  = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    bias = float(np.mean(d))
    mape = float(np.mean(np.abs(d) / np.maximum(1e-6, np.abs(g))))
    r = float(np.corrcoef(p, g)[0, 1]) if p.size >= 2 else float("nan")
    r2 = float(r * r) if np.isfinite(r) else float("nan")
    return dict(n=int(p.size), mae=mae, rmse=rmse, bias=bias, mape=mape, r=r, r2=r2)

def _rank_best_worst(
    rows: List[dict], k: int = 10
) -> Tuple[List[dict], List[dict]]:
    # Rank by absolute CDR error primarily; fall back to mean of dice if needed.
    with_errors = [r for r in rows if r.get("cdr_abs_error") is not None]
    if with_errors:
        sorted_rows = sorted(with_errors, key=lambda r: r["cdr_abs_error"])
        best = sorted_rows[:k]
        worst = sorted_rows[-k:][::-1]
        return best, worst
    # fallback: rank by mean Dice (desc)
    with_dice = [r for r in rows if (r.get("disc_dice") is not None and r.get("cup_dice") is not None)]
    if with_dice:
        sorted_rows = sorted(with_dice, key=lambda r: 0.5*(r["disc_dice"]+r["cup_dice"]), reverse=True)
        best = sorted_rows[:k]
        worst = sorted_rows[-k:][::-1]
        return best, worst
    return [], []

def _viz_one(bgr, pred_disc, pred_cup, gt_disc, gt_cup, disc_box, cup_box, cdr_text):
    if gt_disc is not None and gt_cup is not None:
        # left: preds, right: GT
        return make_side_by_side(bgr, pred_disc, pred_cup, gt_disc, gt_cup, cdr_text, "GT")
    # only preds available
    return overlay_masks_and_boxes(bgr, pred_disc, pred_cup, disc_box, cup_box, cdr_text=cdr_text)

def _save_img(path: Path, img: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)

# ----------------------- Core pipeline -----------------------

@dataclass
class CLI:
    # You can pass multiple image roots (e.g., general + papilla)
    images_root: List[Path]
    gt_disc_root: Path
    gt_cup_root: Path
    medsam_ckpt: Path
    out_dir: Path
    pad_fracs: List[float]
    device: Optional[str]
    subset_n: int
    subset_seed: int
    recursive: bool

def _parse_args() -> CLI:
    ap = argparse.ArgumentParser(
        description="Evaluate MedSAM with GT disc/cup boxes (with padding sweep) and summarize metrics."
    )
    ap.add_argument("--images-root", action="append", required=True,
                    help="Root of images; repeat to include multiple roots (e.g., papilla).")
    ap.add_argument("--gt-disc-masks", required=True, help="Root of GT disc masks (files share image stems).")
    ap.add_argument("--gt-cup-masks",  required=True, help="Root of GT cup  masks (files share image stems).")
    ap.add_argument("--medsam-ckpt", required=True, help="Path to MedSAM checkpoint.")
    ap.add_argument("--out-dir", required=True, help="Output directory for results.")

    ap.add_argument("--pad-fracs", nargs="+", type=float, default=[0.00, 0.05, 0.10, 0.20, 0.30],
                    help="List of padding fractions to apply to GT boxes.")
    ap.add_argument("--device", default=None, help="CUDA device string or 'cpu'.")
    ap.add_argument("--subset-n", type=int, default=0, help="Optional: sample N images for quick runs.")
    ap.add_argument("--subset-seed", type=int, default=43)
    ap.add_argument("--recursive", action="store_true", help="Recurse when mapping stems from roots.")
    a = ap.parse_args()

    return CLI(
        images_root=[expand(p) for p in a.images_root],
        gt_disc_root=expand(a.gt_disc_masks),
        gt_cup_root=expand(a.gt_cup_masks),
        medsam_ckpt=expand(a.medsam_ckpt),
        out_dir=expand(a.out_dir),
        pad_fracs=list(a.pad_fracs),
        device=a.device,
        subset_n=int(a.subset_n),
        subset_seed=int(a.subset_seed),
        recursive=bool(a.recursive),
    )

def _collect_images(images_roots: List[Path], recursive: bool) -> Dict[str, Path]:
    """Return mapping: stem -> image_path (first match wins across roots)."""
    stem2img: Dict[str, Path] = {}
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    for root in images_roots:
        if not root.exists():
            continue
        for pat in patterns:
            globber = root.rglob(pat) if recursive else root.glob(pat)
            for p in globber:
                stem = p.stem
                if stem not in stem2img:
                    stem2img[stem] = p
    return stem2img

def _collect_masks(root: Path, recursive: bool) -> Dict[str, Path]:
    return stem_map_by_first_match(root)

def _embed(msam: MedSAMModel, img_bgr: np.ndarray) -> Tuple[np.ndarray, int, int]:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    emb, H, W, _ = embed_image_1024(msam, img_rgb)
    return emb, H, W

def _process_image_for_pad(
    msam: MedSAMModel,
    img_path: Path,
    disc_mask_path: Path,
    cup_mask_path: Path,
    pad_frac: float,
    out_pad_dir: Path
) -> Optional[dict]:
    """Run MedSAM with padded GT boxes for one image and compute metrics."""
    # Load image and GT masks
    bgr = load_image_bgr(img_path)
    H, W = bgr.shape[:2]
    gt_disc = cv2.imread(str(disc_mask_path), cv2.IMREAD_GRAYSCALE)
    gt_cup  = cv2.imread(str(cup_mask_path),  cv2.IMREAD_GRAYSCALE)
    if gt_disc is None or gt_cup is None:
        return None
    gt_disc = (gt_disc > 127).astype(np.uint8)
    gt_cup  = (gt_cup  > 127).astype(np.uint8)

    # Build GT boxes from masks
    disc_xyxy = _xyxy_from_mask(gt_disc)
    cup_xyxy  = _xyxy_from_mask(gt_cup)
    if disc_xyxy is None or cup_xyxy is None:
        return None

    # Pad boxes
    disc_xyxy_pad = _pad_box_xyxy(disc_xyxy, pad_frac, W, H)
    cup_xyxy_pad  = _pad_box_xyxy(cup_xyxy,  pad_frac, W, H)

    # MedSAM embedding
    emb, Hx, Wx = _embed(msam, bgr)  # Hx, Wx should equal H, W after _embed_image_1024
    assert Hx == H and Wx == W, "Unexpected size change in embedding path."

    # Predict masks
    pred_disc = medsam_infer(msam, emb, disc_xyxy_pad, H, W)
    pred_cup  = medsam_infer(msam, emb, cup_xyxy_pad,  H, W)

    # Metrics
    d_dice = float(dice(pred_disc, gt_disc))
    c_dice = float(dice(pred_cup,  gt_cup))
    pred_cdr = cdr_from_masks(pred_disc, pred_cup)
    gt_cdr   = cdr_from_masks(gt_disc,  gt_cup)
    cdr_err = None
    if (pred_cdr is not None) and (gt_cdr is not None):
        cdr_err = float(pred_cdr - gt_cdr)

    # Save artifacts
    disc_dir = out_pad_dir / "pred_disc"
    cup_dir  = out_pad_dir / "pred_cup"
    viz_dir  = out_pad_dir / "viz"
    ensure_dir(disc_dir); ensure_dir(cup_dir); ensure_dir(viz_dir)
    disc_png = disc_dir / f"{img_path.stem}.png"
    cup_png  = cup_dir  / f"{img_path.stem}.png"
    save_mask_png(disc_png, pred_disc)
    save_mask_png(cup_png,  pred_cup)

    cdr_txt = f"CDR pred={pred_cdr:.3f}, GT={gt_cdr:.3f}, |err|={(abs(cdr_err) if cdr_err is not None else math.nan):.3f}" \
              if pred_cdr is not None and gt_cdr is not None else "CDR: N/A"

    viz = _viz_one(bgr, pred_disc, pred_cup, gt_disc, gt_cup, disc_xyxy_pad, cup_xyxy_pad, cdr_txt)

    viz_path = viz_dir / f"{img_path.stem}_viz.jpg"
    _save_img(viz_path, viz)

    row = dict(
        stem=img_path.stem,
        image_path=str(img_path),
        gt_disc_mask=str(disc_mask_path),
        gt_cup_mask=str(cup_mask_path),
        pred_disc_path=str(disc_png),
        pred_cup_path=str(cup_png),
        viz_path=str(viz_path),
        pad_frac=float(pad_frac),
        disc_box=list(map(int, disc_xyxy)),
        cup_box=list(map(int, cup_xyxy)),
        disc_box_padded=list(map(int, disc_xyxy_pad)),
        cup_box_padded=list(map(int, cup_xyxy_pad)),
        disc_dice=d_dice,
        cup_dice=c_dice,
        pred_cdr=(float(pred_cdr) if pred_cdr is not None else None),
        gt_cdr=(float(gt_cdr) if gt_cdr is not None else None),
        cdr_error=(float(cdr_err) if cdr_err is not None else None),
        cdr_abs_error=(abs(float(cdr_err)) if cdr_err is not None else None),
    )
    return row

def _write_csv(rows: List[dict], path: Path) -> None:
    import csv
    ensure_dir(path.parent)
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _copy_best_worst(rows: List[dict], out_pad_dir: Path, k: int = 10) -> None:
    best, worst = _rank_best_worst(rows, k=k)
    for tag, subset in (("top10", best), ("bottom10", worst)):
        tgt = out_pad_dir / tag
        ensure_dir(tgt)
        for r in subset:
            src = Path(r["viz_path"])
            if src.exists():
                img = cv2.imread(str(src))
                if img is not None:
                    _save_img(tgt / src.name, img)

def _plot_vs_pad(pads: List[float], metrics: Dict[str, List[float]], out_path: Path) -> None:
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

def main() -> None:
    args = _parse_args()
    ensure_dir(args.out_dir)

    # Collect images and masks
    stem2img = _collect_images(args.images_root, recursive=args.recursive)
    disc_map = _collect_masks(args.gt_disc_root, recursive=args.recursive)
    cup_map  = _collect_masks(args.gt_cup_root,  recursive=args.recursive)

    # Intersect on stems that have BOTH masks and an image
    stems = sorted(set(stem2img) & set(disc_map) & set(cup_map))
    if args.subset_n and args.subset_n > 0 and len(stems) > args.subset_n:
        rng = np.random.RandomState(args.subset_seed)
        rng.shuffle(stems)
        stems = stems[: args.subset_n]

    if not stems:
        raise SystemExit("[ERR] No (image, disc_mask, cup_mask) triplets found. Check roots and file stems.")

    # Load MedSAM
    dev = pick_device(args.device)
    msam = load_medsam(args.medsam_ckpt, dev, variant="vit_b")

    # Aggregate outputs
    all_rows: List[dict] = []
    summary_by_pad: List[dict] = []
    pad_float_list = list(map(float, args.pad_fracs))

    for pad in pad_float_list:
        tag = f"pad_{int(round(pad*100)):03d}"
        out_pad_dir = args.out_dir / tag
        ensure_dir(out_pad_dir)

        pad_rows: List[dict] = []
        for stem in stems:
            row = _process_image_for_pad(
                msam=msam,
                img_path=stem2img[stem],
                disc_mask_path=disc_map[stem],
                cup_mask_path=cup_map[stem],
                pad_frac=pad,
                out_pad_dir=out_pad_dir,
            )
            if row is not None:
                pad_rows.append(row)

        # Save per-pad details
        _write_csv(pad_rows, out_pad_dir / "details.csv")

        # Best/Worst copies
        _copy_best_worst(pad_rows, out_pad_dir, k=10)

        # Stats for this pad
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

        # Append table row for summary_by_pad
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
            "cdr_mape": cdr_stat["mape"],
            "cdr_r":    cdr_stat["r"],
            "cdr_r2":   cdr_stat["r2"],
        })

        # Accumulate rows with pad column
        for r in pad_rows:
            all_rows.append(r)

        print(f"[PAD {pad:.2f}] N={len(pad_rows)} | Disc Dice mean={disc_stat['mean']:.4f} | Cup Dice mean={cup_stat['mean']:.4f} | CDR MAE={cdr_stat['mae']:.4f}")

    # Write combined tables
    _write_csv(all_rows, args.out_dir / "details_all.csv")
    _write_csv(summary_by_pad, args.out_dir / "summary_by_pad.csv")

    # Plot vs pad
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
    for pad in pad_float_list:
        tag = f"pad_{int(round(pad*100)):03d}"
        print(f"  {tag}/details.csv, {tag}/top10/, {tag}/bottom10/")

if __name__ == "__main__":
    main()