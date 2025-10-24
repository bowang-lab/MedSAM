#!/usr/bin/env python3
# src/model/predict_segment.py
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from ultralytics import YOLO

from src.imgpipe.binary_mask_ref import BinaryMaskRef
from src.imgpipe.enums import LabelType, Structure
from src.imgpipe.image import Image
from src.imgpipe.bounding_box import BoundingBox
from src.imgpipe.collector import DatasetCollector, group_by_subject
from src.utils import ensure_dir, stem_map_by_first_match
from src.model.MedSAM_infer import (
    MedSAMModel, medsam_infer, embed_image_1024, load_medsam, pick_device
)
from src.utils import (
    overlay_masks_and_boxes, cdr_from_masks, make_side_by_side, load_image_bgr,
    shrink_box_to_fit_mask, tight_bbox_from_mask, save_mask_png, expand, dice
)

# --------------------------- small helpers ---------------------------

def _save_viz_image(path: Path, bgr: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), bgr)

def _best_box_from_result_any(result) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
    """
    Return the single highest-confidence box from one Ultralytics result (any class).
    No confidence thresholding; returns (xyxy_int, conf) or None if no boxes.
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes
    conf = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()
    i = int(np.argmax(conf))
    x1, y1, x2, y2 = map(float, xyxy[i])
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))), float(conf[i])

def _best_box_for_class(result, cls_id: int) -> Optional[Tuple[Tuple[int,int,int,int], float]]:
    """
    Return the highest-confidence box for a specific class id, ignoring thresholds.
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes
    conf = boxes.conf.detach().cpu().numpy()
    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls  = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros_like(conf)
    idxs = np.where(cls.astype(int) == int(cls_id))[0]
    if idxs.size == 0:
        return None
    best_i = int(idxs[np.argmax(conf[idxs])])
    x1, y1, x2, y2 = map(float, xyxy[best_i])
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))), float(conf[best_i])

def _square_crop_bounds(cx: float, cy: float, side: float, W: int, H: int) -> Tuple[int, int, int, int]:
    half = side / 2.0
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    x1 = int(round(cx + half)); y1 = int(round(cy + half))
    # clamp & adjust
    if x0 < 0:  x1 -= x0; x0 = 0
    if y0 < 0:  y1 -= y0; y0 = 0
    if x1 > W:
        shift = x1 - W; x0 = max(0, x0 - shift); x1 = W
    if y1 > H:
        shift = y1 - H; y0 = max(0, y0 - shift); y1 = H
    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = max(1, min(x1, W))
    y1 = max(1, min(y1, H))
    return x0, y0, x1, y1

def _roi_from_disc_mask(disc_mask: np.ndarray,
                        fallback_box: Tuple[int,int,int,int],
                        pad_frac: float,
                        W: int, H: int) -> Tuple[int,int,int,int]:
    """
    Square ROI centered on disc using the mask's tight bbox (fallback to YOLO disc box if mask empty).
    side = max(dw, dh) * (1 + 2*pad_frac)
    """
    tight = tight_bbox_from_mask(disc_mask)
    if tight is None:
        x1, y1, x2, y2 = fallback_box
    else:
        x1, y1, x2, y2 = tight
    dw = max(1, x2 - x1); dh = max(1, y2 - y1)
    side = max(dw, dh) * (1.0 + 2.0 * pad_frac)
    cx = x1 + dw / 2.0; cy = y1 + dh / 2.0
    return _square_crop_bounds(cx, cy, side, W, H)

def _box_area(b: Tuple[int,int,int,int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

def _mirror_rel_under(src_path: Path, new_root: Path, new_ext: str) -> Path:
    """Mirror below 'images' if present; else just use filename."""
    parts = list(src_path.parts)
    rel = Path(src_path.name)
    if "images" in parts:
        idx = parts.index("images")
        rel = Path(*parts[idx + 1:])
    return (new_root / rel).with_suffix(new_ext)

def _write_pred_labels_for_image(out_labels_root: Path, images_root: Path, img: Image,
                                 overwrite: bool = True, require_both: bool = False) -> Optional[Path]:
    """
    Uses your Image.yolo_lines_2class(use_gt=False) to emit predicted labels (disc=0, cup=1).
    Writes one file mirroring the images tree; returns path or None if skipped.
    """
    lines = list(img.yolo_lines_2class(use_gt=False))
    if require_both:
        classes_present = {int(float(ln.split()[0])) for ln in lines if ln.strip()}
        if not ({0, 1} <= classes_present):
            return None
    outp = _mirror_rel_under(img.image_path, out_labels_root, ".txt")
    if outp.exists() and not overwrite:
        return outp
    ensure_dir(outp.parent)
    with outp.open("w") as f:
        for ln in lines:
            f.write(ln + "\n")
    return outp


# ======================================================================
# High-level pipeline
# ======================================================================

@dataclass
class Args:
    images_root: Path
    weights: Path                 # MULTICLASS YOLO (disc+cup)
    medsam_ckpt: Path
    out_dir: Path

    # derived subpaths
    out_labels: Path
    out_disc: Path
    out_cup: Path
    out_viz: Path
    out_viz_compare: Path
    csv_path: Path

    include: Tuple[str, ...]
    exclude: Tuple[str, ...]
    recursive: bool
    subset_n: int
    subset_seed: int

    conf: float
    iou: float
    device: Optional[str]
    overwrite: bool
    save_viz: bool

    # evaluation (optional)
    eval_enabled: bool
    gt_disc_root: Optional[Path]
    gt_cup_root: Optional[Path]
    viz_compare: bool

    # Optional second model for cup on ROI + ROI padding
    cup_weights: Optional[Path]
    roi_pad_frac: float


@dataclass
class _CollectCfg:
    project_dir: Path
    images_root: Path
    disc_masks: Optional[Path] = None
    cup_masks: Optional[Path] = None
    include_name_contains: Optional[List[str]] = None
    exclude_name_contains: Optional[List[str]] = None
    recursive: bool = True

def _attach_gt_masks(images: List[Image],
                     gt_disc_root: Optional[Path],
                     gt_cup_root: Optional[Path]) -> None:
    disc_map = stem_map_by_first_match(gt_disc_root) if gt_disc_root else {}
    cup_map  = stem_map_by_first_match(gt_cup_root)  if gt_cup_root  else {}
    for img in images:
        stem = img.image_path.stem
        if stem in disc_map:
            img.set_mask(Structure.DISC, LabelType.GT, BinaryMaskRef(path=disc_map[stem]))
        if stem in cup_map:
            img.set_mask(Structure.CUP, LabelType.GT, BinaryMaskRef(path=cup_map[stem]))
        img.ensure_boxes_from_masks()

def collect_images(args: Args) -> List[Image]:
    cfg = _CollectCfg(
        project_dir=args.images_root.parent if args.images_root.parent else Path("."),
        images_root=args.images_root,
        disc_masks=args.gt_disc_root if args.eval_enabled else None,
        cup_masks=args.gt_cup_root  if args.eval_enabled else None,
        include_name_contains=list(args.include) if args.include else None,
        exclude_name_contains=list(args.exclude) if args.exclude else None,
        recursive=args.recursive,
    )
    coll = DatasetCollector(cfg)  # type: ignore[arg-type]
    ds = coll.collect()

    if args.subset_n and args.subset_n > 0:
        by = group_by_subject(ds.images)
        keys = list(by.keys())
        rng = np.random.RandomState(args.subset_seed)
        rng.shuffle(keys)
        picked: List[Image] = []
        for k in keys:
            if len(picked) >= args.subset_n:
                break
            picked.extend(by[k])
        ds.images = picked[:args.subset_n]

    if args.eval_enabled:
        _attach_gt_masks(ds.images, args.gt_disc_root, args.gt_cup_root)

    return ds.images


def build_cup_box_from_disc_mask(disc_mask: np.ndarray,
                                 fallback_box: Tuple[int,int,int,int]) -> Optional[Tuple[int,int,int,int]]:
    tight = tight_bbox_from_mask(disc_mask)
    if tight is None:
        return None
    box = shrink_box_to_fit_mask(disc_mask, tight, step_frac=0.02, max_iter=300)
    if box is None:
        box = shrink_box_to_fit_mask(disc_mask, fallback_box, step_frac=0.02, max_iter=300)
    return box


def _process_one_image(
    img: Image,
    mc_model: YOLO,
    msam: MedSAMModel,
    out_labels_dir: Path,
    images_root: Path,
    out_disc_dir: Path,
    out_cup_dir: Path,
    out_viz_dir: Path,
    out_viz_compare_dir: Path,
    save_viz: bool = True,
    eval_enabled: bool = False,
    viz_compare: bool = False,
    # Optional cup-on-ROI model + ROI pad
    cup_yolo_model: Optional[YOLO] = None,
    roi_pad_frac: float = 0,
    conf: float = 0.25,   # kept for signature compat; we ignore thresholding logic
    iou: float = 0.50,
    overwrite: bool = True,
) -> Tuple[
    Optional[float], Optional[float],           # pred_cd_ratio, gt_cd_ratio
    Optional[Path], Optional[Path],             # disc_mask_path, cup_mask_path
    Optional[Path], Optional[Path],             # viz_path, viz_compare_path
    Optional[float], Optional[float],           # disc_box_iou, cup_box_iou
    Optional[float], Optional[float],           # disc_dice, cup_dice
]:
    # A) Predict disc & cup bounding boxes via MULTICLASS detector with conf=0.0 (no threshold)
    img_bgr = load_image_bgr(img.image_path)
    mc_res = mc_model.predict(source=img_bgr, conf=0.0, iou=iou, device=msam.device, verbose=False)
    if not mc_res:
        # no detections at all → cannot proceed
        return None, None, None, None, None, None, None, None, None, None
    mc_res0 = mc_res[0]

    # Always take the best per class (disc=0, cup=1), ignoring conf thresholds
    disc_best = _best_box_for_class(mc_res0, cls_id=0)
    cup_best  = _best_box_for_class(mc_res0, cls_id=1)

    # Require a disc box to proceed (MedSAM needs a prompt)
    if disc_best is None:
        # fallback: best box of any class (rare edge); use as disc proxy
        disc_best = _best_box_from_result_any(mc_res0)
        if disc_best is None:
            return None, None, None, None, None, None, None, None, None, None

    (dx1, dy1, dx2, dy2), _disc_conf = disc_best
    disc_bb = BoundingBox(float(dx1), float(dy1), float(dx2), float(dy2))
    img.set_box(Structure.DISC, LabelType.PRED, disc_bb)

    # Optionally attach initial cup candidate (may be None)
    cup_det_xyxy: Optional[Tuple[int,int,int,int]] = None
    if cup_best is not None:
        (cx1, cy1, cx2, cy2), _ = cup_best
        cup_det_xyxy = (int(cx1), int(cy1), int(cx2), int(cy2))

    # Write current predicted labels (disc present; cup maybe later)
    _write_pred_labels_for_image(out_labels_dir, images_root, img, overwrite=overwrite, require_both=False)

    # B) MedSAM disc → mask
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    emb, H, W, _ = embed_image_1024(msam, img_rgb)

    dxyxy = (int(round(disc_bb.x1)), int(round(disc_bb.y1)),
             int(round(disc_bb.x2)), int(round(disc_bb.y2)))
    pred_disc_mask = medsam_infer(msam, emb, dxyxy, H, W)
    img.set_mask(Structure.DISC, LabelType.PRED, BinaryMaskRef(array=pred_disc_mask))
    seg_disc_path = out_disc_dir / f"{img.image_path.stem}.png"
    save_mask_png(seg_disc_path, pred_disc_mask)

    # C1) Cup from disc mask (fallback/“larger”)
    cup_from_mask_xyxy = build_cup_box_from_disc_mask(pred_disc_mask, dxyxy)

    # C2) Optional: Cup YOLO on disc ROI (candidate, often smaller) — run with conf=0.0, ignore thresholds
    cup_from_roi_xyxy: Optional[Tuple[int,int,int,int]] = None
    if cup_yolo_model is not None:
        rx0, ry0, rx1, ry1 = _roi_from_disc_mask(pred_disc_mask, dxyxy, pad_frac=roi_pad_frac, W=W, H=H)
        roi = img_bgr[ry0:ry1, rx0:rx1].copy()
        det = cup_yolo_model.predict(source=roi, conf=0.0, iou=iou, device=msam.device, verbose=False)
        if det:
            best_cup_any = _best_box_from_result_any(det[0])
            if best_cup_any is not None:
                (cx1, cy1, cx2, cy2), _ = best_cup_any
                cup_from_roi_xyxy = (rx0 + cx1, ry0 + cy1, rx0 + cx2, ry0 + cy2)

    # D) Choose final cup box:
    # Candidates: {MC cup (if any), ROI cup (if any), mask-derived cup (if any)} → pick the smallest area.
    candidates: List[Tuple[int,int,int,int]] = []
    for c in (cup_det_xyxy, cup_from_roi_xyxy, cup_from_mask_xyxy):
        if c is not None:
            candidates.append(c)
    final_cup_xyxy: Optional[Tuple[int,int,int,int]] = min(candidates, key=_box_area) if candidates else None

    # If still no cup: output disc-only results
    if final_cup_xyxy is None:
        pred_cd_ratio = cdr_from_masks(pred_disc_mask, None)
        gt_cd_ratio = None
        if eval_enabled and img.gt_disc_mask and img.gt_cup_mask:
            gt_cd_ratio = cdr_from_masks(img.gt_disc_mask.load(), img.gt_cup_mask.load())

        viz_path = (out_viz_dir / f"{img.image_path.stem}_viz.jpg") if save_viz else None
        if viz_path:
            viz = overlay_masks_and_boxes(img_bgr, pred_disc_mask, None, dxyxy, None,
                                          cdr_text=("CDR: N/A" if pred_cd_ratio is None else f"CDR: {pred_cd_ratio:.3f}"))
            _save_viz_image(viz_path, viz)

        viz_compare_path = None
        if viz_compare and eval_enabled and img.gt_disc_mask and img.gt_cup_mask:
            gt_disc = img.gt_disc_mask.load()
            gt_cup  = img.gt_cup_mask.load()
            left_text = "Pred CDR: N/A"
            right_text = f"GT CDR: {(cdr_from_masks(gt_disc, gt_cup) or 0):.3f}"
            comp = make_side_by_side(img_bgr, pred_disc_mask, None, gt_disc, gt_cup, left_text, right_text)
            viz_compare_path = out_viz_compare_dir / f"{img.image_path.stem}_compare.jpg"
            _save_viz_image(viz_compare_path, comp)

        disc_box_iou = img.disc_iou() if eval_enabled else None
        disc_dice = dice(pred_disc_mask, img.gt_disc_mask.load()) if (eval_enabled and img.gt_disc_mask) else None
        return pred_cd_ratio, gt_cd_ratio, seg_disc_path, None, viz_path, viz_compare_path, disc_box_iou, None, disc_dice, None

    # E) MedSAM cup → mask using final cup box
    cup_box = BoundingBox(*map(float, final_cup_xyxy))
    img.set_box(Structure.CUP, LabelType.PRED, cup_box)
    pred_cup_mask = medsam_infer(msam, emb, final_cup_xyxy, H, W)
    img.set_mask(Structure.CUP, LabelType.PRED, BinaryMaskRef(array=pred_cup_mask))
    seg_cup_path = out_cup_dir / f"{img.image_path.stem}.png"
    save_mask_png(seg_cup_path, pred_cup_mask)

    # Re-write labels so both disc and cup are present
    _write_pred_labels_for_image(out_labels_dir, images_root, img, overwrite=overwrite, require_both=False)

    # F) CDRs
    pred_cd_ratio = cdr_from_masks(pred_disc_mask, pred_cup_mask)
    gt_cd_ratio = None
    gt_disc = gt_cup = None
    if eval_enabled and img.gt_disc_mask and img.gt_cup_mask:
        gt_disc = img.gt_disc_mask.load()
        gt_cup  = img.gt_cup_mask.load()
        gt_cd_ratio = cdr_from_masks(gt_disc, gt_cup)

    # G) Viz (+ optional side-by-side)
    cdr_text = f"CDR: {pred_cd_ratio:.3f}" if pred_cd_ratio is not None else "CDR: N/A"
    viz = overlay_masks_and_boxes(img_bgr, pred_disc_mask, pred_cup_mask, dxyxy, final_cup_xyxy, cdr_text=cdr_text)
    viz_path = (out_viz_dir / f"{img.image_path.stem}_viz.jpg") if save_viz else None
    if viz_path:
        _save_viz_image(viz_path, viz)

    viz_compare_path = None
    if viz_compare and eval_enabled and gt_disc is not None and gt_cup is not None:
        left_text  = f"Pred CDR: {pred_cd_ratio:.3f}" if pred_cd_ratio is not None else "Pred CDR: N/A"
        right_text = f"GT CDR: {gt_cd_ratio:.3f}"      if gt_cd_ratio is not None else "GT CDR: N/A"
        comp = make_side_by_side(img_bgr, pred_disc_mask, pred_cup_mask, gt_disc, gt_cup, left_text, right_text)
        viz_compare_path = out_viz_compare_dir / f"{img.image_path.stem}_compare.jpg"
        _save_viz_image(viz_compare_path, comp)

    # H) Optional mask/box metrics
    disc_box_iou = img.disc_iou() if eval_enabled else None
    cup_box_iou  = img.cup_iou()  if eval_enabled else None
    disc_dice = dice(pred_disc_mask, img.gt_disc_mask.load()) if (eval_enabled and img.gt_disc_mask) else None
    cup_dice  = dice(pred_cup_mask,  img.gt_cup_mask.load())  if (eval_enabled and img.gt_cup_mask)  else None

    return pred_cd_ratio, gt_cd_ratio, seg_disc_path, seg_cup_path, viz_path, viz_compare_path, disc_box_iou, cup_box_iou, disc_dice, cup_dice


def _write_summary_csv(rows: List[dict], csv_path: Path) -> None:
    ensure_dir(csv_path.parent)
    fields = [
        "image_path", "label_path",
        "disc_mask_path", "cup_mask_path",
        "viz_path", "viz_compare_path",
        "pred_cd_ratio", "gt_cd_ratio", "cdr_error", "cdr_abs_error", "cdr_ape",
        "disc_box_iou", "cup_box_iou",
        "disc_dice", "cup_dice",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ======================================================================
# CLI + main
# ======================================================================

def _derive_outputs(out_dir: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    labels = out_dir / "labels"
    disc   = out_dir / "disc"
    cup    = out_dir / "cup"
    viz    = out_dir / "viz"
    vizc   = out_dir / "viz_compare"
    csvp   = out_dir / "summary.csv"
    for p in (labels, disc, cup, viz, vizc, csvp.parent):
        ensure_dir(p)
    return labels, disc, cup, viz, vizc, csvp

@dataclass
class _CLI:
    images_root: Path
    weights: Path
    medsam_ckpt: Path
    out_dir: Path
    include: Tuple[str, ...]
    exclude: Tuple[str, ...]
    recursive: bool
    subset_n: int
    subset_seed: int
    conf: float
    iou: float
    device: Optional[str]
    overwrite: bool
    save_viz: bool
    eval_enabled: bool
    gt_disc_root: Optional[Path]
    gt_cup_root: Optional[Path]
    viz_compare: bool
    cup_weights: Optional[Path]
    roi_pad_frac: float

def _parse_args() -> _CLI:
    ap = argparse.ArgumentParser(
        description="Multiclass YOLO (disc+cup) → optional cup-on-ROI YOLO → MedSAM disc/cup → CDR (always pick best one per class)."
    )
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--weights", required=True, help="Multiclass YOLO weights (disc=0, cup=1)")
    ap.add_argument("--medsam-ckpt", required=True)
    ap.add_argument("--out-dir", required=True)

    # Optional second model for cup on ROI
    ap.add_argument("--cup-weights", default="", help="Optional YOLO weights for cup detection on disc ROI")
    ap.add_argument("--roi-pad-frac", type=float, default=0.08, help="Padding fraction for disc ROI crop")

    ap.add_argument("--include", nargs="*", default=[])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--subset-n", type=int, default=0)
    ap.add_argument("--subset-seed", type=int, default=43)
    ap.add_argument("--conf", type=float, default=0.25)  # kept for API symmetry; not used for thresholding
    ap.add_argument("--iou",  type=float, default=0.50)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-overwrite", action="store_true")
    ap.add_argument("--no-viz", action="store_true")

    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--gt-disc-masks", default="")
    ap.add_argument("--gt-cup-masks",  default="")
    ap.add_argument("--viz-compare", action="store_true", help="Also save side-by-side Pred vs GT overlays (requires --eval)")

    a = ap.parse_args()
    return _CLI(
        images_root=expand(a.images_root),
        weights=expand(a.weights),
        medsam_ckpt=expand(a.medsam_ckpt),
        out_dir=expand(a.out_dir),
        include=tuple(a.include or []),
        exclude=tuple(a.exclude or []),
        recursive=bool(a.recursive),
        subset_n=int(a.subset_n),
        subset_seed=int(a.subset_seed),
        conf=float(a.conf),
        iou=float(a.iou),
        device=a.device,
        overwrite=not bool(a.no_overwrite),
        save_viz=not bool(a.no_viz),
        eval_enabled=bool(a.eval),
        gt_disc_root=(expand(a.gt_disc_masks) if a.gt_disc_masks else None),
        gt_cup_root=(expand(a.gt_cup_masks) if a.gt_cup_masks else None),
        viz_compare=bool(a.viz_compare),
        cup_weights=(expand(a.cup_weights) if a.cup_weights else None),
        roi_pad_frac=float(a.roi_pad_frac),
    )

def main() -> None:
    cli = _parse_args()

    out_labels, out_disc, out_cup, out_viz, out_viz_compare, csv_path = _derive_outputs(cli.out_dir)

    args = Args(
        images_root=cli.images_root,
        weights=cli.weights,
        medsam_ckpt=cli.medsam_ckpt,
        out_dir=cli.out_dir,
        out_labels=out_labels,
        out_disc=out_disc,
        out_cup=out_cup,
        out_viz=out_viz,
        out_viz_compare=out_viz_compare,
        csv_path=csv_path,
        include=cli.include,
        exclude=cli.exclude,
        recursive=cli.recursive,
        subset_n=cli.subset_n,
        subset_seed=cli.subset_seed,
        conf=cli.conf,
        iou=cli.iou,
        device=cli.device,
        overwrite=cli.overwrite,
        save_viz=cli.save_viz,
        eval_enabled=cli.eval_enabled,
        gt_disc_root=cli.gt_disc_root,
        gt_cup_root=cli.gt_cup_root,
        viz_compare=cli.viz_compare,
        cup_weights=cli.cup_weights,
        roi_pad_frac=cli.roi_pad_frac,
    )

    images = collect_images(args)
    if not images:
        raise SystemExit("[ERR] No images collected.")

    # Multiclass YOLO model (disc+cup). We always run with conf=0.0 and pick top per class.
    mc_model = YOLO(str(args.weights))

    # MedSAM
    dev = pick_device(args.device)
    msam = load_medsam(args.medsam_ckpt, dev, variant="vit_b")

    # Optional cup YOLO model (loaded once) — also run with conf=0.0
    cup_yolo_model = YOLO(str(args.cup_weights)) if args.cup_weights else None

    # Aggregates
    disc_iou_vals: List[float] = []
    cup_iou_vals: List[float]  = []
    disc_dice_vals: List[float] = []
    cup_dice_vals: List[float]  = []
    cdr_pred_vals: List[float]  = []
    cdr_gt_vals:   List[float]  = []

    rows: List[dict] = []
    for img in images:
        (pred_cd_ratio, gt_cd_ratio, seg_disc_path, seg_cup_path, viz_path, viz_compare_path,
         disc_box_iou, cup_box_iou, disc_dice, cup_dice) = _process_one_image(
            img, mc_model, msam,
            out_labels_dir=args.out_labels, images_root=args.images_root,
            out_disc_dir=args.out_disc, out_cup_dir=args.out_cup,
            out_viz_dir=args.out_viz, out_viz_compare_dir=args.out_viz_compare,
            save_viz=args.save_viz, eval_enabled=args.eval_enabled, viz_compare=args.viz_compare,
            cup_yolo_model=cup_yolo_model, roi_pad_frac=args.roi_pad_frac,
            conf=args.conf, iou=args.iou, overwrite=args.overwrite,
        )

        rel = img.image_path.resolve().relative_to(args.images_root.resolve())
        label_path = (args.out_labels / rel).with_suffix(".txt")

        if disc_box_iou is not None: disc_iou_vals.append(disc_box_iou)
        if cup_box_iou  is not None: cup_iou_vals.append(cup_box_iou)
        if disc_dice    is not None: disc_dice_vals.append(disc_dice)
        if cup_dice     is not None: cup_dice_vals.append(cup_dice)
        if (pred_cd_ratio is not None) and (gt_cd_ratio is not None):
            cdr_pred_vals.append(pred_cd_ratio)
            cdr_gt_vals.append(gt_cd_ratio)

        err = abs_err = ape = ""
        if (pred_cd_ratio is not None) and (gt_cd_ratio is not None):
            e = pred_cd_ratio - gt_cd_ratio
            err = f"{e:.6f}"
            abs_err = f"{abs(e):.6f}"
            denom = max(1e-6, abs(gt_cd_ratio))
            ape = f"{abs(e)/denom:.6f}"

        rows.append({
            "image_path": str(img.image_path),
            "label_path": str(label_path) if label_path.exists() else "",
            "disc_mask_path": str(seg_disc_path) if seg_disc_path else "",
            "cup_mask_path":  str(seg_cup_path)  if seg_cup_path  else "",
            "viz_path":       str(viz_path)      if viz_path      else "",
            "viz_compare_path": str(viz_compare_path) if viz_compare_path else "",
            "pred_cd_ratio": f"{pred_cd_ratio:.6f}" if pred_cd_ratio is not None else "",
            "gt_cd_ratio":   f"{gt_cd_ratio:.6f}"   if gt_cd_ratio   is not None else "",
            "cdr_error":     err,
            "cdr_abs_error": abs_err,
            "cdr_ape":       ape,
            "disc_box_iou": f"{disc_box_iou:.6f}" if disc_box_iou is not None else "",
            "cup_box_iou":  f"{cup_box_iou:.6f}"  if cup_box_iou  is not None else "",
            "disc_dice":    f"{disc_dice:.6f}"    if disc_dice    is not None else "",
            "cup_dice":     f"{cup_dice:.6f}"     if cup_dice     is not None else "",
        })

    _write_summary_csv(rows, args.csv_path)

    print(f"[OK] Processed {len(rows)} images.")
    print(f"[OK] Outputs → {args.out_dir}")
    print(f"  labels/       → {args.out_labels}")
    print(f"  disc/         → {args.out_disc}")
    print(f"  cup/          → {args.out_cup}")
    print(f"  viz/          → {args.out_viz}")
    if args.viz_compare:
        print(f"  viz_compare/  → {args.out_viz_compare}")
    print(f"  summary       → {args.csv_path}")

    # ---------- CDR metrics ----------
    if args.eval_enabled and cdr_pred_vals and cdr_gt_vals:
        pred = np.asarray(cdr_pred_vals, dtype=np.float64)
        gt   = np.asarray(cdr_gt_vals,   dtype=np.float64)
        diff = pred - gt
        mae  = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        bias = float(np.mean(diff))
        mape = float(np.mean(np.abs(diff) / np.maximum(1e-6, np.abs(gt))))
        r    = float(np.corrcoef(pred, gt)[0, 1]) if pred.size >= 2 else float("nan")
        r2   = float(r * r) if np.isfinite(r) else float("nan")
        md   = float(np.mean(diff))
        sd   = float(np.std(diff, ddof=1)) if diff.size >= 2 else float("nan")
        loa_lo = md - 1.96 * sd if np.isfinite(sd) else float("nan")
        loa_hi = md + 1.96 * sd if np.isfinite(sd) else float("nan")

        print("\n[CDR] Accuracy over items with GT present")
        print(f"  N pairs        : {pred.size}")
        print(f"  MAE            : {mae:.4f}")
        print(f"  RMSE           : {rmse:.4f}")
        print(f"  Bias (mean err): {bias:.4f}")
        print(f"  MAPE           : {mape:.4%}")
        print(f"  Pearson r      : {r:.4f}")
        print(f"  R^2            : {r2:.4f}")
        print(f"  Bland–Altman   : mean={md:.4f}, LoA=({loa_lo:.4f}, {loa_hi:.4f})")
    elif args.eval_enabled:
        print("[CDR] --eval set, but no valid GT/pred CDR pairs were found. "
              "Ensure both --gt-disc-masks and --gt-cup-masks are provided and match image stems.")


if __name__ == "__main__":
    main()