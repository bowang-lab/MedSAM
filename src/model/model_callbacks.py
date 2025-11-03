# model_callbacks.py
from __future__ import annotations
import inspect
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.imgpipe.normalized_box import NormalizedBox
from src.model.canvas_viz import save_val_canvas_debug

# =============================
# Config
# =============================
CLASS_NAMES: Tuple[str, ...] = ("disc", "cup")   # id 0->disc, 1->cup
LOG_FIRST_VAL_BATCH_ONLY = False                 # keep False to accumulate all; controls printing only
VAL_CANVAS_DIR = "runs/val_canvases"             # debug canvases (optional)

# =============================
# Utilities to peek loop locals
# =============================

def _find_loop_locals(expected: Tuple[str, ...]) -> Optional[dict]:
    """
    Walk up a few frames to find a scope (trainer/validator loop) that carries keys like
    'batch', 'preds', 'batch_i'. This is robust across minor Ultralytics changes.
    """
    f = inspect.currentframe()
    for _ in range(7):  # current + 6 parents
        if f is None:
            break
        loc = f.f_locals
        if isinstance(loc, dict) and all(k in loc for k in expected):
            return loc
        f = f.f_back
    return None

def _resolve_paths(batch: dict) -> Optional[Sequence[str]]:
    if not isinstance(batch, dict):
        return None
    for k in ("im_file", "paths", "path"):
        if k in batch:
            v = batch[k]
            return v if isinstance(v, (list, tuple)) else [v]
    return None

def _canvas_shape_from_batch(batch: dict) -> Tuple[Optional[int], Optional[int]]:
    imgs = batch.get("img")
    if imgs is None or not hasattr(imgs, "shape") or imgs.ndim != 4:
        return None, None
    H, W = int(imgs.shape[-2]), int(imgs.shape[-1])
    return H, W

def _to_numpy(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            return None
    return None

def _gt_arrays_from_batch(batch: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns:
      bboxes: (M,4) normalized xywh (relative to the canvas)
      cls:    (M,) or (M,1) class ids
      bidx:   (M,) image index within the batch for each GT row
    """
    bboxes = _to_numpy(batch.get("bboxes"))
    cls    = _to_numpy(batch.get("cls"))
    bidx   = _to_numpy(batch.get("batch_idx"))
    return bboxes, cls, bidx

def _image_id_from_pred_item(pred_item: dict, fallback_i: int) -> int:
    bidx = pred_item.get("batch_idx", None)
    if isinstance(bidx, torch.Tensor) and bidx.numel() > 0:
        return int(bidx.view(-1)[0].item())
    if isinstance(bidx, (list, tuple)) and len(bidx) > 0:
        return int(bidx[0])
    return int(fallback_i)

def _iter_top1_per_class(
    bboxes_xyxy: torch.Tensor, confs: torch.Tensor, clss: torch.Tensor
) -> Iterable[Tuple[float, float, float, float, int, float]]:
    """
    Yield top-1 prediction per class as (x1,y1,x2,y2,cid,conf) in *canvas pixels*.
    """
    if bboxes_xyxy is None or confs is None or clss is None:
        return
    if not (isinstance(bboxes_xyxy, torch.Tensor) and isinstance(confs, torch.Tensor) and isinstance(clss, torch.Tensor)):
        return
    if bboxes_xyxy.numel() == 0:
        return
    if confs.ndim == 1:
        confs = confs.unsqueeze(1)
    if clss.ndim == 1:
        clss = clss.unsqueeze(1)
    dets = torch.cat([bboxes_xyxy, confs, clss], dim=1)  # [N,6] -> x1,y1,x2,y2,conf,cls
    for c in dets[:, 5].unique():
        det_c = dets[dets[:, 5] == c]
        top = det_c[det_c[:, 4].argmax()]
        x1, y1, x2, y2, conf, cid = top.tolist()
        yield float(x1), float(y1), float(x2), float(y2), int(cid), float(conf)

def _gt_norm_for_image_np(
    bboxes_gt_np: Optional[np.ndarray],
    cls_gt_np: Optional[np.ndarray],
    bidx_np: Optional[np.ndarray],
    image_id: int,
) -> List[Tuple[float, float, float, float, int]]:
    """
    Filter GT rows belonging to image_id and return normalized (xc,yc,w,h,cid).
    """
    out: List[Tuple[float, float, float, float, int]] = []
    if bboxes_gt_np is None or cls_gt_np is None or bidx_np is None:
        return out
    if bboxes_gt_np.size == 0 or cls_gt_np.size == 0 or bidx_np.size == 0:
        return out

    cls_flat = cls_gt_np.reshape(-1)
    mask = (bidx_np.astype(np.int64) == int(image_id))
    if not np.any(mask):
        return out

    xywh = bboxes_gt_np[mask]                 # (K,4) normalized (canvas)
    cls_sel = cls_flat[mask].astype(np.int64)

    for i in range(xywh.shape[0]):
        xc, yc, w, h = map(float, xywh[i])
        out.append((xc, yc, w, h, int(cls_sel[i])))
    return out

def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))

def _pred_xyxy_to_norm_box(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float, float, float, float]:
    """
    Convert prediction xyxy (canvas px) -> normalized (xc,yc,w,h) in [0,1].
    """
    x1c, x2c = max(0.0, min(x1, W)), max(0.0, min(x2, W))
    y1c, y2c = max(0.0, min(y1, H)), max(0.0, min(y2, H))
    if x2c < x1c:
        x1c, x2c = x2c, x1c
    if y2c < y1c:
        y1c, y2c = y2c, y1c
    w = max(0.0, x2c - x1c)
    h = max(0.0, y2c - y1c)
    xc = x1c + 0.5 * w
    yc = y1c + 0.5 * h
    xn = _clip01(xc / max(W, 1e-9))
    yn = _clip01(yc / max(H, 1e-9))
    wn = _clip01(w  / max(W, 1e-9))
    hn = _clip01(h  / max(H, 1e-9))
    return xn, yn, wn, hn

def _gt_row_to_norm_box(xc: float, yc: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return _clip01(xc), _clip01(yc), _clip01(w), _clip01(h)

# =============================
# Metric helpers
# =============================

def _pair_metrics(pred_nb: NormalizedBox, gt_nb: NormalizedBox) -> Tuple[float, float, float]:
    """
    Returns (dice, iou, box_loss). Dice in [0,1], IoU in [0,1], box_loss >= 0.
    Dice loss is computed later as (1 - dice).
    """
    dice = float(pred_nb.dice(gt_nb))
    iou  = float(pred_nb.iou(gt_nb))
    # box_loss is asymmetric: pred vs gt (match YOLO convention)
    box_loss = float(pred_nb.box_loss(gt_nb))  # relies on your NormalizedBox.box_loss()
    return dice, iou, box_loss

class MetricAccumulator:
    """
    Accumulates sums and counts for per-class and combined metrics within an epoch.
    Stores: dice, iou, boxloss for classes {0,1} and 'both'.
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum_dice  = {0: 0.0, 1: 0.0, "both": 0.0}
        self.sum_iou   = {0: 0.0, 1: 0.0, "both": 0.0}
        self.sum_box   = {0: 0.0, 1: 0.0, "both": 0.0}
        self.count     = {0: 0,   1: 0,   "both": 0}

    def update_for_class(self, cls_id: int, dice: float, iou: float, boxloss: float) -> None:
        self.sum_dice[cls_id] += dice
        self.sum_iou[cls_id]  += iou
        self.sum_box[cls_id]  += boxloss
        self.count[cls_id]    += 1
        # combined
        self.sum_dice["both"] += dice
        self.sum_iou["both"]  += iou
        self.sum_box["both"]  += boxloss
        self.count["both"]    += 1

    def means(self) -> Dict[str, float]:
        """
        Returns a flat dict of mean values:
          dice_disc_mean, dice_cup_mean, dice_both_mean,
          iou_disc_mean,  iou_cup_mean,  iou_both_mean,
          box_disc_mean,  box_cup_mean,  box_both_mean,
          dice_loss_disc_mean, dice_loss_cup_mean, dice_loss_both_mean
        """
        out: Dict[str, float] = {}
        for name, cid in (("disc", 0), ("cup", 1), ("both", "both")):
            n = max(1, self.count[cid])  # avoid div by zero; means will be 0 if no matches
            dice_mean = self.sum_dice[cid] / n if self.count[cid] > 0 else float("nan")
            iou_mean  = self.sum_iou[cid]  / n if self.count[cid] > 0 else float("nan")
            box_mean  = self.sum_box[cid]  / n if self.count[cid] > 0 else float("nan")
            out[f"dice_{name}_mean"]      = dice_mean
            out[f"iou_{name}_mean"]       = iou_mean
            out[f"box_{name}_mean"]       = box_mean
            out[f"dice_loss_{name}_mean"] = (1.0 - dice_mean) if np.isfinite(dice_mean) else float("nan")
        return out

# Keep separate accumulators for train/val phases per epoch
_train_acc: Optional[MetricAccumulator] = None
_val_acc:   Optional[MetricAccumulator] = None

def _ensure_phase_accumulator(phase: str) -> MetricAccumulator:
    global _train_acc, _val_acc
    if phase == "train":
        if _train_acc is None:
            _train_acc = MetricAccumulator()
        return _train_acc
    else:
        if _val_acc is None:
            _val_acc = MetricAccumulator()
        return _val_acc

def _reset_phase_accumulator(phase: str) -> None:
    acc = _ensure_phase_accumulator(phase)
    acc.reset()

# =============================
# Per-batch processing (shared)
# =============================

def _build_nb_maps_for_item(
    pred_item: dict, H: Optional[int], W: Optional[int],
    gts_norm_rows: List[Tuple[float, float, float, float, int]]
) -> Tuple[Dict[int, NormalizedBox], Dict[int, NormalizedBox]]:
    """
    Returns (pred_norm_boxes_by_class, gt_norm_boxes_by_class) with at most one entry per class.
    """
    pred_norm_boxes_by_class: Dict[int, NormalizedBox] = {}
    gt_norm_boxes_by_class: Dict[int, NormalizedBox] = {}

    # Predictions: keep top-1 per class if canvas size known
    if H is not None and W is not None:
        bxyxy = pred_item.get("bboxes"); confs = pred_item.get("conf"); clss = pred_item.get("cls")
        for (x1, y1, x2, y2, cid, _conf) in _iter_top1_per_class(bxyxy, confs, clss):
            if cid in pred_norm_boxes_by_class:
                continue
            xn, yn, wn, hn = _pred_xyxy_to_norm_box(x1, y1, x2, y2, W, H)
            try:
                pred_norm_boxes_by_class[cid] = NormalizedBox(xn, yn, wn, hn)
            except ValueError:
                pass

    # GT: keep first per class
    for (xc, yc, w, h, cid) in gts_norm_rows:
        if cid in gt_norm_boxes_by_class:
            continue
        xn, yn, wn, hn = _gt_row_to_norm_box(xc, yc, w, h)
        try:
            gt_norm_boxes_by_class[cid] = NormalizedBox(xn, yn, wn, hn)
        except ValueError:
            pass

    return pred_norm_boxes_by_class, gt_norm_boxes_by_class

def _process_one_pred_item_for_metrics(
    pred_item: dict,
    batch: dict,
    acc: MetricAccumulator,
    log_visual: bool,
    phase: str
) -> None:
    """
    Extract per-class top-1 predictions and GT for one image, compute metrics, and update accumulator.
    """
    paths = _resolve_paths(batch)
    H, W = _canvas_shape_from_batch(batch)
    bboxes_gt_np, cls_gt_np, bidx_np = _gt_arrays_from_batch(batch)

    # Which image index this pred record belongs to?
    img_id = _image_id_from_pred_item(pred_item, fallback_i=0)
    gts_norm = _gt_norm_for_image_np(bboxes_gt_np, cls_gt_np, bidx_np, image_id=img_id)

    pred_map, gt_map = _build_nb_maps_for_item(pred_item, H, W, gts_norm)

    # For disc(0) and cup(1), compute metrics if we have both pred and gt
    for cid in (0, 1):
        pred_nb = pred_map.get(cid, None)
        gt_nb   = gt_map.get(cid, None)
        if pred_nb is not None and gt_nb is not None:
            dice, iou, box_loss = _pair_metrics(pred_nb, gt_nb)
            acc.update_for_class(cid, dice, iou, box_loss)

    # Optional: save a canvas visualization for val phase
    if log_visual and phase == "val" and "img" in batch and H is not None and W is not None:
        try:
            fname = None
            paths_seq = paths if paths else []
            if paths_seq and 0 <= img_id < len(paths_seq):
                fname = paths_seq[img_id]
            preds_px = []
            bxyxy = pred_item.get("bboxes"); confs = pred_item.get("conf"); clss = pred_item.get("cls")
            for (x1, y1, x2, y2, cid, conf) in _iter_top1_per_class(bxyxy, confs, clss):
                preds_px.append((x1, y1, x2, y2, cid, conf))
            out_path = save_val_canvas_debug(VAL_CANVAS_DIR, batch["img"][img_id], fname, preds_px, gts_norm, CLASS_NAMES)
            print(f"  [SAVE] {out_path}", flush=True)
        except Exception as e:
            print(f"  [WARN] canvas save failed: {e!r}", flush=True)

def _process_phase_batch(phase: str, container) -> None:
    """
    Shared handler for train and val batch-end events.
    """
    expected = ("batch", "preds", "batch_i")
    loc = _find_loop_locals(expected)
    if not loc:
        return
    preds = loc.get("preds")
    batch = loc.get("batch")
    batch_i = loc.get("batch_i")

    if preds is None or batch is None:
        return
    if not isinstance(preds, (list, tuple)):
        preds = [preds]

    acc = _ensure_phase_accumulator(phase)

    # Only reduce console noise for val; metric accumulation always runs
    log_visual = (phase == "val") and ((not LOG_FIRST_VAL_BATCH_ONLY) or (batch_i in (0, None)))
    log_visual = False

    if log_visual:
        print(f"\n---- {phase.capitalize()} batch {batch_i} ----", flush=True)

    for i, pred_item in enumerate(preds):
        if not isinstance(pred_item, dict):
            continue
        _process_one_pred_item_for_metrics(pred_item, batch, acc, log_visual, phase)

# =============================
# CSV & plotting
# =============================

def _csv_path(trainer) -> Path:
    return Path(trainer.save_dir) / "custom_metrics.csv"

def _ensure_csv_with_header(csv_fp: Path) -> None:
    if csv_fp.exists():
        return
    csv_fp.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        # train means
        "train.dice_disc", "train.dice_cup", "train.dice_both",
        "train.dice_loss_disc", "train.dice_loss_cup", "train.dice_loss_both",
        "train.iou_disc",  "train.iou_cup",  "train.iou_both",
        "train.box_disc",  "train.box_cup",  "train.box_both",
        # val means
        "val.dice_disc", "val.dice_cup", "val.dice_both",
        "val.dice_loss_disc", "val.dice_loss_cup", "val.dice_loss_both",
        "val.iou_disc",  "val.iou_cup",  "val.iou_both",
        "val.box_disc",  "val.box_cup",  "val.box_both",
    ]
    with csv_fp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

def _append_epoch_row(trainer, train_means: Dict[str, float], val_means: Dict[str, float]) -> None:
    csv_fp = _csv_path(trainer)
    _ensure_csv_with_header(csv_fp)

    row = {"epoch": int(getattr(trainer, "epoch", -1))}
    # flatten wanted keys in stable order
    mapping = [
        ("train.dice_disc",        "dice_disc_mean"),
        ("train.dice_cup",         "dice_cup_mean"),
        ("train.dice_both",        "dice_both_mean"),
        ("train.dice_loss_disc",   "dice_loss_disc_mean"),
        ("train.dice_loss_cup",    "dice_loss_cup_mean"),
        ("train.dice_loss_both",   "dice_loss_both_mean"),
        ("train.iou_disc",         "iou_disc_mean"),
        ("train.iou_cup",          "iou_cup_mean"),
        ("train.iou_both",         "iou_both_mean"),
        ("train.box_disc",         "box_disc_mean"),
        ("train.box_cup",          "box_cup_mean"),
        ("train.box_both",         "box_both_mean"),

        ("val.dice_disc",        "dice_disc_mean"),
        ("val.dice_cup",         "dice_cup_mean"),
        ("val.dice_both",        "dice_both_mean"),
        ("val.dice_loss_disc",   "dice_loss_disc_mean"),
        ("val.dice_loss_cup",    "dice_loss_cup_mean"),
        ("val.dice_loss_both",   "dice_loss_both_mean"),
        ("val.iou_disc",         "iou_disc_mean"),
        ("val.iou_cup",          "iou_cup_mean"),
        ("val.iou_both",         "iou_both_mean"),
        ("val.box_disc",         "box_disc_mean"),
        ("val.box_cup",          "box_cup_mean"),
        ("val.box_both",         "box_both_mean"),
    ]
    for out_key, src_key in mapping[:12]:
        row[out_key] = float(train_means.get(src_key, float("nan")))
    for out_key, src_key in mapping[12:]:
        row[out_key] = float(val_means.get(src_key, float("nan")))

    with csv_fp.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

def _load_csv_for_plots(csv_fp: Path):
    import pandas as pd
    if not csv_fp.exists():
        return None
    return pd.read_csv(csv_fp)

def _plot_metrics(trainer) -> None:
    """
    Create three figures (Dice, IoU, Box loss) vs epoch, each with 6 lines:
    train/val × disc/cup/both.
    """
    import pandas as pd
    csv_fp = _csv_path(trainer)
    df = _load_csv_for_plots(csv_fp)
    if df is None or df.empty:
        return

    # Define plot specs
    plots = [
        ("dice_vs_epoch.png",
         ["train.dice_disc", "train.dice_cup", "train.dice_both",
          "val.dice_disc",   "val.dice_cup",   "val.dice_both"],
         "Dice"),

        ("iou_vs_epoch.png",
         ["train.iou_disc", "train.iou_cup", "train.iou_both",
          "val.iou_disc",   "val.iou_cup",   "val.iou_both"],
         "IoU"),

        ("boxloss_vs_epoch.png",
         ["train.box_disc", "train.box_cup", "train.box_both",
          "val.box_disc",   "val.box_cup",   "val.box_both"],
         "Box Loss"),
    ]

    for fname, cols, ylabel in plots:
        plt.figure()
        for col in cols:
            if col in df.columns:
                plt.plot(df["epoch"], df[col], label=col)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Epoch")
        plt.legend()
        plt.grid(True, linewidth=0.3, alpha=0.6)
        out_path = Path(trainer.save_dir) / fname
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

# =============================
# Ultralytics callback entry points
# =============================

def on_train_start(trainer) -> None:
    # Prepare CSV and reset accumulators
    _ensure_csv_with_header(_csv_path(trainer))
    _reset_phase_accumulator("train")
    _reset_phase_accumulator("val")

def on_train_epoch_start(trainer) -> None:
    print("[INFO] ON TRAIN EPOCH START")
    _reset_phase_accumulator("train")

def on_train_batch_end(trainer) -> None:
    print("[INFO] ON TRAIN BATCH END")
    _process_phase_batch("train", trainer)

def on_val_start(validator) -> None:
    print("[INFO] ON VAL EPOCH START")
    _reset_phase_accumulator("val")

def on_val_batch_end(validator) -> None:
    print("[INFO] ON VAL BATCH END")
    _process_phase_batch("val", validator)

def on_fit_epoch_end(trainer) -> None:
    print("[INFO] ON FIT EPOCH END")
    """
    After validation ends, harvest both accumulators, write CSV, and plot.
    This runs once per epoch.
    """
    # Compute epoch means
    train_means = _ensure_phase_accumulator("train").means()
    val_means   = _ensure_phase_accumulator("val").means()

    # Persist per-epoch metrics
    _append_epoch_row(trainer, train_means, val_means)

    # Produce figures
    _plot_metrics(trainer)

    # (Optional) print short console summary
    ep = int(getattr(trainer, "epoch", -1))
    def _fmt(x: float) -> str:
        return "nan" if not np.isfinite(x) else f"{x:.4f}"
    print(f"\n==== Custom metrics (epoch {ep}) ====")
    print(" Train  | dice_disc:", _fmt(train_means["dice_disc_mean"]),
          " dice_cup:", _fmt(train_means["dice_cup_mean"]),
          " dice_both:", _fmt(train_means["dice_both_mean"]),
          " iou_both:", _fmt(train_means["iou_both_mean"]),
          " box_both:", _fmt(train_means["box_both_mean"]))
    print(" Val    | dice_disc:", _fmt(val_means["dice_disc_mean"]),
          " dice_cup:", _fmt(val_means["dice_cup_mean"]),
          " dice_both:", _fmt(val_means["dice_both_mean"]),
          " iou_both:", _fmt(val_means["iou_both_mean"]),
          " box_both:", _fmt(val_means["box_both_mean"]))