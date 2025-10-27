#!/usr/bin/env python3
# predict_visualize.py
"""
Two-stage inference + visualization for a single fundus image.

Stage A: disc-only detector (single class 'disc')
Stage B: cup-only detector, run on a square ROI around the disc (with padding)

Outputs:
- Annotated image with disc (red), cup (blue), and (optional) ROI (green)
- Console printout with boxes, confidences, and vertical CDR

Examples
--------
python predict_visualize.py \
  --image /path/to/fundus.png \
  --stageA /path/to/runs/detect/stageA_disc_only/weights/best.pt \
  --stageB /path/to/runs/detect/stageB_cup_roi/weights/best.pt \
  --od_pad_pct 0.08 \
  --out ./predictions/fundus_annot.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# -------------------- small utils --------------------

def _expand(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def load_image(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise SystemExit(f"[ERR] failed to read image: {path}")
    return im

def draw_box(img: np.ndarray, box_xyxy: Tuple[int,int,int,int], color, label: str="", thickness: int=2):
    x1,y1,x2,y2 = map(int, box_xyxy)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(0, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def square_crop_bounds(cx: float, cy: float, side: float, W: int, H: int) -> Tuple[int,int,int,int]:
    """Square ROI centered (cx,cy) with side, clamped to image bounds."""
    half = side / 2.0
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    x1 = int(round(cx + half)); y1 = int(round(cy + half))
    if x0 < 0:  x1 -= x0; x0 = 0
    if y0 < 0:  y1 -= y0; y0 = 0
    if x1 > W:
        shift = x1 - W; x0 = max(0, x0 - shift); x1 = W
    if y1 > H:
        shift = y1 - H; y0 = max(0, y0 - shift); y1 = H
    x0 = clamp(x0, 0, max(0, W-1)); y0 = clamp(y0, 0, max(0, H-1))
    x1 = clamp(x1, 1, W);           y1 = clamp(y1, 1, H)
    return x0, y0, x1, y1

def to_int_box(b: np.ndarray) -> Tuple[int,int,int,int]:
    x1,y1,x2,y2 = map(float, b)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def box_h(box_xyxy: Tuple[int,int,int,int]) -> int:
    return max(0, int(box_xyxy[3] - box_xyxy[1]))


# -------------------- config / args --------------------

@dataclass
class PredictConfig:
    image_path: Path
    stageA_weights: Path
    stageB_weights: Path
    od_pad_pct: float = 0.08
    confA: float = 0.25
    confB: float = 0.10
    iouA: float = 0.50
    iouB: float = 0.50
    show_roi: bool = True
    out_path: Optional[Path] = None
    device: Optional[str] = None  # e.g. "0" or "cpu"

def parse_args() -> PredictConfig:
    ap = argparse.ArgumentParser(description="Two-stage (disc→cup ROI) prediction + visualization for one image.")
    ap.add_argument("--image", required=True, help="Path to a fundus image.")
    ap.add_argument("--stageA", required=True, help="Path to Stage A (disc-only) weights, e.g., best.pt")
    ap.add_argument("--stageB", required=True, help="Path to Stage B (cup-ROI) weights, e.g., best.pt")
    ap.add_argument("--od_pad_pct", type=float, default=0.08, help="Square ROI padding around disc (fraction of disc side)")
    ap.add_argument("--confA", type=float, default=0.25, help="Disc confidence threshold")
    ap.add_argument("--confB", type=float, default=0.10, help="Cup confidence threshold (on ROI)")
    ap.add_argument("--iouA", type=float, default=0.50, help="Disc NMS IoU threshold")
    ap.add_argument("--iouB", type=float, default=0.50, help="Cup NMS IoU threshold (ROI)")
    ap.add_argument("--no-show_roi", action="store_true", help="Do not draw ROI square")
    ap.add_argument("--out", default="", help="Where to save the annotated image (default: alongside input)")
    ap.add_argument("--device", default=None, help="Ultralytics device string, e.g. '0' or 'cpu'")
    args = ap.parse_args()

    image = _expand(args.image)
    stageA = _expand(args.stageA)
    stageB = _expand(args.stageB)
    outp = _expand(args.out) if args.out else None

    if not image.exists():
        raise SystemExit(f"[ERR] image not found: {image}")
    if not stageA.exists():
        raise SystemExit(f"[ERR] Stage A weights not found: {stageA}")
    if not stageB.exists():
        raise SystemExit(f"[ERR] Stage B weights not found: {stageB}")

    return PredictConfig(
        image_path=image,
        stageA_weights=stageA,
        stageB_weights=stageB,
        od_pad_pct=float(args.od_pad_pct),
        confA=float(args.confA),
        confB=float(args.confB),
        iouA=float(args.iouA),
        iouB=float(args.iouB),
        show_roi=(not args.no_show_roi),
        out_path=outp,
        device=args.device,
    )


# -------------------- inference helpers --------------------

def best_box_from_results(results, class_id: Optional[int], conf_thres: float) -> Optional[Tuple[Tuple[int,int,int,int], float]]:
    """
    From a single Ultralytics result, return the highest-confidence box (xyxy int) and its conf.
    If class_id is None, consider all classes (useful for single-class models).
    """
    if results is None or len(results.boxes) == 0:
        return None
    boxes = results.boxes
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    cls  = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros_like(conf)

    # Filter by class (if requested) and confidence
    inds = np.arange(len(conf))
    if class_id is not None:
        inds = inds[cls.astype(int) == int(class_id)]
    inds = [i for i in inds if conf[i] >= conf_thres]
    if not inds:
        return None

    # Pick best confidence
    i_best = max(inds, key=lambda i: conf[i])
    return to_int_box(xyxy[i_best]), float(conf[i_best])

def run_stageA_disc(modelA: YOLO, img_bgr: np.ndarray, conf: float, iou: float, device: Optional[str]) -> Optional[Tuple[Tuple[int,int,int,int], float]]:
    pred = modelA.predict(source=img_bgr, conf=conf, iou=iou, device=device, verbose=False)
    if not pred:
        return None
    return best_box_from_results(pred[0], class_id=0, conf_thres=conf)  # disc = class 0

def run_stageB_cup(modelB: YOLO, roi_bgr: np.ndarray, conf: float, iou: float, device: Optional[str]) -> Optional[Tuple[Tuple[int,int,int,int], float]]:
    pred = modelB.predict(source=roi_bgr, conf=conf, iou=iou, device=device, verbose=False)
    if not pred:
        return None
    # Stage B is single-class cup (index 0 in its dataset)
    return best_box_from_results(pred[0], class_id=None, conf_thres=conf)


# -------------------- main logic --------------------

def main():
    cfg = parse_args()

    # Load image
    im = load_image(cfg.image_path)
    H, W = im.shape[:2]

    # Load models
    modelA = YOLO(str(cfg.stageA_weights))
    modelB = YOLO(str(cfg.stageB_weights))

    # ---- Stage A: disc detection ----
    disc = run_stageA_disc(modelA, im, conf=cfg.confA, iou=cfg.iouA, device=cfg.device)
    overlay = im.copy()
    disc_box = None
    disc_conf = None

    if disc is None:
        print("[WARN] No disc detected. Will attempt cup on full image.")
    else:
        disc_box, disc_conf = disc
        draw_box(overlay, disc_box, (0, 0, 255), f"disc {disc_conf:.2f}")  # red

    # ---- ROI around disc (square with padding), else full image ----
    roi_x0 = roi_y0 = 0
    roi_x1 = W
    roi_y1 = H
    if disc_box is not None:
        dx1, dy1, dx2, dy2 = disc_box
        dw = max(1, dx2 - dx1)
        dh = max(1, dy2 - dy1)
        side = max(dw, dh) * (1.0 + 2.0 * cfg.od_pad_pct)
        cx = dx1 + dw / 2.0
        cy = dy1 + dh / 2.0
        roi_x0, roi_y0, roi_x1, roi_y1 = square_crop_bounds(cx, cy, side, W, H)
        if cfg.show_roi:
            draw_box(overlay, (roi_x0, roi_y0, roi_x1, roi_y1), (0, 200, 0), "ROI", thickness=1)  # green

    roi = im[roi_y0:roi_y1, roi_x0:roi_x1].copy()

    # ---- Stage B: cup detection on ROI (or full image if disc missing) ----
    cup = run_stageB_cup(modelB, roi, conf=cfg.confB, iou=cfg.iouB, device=cfg.device)
    cup_box_global = None
    cup_conf = None

    if cup is None:
        print("[WARN] No cup detected.")
    else:
        # map ROI-local cup box back to full-image coords
        (cx1, cy1, cx2, cy2), cup_conf = cup
        cup_box_global = (roi_x0 + cx1, roi_y0 + cy1, roi_x0 + cx2, roi_y0 + cy2)
        draw_box(overlay, cup_box_global, (255, 0, 0), f"cup {cup_conf:.2f}")  # blue

    # ---- CDR (vertical) ----
    cdr = None
    if disc_box is not None and cup_box_global is not None:
        d_h = float(box_h(disc_box))
        c_h = float(box_h(cup_box_global))
        if d_h > 1e-6:
            cdr = c_h / d_h
            # Put CDR text in the top-left corner
            cv2.putText(overlay, f"CDR (vertical): {cdr:.3f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2, cv2.LINE_AA)

    # ---- Save result ----
    if cfg.out_path is None or str(cfg.out_path).strip() == "":
        out_path = cfg.image_path.parent / f"{cfg.image_path.stem}_annot.png"
    else:
        out_path = cfg.out_path
    ensure_dir(out_path)
    cv2.imwrite(str(out_path), overlay)

    # ---- Print summary ----
    print("\n=== Prediction Summary ===")
    print(f"Input : {cfg.image_path}")
    print(f"Output: {out_path}")
    if disc_box is not None:
        print(f"Disc: box={disc_box} conf={disc_conf:.3f}")
    else:
        print("Disc: not detected")
    if cup_box_global is not None:
        print(f"Cup : box={cup_box_global} conf={cup_conf:.3f}")
    else:
        print("Cup : not detected")
    if cdr is not None:
        print(f"CDR : {cdr:.4f}")
    print("==========================\n")


if __name__ == "__main__":
    main()