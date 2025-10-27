# predict_save_labels.py
from __future__ import annotations

"""
Batch two-stage inference (disc -> cup ROI) that saves YOLO label files.

- Stage A: disc-only detector on full image.
- Stage B: cup-only detector on a square ROI centered at the disc with padding.

Outputs:
- For each image, a YOLO .txt label file under --out-labels mirroring the images tree.
  Format: "<class> <xc> <yc> <w> <h>" normalized to original image size.
  Classes: 0=disc, 1=cup.

Notes:
- Uses your package classes: ImageFactory, Image, BoundingBox, Structure, LabelType.
- Defaults to processing only filenames that contain "PAPILA" (configurable via --include/--exclude).
- If --require-both is set, label file is written only when both disc and cup are detected.

This script does NOT draw or save annotated images; it focuses on labels for downstream evaluation.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from src.imgpipe.image_factory import ImageFactory
from  src.imgpipe.image import Image
from  src.imgpipe.bounding_box import BoundingBox
from  src.imgpipe.enums import LabelType, Structure


# ----------------------------- small utils -----------------------------

def _expand(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()

def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _load_image_bgr(p: Path) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Failed to read image: {p}")
    return im

def _to_int_box(xyxy: np.ndarray | Iterable[float]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, xyxy)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def _square_crop_bounds(cx: float, cy: float, side: float, W: int, H: int) -> Tuple[int, int, int, int]:
    half = side / 2.0
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    x1 = int(round(cx + half)); y1 = int(round(cy + half))
    if x0 < 0:  x1 -= x0; x0 = 0
    if y0 < 0:  y1 -= y0; y0 = 0
    if x1 > W:
        shift = x1 - W; x0 = max(0, x0 - shift); x1 = W
    if y1 > H:
        shift = y1 - H; y0 = max(0, y0 - shift); y1 = H
    x0 = max(0, min(x0, max(0, W - 1)))
    y0 = max(0, min(y0, max(0, H - 1)))
    x1 = max(1, min(x1, W))
    y1 = max(1, min(y1, H))
    return x0, y0, x1, y1


# ----------------------------- config types -----------------------------

@dataclass(frozen=True)
class PredictConfig:
    images_root: Path
    out_labels: Path
    stageA_weights: Path
    stageB_weights: Path

    # Filtering
    include_name_contains: Tuple[str, ...] = ("PAPILA",)
    exclude_name_contains: Tuple[str, ...] = ()
    recursive: bool = True

    # Detection thresholds
    confA: float = 0.25
    iouA: float = 0.50
    confB: float = 0.10
    iouB: float = 0.50

    # ROI config
    od_pad_pct: float = 0.08  # fraction of disc side added on EACH side (total ~ (1+2p)*side)

    # Class ids (adjust if your training differs)
    disc_class_id_stageA: int = 0  # disc in Stage A model
    cup_class_id_stageB: Optional[int] = None  # None for single-class cup model; else class index

    # Behavior
    device: Optional[str] = None  # e.g., "0" or "cpu"
    require_both: bool = False    # write label only if both disc and cup are detected
    overwrite: bool = True        # overwrite existing .txt files


# ----------------------------- YOLO helpers -----------------------------

def _best_box_from_results(result, class_id: Optional[int], conf_thres: float
                           ) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
    """
    From a single Ultralytics result, return the highest-confidence (xyxy int) box and its conf.
    If class_id is None, consider all classes (useful for single-class models).
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    cls  = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros_like(conf)

    inds = np.arange(len(conf))
    if class_id is not None:
        inds = inds[cls.astype(int) == int(class_id)]
    inds = [i for i in inds if conf[i] >= conf_thres]
    if not inds:
        return None
    i_best = max(inds, key=lambda i: conf[i])
    return _to_int_box(xyxy[i_best]), float(conf[i_best])

def _predict_one(model: YOLO, img_bgr: np.ndarray, conf: float, iou: float, device: Optional[str]):
    preds = model.predict(source=img_bgr, conf=conf, iou=iou, device=device, verbose=False)
    return preds[0] if preds else None


# ----------------------------- core predictor -----------------------------

class TwoStagePredictor:
    def __init__(self, cfg: PredictConfig) -> None:
        self.cfg = cfg
        self.modelA = YOLO(str(cfg.stageA_weights))
        self.modelB = YOLO(str(cfg.stageB_weights))

    def predict_boxes(self, img: Image) -> Tuple[Optional[BoundingBox], Optional[BoundingBox]]:
        """
        Returns (disc_box_global, cup_box_global) in original image coordinates.
        Any of them can be None if not detected.
        """
        im_bgr = _load_image_bgr(img.image_path)
        H, W = im_bgr.shape[:2]

        # Stage A: disc on full image
        resA = _predict_one(self.modelA, im_bgr, self.cfg.confA, self.cfg.iouA, self.cfg.device)
        best_disc = _best_box_from_results(resA, self.cfg.disc_class_id_stageA, self.cfg.confA)

        disc_box_xyxy: Optional[Tuple[int, int, int, int]] = None
        if best_disc is not None:
            disc_box_xyxy, _ = best_disc

        # ROI (square) around disc; fall back to full image if missing
        if disc_box_xyxy is not None:
            x1, y1, x2, y2 = disc_box_xyxy
            dw = max(1, x2 - x1)
            dh = max(1, y2 - y1)
            side = max(dw, dh) * (1.0 + 2.0 * self.cfg.od_pad_pct)
            cx = x1 + dw / 2.0
            cy = y1 + dh / 2.0
            rx0, ry0, rx1, ry1 = _square_crop_bounds(cx, cy, side, W, H)
        else:
            rx0, ry0, rx1, ry1 = 0, 0, W, H

        roi = im_bgr[ry0:ry1, rx0:rx1].copy()

        # Stage B: cup on ROI (or full image)
        resB = _predict_one(self.modelB, roi, self.cfg.confB, self.cfg.iouB, self.cfg.device)
        best_cup = _best_box_from_results(resB, self.cfg.cup_class_id_stageB, self.cfg.confB)

        cup_box_global: Optional[Tuple[int, int, int, int]] = None
        if best_cup is not None:
            (cx1, cy1, cx2, cy2), _ = best_cup
            cup_box_global = (rx0 + cx1, ry0 + cy1, rx0 + cx2, ry0 + cy2)

        # Wrap in your BoundingBox class for consistent downstream use
        disc_bb = BoundingBox(*map(float, disc_box_xyxy)) if disc_box_xyxy is not None else None
        cup_bb  = BoundingBox(*map(float, cup_box_global)) if cup_box_global is not None else None
        return disc_bb, cup_bb


# ----------------------------- writer -----------------------------

class LabelWriter:
    """
    Writes YOLO-format labels for PRED boxes using your Image API:
      - Calls img.set_box(..., kind=LabelType.PRED, ...)
      - Uses img.yolo_lines_2class(use_gt=False) to emit lines (0=disc, 1=cup)
    """

    def __init__(self, out_root: Path, images_root: Path, overwrite: bool) -> None:
        self.out_root = out_root
        self.images_root = images_root
        self.overwrite = overwrite

    def _label_path_for(self, img: Image) -> Path:
        # Mirror the images tree under out_root; replace extension with .txt
        rel = Path(img.image_path).resolve().relative_to(self.images_root.resolve())
        return self.out_root.joinpath(rel).with_suffix(".txt")

    def write(self, img: Image, require_both: bool = False) -> Optional[Path]:
        lines = list(img.yolo_lines_2class(use_gt=False))
        if require_both:
            # Ensure both classes present (0 and 1)
            classes_present = {int(float(ln.split()[0])) for ln in lines if ln.strip()}
            if not ({0, 1} <= classes_present):
                return None

        outp = self._label_path_for(img)
        if outp.exists() and not self.overwrite:
            return outp
        _ensure_parent(outp)
        with outp.open("w") as f:
            for ln in lines:
                f.write(ln + "\n")
        return outp


# ----------------------------- dataset runner -----------------------------

class DatasetPredictor:
    def __init__(self, cfg: PredictConfig) -> None:
        self.cfg = cfg
        self.predictor = TwoStagePredictor(cfg)
        self.writer = LabelWriter(cfg.out_labels, cfg.images_root, cfg.overwrite)

    def _collect_images(self) -> List[Image]:
        inc = list(self.cfg.include_name_contains) if self.cfg.include_name_contains else None
        exc = list(self.cfg.exclude_name_contains) if self.cfg.exclude_name_contains else None
        factory = ImageFactory(
            dataset_name=self.cfg.images_root.name or "dataset",
            images_root=self.cfg.images_root,
            disc_masks_root=None,   # not required for prediction
            cup_masks_root=None,    # not required for prediction
            include_name_contains=inc,
            exclude_name_contains=exc,
            recursive=self.cfg.recursive,
        )
        return factory.collect()

    def run(self) -> None:
        items = self._collect_images()
        n_total = len(items)
        n_pred  = 0
        n_both  = 0
        n_saved = 0

        for img in items:
            try:
                disc_bb, cup_bb = self.predictor.predict_boxes(img)
                if disc_bb is not None:
                    img.set_box(Structure.DISC, LabelType.PRED, disc_bb)
                if cup_bb is not None:
                    img.set_box(Structure.CUP, LabelType.PRED, cup_bb)

                n_pred += 1
                if disc_bb is not None and cup_bb is not None:
                    n_both += 1

                outp = self.writer.write(img, require_both=self.cfg.require_both)
                if outp is not None:
                    n_saved += 1

            except Exception as e:
                print(f"[WARN] Skipping {img.image_path}: {e}")

        # Console summary
        print("---- Prediction Label Summary ----")
        print(f"Images scanned     : {n_total}")
        print(f"Images inferred    : {n_pred}")
        print(f"Both disc & cup    : {n_both}")
        print(f"Labels written     : {n_saved}")
        print(f"Output labels root : {self.cfg.out_labels}")
        print("----------------------------------")


# ----------------------------- CLI -----------------------------

def _parse_args() -> PredictConfig:
    ap = argparse.ArgumentParser(description="Run two-stage disc→cup ROI inference and save YOLO labels.")
    ap.add_argument("--images-root", required=True, help="Root directory of fundus images.")
    ap.add_argument("--out-labels",  required=True, help="Output root for YOLO labels (.txt).")
    ap.add_argument("--stageA", required=True, help="Path to Stage A (disc-only) weights, e.g., best.pt")
    ap.add_argument("--stageB", required=True, help="Path to Stage B (cup-ROI) weights, e.g., best.pt")

    # Filtering
    ap.add_argument("--include", nargs="*", default=["PAPILA"], help="Case-insensitive substrings to include.")
    ap.add_argument("--exclude", nargs="*", default=[], help="Case-insensitive substrings to exclude.")
    ap.add_argument("--recursive", action="store_true", help="Recurse through subfolders.")

    # Thresholds & ROI
    ap.add_argument("--confA", type=float, default=0.25, help="Disc confidence threshold.")
    ap.add_argument("--iouA",  type=float, default=0.50, help="Disc NMS IoU threshold.")
    ap.add_argument("--confB", type=float, default=0.10, help="Cup confidence threshold.")
    ap.add_argument("--iouB",  type=float, default=0.50, help="Cup NMS IoU threshold.")
    ap.add_argument("--od-pad-pct", type=float, default=0.08, help="Disc ROI padding fraction per side.")

    # Classes & device
    ap.add_argument("--disc-class-id-stageA", type=int, default=0, help="Disc class id in Stage A model.")
    ap.add_argument("--cup-class-id-stageB", type=int, default=-1,
                    help="Cup class id in Stage B model; set -1 for single-class model.")
    ap.add_argument("--device", default=None, help="Ultralytics device string, e.g., '0' or 'cpu'.")

    # Behavior
    ap.add_argument("--require-both", action="store_true", help="Write label only if both disc and cup are detected.")
    ap.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing label files.")

    args = ap.parse_args()

    images_root = _expand(args.images_root)
    out_labels  = _expand(args.out_labels)
    stageA      = _expand(args.stageA)
    stageB      = _expand(args.stageB)

    if not images_root.exists():
        raise SystemExit(f"[ERR] images root not found: {images_root}")
    if not stageA.exists():
        raise SystemExit(f"[ERR] Stage A weights not found: {stageA}")
    if not stageB.exists():
        raise SystemExit(f"[ERR] Stage B weights not found: {stageB}")

    cup_id = None if int(args.cup_class_id_stageB) < 0 else int(args.cup_class_id_stageB)

    return PredictConfig(
        images_root=images_root,
        out_labels=out_labels,
        stageA_weights=stageA,
        stageB_weights=stageB,
        include_name_contains=tuple(args.include or []),
        exclude_name_contains=tuple(args.exclude or []),
        recursive=bool(args.recursive),
        confA=float(args.confA),
        iouA=float(args.iouA),
        confB=float(args.confB),
        iouB=float(args.iouB),
        od_pad_pct=float(args.od_pad_pct),
        disc_class_id_stageA=int(args.disc_class_id_stageA),
        cup_class_id_stageB=cup_id,
        device=args.device,
        require_both=bool(args.require_both),
        overwrite=not bool(args.no_overwrite),
    )


def main() -> None:
    cfg = _parse_args()
    runner = DatasetPredictor(cfg)
    runner.run()


if __name__ == "__main__":
    main()