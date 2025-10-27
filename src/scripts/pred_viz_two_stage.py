# predict_visualize_dataset.py
from __future__ import annotations

"""
Batch visualization of two-stage (disc -> cup ROI) predictions.

- Stage A: disc-only detector on the full image.
- Stage B: cup-only detector on a square ROI centered at the disc (padding configurable).

Outputs per image:
- Annotated .png with rectangles: disc (red), cup (blue), optional ROI (green).
- Optional YOLO label .txt (0=disc, 1=cup) if --out-labels is provided.

Subset control:
- --num-samples N: visualize a random subset of N images (N=0 -> all).
- --seed: RNG seed for reproducible sampling.
"""

import argparse
import random
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

def _mirror_path(root_src: Path, root_dst: Path, file_in_src_tree: Path, new_suffix: str) -> Path:
    rel = file_in_src_tree.resolve().relative_to(root_src.resolve())
    return root_dst.joinpath(rel).with_suffix(new_suffix)

def _load_image_bgr(p: Path) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Failed to read image: {p}")
    return im

def _draw_box(img: np.ndarray, box: Tuple[int,int,int,int], color: Tuple[int,int,int], label: str="", thickness: int=2):
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(0, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def _to_int_box(xyxy: Iterable[float]) -> Tuple[int,int,int,int]:
    x1,y1,x2,y2 = map(float, xyxy)
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def _square_crop_bounds(cx: float, cy: float, side: float, W: int, H: int) -> Tuple[int,int,int,int]:
    half = side / 2.0
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    x1 = int(round(cx + half)); y1 = int(round(cy + half))
    if x0 < 0:  x1 -= x0; x0 = 0
    if y0 < 0:  y1 -= y0; y0 = 0
    if x1 > W:
        shift = x1 - W; x0 = max(0, x0 - shift); x1 = W
    if y1 > H:
        shift = y1 - H; y0 = max(0, y0 - shift); y1 = H
    x0 = max(0, min(x0, max(0, W-1)))
    y0 = max(0, min(y0, max(0, H-1)))
    x1 = max(1, min(x1, W))
    y1 = max(1, min(y1, H))
    return x0, y0, x1, y1

def _box_h(box_xyxy: Tuple[int,int,int,int]) -> int:
    return max(0, int(box_xyxy[3] - box_xyxy[1]))


# ----------------------------- config types -----------------------------

@dataclass(frozen=True)
class VizConfig:
    images_root: Path
    out_viz: Path
    stageA_weights: Path
    stageB_weights: Path

    # Optional label output
    out_labels: Optional[Path] = None

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
    od_pad_pct: float = 0.08  # fraction of disc side per side

    # Class ids
    disc_class_id_stageA: int = 0
    cup_class_id_stageB: Optional[int] = None  # None => single-class cup model

    # Behavior
    device: Optional[str] = None
    draw_roi: bool = True
    require_both: bool = False   # affects label writing only
    overwrite: bool = True

    # Subset control
    num_samples: int = 0         # 0 => all
    seed: Optional[int] = None


# ----------------------------- YOLO helpers -----------------------------

def _best_box_from_results(result, class_id: Optional[int], conf_thres: float
                           ) -> Optional[Tuple[Tuple[int,int,int,int], float]]:
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    cls  = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros_like(conf)

    idx = np.arange(len(conf))
    if class_id is not None:
        idx = idx[cls.astype(int) == int(class_id)]
    idx = [i for i in idx if conf[i] >= conf_thres]
    if not idx:
        return None
    i_best = max(idx, key=lambda i: conf[i])
    return _to_int_box(xyxy[i_best]), float(conf[i_best])

def _predict_one(model: YOLO, img_bgr: np.ndarray, conf: float, iou: float, device: Optional[str]):
    preds = model.predict(source=img_bgr, conf=conf, iou=iou, device=device, verbose=False)
    return preds[0] if preds else None


# ----------------------------- core predictor -----------------------------

class TwoStagePredictor:
    def __init__(self, cfg: VizConfig) -> None:
        self.cfg = cfg
        self.modelA = YOLO(str(cfg.stageA_weights))
        self.modelB = YOLO(str(cfg.stageB_weights))

    def predict_boxes(self, img: Image) -> Tuple[Optional[BoundingBox], Optional[BoundingBox], Optional[Tuple[int,int,int,int]], Optional[float], Optional[float]]:
        """
        Returns:
          disc_bb (global), cup_bb (global),
          roi_xyxy (global or None), disc_conf, cup_conf
        """
        im_bgr = _load_image_bgr(img.image_path)
        H, W = im_bgr.shape[:2]

        # Stage A: disc on full image
        resA = _predict_one(self.modelA, im_bgr, self.cfg.confA, self.cfg.iouA, self.cfg.device)
        best_disc = _best_box_from_results(resA, self.cfg.disc_class_id_stageA, self.cfg.confA)

        disc_box_xyxy: Optional[Tuple[int,int,int,int]] = None
        disc_conf: Optional[float] = None
        if best_disc is not None:
            disc_box_xyxy, disc_conf = best_disc

        # ROI (square) around disc; fallback to full image
        if disc_box_xyxy is not None:
            x1, y1, x2, y2 = disc_box_xyxy
            dw = max(1, x2 - x1)
            dh = max(1, y2 - y1)
            side = max(dw, dh) * (1.0 + 2.0 * self.cfg.od_pad_pct)
            cx = x1 + dw / 2.0
            cy = y1 + dh / 2.0
            rx0, ry0, rx1, ry1 = _square_crop_bounds(cx, cy, side, W, H)
            roi_xyxy = (rx0, ry0, rx1, ry1)
        else:
            roi_xyxy = (0, 0, W, H)

        roi = im_bgr[roi_xyxy[1]:roi_xyxy[3], roi_xyxy[0]:roi_xyxy[2]].copy()

        # Stage B: cup on ROI
        resB = _predict_one(self.modelB, roi, self.cfg.confB, self.cfg.iouB, self.cfg.device)
        best_cup = _best_box_from_results(resB, self.cfg.cup_class_id_stageB, self.cfg.confB)

        cup_box_global: Optional[Tuple[int,int,int,int]] = None
        cup_conf: Optional[float] = None
        if best_cup is not None:
            (cx1, cy1, cx2, cy2), cup_conf = best_cup
            cup_box_global = (roi_xyxy[0] + cx1, roi_xyxy[1] + cy1,
                              roi_xyxy[0] + cx2, roi_xyxy[1] + cy2)

        disc_bb = BoundingBox(*map(float, disc_box_xyxy)) if disc_box_xyxy is not None else None
        cup_bb  = BoundingBox(*map(float, cup_box_global)) if cup_box_global is not None else None
        return disc_bb, cup_bb, roi_xyxy if disc_box_xyxy is not None else None, disc_conf, cup_conf


# ----------------------------- label writer (optional) -----------------------------

class LabelWriter:
    """
    Writes YOLO-format labels for predicted boxes via Image.yolo_lines_2class(use_gt=False).
    """
    def __init__(self, out_root: Path, images_root: Path, overwrite: bool) -> None:
        self.out_root = out_root
        self.images_root = images_root
        self.overwrite = overwrite

    def write(self, img: Image, require_both: bool = False) -> Optional[Path]:
        lines = list(img.yolo_lines_2class(use_gt=False))
        if require_both:
            classes_present = {int(float(ln.split()[0])) for ln in lines if ln.strip()}
            if not ({0, 1} <= classes_present):
                return None
        outp = _mirror_path(self.images_root, self.out_root, img.image_path, ".txt")
        if outp.exists() and not self.overwrite:
            return outp
        _ensure_parent(outp)
        with outp.open("w") as f:
            for ln in lines:
                f.write(ln + "\n")
        return outp


# ----------------------------- visualizer -----------------------------

class Visualizer:
    def __init__(self, cfg: VizConfig) -> None:
        self.cfg = cfg
        self.predictor = TwoStagePredictor(cfg)
        self.label_writer = LabelWriter(cfg.out_labels, cfg.images_root, cfg.overwrite) if cfg.out_labels else None

    def _collect_images(self) -> List[Image]:
        inc = list(self.cfg.include_name_contains) if self.cfg.include_name_contains else None
        exc = list(self.cfg.exclude_name_contains) if self.cfg.exclude_name_contains else None
        factory = ImageFactory(
            dataset_name=self.cfg.images_root.name or "dataset",
            images_root=self.cfg.images_root,
            disc_masks_root=None,
            cup_masks_root=None,
            include_name_contains=inc,
            exclude_name_contains=exc,
            recursive=self.cfg.recursive,
        )
        items = factory.collect()
        # subset selection
        if self.cfg.num_samples and self.cfg.num_samples > 0 and len(items) > self.cfg.num_samples:
            if self.cfg.seed is not None:
                random.seed(self.cfg.seed)
            items = random.sample(items, self.cfg.num_samples)
        return items

    def _annotate(self, img: Image, disc_bb: Optional[BoundingBox], cup_bb: Optional[BoundingBox],
                  roi_xyxy: Optional[Tuple[int,int,int,int]], disc_conf: Optional[float], cup_conf: Optional[float]) -> np.ndarray:
        canvas = _load_image_bgr(img.image_path).copy()

        # ROI
        if self.cfg.draw_roi and roi_xyxy is not None:
            _draw_box(canvas, roi_xyxy, (0, 200, 0), "ROI", thickness=1)  # green

        # Disc & Cup
        if disc_bb is not None:
            _draw_box(canvas, (int(disc_bb.x1), int(disc_bb.y1), int(disc_bb.x2), int(disc_bb.y2)),
                      (0, 0, 255), f"disc {disc_conf:.2f}" if disc_conf is not None else "disc", thickness=2)  # red

        if cup_bb is not None:
            _draw_box(canvas, (int(cup_bb.x1), int(cup_bb.y1), int(cup_bb.x2), int(cup_bb.y2)),
                      (255, 0, 0), f"cup {cup_conf:.2f}" if cup_conf is not None else "cup", thickness=2)  # blue

        # CDR text (vertical)
        if disc_bb is not None and cup_bb is not None:
            d_h = _box_h((int(disc_bb.x1), int(disc_bb.y1), int(disc_bb.x2), int(disc_bb.y2)))
            c_h = _box_h((int(cup_bb.x1), int(cup_bb.y1), int(cup_bb.x2), int(cup_bb.y2)))
            if d_h > 0:
                cdr = c_h / float(d_h)
                cv2.putText(canvas, f"CDR_v: {cdr:.3f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2, cv2.LINE_AA)
        return canvas

    def run(self) -> None:
        items = self._collect_images()
        n = len(items)
        n_saved_viz = 0
        n_saved_lbl = 0

        for img in items:
            try:
                disc_bb, cup_bb, roi_xyxy, disc_conf, cup_conf = self.predictor.predict_boxes(img)

                # attach to Image (so yolo_lines_2class(use_gt=False) works)
                if disc_bb is not None:
                    img.set_box(Structure.DISC, LabelType.PRED, disc_bb)
                if cup_bb is not None:
                    img.set_box(Structure.CUP, LabelType.PRED, cup_bb)

                # write labels if requested
                if self.label_writer is not None:
                    out_lbl = self.label_writer.write(img, require_both=self.cfg.require_both)
                    if out_lbl is not None:
                        n_saved_lbl += 1

                # draw and save visualization
                overlay = self._annotate(img, disc_bb, cup_bb, roi_xyxy, disc_conf, cup_conf)
                out_png = _mirror_path(self.cfg.images_root, self.cfg.out_viz, img.image_path, ".png")
                _ensure_parent(out_png)
                cv2.imwrite(str(out_png), overlay)
                n_saved_viz += 1

            except Exception as e:
                print(f"[WARN] Skipping {img.image_path}: {e}")

        # summary
        print("---- Visualization Summary ----")
        print(f"Images selected    : {n}")
        print(f"Annotated saved    : {n_saved_viz}")
        if self.label_writer is not None:
            print(f"Label files saved  : {n_saved_lbl}")
            print(f"Labels root        : {self.cfg.out_labels}")
        print(f"Viz root           : {self.cfg.out_viz}")
        print("--------------------------------")


# ----------------------------- CLI -----------------------------

def _parse_args() -> VizConfig:
    ap = argparse.ArgumentParser(description="Visualize two-stage disc→cup ROI predictions over a dataset.")
    ap.add_argument("--images-root", required=True, help="Root directory of fundus images.")
    ap.add_argument("--out-viz", required=True, help="Output root for annotated images (.png).")
    ap.add_argument("--stageA", required=True, help="Path to Stage A (disc-only) weights, e.g., best.pt")
    ap.add_argument("--stageB", required=True, help="Path to Stage B (cup-ROI) weights, e.g., best.pt")

    # Optional YOLO labels
    ap.add_argument("--out-labels", default="", help="Output root for label .txt (optional).")

    # Filtering
    ap.add_argument("--include", nargs="*", default=["PAPILA"], help="Case-insensitive substrings to include.")
    ap.add_argument("--exclude", nargs="*", default=[], help="Case-insensitive substrings to exclude.")
    ap.add_argument("--recursive", action="store_true", help="Recurse subfolders.")

    # Thresholds & ROI
    ap.add_argument("--confA", type=float, default=0.25, help="Disc confidence threshold.")
    ap.add_argument("--iouA",  type=float, default=0.50, help="Disc NMS IoU threshold.")
    ap.add_argument("--confB", type=float, default=0.10, help="Cup confidence threshold.")
    ap.add_argument("--iouB",  type=float, default=0.50, help="Cup NMS IoU threshold.")
    ap.add_argument("--od-pad-pct", type=float, default=0.08, help="Disc ROI padding fraction per side.")
    ap.add_argument("--draw-roi", action="store_true", help="Draw ROI rectangle on the visualization.")

    # Classes & device
    ap.add_argument("--disc-class-id-stageA", type=int, default=0, help="Disc class id in Stage A model.")
    ap.add_argument("--cup-class-id-stageB", type=int, default=-1,
                    help="Cup class id in Stage B model; set -1 for single-class model.")
    ap.add_argument("--device", default=None, help="Ultralytics device string, e.g., '0', 'cpu', or 'mps'.")

    # Behavior
    ap.add_argument("--require-both", action="store_true", help="For labels: write only if both disc and cup detected.")
    ap.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing outputs.")

    # Subset control
    ap.add_argument("--num-samples", type=int, default=0, help="Random subset size; 0 = visualize all images.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")

    args = ap.parse_args()

    images_root = _expand(args.images_root)
    out_viz     = _expand(args.out_viz)
    stageA      = _expand(args.stageA)
    stageB      = _expand(args.stageB)
    out_labels  = _expand(args.out_labels) if args.out_labels else None

    if not images_root.exists():
        raise SystemExit(f"[ERR] images root not found: {images_root}")
    if not stageA.exists():
        raise SystemExit(f"[ERR] Stage A weights not found: {stageA}")
    if not stageB.exists():
        raise SystemExit(f"[ERR] Stage B weights not found: {stageB}")

    cup_id = None if int(args.cup_class_id_stageB) < 0 else int(args.cup_class_id_stageB)

    return VizConfig(
        images_root=images_root,
        out_viz=out_viz,
        stageA_weights=stageA,
        stageB_weights=stageB,
        out_labels=out_labels,
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
        draw_roi=bool(args.draw_roi),
        require_both=bool(args.require_both),
        overwrite=not bool(args.no_overwrite),
        num_samples=int(args.num_samples),
        seed=None if args.seed is None else int(args.seed),
    )


def main() -> None:
    cfg = _parse_args()
    viz = Visualizer(cfg)
    viz.run()


if __name__ == "__main__":
    main()