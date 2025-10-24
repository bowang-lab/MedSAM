#!/usr/bin/env python3
# src/model/predict_bounding_box.py
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Iterable

import cv2
import numpy as np
from ultralytics import YOLO

# --- Your package imports ---
from src.imgpipe.config import PipelineConfig
from src.imgpipe.collector import DatasetCollector
from src.utils import ensure_dir

# OOP classes used by the new API
from src.imgpipe.image import Image as ImageRec
from src.imgpipe.bounding_box import BoundingBox
from src.imgpipe.enums import LabelType, Structure


# ========================== Datatypes ==========================

@dataclass(frozen=True)
class PredictionBox:
    """Container for a single detection box."""
    xyxy: Tuple[int, int, int, int]           # (x1,y1,x2,y2) in pixels
    yolo: Tuple[float, float, float, float]   # (xc,yc,w,h) normalized
    conf: float
    class_id: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Predictions:
    """All detections for one image."""
    width: int
    height: int
    boxes: List[PredictionBox]

    def best_per_class(self) -> Dict[int, PredictionBox]:
        """Return the top-confidence box per class."""
        best: Dict[int, PredictionBox] = {}
        for b in self.boxes:
            if b.class_id not in best or b.conf > best[b.class_id].conf:
                best[b.class_id] = b
        return best

    def filter_classes(self, keep: Optional[Iterable[int]]) -> "Predictions":
        if keep is None:
            return self
        kset = set(int(c) for c in keep)
        return Predictions(self.width, self.height, [b for b in self.boxes if b.class_id in kset])

    def __len__(self) -> int:
        return len(self.boxes)


# ========================== Small utils ==========================

PathLike = Union[str, Path]
ImageLike = Union[PathLike, np.ndarray]

def _expand(p: PathLike) -> Path:
    return Path(p).expanduser().resolve()

def _load_image_bgr(img: ImageLike) -> np.ndarray:
    if isinstance(img, (str, Path)):
        im = cv2.imread(str(_expand(img)), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Failed to read image: {img}")
        return im
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 3:
            return img
        raise ValueError("np.ndarray image must be HxWx3 or HxW.")
    raise TypeError("image must be a path or a numpy.ndarray")

def _xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[float, float, float, float]:
    bw = max(0, x2 - x1); bh = max(0, y2 - y1)
    cx = x1 + bw / 2.0; cy = y1 + bh / 2.0
    return (cx / W, cy / H, bw / W, bh / H)

def _all_boxes_from_result(result, conf_thres: float, class_filter: Optional[Iterable[int]]
                           ) -> List[Tuple[Tuple[int, int, int, int], float, int]]:
    """
    From one Ultralytics result, return a list of (xyxy_int, conf, cls_id) for *all* boxes
    filtered by confidence and (optionally) by class ids.
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    cls  = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros_like(conf, dtype=np.int32)

    idxs = [i for i in range(len(conf)) if conf[i] >= conf_thres]
    if class_filter is not None:
        keep = set(int(c) for c in class_filter)
        idxs = [i for i in idxs if int(cls[i]) in keep]

    out = []
    for i in idxs:
        x1, y1, x2, y2 = map(float, xyxy[i])
        out.append(((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                    float(conf[i]), int(cls[i])))
    return out

def _draw_annotated_multi(image_bgr: np.ndarray, preds: Predictions,
                          class_names: Optional[Dict[int, str]] = None) -> np.ndarray:
    """
    Draw all predictions (per-class colors). Defaults: 0=disc (cyan), 1=cup (magenta), others=red.
    """
    color_map = {
        0: (255, 255, 0),   # cyan-ish in BGR
        1: (255, 0, 255),   # magenta
    }
    out = image_bgr.copy()
    for b in preds.boxes:
        x1, y1, x2, y2 = b.xyxy
        color = color_map.get(b.class_id, (0, 0, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        name = class_names.get(b.class_id, str(b.class_id)) if class_names else str(b.class_id)
        cv2.putText(out, f"{name} {b.conf:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out

def _write_yolo_label(path: Path, rows: List[Tuple[int, Tuple[float, float, float, float]]]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        for cls_id, (xc, yc, w, h) in rows:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# ========================== Multiclass predictor ==========================

class MultiClassPredictor:
    """
    Loads YOLO once; exposes per-image inference that returns all boxes.
    Optionally filter by classes (e.g., [0,1]) and confidence threshold.
    """

    def __init__(self, *,
                 weights: PathLike,
                 conf: float = 0.25,
                 iou: float = 0.50,
                 device: Optional[str] = None,
                 classes: Optional[Iterable[int]] = None) -> None:
        self.weights = _expand(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.classes = set(int(c) for c in classes) if classes is not None else None
        self.model = YOLO(str(self.weights))

    def predict_ndarray(self, img_bgr: np.ndarray) -> Predictions:
        H, W = img_bgr.shape[:2]
        res = self.model.predict(source=img_bgr, conf=self.conf, iou=self.iou,
                                 device=self.device, verbose=False)
        boxes: List[PredictionBox] = []
        if res:
            for (xyxy, score, cls_id) in _all_boxes_from_result(res[0], self.conf, self.classes):
                yolo_box = _xyxy_to_yolo(*xyxy, W, H)
                boxes.append(PredictionBox(xyxy, yolo_box, score, int(cls_id)))
        return Predictions(W, H, boxes)

    def predict_path(self, path: PathLike) -> Predictions:
        return self.predict_ndarray(_load_image_bgr(path))

    # ----- OOP convenience: attach best disc/cup boxes to ImageRec -----

    def predict_one_image_to_imageobj(self, img: ImageRec) -> Predictions:
        """
        Runs inference, attaches best disc (cls=0) and cup (cls=1) boxes to `img`
        as inter_pred_disc_box / inter_pred_cup_box, and returns all predictions.
        """
        preds = self.predict_path(img.image_path)
        best = preds.best_per_class()
        # Map YOLO cls → Structure
        for cls_id, box in best.items():
            bb = BoundingBox(float(box.xyxy[0]), float(box.xyxy[1]),
                             float(box.xyxy[2]), float(box.xyxy[3]))
            if cls_id == 0:
                img.set_box(Structure.DISC, LabelType.PRED, bb)
            elif cls_id == 1:
                img.set_box(Structure.CUP, LabelType.PRED, bb)
            else:
                # ignore other classes for ImageRec (your Image schema is 2-class)
                pass
        return preds


# ========================== Dataset runner ==========================

def run_on_config(
    *,
    config: PathLike,
    weights: PathLike,
    classes: Optional[Iterable[int]] = None,     # None = all classes
    conf: float = 0.25,
    iou: float = 0.50,
    device: Optional[str] = None,
    # optional overrides for quick toy runs
    subset_n: Optional[int] = None,
    subset_seed: int = 43,
    # optional outputs
    save_labels: Optional[PathLike] = None,      # root to mirror .txt labels (preds)
    save_annot: Optional[PathLike] = None,       # root to mirror annotated images
    write_empty: bool = False,                   # write empty .txt when no detections
    require_both: bool = False,                  # only write label if both disc & cup present
    class_names: Optional[Dict[int, str]] = None # for pretty annotations
) -> Dict[str, int]:
    """
    Use DatasetCollector with a PipelineConfig to run predictions over a dataset (or subset).
    For each image:
      - run multiclass inference
      - attach best disc/cup boxes to ImageRec (for downstream code)
      - optionally write YOLO label(s) and annotated images

    Returns: summary dict with counts.
    """
    cfg = PipelineConfig.load(_expand(config))

    # subset override (patient-wise via DatasetCollector)
    if subset_n is not None and subset_n > 0:
        cfg.subset_n = int(subset_n)
        cfg.subset_seed = int(subset_seed)
    else:
        cfg.subset_n = int(getattr(cfg, "subset_n", 0) or 0)

    collector = DatasetCollector(cfg)
    ds_full = collector.collect()
    ds, _outs = collector.subset_if_enabled(ds_full)  # may be pass-through

    predictor = MultiClassPredictor(weights=weights, conf=conf, iou=iou, device=device, classes=classes)
    save_labels_path = _expand(save_labels) if save_labels else None
    save_annot_path  = _expand(save_annot) if save_annot else None

    n_total = len(ds.images)
    n_pred_any  = 0
    n_pred_disc = 0
    n_pred_cup  = 0
    n_saved_lbl = 0
    n_saved_img = 0
    n_empty_lbl = 0

    for im in ds.images:
        img_path = im.image_path
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] cannot read {img_path}")
            continue

        preds = predictor.predict_one_image_to_imageobj(im)
        best = preds.best_per_class()
        has_any = len(preds) > 0
        if has_any:
            n_pred_any += 1
        if 0 in best:
            n_pred_disc += 1
        if 1 in best:
            n_pred_cup += 1

        # ---------- write YOLO labels (optional) ----------
        if save_labels_path is not None:
            rows: List[Tuple[int, Tuple[float, float, float, float]]] = []
            # choose the best box per class (for a single-object-per-class label)
            for cls_id, box in best.items():
                rows.append((cls_id, box.yolo))

            # optional guard: require both disc & cup present
            if require_both and not ({0, 1} <= set(best.keys())):
                if write_empty:
                    # still emit an explicit empty file
                    out_lbl = _mirror_rel_under(img_path, save_labels_path, new_ext=".txt")
                    ensure_dir(out_lbl.parent); out_lbl.write_text("")
                    n_empty_lbl += 1
                # skip further writing
            else:
                out_lbl = _mirror_rel_under(img_path, save_labels_path, new_ext=".txt")
                _write_yolo_label(out_lbl, rows)
                n_saved_lbl += 1

            if not rows and write_empty:
                out_lbl = _mirror_rel_under(img_path, save_labels_path, new_ext=".txt")
                ensure_dir(out_lbl.parent); out_lbl.write_text("")
                n_empty_lbl += 1

        # ---------- write annotated image (optional) ----------
        if save_annot_path is not None and has_any:
            out_img = _mirror_rel_under(img_path, save_annot_path, new_ext=".jpg")
            ensure_dir(out_img.parent)
            anno = _draw_annotated_multi(img_bgr, preds, class_names=class_names)
            cv2.imwrite(str(out_img), anno)
            n_saved_img += 1

    summary = dict(
        total=n_total,
        predicted_any=n_pred_any,
        predicted_disc=n_pred_disc,
        predicted_cup=n_pred_cup,
        labels_written=n_saved_lbl,
        empty_labels_written=n_empty_lbl,
        annotated_written=n_saved_img,
        subset_n=int(getattr(cfg, "subset_n", 0) or 0),
    )
    print("[SUMMARY]", summary)
    return summary


# ========================== Single-image helper ==========================

def predict_bounding_boxes(
    image: ImageLike,
    weights: PathLike,
    *,
    classes: Optional[Iterable[int]] = None,  # None = all
    conf: float = 0.25,
    iou: float = 0.50,
    device: Optional[str] = None,
) -> Predictions:
    """
    Single-image prediction. Returns *all* boxes above threshold (optionally class-filtered).
    """
    img_bgr = _load_image_bgr(image)
    H, W = img_bgr.shape[:2]
    model = YOLO(str(_expand(weights)))
    res = model.predict(source=img_bgr, conf=conf, iou=iou, device=device, verbose=False)

    boxes: List[PredictionBox] = []
    if res:
        for (xyxy, score, cls_id) in _all_boxes_from_result(res[0], conf, classes):
            yolo_box = _xyxy_to_yolo(*xyxy, W, H)
            boxes.append(PredictionBox(xyxy, yolo_box, score, int(cls_id)))
    return Predictions(W, H, boxes)


# ========================== Helpers (paths/rel mirroring) ==========================

def _mirror_rel_under(src_path: Path, new_root: Path, new_ext: str) -> Path:
    """Mirror the path structure below the first 'images' component if present, else use filename only."""
    parts = list(src_path.parts)
    rel = Path(src_path.name)
    if "images" in parts:
        idx = parts.index("images")
        rel = Path(*parts[idx + 1:])
    out_path = (new_root / rel).with_suffix(new_ext)
    return out_path


# ========================== CLI ==========================

def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Multiclass YOLO bounding-box prediction.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # single image
    one = sub.add_parser("one", help="Predict one image")
    one.add_argument("--image", required=True)
    one.add_argument("--weights", required=True)
    one.add_argument("--classes", nargs="*", type=int, default=None, help="Limit to class ids (space separated)")
    one.add_argument("--conf", type=float, default=0.25)
    one.add_argument("--iou", type=float, default=0.50)
    one.add_argument("--device", default=None)
    one.add_argument("--out", default="", help="Optional annotated image path")
    one.add_argument("--names", nargs="*", type=str, default=None,
                     help='Optional class names in order, e.g. "--names disc cup"')

    # dataset via config + collector
    ds = sub.add_parser("dataset", help="Predict an entire dataset via PipelineConfig + DatasetCollector")
    ds.add_argument("--config", required=True, help="pipeline YAML")
    ds.add_argument("--weights", required=True, help="YOLO .pt weights")
    ds.add_argument("--classes", nargs="*", type=int, default=None, help="Limit to class ids (space separated)")
    ds.add_argument("--conf", type=float, default=0.25)
    ds.add_argument("--iou", type=float, default=0.50)
    ds.add_argument("--device", default=None)
    ds.add_argument("--subset-n", type=int, default=0, help="patient-wise subset size (0 = all)")
    ds.add_argument("--subset-seed", type=int, default=43)
    ds.add_argument("--save-labels", default="", help="Optional: root to write YOLO labels")
    ds.add_argument("--save-annot", default="", help="Optional: root to write annotated images")
    ds.add_argument("--write-empty", action="store_true", help="Write empty .txt when no detection")
    ds.add_argument("--require-both", action="store_true",
                    help="Only write label if both disc and cup are detected")
    ds.add_argument("--names", nargs="*", type=str, default=None,
                    help='Optional class names in order, e.g. "--names disc cup"')

    return ap.parse_args()


def main() -> None:
    args = _cli()

    if args.cmd == "one":
        preds = predict_bounding_boxes(
            image=args.image,
            weights=args.weights,
            classes=args.classes,
            conf=float(args.conf),
            iou=float(args.iou),
            device=args.device,
        )
        if len(preds) == 0:
            print("[INFO] No detections above threshold.")
            return

        # print all
        for b in preds.boxes:
            (x1, y1, x2, y2) = b.xyxy
            (xc, yc, w, h) = b.yolo
            print(f"cls={b.class_id} conf={b.conf:.3f} "
                  f"xyxy=({x1},{y1},{x2},{y2}) yolo=({xc:.4f},{yc:.4f},{w:.4f},{h:.4f})")

        if args.out:
            img_bgr = _load_image_bgr(args.image)
            names = {i: n for i, n in enumerate(args.names)} if args.names else None
            anno = _draw_annotated_multi(img_bgr, preds, class_names=names)
            outp = _expand(args.out)
            ensure_dir(outp.parent)
            cv2.imwrite(str(outp), anno)
            print(f"[OK] Saved: {outp}")
        return

    if args.cmd == "dataset":
        names = {i: n for i, n in enumerate(args.names)} if args.names else None
        run_on_config(
            config=args.config,
            weights=args.weights,
            classes=args.classes,
            conf=float(args.conf),
            iou=float(args.iou),
            device=args.device,
            subset_n=(None if int(args.subset_n) <= 0 else int(args.subset_n)),
            subset_seed=int(args.subset_seed),
            save_labels=(None if not args.save_labels else args.save_labels),
            save_annot=(None if not args.save_annot else args.save_annot),
            write_empty=bool(args.write_empty),
            require_both=bool(args.require_both),
            class_names=names,
        )
        return


if __name__ == "__main__":
    main()