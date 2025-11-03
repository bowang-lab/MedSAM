#!/usr/bin/env python3
# src.modelpredict_bounding_box.py

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict

import cv2
import numpy as np
from ultralytics import YOLO

# --- Your package imports ---
from src.imgpipe.config import PipelineConfig
from src.imgpipe.collector import DatasetCollector
from src.utils import ensure_dir

# OOP classes used by the new API
from src.imgpipe.image import Image as ImageRec
from src.imgpipe.normalized_box import BoundingBox
from src.imgpipe.enums import LabelType, Structure


# ========================== Datatypes ==========================

@dataclass(frozen=True)
class Prediction:
    """Container for a single best box prediction."""
    image_bgr: np.ndarray
    width: int
    height: int
    xyxy: Tuple[int, int, int, int]           # (x1,y1,x2,y2)
    yolo: Tuple[float, float, float, float]   # (xc,yc,w,h) normalized
    conf: float
    class_id: int

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("image_bgr", None)  # don't dump image bytes
        return d

    def __str__(self) -> str:
        x1, y1, x2, y2 = self.xyxy
        xc, yc, w, h = self.yolo
        return (f"Prediction(xyxy=({x1},{y1},{x2},{y2}), "
                f"yolo=({xc:.4f},{yc:.4f},{w:.4f},{h:.4f}), "
                f"conf={self.conf:.3f}, class_id={self.class_id}, "
                f"size=({self.width}x{self.height}))")


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

def _best_box_from_result(result, class_id: Optional[int], conf_thres: float
                          ) -> Optional[Tuple[Tuple[int, int, int, int], float, int]]:
    """
    From one Ultralytics result, return (xyxy_int, conf, cls_id) for the best box
    filtered by optional class_id and conf threshold.
    """
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    cls  = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros_like(conf, dtype=np.int32)

    idxs = np.arange(len(conf))
    if class_id is not None and class_id >= 0:
        idxs = idxs[cls.astype(int) == int(class_id)]
    idxs = [i for i in idxs if conf[i] >= conf_thres]
    if not idxs:
        return None
    i_best = max(idxs, key=lambda i: conf[i])
    x1, y1, x2, y2 = map(float, xyxy[i_best])
    return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))), float(conf[i_best]), int(cls[i_best])

def _draw_annotated(image_bgr: np.ndarray, pred: Prediction, label: Optional[str] = None) -> np.ndarray:
    x1, y1, x2, y2 = pred.xyxy
    out = image_bgr.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    text = label if label is not None else f"id={pred.class_id} conf={pred.conf:.2f}"
    cv2.putText(out, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    return out

def _write_yolo_label(path: Path, class_id: int, yolo_box: Tuple[float, float, float, float]) -> None:
    ensure_dir(path.parent)
    xc, yc, w, h = yolo_box
    path.write_text(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# ========================== NEW OOP API ==========================

class BoundingBoxPredictor:
    """
    Loads YOLO once; exposes per-image prediction that returns your OOP BoundingBox.
    """

    def __init__(self, *, weights: PathLike, conf: float = 0.25, iou: float = 0.50,
                 device: Optional[str] = None, class_id: Optional[int] = None) -> None:
        self.weights = _expand(weights)
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.class_id = class_id if (class_id is not None and class_id >= 0) else None
        self.model = YOLO(str(self.weights))

    def predict_ndarray(self, img_bgr: np.ndarray) -> Optional[Prediction]:
        H, W = img_bgr.shape[:2]
        res = self.model.predict(source=img_bgr, conf=self.conf, iou=self.iou,
                                 device=self.device, verbose=False)
        if not res:
            return None
        best = _best_box_from_result(res[0], self.class_id, self.conf)
        if best is None:
            return None
        (x1, y1, x2, y2), score, cls = best
        yolo_box = _xyxy_to_yolo(x1, y1, x2, y2, W, H)
        return Prediction(img_bgr, W, H, (x1, y1, x2, y2), yolo_box, score, int(cls))

    def predict_path(self, path: PathLike) -> Optional[Prediction]:
        return self.predict_ndarray(_load_image_bgr(path))

    def predict_one_image_to_box(self, img: ImageRec) -> Optional[BoundingBox]:
        """
        Accepts your OOP Image; returns a BoundingBox or None.
        """
        pred = self.predict_path(img.image_path)
        if pred is None:
            return None
        x1, y1, x2, y2 = pred.xyxy
        return BoundingBox(float(x1), float(y1), float(x2), float(y2))


class LabelWriter:
    """
    Mirrors the images tree under out_root and writes YOLO labels using Image.yolo_lines_2class(use_gt=False).
    """

    def __init__(self, out_root: PathLike, images_root: PathLike, overwrite: bool = True) -> None:
        self.out_root = _expand(out_root)
        self.images_root = _expand(images_root)
        self.overwrite = bool(overwrite)

    def _label_path_for(self, img: ImageRec) -> Path:
        rel = img.image_path.resolve().relative_to(self.images_root.resolve())
        return (self.out_root / rel).with_suffix(".txt")

    def write(self, img: ImageRec, require_both: bool = False) -> Optional[Path]:
        lines = list(img.yolo_lines_2class(use_gt=False))
        if require_both:
            classes_present = {int(float(ln.split()[0])) for ln in lines if ln.strip()}
            if not ({0, 1} <= classes_present):
                return None
        outp = self._label_path_for(img)
        if outp.exists() and not self.overwrite:
            return outp
        ensure_dir(outp.parent)
        with outp.open("w") as f:
            for ln in lines:
                f.write(ln + "\n")
        return outp


# ========================== Legacy Public API ==========================

def predict_bounding_box(
    image: ImageLike,
    weights: PathLike,
    *,
    class_id: Optional[int] = None,
    conf: float = 0.25,
    iou: float = 0.50,
    device: Optional[str] = None,
) -> Optional[Prediction]:
    """
    Single-image prediction. Returns the best box or None.
    """
    img_bgr = _load_image_bgr(image)
    H, W = img_bgr.shape[:2]
    model = YOLO(str(_expand(weights)))
    res = model.predict(source=img_bgr, conf=conf, iou=iou, device=device, verbose=False)
    if not res:
        return None

    best = _best_box_from_result(res[0], class_id=None if (class_id is None or class_id < 0) else int(class_id),
                                 conf_thres=conf)
    if best is None:
        return None
    (x1, y1, x2, y2), score, cls = best
    yolo_box = _xyxy_to_yolo(x1, y1, x2, y2, W, H)
    return Prediction(
        image_bgr=img_bgr,
        width=W,
        height=H,
        xyxy=(x1, y1, x2, y2),
        yolo=yolo_box,
        conf=score,
        class_id=int(cls),
    )


def run_on_config(
    *,
    config: PathLike,
    weights: PathLike,
    class_id: int | None = -1,
    conf: float = 0.25,
    iou: float = 0.50,
    device: Optional[str] = None,
    # optional overrides for quick toy runs
    subset_n: Optional[int] = None,
    subset_seed: int = 43,
    # optional outputs
    save_labels: Optional[PathLike] = None,      # root to mirror .txt labels
    save_annot: Optional[PathLike] = None,       # root to mirror annotated images
    write_empty: bool = False,                   # if True, write empty .txt when no detection
) -> Dict[str, int]:
    """
    Use DatasetCollector with a PipelineConfig to run predictions over a dataset (or subset).

    Returns: summary dict with counts.
    """
    cfg = PipelineConfig.load(_expand(config))

    # quick subset override (patient-wise via DatasetCollector.subset_if_enabled)
    if subset_n is not None and subset_n > 0:
        cfg.subset_n = int(subset_n)
        cfg.subset_seed = int(subset_seed)
    else:
        cfg.subset_n = int(getattr(cfg, "subset_n", 0) or 0)

    collector = DatasetCollector(cfg)
    ds_full = collector.collect()
    ds, _outs = collector.subset_if_enabled(ds_full)  # may be pass-through if subset_n<=0

    model = YOLO(str(_expand(weights)))
    save_labels_path = _expand(save_labels) if save_labels else None
    save_annot_path  = _expand(save_annot) if save_annot else None

    n_total = len(ds.images)
    n_pred  = 0
    n_saved_lbl = 0
    n_saved_img = 0
    n_empty_lbl = 0

    for im in ds.images:
        img_bgr = cv2.imread(str(im.image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] cannot read {im.image_path}")
            continue

        H, W = img_bgr.shape[:2]
        res = model.predict(source=img_bgr, conf=conf, iou=iou, device=device, verbose=False)
        pred_obj: Optional[Prediction] = None
        if res:
            best = _best_box_from_result(res[0], class_id=None if (class_id is None or class_id < 0) else int(class_id),
                                         conf_thres=conf)
            if best is not None:
                (x1, y1, x2, y2), score, cls = best
                yolo_box = _xyxy_to_yolo(x1, y1, x2, y2, W, H)
                pred_obj = Prediction(img_bgr, W, H, (x1, y1, x2, y2), yolo_box, score, int(cls))
                n_pred += 1

                # Attach to Image (disc as class 0 by convention) so downstream code can reuse it.
                im.set_box(Structure.DISC, LabelType.PRED,
                           BoundingBox(float(x1), float(y1), float(x2), float(y2)))

        # write YOLO label (optional) using your OOP writer when possible
        if save_labels_path is not None:
            if pred_obj is not None:
                # Mirror input tree under save_labels
                try:
                    # Try to infer a root called 'images' to preserve deeper structure; else use filename only
                    parts = list(im.image_path.parts)
                    rel = Path(im.image_path.name)
                    if "images" in parts:
                        idx = parts.index("images")
                        rel = Path(*parts[idx+1:])
                    out_lbl = save_labels_path / rel.with_suffix(".txt")
                except Exception:
                    out_lbl = save_labels_path / (im.image_path.stem + ".txt")

                _write_yolo_label(out_lbl, pred_obj.class_id, pred_obj.yolo)
                n_saved_lbl += 1
            else:
                if write_empty:
                    try:
                        parts = list(im.image_path.parts)
                        rel = Path(im.image_path.name)
                        if "images" in parts:
                            idx = parts.index("images")
                            rel = Path(*parts[idx+1:])
                        out_lbl = save_labels_path / rel.with_suffix(".txt")
                    except Exception:
                        out_lbl = save_labels_path / (im.image_path.stem + ".txt")
                    ensure_dir(out_lbl.parent)
                    out_lbl.write_text("")  # explicit negative
                    n_empty_lbl += 1

        # write annotated image (optional)
        if save_annot_path is not None and pred_obj is not None:
            try:
                parts = list(im.image_path.parts)
                rel = Path(im.image_path.name)
                if "images" in parts:
                    idx = parts.index("images")
                    rel = Path(*parts[idx+1:])
                out_img = (save_annot_path / rel).with_suffix(".jpg")
                ensure_dir(out_img.parent)
                cv2.imwrite(str(out_img), _draw_annotated(img_bgr, pred_obj))
                n_saved_img += 1
            except Exception as e:
                print(f"[WARN] failed to save annotation for {im.image_path}: {e}")

    summary = dict(
        total=n_total,
        predicted=n_pred,
        labels_written=n_saved_lbl,
        empty_labels_written=n_empty_lbl,
        annotated_written=n_saved_img,
        subset_n=int(getattr(cfg, "subset_n", 0) or 0),
    )
    print("[SUMMARY]", summary)
    return summary


# ========================== CLI ==========================

def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Single-stage YOLO bounding-box prediction.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # single image
    one = sub.add_parser("one", help="Predict one image")
    one.add_argument("--image", required=True)
    one.add_argument("--weights", required=True)
    one.add_argument("--class-id", type=int, default=-1)
    one.add_argument("--conf", type=float, default=0.25)
    one.add_argument("--iou", type=float, default=0.50)
    one.add_argument("--device", default=None)
    one.add_argument("--out", default="", help="Optional annotated image path")

    # dataset via config + collector
    ds = sub.add_parser("dataset", help="Predict an entire dataset via PipelineConfig + DatasetCollector")
    ds.add_argument("--config", required=True, help="pipeline YAML")
    ds.add_argument("--weights", required=True, help="YOLO .pt weights")
    ds.add_argument("--class-id", type=int, default=-1)
    ds.add_argument("--conf", type=float, default=0.25)
    ds.add_argument("--iou", type=float, default=0.50)
    ds.add_argument("--device", default=None)
    ds.add_argument("--subset-n", type=int, default=0, help="patient-wise subset size (0 = all)")
    ds.add_argument("--subset-seed", type=int, default=43)
    ds.add_argument("--save-labels", default="", help="Optional: root to write YOLO labels")
    ds.add_argument("--save-annot", default="", help="Optional: root to write annotated images")
    ds.add_argument("--write-empty", action="store_true", help="Write empty .txt when no detection")

    return ap.parse_args()


def main() -> None:
    args = _cli()
    if args.cmd == "one":
        pred = predict_bounding_box(
            image=args.image,
            weights=args.weights,
            class_id=(None if args.class_id < 0 else int(args.class_id)),
            conf=float(args.conf),
            iou=float(args.iou),
            device=args.device,
        )
        if pred is None:
            print("[INFO] No detection above threshold.")
            return
        print(pred)
        if args.out:
            outp = _expand(args.out)
            ensure_dir(outp.parent)
            cv2.imwrite(str(outp), _draw_annotated(pred.image_bgr, pred))
            print(f"[OK] Saved: {outp}")
        return

    if args.cmd == "dataset":
        run_on_config(
            config=args.config,
            weights=args.weights,
            class_id=(None if args.class_id < 0 else int(args.class_id)),
            conf=float(args.conf),
            iou=float(args.iou),
            device=args.device,
            subset_n=(None if int(args.subset_n) <= 0 else int(args.subset_n)),
            subset_seed=int(args.subset_seed),
            save_labels=(None if not args.save_labels else args.save_labels),
            save_annot=(None if not args.save_annot else args.save_annot),
            write_empty=bool(args.write_empty),
        )
        return


if __name__ == "__main__":
    main()