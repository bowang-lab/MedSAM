# evaluate_performance.py
from __future__ import annotations

"""
Evaluate vertical cup-to-disc ratio (CDR) error between ground-truth (from masks)
and *intermediate* predictions (from YOLO bounding boxes).

Usage (module):
    python -m yourpkg.evaluate_performance \
        --images-root /path/to/images \
        --gt-disc-masks /path/to/gt_disc_masks \
        --gt-cup-masks  /path/to/gt_cup_masks  \
        --pred-labels   /path/to/yolo_pred_labels \
        --include PAPILA \
        --loss l1 \
        --out-csv results/papila_eval.csv

Design:
- ImageFactory supplies Image objects with GT boxes pre-populated (from masks).
- We attach intermediate predicted boxes by parsing YOLO label files (class 0=disc, 1=cup).
- Vertical CDR = (cup_box_height / disc_box_height). Optionally clipped to [0,1].
- Loss: L1 (default), L2, or Huber(delta).
- Clean extension points:
    * PredictionSource: "intermediate" (YOLO boxes) today; "medsam" in future.
    * load_predicted_masks_for_medsam(...) stub to be wired to your MedSAM outputs.

Notes:
- No assumptions about utils not shown in your codebase; YOLO parsing implemented here.
- Keep class/funcs small and testable.
"""

import argparse
import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# --- Package-local imports (same package as your existing code) ---
from src.imgpipe.image_factory import ImageFactory
from  src.imgpipe.image import Image
from  src.imgpipe.bounding_box import BoundingBox
from  src.imgpipe.enums import LabelType, Structure


# ----------------------- Small Utilities ----------------------- #

def _ensure_dir(p: Optional[Path]) -> None:
    if p is not None:
        p.parent.mkdir(parents=True, exist_ok=True)


def _isfinite(x: Optional[float]) -> bool:
    return x is not None and math.isfinite(x)


# ------------------------ Config & Types ----------------------- #

@dataclass(frozen=True)
class EvalConfig:
    images_root: Path
    gt_disc_masks: Optional[Path]
    gt_cup_masks: Optional[Path]

    # Predictions
    pred_labels_dir: Optional[Path] = None  # YOLO label files (.txt) for full image predictions

    # Filtering
    include_name_contains: Tuple[str, ...] = ("PAPILA",)
    exclude_name_contains: Tuple[str, ...] = ()

    # Compute options
    loss: str = "l1"                        # l1 | l2 | huber
    huber_delta: float = 0.05               # used if loss == huber
    clip_ratio_01: bool = True              # clip predicted/gt CDR into [0,1]

    # Output
    out_csv: Optional[Path] = None
    recursive: bool = True
    strict: bool = False                    # if missing GT or pred -> raise, else skip sample


@dataclass(frozen=True)
class EvalRow:
    uid: str
    dataset: str
    subject_id: str
    image_path: str
    gt_cdr_v: Optional[float]
    pred_cdr_v: Optional[float]
    abs_error: Optional[float]
    sq_error: Optional[float]


@dataclass(frozen=True)
class EvalSummary:
    n_total: int
    n_used: int
    n_missing_gt: int
    n_missing_pred: int
    mae: Optional[float]
    mse: Optional[float]
    rmse: Optional[float]
    huber: Optional[float]  # if requested
    pearson_r: Optional[float]


# ------------------------ Core Calculators --------------------- #

class CDRCalculator:
    """Compute vertical CDR from boxes."""
    @staticmethod
    def vertical_ratio_from_boxes(
        disc: Optional[BoundingBox],
        cup: Optional[BoundingBox],
        *,
        clip_01: bool = True,
        eps: float = 1e-9,
    ) -> Optional[float]:
        if disc is None or cup is None:
            return None
        dh = max(0.0, disc.y2 - disc.y1)
        ch = max(0.0, cup.y2 - cup.y1)
        if dh <= eps:
            return None
        r = ch / dh
        if clip_01:
            r = float(min(1.0, max(0.0, r)))
        return float(r)


class LossComputer:
    """Pointwise losses + reducers."""
    @staticmethod
    def pointwise(loss: str, y_true: float, y_pred: float, huber_delta: float = 0.05) -> float:
        diff = y_pred - y_true
        if loss == "l1":
            return abs(diff)
        if loss == "l2":
            return diff * diff
        if loss == "huber":
            a = abs(diff)
            if a <= huber_delta:
                return 0.5 * diff * diff
            return huber_delta * (a - 0.5 * huber_delta)
        raise ValueError(f"Unknown loss '{loss}'")

    @staticmethod
    def reduce_mae(vals: List[float]) -> Optional[float]:
        return float(np.mean(vals)) if vals else None

    @staticmethod
    def reduce_mse(vals: List[float]) -> Optional[float]:
        return float(np.mean([v * v for v in vals])) if vals else None

    @staticmethod
    def reduce_custom(vals: List[float]) -> Optional[float]:
        return float(np.mean(vals)) if vals else None

    @staticmethod
    def pearson_r(y: List[float], p: List[float]) -> Optional[float]:
        if len(y) < 2 or len(y) != len(p):
            return None
        yv = np.asarray(y, dtype=float)
        pv = np.asarray(p, dtype=float)
        if np.any(~np.isfinite(yv)) or np.any(~np.isfinite(pv)):
            return None
        y_mean = yv.mean()
        p_mean = pv.mean()
        num = np.sum((yv - y_mean) * (pv - p_mean))
        den = math.sqrt(np.sum((yv - y_mean) ** 2) * np.sum((pv - p_mean) ** 2))
        if den <= 0:
            return None
        return float(num / den)


# ------------------------ IO: YOLO Labels ---------------------- #

class YoloLabelReader:
    """
    Minimal YOLO .txt reader. Assumes normalized xywh lines:
        <class_id> <x_center> <y_center> <width> <height> [score... ignored]
    We take the FIRST instance of each required class.
    """
    def __init__(self, labels_dir: Path) -> None:
        self.labels_dir = labels_dir

    def read_first_xywhn(self, stem: str, class_id: int) -> Optional[Tuple[float, float, float, float]]:
        lp = self.labels_dir / f"{stem}.txt"
        if not lp.exists():
            return None
        try:
            for ln in lp.read_text().splitlines():
                parts = ln.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(float(parts[0]))
                except Exception:
                    continue
                if cid == class_id:
                    x, y, w, h = map(float, parts[1:5])
                    return x, y, w, h
        except Exception:
            return None
        return None

    def to_box(
        self, xywhn: Tuple[float, float, float, float], W: int, H: int
    ) -> BoundingBox:
        x, y, w, h = xywhn
        return BoundingBox.from_yolo_norm(x, y, w, h, img_w=W, img_h=H)


# ------------------------ Evaluator ---------------------------- #

class Evaluator:
    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg

    # ---- Data loading ----
    def _load_images(self) -> List[Image]:
        inc = list(self.cfg.include_name_contains) if self.cfg.include_name_contains else None
        exc = list(self.cfg.exclude_name_contains) if self.cfg.exclude_name_contains else None

        factory = ImageFactory(
            dataset_name=self._infer_dataset_name(self.cfg.images_root),
            images_root=self.cfg.images_root,
            disc_masks_root=self.cfg.gt_disc_masks,
            cup_masks_root=self.cfg.gt_cup_masks,
            include_name_contains=inc,
            exclude_name_contains=exc,
            recursive=self.cfg.recursive,
        )
        items = factory.collect()
        return items

    @staticmethod
    def _infer_dataset_name(images_root: Path) -> str:
        # Simple fallback: folder name as dataset id
        return images_root.name or "dataset"

    # ---- Attaching predictions ----
    def _attach_intermediate_boxes_from_yolo(self, items: List[Image]) -> None:
        if self.cfg.pred_labels_dir is None:
            return
        reader = YoloLabelReader(self.cfg.pred_labels_dir)

        for img in items:
            stem = Path(img.image_path).stem
            W, H = img.width, img.height

            disc_xywhn = reader.read_first_xywhn(stem, class_id=0)
            cup_xywhn = reader.read_first_xywhn(stem, class_id=1)

            disc_box = reader.to_box(disc_xywhn, W, H) if disc_xywhn else None
            cup_box = reader.to_box(cup_xywhn, W, H) if cup_xywhn else None

            if disc_box is not None:
                img.set_box(Structure.DISC, LabelType.PRED, disc_box)
            if cup_box is not None:
                img.set_box(Structure.CUP, LabelType.PRED, cup_box)

    # ---- Future: MedSAM plug-in point ----
    def _attach_medsam_predictions(self, items: List[Image]) -> None:
        """
        Placeholder for future integration:
        - Read MedSAM-predicted masks (disc/cup) from provided directories.
        - Convert to boxes -> set img.pred_disc_box / img.pred_cup_box.
        Intentionally left unimplemented to avoid hallucinating your mask layout.
        """
        # Example scaffold (commented):
        # for img in items:
        #     disc_mask_path = ...  # locate by stem
        #     cup_mask_path  = ...
        #     if disc_mask_path: img.pred_disc_box = BoundingBox.from_mask(BinaryMaskRef(path=disc_mask_path).load())
        #     if cup_mask_path:  img.pred_cup_box  = BoundingBox.from_mask(BinaryMaskRef(path=cup_mask_path).load())
        return

    # ---- Compute per-image ratios ----
    def _compute_gt_ratio(self, img: Image) -> Optional[float]:
        return CDRCalculator.vertical_ratio_from_boxes(
            img.gt_disc_box, img.gt_cup_box, clip_01=self.cfg.clip_ratio_01
        )

    def _compute_intermediate_ratio(self, img: Image) -> Optional[float]:
        return CDRCalculator.vertical_ratio_from_boxes(
            img.inter_pred_disc_box, img.inter_pred_cup_box, clip_01=self.cfg.clip_ratio_01
        )

    # ---- Evaluation loop ----
    def evaluate(self) -> Tuple[List[EvalRow], EvalSummary]:
        items = self._load_images()
        self._attach_intermediate_boxes_from_yolo(items)
        # self._attach_medsam_predictions(items)  # keep disabled until you wire paths

        rows: List[EvalRow] = []
        y_true: List[float] = []
        y_pred: List[float] = []
        missing_gt = 0
        missing_pred = 0

        for img in items:
            gt = self._compute_gt_ratio(img)
            pred = self._compute_intermediate_ratio(img)

            if gt is None:
                missing_gt += 1
            if pred is None:
                missing_pred += 1

            if self.cfg.strict:
                if gt is None or pred is None:
                    raise RuntimeError(f"Missing data for {img.image_path}: gt={gt}, pred={pred}")

            abs_err = abs(pred - gt) if (_isfinite(gt) and _isfinite(pred)) else None
            sq_err = (pred - gt) ** 2 if (_isfinite(gt) and _isfinite(pred)) else None

            if _isfinite(gt) and _isfinite(pred):
                y_true.append(float(gt))   # type: ignore[arg-type]
                y_pred.append(float(pred)) # type: ignore[arg-type]

            rows.append(EvalRow(
                uid=img.uid,
                dataset=img.dataset,
                subject_id=img.subject_id,
                image_path=str(img.image_path),
                gt_cdr_v=gt,
                pred_cdr_v=pred,
                abs_error=abs_err,
                sq_error=sq_err,
            ))

        mae = LossComputer.reduce_mae([r.abs_error for r in rows if r.abs_error is not None])
        mse = LossComputer.reduce_mae([r.sq_error for r in rows if r.sq_error is not None])
        rmse = math.sqrt(mse) if mse is not None else None

        huber_vals: Optional[List[float]] = None
        huber_mean: Optional[float] = None
        if self.cfg.loss == "huber" and y_true and y_pred:
            huber_vals = [
                LossComputer.pointwise("huber", t, p, self.cfg.huber_delta)
                for t, p in zip(y_true, y_pred)
            ]
            huber_mean = LossComputer.reduce_custom(huber_vals)

        summary = EvalSummary(
            n_total=len(items),
            n_used=len(y_true),
            n_missing_gt=missing_gt,
            n_missing_pred=missing_pred,
            mae=mae,
            mse=mse,
            rmse=rmse,
            huber=huber_mean,
            pearson_r=LossComputer.pearson_r(y_true, y_pred),
        )

        if self.cfg.out_csv:
            self._write_csv(self.cfg.out_csv, rows, summary)

        return rows, summary

    # ---- Output helpers ----
    def _write_csv(self, out_path: Path, rows: List[EvalRow], summary: EvalSummary) -> None:
        _ensure_dir(out_path)
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            # header
            w.writerow(["uid", "dataset", "subject_id", "image_path",
                        "gt_cdr_v", "pred_cdr_v", "abs_error", "sq_error"])
            # rows
            for r in rows:
                w.writerow([
                    r.uid, r.dataset, r.subject_id, r.image_path,
                    _fmt(r.gt_cdr_v), _fmt(r.pred_cdr_v),
                    _fmt(r.abs_error), _fmt(r.sq_error),
                ])
            # summary block (as comment-style tail)
            w.writerow([])
            w.writerow(["# Summary"])
            for k, v in asdict(summary).items():
                w.writerow([k, _fmt(v)])


def _fmt(x: Optional[float]) -> str:
    if x is None or not math.isfinite(x):
        return ""
    return f"{x:.6f}"


# ------------------------ CLI ------------------------------- #

def _parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate vertical CDR error for intermediate YOLO predictions.")
    p.add_argument("--images-root", type=Path, required=True, help="Root of original fundus images.")
    p.add_argument("--gt-disc-masks", type=Path, default=None, help="Root of GT disc masks (binary).")
    p.add_argument("--gt-cup-masks", type=Path, default=None, help="Root of GT cup masks (binary).")
    p.add_argument("--pred-labels", type=Path, default=None, help="Directory of YOLO label .txt files (predicted).")

    p.add_argument("--include", nargs="*", default=["PAPILA"], help="Case-insensitive name filters to INCLUDE.")
    p.add_argument("--exclude", nargs="*", default=[], help="Case-insensitive name filters to EXCLUDE.")
    p.add_argument("--recursive", action="store_true", help="Recurse through subfolders when collecting images.")

    p.add_argument("--loss", choices=["l1", "l2", "huber"], default="l1", help="Loss to compute for report.")
    p.add_argument("--huber-delta", type=float, default=0.05, help="Delta for Huber loss.")
    p.add_argument("--no-clip", action="store_true", help="Do not clip CDR into [0,1].")
    p.add_argument("--strict", action="store_true", help="Raise on missing GT or prediction instead of skipping.")

    p.add_argument("--out-csv", type=Path, default=None, help="Optional path to write per-sample results + summary CSV.")

    args = p.parse_args()

    return EvalConfig(
        images_root=args.images_root,
        gt_disc_masks=args.gt_disc_masks,
        gt_cup_masks=args.gt_cup_masks,
        pred_labels_dir=args.pred_labels,
        include_name_contains=tuple(args.include or []),
        exclude_name_contains=tuple(args.exclude or []),
        loss=args.loss,
        huber_delta=float(args.huber_delta),
        clip_ratio_01=not args.no_clip,
        out_csv=args.out_csv,
        recursive=bool(args.recursive),
        strict=bool(args.strict),
    )


def main() -> None:
    cfg = _parse_args()
    ev = Evaluator(cfg)
    rows, summary = ev.evaluate()

    # Concise console summary
    print("---- Evaluation Summary ----")
    print(f"Total images:      {summary.n_total}")
    print(f"Used (valid pairs):{summary.n_used}")
    print(f"Missing GT:        {summary.n_missing_gt}")
    print(f"Missing Pred:      {summary.n_missing_pred}")
    print(f"MAE:               {_fmt(summary.mae)}")
    print(f"MSE:               {_fmt(summary.mse)}")
    print(f"RMSE:              {_fmt(summary.rmse)}")
    if cfg.loss == "huber":
        print(f"Huber (mean):      {_fmt(summary.huber)}")
    print(f"Pearson r:         {_fmt(summary.pearson_r)}")
    if cfg.out_csv:
        print(f"Wrote CSV:         {cfg.out_csv}")


if __name__ == "__main__":
    main()