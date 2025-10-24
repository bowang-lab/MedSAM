#!/usr/bin/env python3
# train_cup.py
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import yaml
from ultralytics import YOLO

from src.utils import ensure_dir, need, expand, ultralytics_device_arg


# ---------------- Rectangle-based Dice (proxy) ----------------

def _rect_to_mask(H: int, W: int, bbox_xywh) -> np.ndarray:
    x, y, w, h = bbox_xywh
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(W, int(round(x + w)))
    y2 = min(H, int(round(y + h)))
    m = np.zeros((H, W), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 1
    return m


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    s = a.sum() + b.sum()
    return (2.0 * inter / s) if s > 0 else float("nan")


def _compute_rect_dice_from_eval_dir(eval_dir: Path) -> Tuple[Optional[float], Dict[str, float]]:
    """
    Computes mean rectangle-based Dice (and per-class) from Ultralytics eval artifacts.
    Requires 'predictions.json' and 'labels.json' in eval_dir.
    Returns (mean_dice, per_class_dict). If unavailable, returns (None, {}).
    """
    pred_json = eval_dir / "predictions.json"
    labels_json = eval_dir / "labels.json"
    if not pred_json.exists() or not labels_json.exists():
        return None, {}

    preds = json.loads(pred_json.read_text())
    gts = json.loads(labels_json.read_text())

    by_img_pred, by_img_gt = {}, {}
    for p in preds:
        by_img_pred.setdefault(p["image_id"], []).append(p)
    for g in gts:
        by_img_gt.setdefault(g["image_id"], []).append(g)

    per_class: Dict[int, List[float]] = {}
    for img_id, gboxes in by_img_gt.items():
        H = gboxes[0].get("height")
        W = gboxes[0].get("width")
        if H is None or W is None:
            continue
        pboxes = by_img_pred.get(img_id, [])
        used = set()

        for g in gboxes:
            gc = g["category_id"]
            gb = g["bbox"]  # xywh
            gmask = _rect_to_mask(H, W, gb)
            best_dice = float("nan")
            best_j = -1
            for j, pr in enumerate(pboxes):
                if j in used or pr["category_id"] != gc:
                    continue
                pmask = _rect_to_mask(H, W, pr["bbox"])
                d = _dice(gmask, pmask)
                if not np.isnan(d) and (np.isnan(best_dice) or d > best_dice):
                    best_dice, best_j = d, j
            if best_j >= 0:
                used.add(best_j)
                per_class.setdefault(gc, []).append(float(best_dice))

    per_class_mean = {str(k): float(np.nanmean(v)) for k, v in per_class.items() if v}
    all_vals: List[float] = []
    for v in per_class.values():
        all_vals.extend([x for x in v if np.isfinite(x)])
    mean_dice = float(np.mean(all_vals)) if all_vals else None

    # Also write a sidecar summary if desired
    try:
        (eval_dir / "dice_rect_summary.json").write_text(json.dumps(
            {"mean": mean_dice, "per_class": per_class_mean}, indent=2))
    except Exception:
        pass

    return mean_dice, per_class_mean


def _pick(d: dict, keys: Tuple[str, ...]):
    for k in keys:
        if k in d:
            return d[k]
    return None


# --- config ---
@dataclass
class CupROITrainCfg:
    # required paths
    data_root: Path                  # contains data.yaml + images/labels
    runs_root: Path

    # model selection
    amp: bool = True
    freeze: int = 0                    # freeze first N layers (0 = none)
    weights: Optional[str] = None      # /path/best.pt or a hub tag like yolo12x.pt
    family: str = "auto"               # auto|yolo12|yolo11|yolov8
    size: str = "l"                    # n|s|m|l|x

    # training knobs
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    name: str = "stageB_cup_roi_modern"
    workers: int = 8
    seed: int = 1337
    optimizer: str = "AdamW"
    cos_lr: bool = True
    patience: int = 50
    pretrained: bool = True

    # augs
    hsv_h: float = 0.015
    hsv_s: float = 0.70
    hsv_v: float = 0.40
    degrees: float = 10.0
    translate: float = 0.10
    scale: float = 0.50
    shear: float = 2.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 0.50
    mixup: float = 0.10
    copy_paste: float = 0.0
    erasing: float = 0.40

    device: Optional[str] = None       # "0"|"0,1"| "cpu" | "mps"

    # eval-only helpers (when you want to re-run test on existing weights)
    eval_ckpt: Optional[Path] = None   # path to .pt; if set, skip training
    eval_split: str = "test"           # "test" or "val"
    eval_plots: bool = True            # save PR/F1/confusion plots


# --- trainer ---
class CupROITrainer:
    def __init__(self, cfg: CupROITrainCfg):
        self.cfg = cfg
        self.yaml_path = cfg.data_root / "data.yaml"
        need(cfg.data_root, "ROI dataset root")
        need(self.yaml_path, "data.yaml")
        ensure_dir(cfg.runs_root)
        self.device = cfg.device or ultralytics_device_arg()

        # build/attach model (weights may be hub tag or local .pt)
        self.model = YOLO(self._resolve_weights())

    def _resolve_weights(self) -> str:
        if self.cfg.weights:
            return str(expand(self.cfg.weights))
        fam = (self.cfg.family or "auto").lower()
        size = (self.cfg.size or "x").lower()
        if fam in ("auto", "yolo12"):
            return f"yolo12{size}.pt"
        if fam in ("yolo11",):
            return f"yolo11{size}.pt"
        return f"yolov8{size}.pt"

    def train(self) -> None:
        c = self.cfg
        overrides = dict(
            data=str(self.yaml_path),
            epochs=c.epochs,
            imgsz=c.imgsz,
            batch=c.batch,
            device=self.device,
            project=str(c.runs_root),
            name=c.name,
            workers=c.workers,
            seed=c.seed,
            optimizer=c.optimizer,
            cos_lr=c.cos_lr,
            patience=c.patience,
            pretrained=c.pretrained,
            single_cls=True,  # cup-only
            amp=c.amp,
            freeze=c.freeze,
            # augs
            hsv_h=c.hsv_h, hsv_s=c.hsv_s, hsv_v=c.hsv_v,
            degrees=c.degrees, translate=c.translate, scale=c.scale,
            shear=c.shear, perspective=c.perspective,
            flipud=c.flipud, fliplr=c.fliplr,
            mosaic=c.mosaic, mixup=c.mixup, copy_paste=c.copy_paste, erasing=c.erasing,
        )
        print(f"[INFO] Device: {self.device}")
        self.model.train(**overrides)

    # ---------- evaluation helpers ----------
    def _choose_split(self) -> str:
        y = yaml.safe_load(self.yaml_path.read_text()) or {}
        return "test" if "test" in y else "val"

    def _find_ckpt_in_run(self) -> Optional[Path]:
        run_dir = self.cfg.runs_root / self.cfg.name / "weights"
        best = run_dir / "best.pt"
        last = run_dir / "last.pt"
        if best.exists():
            return best
        if last.exists():
            return last
        return None

    def _eval_and_collect(
        self,
        ckpt: Path,
        split: str,
        out_root: Path,
        plots: bool = True
    ) -> Dict[str, Any]:
        """
        Run YOLO val() on the given checkpoint and split, copy plots into the run directory,
        and return a metrics dict. Also writes metrics JSON to out_root.
        Additionally computes:
          - box_loss/cls_loss/dfl_loss (robust key matching)
          - rectangle-based Dice vs GT using predictions.json + labels.json
        """
        tester = YOLO(str(ckpt))
        print(f"[EVAL] Evaluating '{ckpt.name}' on split='{split}' …")
        res = tester.val(
            data=str(self.yaml_path),
            split=split,
            imgsz=self.cfg.imgsz,
            device=self.device,
            plots=plots,
            save_json=True,  # COCO json (per-class metrics-friendly)
        )

        # metrics from Ultralytics
        try:
            base_metrics = getattr(res, "results_dict", None) or {}
            base_metrics = {k: float(v) for k, v in base_metrics.items()}
        except Exception:
            base_metrics = {}

        # Add robustly-parsed losses (keys differ across versions)
        box_loss = _pick(base_metrics, ("loss/box", "box_loss", "box"))
        cls_loss = _pick(base_metrics, ("loss/cls", "cls_loss", "cls"))
        dfl_loss = _pick(base_metrics, ("loss/dfl", "dfl_loss", "dfl"))
        if box_loss is not None:
            base_metrics["box_loss"] = float(box_loss)
        if cls_loss is not None:
            base_metrics["cls_loss"] = float(cls_loss)
        if dfl_loss is not None:
            base_metrics["dfl_loss"] = float(dfl_loss)

        # Where Ultralytics saved eval artifacts (plots, predictions.json, etc.)
        eval_save_dir = Path(getattr(res, "save_dir", out_root / f"{split}_eval_tmp"))

        # Compute rectangle-based Dice summary from eval artifacts
        try:
            dice_mean, dice_per_class = _compute_rect_dice_from_eval_dir(eval_save_dir)
            if dice_mean is not None:
                base_metrics["dice_rect_mean"] = float(dice_mean)
            if dice_per_class:
                # add as nested dict
                base_metrics["dice_rect_per_class"] = {k: float(v) for k, v in dice_per_class.items()}
        except Exception as e:
            print(f"[WARN] Dice metric computation failed: {e}")

        # Consolidate within this run directory under a split-specific folder
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target_dir = out_root / f"eval_{split}" / stamp
        ensure_dir(target_dir)

        # Copy any artifacts (PR curves, confusion matrix, labels, predictions.json, etc.)
        try:
            for p in eval_save_dir.glob("*"):
                if p.is_file():
                    shutil.copy2(p, target_dir / p.name)
                elif p.is_dir():
                    shutil.copytree(p, target_dir / p.name, dirs_exist_ok=True)
        except Exception as e:
            print(f"[WARN] Could not consolidate eval artifacts: {e}")

        # Write metrics JSON alongside the plots
        metrics_json = target_dir / "metrics.json"
        try:
            metrics_json.write_text(json.dumps(base_metrics, indent=2))
            print(f"[EVAL] Metrics JSON → {metrics_json}")
        except Exception as e:
            print(f"[WARN] Failed to write metrics.json: {e}")

        # Also write/update a convenience symlink "latest" for quick checks
        try:
            latest = out_root / f"eval_{split}" / "latest"
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(target_dir.name)  # relative symlink
        except Exception:
            pass

        # Console summary (including losses + dice if present)
        if base_metrics:
            printable = {k: v for k, v in base_metrics.items() if isinstance(v, (int, float))}
            print("[EVAL] Summary metrics:", printable)
            if "box_loss" in base_metrics or "dice_rect_mean" in base_metrics:
                bl = base_metrics.get("box_loss", None)
                cl = base_metrics.get("cls_loss", None)
                dl = base_metrics.get("dfl_loss", None)
                dm = base_metrics.get("dice_rect_mean", None)
                parts = []
                if bl is not None: parts.append(f"box_loss={bl:.4f}")
                if cl is not None: parts.append(f"cls_loss={cl:.4f}")
                if dl is not None: parts.append(f"dfl_loss={dl:.4f}")
                if dm is not None: parts.append(f"dice_rect_mean={dm:.4f}")
                if parts:
                    print("[EVAL] " + " | ".join(parts))
        else:
            print("[EVAL] Finished; see plots/metrics in:", target_dir)

        return base_metrics

    def validate_and_save(self, ckpt: Optional[Path] = None, split: Optional[str] = None) -> None:
        """Evaluate (default to TEST if present) and save artifacts/metrics under the same run dir."""
        out_root = self.cfg.runs_root / self.cfg.name
        ensure_dir(out_root)

        split = split or self._choose_split()
        ckpt_path = ckpt or self._find_ckpt_in_run()
        if ckpt_path is None:
            print("[WARN] No checkpoint found to evaluate.")
            return

        self._eval_and_collect(ckpt=ckpt_path, split=split, out_root=out_root, plots=self.cfg.eval_plots)

    # ---------- top-level entry points ----------
    def run_train_then_test(self) -> None:
        self.train()
        print("[INFO] Training complete. Running evaluation on held-out split …")
        self.validate_and_save(split="test")  # will auto-fallback to val if 'test' not in YAML

    def run_eval_only(self) -> None:
        # eval-only mode requires an explicit checkpoint path
        ckpt = self.cfg.eval_ckpt
        if not ckpt:
            raise SystemExit("[ERR] --eval-ckpt PATH is required for eval-only mode.")
        ckpt = expand(ckpt)
        need(ckpt, "eval checkpoint (*.pt)")
        print(f"[INFO] Eval-only mode. Using device: {self.device}")
        self.validate_and_save(ckpt=ckpt, split=self.cfg.eval_split)


# --- CLI ---
def parse_args() -> CupROITrainCfg:
    ap = argparse.ArgumentParser(
        description="Train cup-only YOLO on ROI crops and/or evaluate checkpoints with test metrics, PR curves, and Dice."
    )

    # required-ish
    ap.add_argument("--data-root", default=None,
                    help="Root with data.yaml and images/labels. Default: ./bounding_box/data/yolo_split_cupROI")
    ap.add_argument("--runs-root", default=None,
                    help="Ultralytics runs root. Default: ./bounding_box/runs/detect")

    # model selection
    ap.add_argument("--weights", default=None, help="Explicit weights path or hub tag (e.g., yolo12x.pt)")
    ap.add_argument("--family", default="auto", choices=["auto", "yolo12", "yolo11", "yolov8"])
    ap.add_argument("--size", default="x", choices=["n", "s", "m", "l", "x"])

    # training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--freeze", type=int, default=0)
    ap.add_argument("--amp", type=lambda v: str(v).lower() not in {"0", "false", "no"}, default=True)
    ap.add_argument("--name", default="stageB_cup_roi_modern")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--cos-lr", action="store_true", default=True)
    ap.add_argument("--no-cos-lr", dest="cos_lr", action="store_false")
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--device", default=None)

    # augs
    ap.add_argument("--hsv_h", type=float, default=0.015)
    ap.add_argument("--hsv_s", type=float, default=0.70)
    ap.add_argument("--hsv_v", type=float, default=0.40)
    ap.add_argument("--degrees", type=float, default=10.0)
    ap.add_argument("--translate", type=float, default=0.10)
    ap.add_argument("--scale", type=float, default=0.50)
    ap.add_argument("--shear", type=float, default=2.0)
    ap.add_argument("--perspective", type=float, default=0.0)
    ap.add_argument("--flipud", type=float, default=0.0)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--mosaic", type=float, default=0.50)
    ap.add_argument("--mixup", type=float, default=0.10)
    ap.add_argument("--copy_paste", type=float, default=0.0)
    ap.add_argument("--erasing", type=float, default=0.40)

    # eval-only mode
    ap.add_argument("--eval-ckpt", default="", help="If set, skip training and only evaluate this checkpoint (.pt).")
    ap.add_argument("--eval-split", default="test", choices=["test", "val"], help="Split to evaluate.")
    ap.add_argument("--no-eval-plots", dest="eval_plots", action="store_false", help="Disable PR/F1/confusion plots.")

    args = ap.parse_args()

    project = Path(".").resolve()
    default_data = project / "bounding_box" / "data" / "yolo_split_cupROI"
    default_runs = project / "bounding_box" / "runs" / "detect"

    data_root = expand(args.data_root) if args.data_root else default_data
    runs_root = expand(args.runs_root) if args.runs_root else default_runs
    need(data_root, "ROI dataset root")
    need(data_root / "data.yaml", "data.yaml")
    ensure_dir(runs_root)

    eval_ckpt = expand(args.eval_ckpt) if args.eval_ckpt else None

    return CupROITrainCfg(
        data_root=data_root,
        runs_root=runs_root,
        weights=args.weights,
        family=args.family,
        size=args.size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        freeze=args.freeze,
        amp=bool(args.amp),
        name=args.name,
        workers=args.workers,
        seed=args.seed,
        hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
        degrees=args.degrees, translate=args.translate, scale=args.scale,
        shear=args.shear, perspective=args.perspective,
        flipud=args.flipud, fliplr=args.fliplr,
        mosaic=args.mosaic, mixup=args.mixup, copy_paste=args.copy_paste, erasing=args.erasing,
        optimizer=args.optimizer, cos_lr=bool(args.cos_lr),
        patience=args.patience, pretrained=not (not args.pretrained),  # keep boolean
        device=args.device,
        eval_ckpt=eval_ckpt,
        eval_split=args.eval_split,
        eval_plots=bool(args.eval_plots if hasattr(args, "eval_plots") else True),
    )


def main():
    cfg = parse_args()
    trainer = CupROITrainer(cfg)

    if cfg.eval_ckpt:
        # Re-run test/val on an existing checkpoint, save curves + metrics under the same run root/name
        print("[MODE] Eval-only")
        trainer.run_eval_only()
    else:
        # Standard: train then evaluate on test (or val if test not present)
        print("[MODE] Train → Test")
        trainer.run_train_then_test()
        print("[OK] Done.")


if __name__ == "__main__":
    main()