#!/usr/bin/env python3
# File: src/scripts/predict_summarize.py

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image as PILImage, ImageFile
from tqdm import tqdm

# Use your unified Image class
from src.imgpipe.image import Image
from src.model.predictor import (
    Predictor,
    PredictorConfig,
    YoloPredictorConfig,
    MedSamPredictorConfig,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# Helpers
# =========================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _is_finite(x: Any) -> bool:
    try:
        return (x is not None) and bool(np.isfinite(float(x)))
    except Exception:
        return False


def _mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not xs:
        return None, None
    arr = np.asarray(xs, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, std


def _relative_subdir(image_path: Path, images_root: Optional[Path]) -> Path:
    p = Path(image_path)
    if images_root is not None:
        try:
            return p.parent.relative_to(images_root)
        except Exception:
            pass
    for parent in p.parents:
        if parent.name.lower() == "fundus":
            try:
                return p.parent.relative_to(parent)
            except Exception:
                return Path()
    return Path()


def _ensure_scalar_metrics(img: Image) -> None:
    # CDR
    try:
        if not _is_finite(getattr(img, "gt_cdr", None)):
            gt_cdr = img.cdr(use_pred=False, axis="vertical")
            if _is_finite(gt_cdr):
                img.gt_cdr = float(gt_cdr)
        if not _is_finite(getattr(img, "pred_cdr", None)):
            pr_cdr = img.cdr(use_pred=True, axis="vertical")
            if _is_finite(pr_cdr):
                img.pred_cdr = float(pr_cdr)
    except Exception:
        pass

    # RDR
    try:
        if not _is_finite(getattr(img, "gt_rdr", None)):
            gt_r = img.rim_metrics(use_pred=False) or {}
            v = gt_r.get("rim_over_disc")
            if _is_finite(v):
                img.gt_rdr = float(v)
        if not _is_finite(getattr(img, "pred_rdr", None)):
            pr_r = img.rim_metrics(use_pred=True) or {}
            v = pr_r.get("rim_over_disc")
            if _is_finite(v):
                img.pred_rdr = float(v)
    except Exception:
        pass


def _pack_pred_masks_for_parquet(img: Image) -> None:
    """
    Align predicted masks to image size, then pack them into BinaryMaskRef so
    Image.to_dict(include_mask_bytes=True) can serialize bytes into Parquet.
    """
    for attr in ("pred_disc_mask", "pred_cup_mask"):
        mref = getattr(img, attr, None)
        if mref is None:
            continue
        try:
            arr = img._mask_to_image_size(mref)
            if arr is None:
                continue
            mref.array = arr
            if hasattr(mref, "pack_inplace"):
                mref.pack_inplace()
            # Optional: keep parquet self-contained
            mref.path = None
        except Exception:
            pass


def _save_pred_masks_for_image(
        img: Image,
        *,
        oc_mask_root: Path,
        od_mask_root: Path,
        rel_subdir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Write predicted masks as PNGs and update mref.path. Returns (disc_path, cup_path)."""
    disc_path: Optional[Path] = None
    cup_path: Optional[Path] = None

    for attr_name, out_root, is_disc in (
            ("pred_disc_mask", od_mask_root, True),
            ("pred_cup_mask", oc_mask_root, False),
    ):
        mref = getattr(img, attr_name, None)
        if mref is None:
            continue

        try:
            arr = img._mask_to_image_size(mref)
        except Exception:
            arr = None
        if arr is None:
            continue

        out_path = out_root / rel_subdir / img.image_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        m_uint8 = (arr.astype(bool)).astype("uint8") * 255
        PILImage.fromarray(m_uint8).save(str(out_path))

        try:
            mref.path = out_path
            if hasattr(mref, "array"):
                mref.array = None
        except Exception:
            pass

        if is_disc:
            disc_path = out_path
        else:
            cup_path = out_path

    return disc_path, cup_path


def _gather_image_files(images_dir: Path, pattern: str) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in images_dir.rglob(pattern) if p.suffix.lower() in exts)


def _make_images_from_dir(
        images_dir: Path,
        pattern: str = "*",
        dataset: str = "inference",
        split: Optional[str] = None,
) -> List[Image]:
    """Scans directory and creates lightweight Image objects."""
    img_paths = _gather_image_files(images_dir, pattern)
    if not img_paths:
        raise RuntimeError(f"No images found under {images_dir} with pattern '{pattern}'.")

    images: List[Image] = []
    for p in img_paths:
        try:
            # Fast header read
            with PILImage.open(p) as im:
                W, H = im.size
        except Exception as e:
            logging.warning("Skipping unreadable image %s: %r", p, e)
            continue

        # Construct Image directly
        im_obj = Image.from_path(
            image_path=p,
            dataset=dataset,
            subject_id=p.stem,
            uid=p.stem,
            split=split,
            width=W,
            height=H,
        )
        images.append(im_obj)
    return images


# =========================
# Schema helpers (for summarization only)
# =========================

def append_summary_row_to_csv(csv_path: Path, pred_ref: Path, summary: Dict[str, Any]) -> None:
    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    md = summary["mask_dice_stats"]
    bd = summary["box_dice_stats"]
    me = summary["metric_error"]
    se = summary.get("scalar_error", {})

    header = [
        "pred_path",
        "n_images",
        "mask_dice_disc_mean", "mask_dice_disc_std",
        "mask_dice_cup_mean", "mask_dice_cup_std",
        "box_dice_disc_mean", "box_dice_disc_std",
        "box_dice_cup_mean", "box_dice_cup_std",
        "cdr_v_mae_mean", "cdr_v_mae_std",
        "rim_over_disc_mae_mean", "rim_over_disc_mae_std",
        "I_over_S_mae_mean", "I_over_S_mae_std",
        "cdr_mae_mean", "cdr_mae_std",
        "rdr_mae_mean", "rdr_mae_std",
    ]

    row = [
        str(pred_ref),
        summary["counts"]["images"],
        md["disc"]["mean"], md["disc"]["std"],
        md["cup"]["mean"], md["cup"]["std"],
        bd["disc"]["mean"], bd["disc"]["std"],
        bd["cup"]["mean"], bd["cup"]["std"],
        me["cdr_v"]["mae_mean"], me["cdr_v"]["mae_std"],
        me["rim_over_disc"]["mae_mean"], me["rim_over_disc"]["mae_std"],
        me["I_over_S"]["mae_mean"], me["I_over_S"]["mae_std"],
        (se.get("cdr_mae") or {}).get("mae_mean"),
        (se.get("cdr_mae") or {}).get("mae_std"),
        (se.get("rdr_mae") or {}).get("mae_mean"),
        (se.get("rdr_mae") or {}).get("mae_std"),
    ]

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)


# =========================
# Distributed / Multi-GPU helpers
# =========================

@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int


def _get_dist_info() -> DistInfo:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return DistInfo(
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            local_rank=int(os.environ.get("LOCAL_RANK", "0"))
        )
    if "SLURM_PROCID" in os.environ:
        return DistInfo(
            rank=int(os.environ["SLURM_PROCID"]),
            world_size=int(os.environ["SLURM_NTASKS"]),
            local_rank=int(os.environ.get("SLURM_LOCALID", "0"))
        )
    return DistInfo(rank=0, world_size=1, local_rank=0)


def _maybe_init_torch_distributed(dist: DistInfo) -> bool:
    if dist.world_size <= 1 or not torch.distributed.is_available():
        return False
    if torch.distributed.is_initialized():
        return True
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    try:
        torch.distributed.init_process_group(
            backend=backend, init_method="env://", rank=dist.rank, world_size=dist.world_size
        )
        return True
    except Exception:
        return False


def _barrier_if_possible() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
        except Exception:
            pass


def _device_for_rank(base_device: str, local_rank: int) -> str:
    d = (base_device or "").strip().lower()
    if not d.startswith("cuda"):
        return base_device
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        if "," not in cvd:
            return "cuda:0"
    if d == "cuda":
        return f"cuda:{local_rank}"
    return base_device


def _pin_torch_cuda_device(device: str) -> None:
    if not torch.cuda.is_available():
        return
    d = (device or "").strip().lower()
    if not d.startswith("cuda"):
        return
    idx = 0
    if ":" in d:
        try:
            idx = int(d.split(":")[1])
        except Exception:
            idx = 0
    if idx < torch.cuda.device_count():
        torch.cuda.set_device(idx)


def _ultralytics_device_str_for_current_process(base_device: str) -> str:
    d = (base_device or "").strip().lower()
    if d.startswith("cuda") and torch.cuda.is_available():
        return str(torch.cuda.current_device())
    return base_device


def _dataset_dir(out_dir: Path) -> Path:
    return out_dir / "predictions.dataset"


def _rank_dataset_dir(dataset_dir: Path, rank: int) -> Path:
    return dataset_dir / f"gpu={rank:03d}"


def _rank_progress_path(out_dir: Path, rank: int) -> Path:
    return out_dir / "progress" / f"gpu_{rank:03d}.jsonl"


def _rank_part_path(rank_dir: Path, part_idx: int) -> Path:
    return rank_dir / f"part-{part_idx:06d}.parquet"


def _load_done_set(progress_file: Path) -> Set[str]:
    done: Set[str] = set()
    if not progress_file.exists():
        return done
    with open(progress_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = rec.get("image_path") or rec.get("uid")
                if key:
                    done.add(str(key))
            except Exception:
                continue
    return done


def _append_done(progress_file: Path, *, image_path: str, uid: str) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"image_path": image_path, "uid": uid}) + "\n")


def _existing_part_count(rank_dir: Path) -> int:
    if not rank_dir.exists():
        return 0
    return len(list(rank_dir.glob("part-*.parquet")))


def _iter_sharded_images(it: Iterable[Image], *, rank: int, world_size: int) -> Iterable[Image]:
    for i, img in enumerate(it):
        if (i % world_size) == rank:
            yield img


# =========================
# Summarization Logic (Refactored to use Image.iter_parquet)
# =========================

def summarize_predictions_dataset(
        dataset_dir: Path,
        splits: Optional[Set[str]] = None,
        datasets: Optional[Set[str]] = None,
        batch_size: int = 2048,
) -> Dict[str, Any]:
    """
    Stream the entire dataset_dir using Image.iter_parquet and compute metrics.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Predictions dataset directory not found: {dataset_dir}")

    n_images = 0
    det_disc = det_cup = seg_disc = seg_cup = 0

    # Metrics storage
    mask_disc_vals: List[float] = []
    mask_cup_vals: List[float] = []
    box_disc_vals: List[float] = []
    box_cup_vals: List[float] = []

    # Absolute errors for scalar metrics
    err_abs_adv: Dict[str, List[float]] = {
        "cdr_v": [], "cdr_h": [], "rim_over_disc": [], "I_over_S": []
    }
    cdr_abs_err: List[float] = []
    rdr_abs_err: List[float] = []

    # Stream efficiently from directory
    for img in Image.iter_parquet(dataset_dir, batch_size=batch_size):
        # Filtering
        if datasets and getattr(img, "dataset", None) not in datasets:
            continue
        if splits and getattr(img, "split", None) not in splits:
            continue

        n_images += 1

        # Detection counts
        if getattr(img, "inter_pred_disc_box", None): det_disc += 1
        if getattr(img, "inter_pred_cup_box", None): det_cup += 1
        if getattr(img, "pred_disc_mask", None): seg_disc += 1
        if getattr(img, "pred_cup_mask", None): seg_cup += 1

        # Mask Dice (use cached or compute from mask bytes)
        try:
            img.update_mask_dice(fallback_to_boxes=True)
        except Exception:
            pass

        if _is_finite(img.mask_dice_disc): mask_disc_vals.append(float(img.mask_dice_disc))
        if _is_finite(img.mask_dice_cup): mask_cup_vals.append(float(img.mask_dice_cup))

        # Box Dice
        if img.pred_disc_box and img.gt_disc_box:
            box_disc_vals.append(float(img.pred_disc_box.dice(img.gt_disc_box)))
        if img.pred_cup_box and img.gt_cup_box:
            box_cup_vals.append(float(img.pred_cup_box.dice(img.gt_cup_box)))

        # CDR Error
        try:
            gt_cdr = img.gt_cdr if _is_finite(img.gt_cdr) else img.cdr(use_pred=False, axis="vertical")
            pr_cdr = img.pred_cdr if _is_finite(img.pred_cdr) else img.cdr(use_pred=True, axis="vertical")
            if _is_finite(gt_cdr) and _is_finite(pr_cdr):
                cdr_abs_err.append(abs(float(pr_cdr) - float(gt_cdr)))
        except Exception:
            pass

        # RDR Error
        try:
            gt_rdr = img.gt_rdr if _is_finite(img.gt_rdr) else (img.rim_metrics(use_pred=False) or {}).get(
                "rim_over_disc")
            pr_rdr = img.pred_rdr if _is_finite(img.pred_rdr) else (img.rim_metrics(use_pred=True) or {}).get(
                "rim_over_disc")
            if _is_finite(gt_rdr) and _is_finite(pr_rdr):
                rdr_abs_err.append(abs(float(pr_rdr) - float(gt_rdr)))
        except Exception:
            pass

        # Advanced Metrics Error (Rim, I/S, etc)
        try:
            metrics = img.metrics_summary()
            gt_m = metrics.get("gt", {})
            pr_m = metrics.get("pred", {})
            for key in ("cdr_v", "cdr_h", "rim_over_disc", "I_over_S"):
                gv, pv = gt_m.get(key), pr_m.get(key)
                if _is_finite(gv) and _is_finite(pv):
                    err_abs_adv[key].append(abs(float(pv) - float(gv)))
        except Exception:
            pass

    # Aggregation
    md_disc_mean, md_disc_std = _mean_std(mask_disc_vals)
    md_cup_mean, md_cup_std = _mean_std(mask_cup_vals)
    bd_disc_mean, bd_disc_std = _mean_std(box_disc_vals)
    bd_cup_mean, bd_cup_std = _mean_std(box_cup_vals)

    metric_error_adv = {
        k: {"mae_mean": m, "mae_std": s, "n": len(err_abs_adv[k])}
        for k, (m, s) in {name: _mean_std(vals) for name, vals in err_abs_adv.items()}.items()
    }

    return {
        "counts": {
            "images": n_images,
            "det_rate_disc": det_disc / max(1, n_images),
            "det_rate_cup": det_cup / max(1, n_images),
            "seg_rate_disc": seg_disc / max(1, n_images),
            "seg_rate_cup": seg_cup / max(1, n_images),
        },
        "mask_dice_stats": {
            "disc": {"mean": md_disc_mean, "std": md_disc_std, "n": len(mask_disc_vals)},
            "cup": {"mean": md_cup_mean, "std": md_cup_std, "n": len(mask_cup_vals)},
        },
        "box_dice_stats": {
            "disc": {"mean": bd_disc_mean, "std": bd_disc_std, "n": len(box_disc_vals)},
            "cup": {"mean": bd_cup_mean, "std": bd_cup_std, "n": len(box_cup_vals)},
        },
        "metric_error": metric_error_adv,
        "scalar_error": {
            "cdr_mae": {"mae_mean": _mean_std(cdr_abs_err)[0], "mae_std": _mean_std(cdr_abs_err)[1],
                        "n": len(cdr_abs_err)},
            "rdr_mae": {"mae_mean": _mean_std(rdr_abs_err)[0], "mae_std": _mean_std(rdr_abs_err)[1],
                        "n": len(rdr_abs_err)},
        },
    }


# =========================
# CLI
# =========================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run YOLO+MedSAM predictions (resumable + multi-GPU) and summarize results.")

    p.add_argument("--images-parquet", type=Path, default=None)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--image-pattern", type=str, default="*")
    p.add_argument("--inference-dataset", type=str, default="inference")
    p.add_argument("--inference-split", type=str, default=None)

    p.add_argument("--yolo-weights", type=Path, default=None)
    p.add_argument("--medsam-checkpoint", type=Path, default=None)

    p.add_argument("--device", type=str, default="cuda", help="cuda | cuda:<id> | cpu | mps")

    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--imgsz", type=int, default=640)

    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--no-save-masks", action="store_true")
    p.add_argument("--box-pad-frac", type=float, default=0.05)

    p.add_argument("--sam-amp", action="store_true", help="Enable CUDA AMP for MedSAM (if supported by predictor).")
    p.add_argument("--sam-no-amp", action="store_true", help="Disable CUDA AMP for MedSAM (if supported by predictor).")
    p.add_argument("--sam-resize-backend", type=str, default="cv2", choices=("cv2", "pil"))

    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--summary-csv", type=Path, default=None)

    p.add_argument("--splits", type=str, default=None)
    p.add_argument("--datasets", type=str, default=None)

    p.add_argument("--parquet-read-batch", type=int, default=1024)
    p.add_argument("--predict-batch-size", type=int, default=128)

    p.add_argument("--yolo-batch", type=int, default=None, help="Ultralytics YOLO batch size for inference.")

    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-recompute", action="store_true")

    p.add_argument(
        "--mask-store",
        type=str,
        choices=("png", "parquet", "both", "none"),
        default="parquet",
        help="Where to store predicted masks. 'parquet' stores packed masks in output parquet; "
             "'png' writes oc_mask/od_mask PNGs; 'both' does both; 'none' stores neither.",
    )

    p.add_argument(
        "--image-store",
        type=str,
        choices=("parquet", "none"),
        default="none",
        help="Whether to embed raw image bytes in the output parquets (large!).",
    )

    p.add_argument("--final-parquet-name", type=str, default="predictions.parquet")
    p.add_argument("--final-parquet-compression", type=str, default="zstd")
    p.add_argument("--final-parquet-batch", type=int, default=2048)

    return p.parse_args(argv)


# =========================
# Main
# =========================

def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args(argv)

    out_dir: Path = _ensure_dir(args.out_dir)
    dist = _get_dist_info()

    # Best-effort distributed init so we can barrier for final concatenation.
    _maybe_init_torch_distributed(dist)

    # Filters
    split_set: Optional[Set[str]] = None
    if args.splits:
        split_set = {s.strip() for s in args.splits.split(",") if s.strip()}

    dataset_set: Optional[Set[str]] = None
    if args.datasets:
        dataset_set = {d.strip() for d in args.datasets.split(",") if d.strip()}

    # Outputs
    dataset_dir = _dataset_dir(out_dir)
    rank_dir = _rank_dataset_dir(dataset_dir, dist.rank)

    save_png_masks = (args.mask_store in ("png", "both")) and (not args.no_save_masks)
    save_parquet_masks = (args.mask_store in ("parquet", "both"))
    save_overlays = bool(args.save_overlays)
    save_parquet_images = (args.image_store == "parquet")

    oc_mask_root = (out_dir / "oc_mask" / f"gpu={dist.rank:03d}") if save_png_masks else None
    od_mask_root = (out_dir / "od_mask" / f"gpu={dist.rank:03d}") if save_png_masks else None
    overlay_root = (out_dir / "overlay" / f"gpu={dist.rank:03d}") if save_overlays else None

    if oc_mask_root is not None: _ensure_dir(oc_mask_root)
    if od_mask_root is not None: _ensure_dir(od_mask_root)
    if overlay_root is not None: _ensure_dir(overlay_root)

    progress_file = _rank_progress_path(out_dir, dist.rank)

    # Resume bookkeeping
    done: Set[str] = set()
    if args.force_recompute:
        logging.info("force-recompute enabled: ignoring any resume state.")
    elif args.resume:
        done = _load_done_set(progress_file)
        logging.info("resume enabled (rank=%d): loaded %d done items", dist.rank, len(done))

    # Input source iterator
    if not (args.images_parquet or args.images_dir):
        raise ValueError("You must pass --images-parquet or --images-dir to run predictions.")
    if args.yolo_weights is None or args.medsam_checkpoint is None:
        raise ValueError("--yolo-weights and --medsam-checkpoint are required.")

    def input_image_iter() -> Iterable[Image]:
        if args.images_parquet is not None:
            logging.info("Reading images from Parquet: %s", args.images_parquet)
            # Use Image class to stream efficiently
            for img in Image.iter_parquet(args.images_parquet, batch_size=args.parquet_read_batch):
                if dataset_set and getattr(img, "dataset", None) not in dataset_set: continue
                if split_set and getattr(img, "split", None) not in split_set: continue
                yield img
        else:
            assert args.images_dir is not None
            logging.info("Creating images from directory: %s", args.images_dir)
            images = _make_images_from_dir(
                images_dir=args.images_dir,
                pattern=args.image_pattern,
                dataset=args.inference_dataset,
                split=args.inference_split,
            )
            for im in images:
                if dataset_set and getattr(im, "dataset", None) not in dataset_set: continue
                if split_set and getattr(im, "split", None) not in split_set: continue
                yield im

    # Device for this rank
    device = _device_for_rank(args.device, dist.local_rank)
    _pin_torch_cuda_device(device)
    yolo_device = _ultralytics_device_str_for_current_process(device)

    # Predictor Setup
    yolo_kwargs: Dict[str, Any] = dict(
        weights=args.yolo_weights, device=yolo_device, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
    )
    if args.yolo_batch is not None:
        yolo_kwargs["batch_size"] = int(args.yolo_batch)

    try:
        yolo_cfg = YoloPredictorConfig(**yolo_kwargs)
    except TypeError:
        yolo_kwargs.pop("batch_size", None)
        yolo_cfg = YoloPredictorConfig(**yolo_kwargs)

    use_amp = True
    if args.sam_no_amp:
        use_amp = False
    elif args.sam_amp:
        use_amp = True

    sam_kwargs: Dict[str, Any] = {"checkpoint": args.medsam_checkpoint, "device": device, "use_amp": use_amp,
                                  "resize_backend": args.sam_resize_backend}
    try:
        sam_cfg = MedSamPredictorConfig(**sam_kwargs)
    except TypeError:
        sam_kwargs.pop("use_amp", None)
        sam_kwargs.pop("resize_backend", None)
        sam_cfg = MedSamPredictorConfig(**sam_kwargs)

    pred_cfg = PredictorConfig(box_pad_frac=args.box_pad_frac)
    predictor = Predictor(yolo_cfg, sam_cfg, pred_cfg)

    images_root = args.images_dir.resolve() if args.images_dir else None
    part_idx = _existing_part_count(rank_dir)

    def process_batch(batch: List[Image]) -> None:
        nonlocal part_idx
        if not batch:
            return

        try:
            preds = predictor.predict(batch)
        except Exception as e:
            logging.error("Batch failed on rank=%d: %r. Trying per-image fallback.", dist.rank, e)
            preds = []
            for img in batch:
                try:
                    preds.extend(predictor.predict([img]))
                except Exception:
                    # Skip bad images, mark done to avoid retry loops
                    if args.resume:
                        ip = str(img.image_path)
                        done.add(ip)
                        _append_done(progress_file, image_path=ip, uid=img.uid)
            if not preds: return

        batch_done_records: List[Tuple[str, str]] = []

        for img in preds:
            try:
                img.update_mask_dice(fallback_to_boxes=True)
            except Exception:
                pass

            _ensure_scalar_metrics(img)
            rel_subdir = _relative_subdir(img.image_path, images_root=images_root)

            if save_png_masks and oc_mask_root and od_mask_root:
                _save_pred_masks_for_image(img=img, oc_mask_root=oc_mask_root, od_mask_root=od_mask_root,
                                           rel_subdir=rel_subdir)

            if save_parquet_masks:
                _pack_pred_masks_for_parquet(img)

            if save_overlays and overlay_root:
                overlay_path = overlay_root / rel_subdir / img.image_path.name
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    img.visualize(show=False, save_path=overlay_path, dpi=140, mask_alpha=0.7)
                    if img.extras is None: img.extras = {}
                    img.extras["overlay_path"] = str(overlay_path)
                except Exception:
                    pass

            batch_done_records.append((str(img.image_path), str(img.uid)))

        # Write part using robust Image class writer
        part_path = _rank_part_path(rank_dir, part_idx)
        Image.save_parquet(
            images=preds,
            path=part_path,
            drop_none=False,
            include_image_bytes=save_parquet_images,
            include_mask_bytes=save_parquet_masks,
            compression="zstd",
            write_batch=1024,
        )
        part_idx += 1

        if args.resume and not args.force_recompute:
            for ip, uid in batch_done_records:
                done.add(ip)
                _append_done(progress_file, image_path=ip, uid=uid)

    # Main Processing Loop
    batch: List[Image] = []
    n_seen = 0

    # Iterate specifically over sharded subset for this rank
    sharded_iter = _iter_sharded_images(input_image_iter(), rank=dist.rank, world_size=dist.world_size)

    for img in tqdm(sharded_iter, desc=f"Predict (rank={dist.rank})", unit="img"):
        n_seen += 1
        if not args.force_recompute and args.resume and str(img.image_path) in done:
            continue

        batch.append(img)
        if len(batch) >= args.predict_batch_size:
            process_batch(batch)
            batch = []

    if batch:
        process_batch(batch)

    logging.info("rank=%d finished. seen=%d parts=%d", dist.rank, n_seen, part_idx)
    _barrier_if_possible()

    # Rank 0: Summarize and Concatenate
    if dist.rank == 0:
        logging.info("Summarizing predictions from dataset %s", dataset_dir)
        summary = summarize_predictions_dataset(
            dataset_dir, splits=split_set, datasets=dataset_set, batch_size=args.parquet_read_batch
        )

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if args.summary_csv:
            append_summary_row_to_csv(args.summary_csv, dataset_dir, summary)

        final_path = out_dir / args.final_parquet_name
        logging.info("Concatenating into single file: %s", final_path)

        # Use Image class to stream from dataset directory and write to single file
        Image.save_parquet(
            Image.iter_parquet(dataset_dir),
            path=final_path,
            include_image_bytes=save_parquet_images,
            include_mask_bytes=save_parquet_masks,
            compression=args.final_parquet_compression,
            write_batch=args.final_parquet_batch
        )
        logging.info("Done.")


if __name__ == "__main__":
    main()