# src/imgpipe/preprocess_pipeline_oop.py
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.imgpipe.collector import DatasetCollector
from src.imgpipe.dataset import Dataset
from src.imgpipe.binary_mask_ref import BinaryMaskRef
from src.imgpipe.enums import LabelType, Structure
from src.utils import ensure_dir, stem_map_by_first_match, load_cfg, p


# ----------------------------- config loader -----------------------------

@dataclass
class _CollectorConfigShim:
    """Tiny shim so DatasetCollector keeps working; we pass YAML in .raw."""
    project_dir: Path
    images_root: Path
    disc_masks: Optional[Path]
    cup_masks: Optional[Path]
    include_name_contains: Optional[List[str]]
    exclude_name_contains: Optional[List[str]]
    recursive: bool
    raw: Dict[str, Any]



# ----------------------------- mask attach -----------------------------

def _build_per_dataset_mask_maps(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Path]]:
    """
    Build {tag: {"disc": stem->path, "cup": stem->path}} using dataset blocks.
    If 'datasets' is missing, return empty and the caller can fall back to top-level roots.
    """
    maps: Dict[str, Dict[str, Dict[str, Path]]] = {}
    for d in cfg.get("datasets", []) or []:
        tag = d.get("tag") or ""
        disc_root = p(d.get("disc_masks")) if d.get("disc_masks") else None
        cup_root  = p(d.get("cup_masks"))  if d.get("cup_masks")  else None
        disc_map = stem_map_by_first_match(disc_root) if disc_root else {}
        cup_map  = stem_map_by_first_match(cup_root)  if cup_root  else {}
        maps[tag] = {"disc": disc_map, "cup": cup_map}
    return maps  # type: ignore[return-value]


def _attach_gt_masks_and_boxes(
    ds: Dataset,
    cfg: Dict[str, Any],
) -> None:
    """
    For each Image, attach GT disc/cup masks (if found) and derive GT boxes.
    Priority:
      1) per-dataset maps (matched by image.dataset)
      2) top-level disc_masks / cup_masks (if provided)
    """
    per_ds = _build_per_dataset_mask_maps(cfg)
    # global fallbacks
    global_disc = stem_map_by_first_match(p(cfg.get("disc_masks"))) if cfg.get("disc_masks") else {}
    global_cup  = stem_map_by_first_match(p(cfg.get("cup_masks")))  if cfg.get("cup_masks")  else {}

    for img in ds.images:
        stem = img.image_path.stem
        tag = getattr(img, "dataset", "") or ""
        # choose map by tag else fallback to global
        disc_map = per_ds.get(tag, {}).get("disc", {}) if per_ds else {}
        cup_map  = per_ds.get(tag, {}).get("cup",  {}) if per_ds else {}
        if not disc_map and global_disc:
            disc_map = global_disc
        if not cup_map and global_cup:
            cup_map = global_cup

        # attach masks if not already set
        if getattr(img, "gt_disc_mask", None) is None and stem in disc_map:
            img.set_mask(Structure.DISC, LabelType.GT, BinaryMaskRef(path=disc_map[stem]))
        if getattr(img, "gt_cup_mask", None) is None and stem in cup_map:
            img.set_mask(Structure.CUP,  LabelType.GT, BinaryMaskRef(path=cup_map[stem]))

        # derive GT boxes from masks
        img.ensure_boxes_from_masks()

# ----------------------------- small helpers -----------------------------

def _dedupe_by_stem(images: List[Any]) -> List[Any]:
    seen: set[str] = set()
    out: List[Any] = []
    for img in images:
        s = img.image_path.stem
        if s in seen:
            continue
        seen.add(s)
        out.append(img)
    return out


def _has_any_gt(img: Any) -> bool:
    return (getattr(img, "gt_disc_mask", None) is not None) or (getattr(img, "gt_cup_mask", None) is not None)


def _write_split_summary_csv(ds: Dataset, out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "split", "has_disc_gt", "has_cup_gt"])
        w.writeheader()
        for img in ds.images:
            w.writerow({
                "image_path": str(img.image_path),
                "split": getattr(img, "split", ""),
                "has_disc_gt": int(img.gt_disc_mask is not None),
                "has_cup_gt":  int(img.gt_cup_mask  is not None),
            })


def _write_data_yaml(yolo_root: Path) -> Path:
    data_yaml = {
        "path": str(yolo_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": ["disc", "cup"],
    }
    p = yolo_root / "data.yaml"
    ensure_dir(p.parent)
    with open(p, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    return p


def _train_multiclass_if_requested(cfg: Dict[str, Any], yolo_root: Path) -> None:
    train_cfg = cfg.get("train")
    if not train_cfg:
        print("[TRAIN] No 'train:' block; skipping.")
        return
    from ultralytics import YOLO
    weights = train_cfg.get("weights")
    if not weights:
        print("[TRAIN] 'weights' missing; skipping.")
        return

    epochs  = int(train_cfg.get("epochs", 100))
    imgsz   = int(train_cfg.get("imgsz", 640))
    batch   = int(train_cfg.get("batch", 16))
    name    = str(train_cfg.get("name", "multiclass_disc_cup"))
    workers = int(train_cfg.get("workers", 8))
    device  = train_cfg.get("device", None)
    seed    = int(train_cfg.get("seed", 1337))
    runs_root = p(cfg.get("outputs", {}).get("runs_root")) or (Path(cfg["project_dir"]) / "runs" / "detect")

    data_yaml = _write_data_yaml(yolo_root)
    print(f"[TRAIN] Ultralytics: data={data_yaml} weights={weights} epochs={epochs} imgsz={imgsz} batch={batch}")
    YOLO(str(weights)).train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(runs_root),
        name=name,
        workers=workers,
        seed=seed,
        single_cls=False,
        pretrained=True,
        optimizer="AdamW",
        cos_lr=True,
        patience=50,
    )
    print("[TRAIN] Done.")

# ------------------------------- main -------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Collect → attach GT masks+boxes → filter (any GT) → dedupe → split → save YOLO (disc/cup) → (optional) train"
    )
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(Path(args.config))

    project_dir = p(cfg["project_dir"])
    yolo_root = p(cfg["outputs"]["yolo_root"])
    if not project_dir or not yolo_root:
        raise SystemExit("[ERR] 'project_dir' and outputs.yolo_root are required.")

    ensure_dir(yolo_root)

    # Collector (uses cfg['datasets'] via .raw)
    shim = _CollectorConfigShim(
        project_dir=project_dir,
        images_root=p(cfg.get("images_root")) or project_dir,  # unused when 'datasets' present
        disc_masks=p(cfg.get("disc_masks")),
        cup_masks=p(cfg.get("cup_masks")),
        include_name_contains=cfg.get("include_name_contains"),
        exclude_name_contains=cfg.get("exclude_name_contains"),
        recursive=bool(cfg.get("recursive", False)),
        raw=cfg,
    )
    collector = DatasetCollector(shim)
    ds_all: Dataset = collector.collect()
    print(f"[COLLECT] images (raw): {len(ds_all.images)}")

    # Attach GT masks and derive GT boxes (so YOLO labels can be written)
    _attach_gt_masks_and_boxes(ds_all, cfg)

    # Filter to items with any GT (disc or cup), then dedupe by stem
    kept = [im for im in ds_all.images if _has_any_gt(im)]
    print(f"[FILTER ] images with any GT: {len(kept)}")
    kept = _dedupe_by_stem(kept)
    print(f"[DEDUP  ] unique by stem   : {len(kept)}")
    if not kept:
        raise SystemExit("[ERR] No images left after filtering+dedupe.")

    ds_f = Dataset(kept)

    # Shuffle/split by patient
    split_cfg = cfg.get("split", {}) or {}
    train, val, test = ds_f.split_by_patient(
        val_frac=float(split_cfg.get("val_frac", 0.15)),
        test_frac=float(split_cfg.get("test_frac", 0.15)),
        seed=int(split_cfg.get("seed", 1337)),
        patient_regex=str(split_cfg.get("patient_regex", "") or ""),
    )
    print(f"[SPLIT  ] train={len(train.images)} | val={len(val.images)} | test={len(test.images)}")

    # Save one YOLO dataset with three splits (multiclass disc/cup)
    combined = Dataset(train.images + val.images + test.images)
    combined.save_as_yolo(yolo_root, write_yaml=False, prefer_copy=bool(cfg.get("prefer_copy", False)))
    data_yaml_path = _write_data_yaml(yolo_root)
    print(f"[SAVE   ] YOLO root → {yolo_root}")
    print(f"[SAVE   ] data.yaml → {data_yaml_path}")

    # Split summary CSV
    summary_csv = yolo_root / "split_summary.csv"
    _write_split_summary_csv(combined, summary_csv)
    print(f"[SAVE   ] split summary → {summary_csv}")

    # Optional training
    _train_multiclass_if_requested(cfg, yolo_root)


if __name__ == "__main__":
    main()