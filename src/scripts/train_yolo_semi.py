#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from ultralytics import YOLO

from src.imgpipe.image import Image  # your Image class w/ Parquet IO


SEED_DEFAULT = 42


# -------------------------
# Utilities
# -------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)


def get_next_run_name(run_root: Path, base: str = "ss") -> str:
    i = 1
    while (run_root / f"{base}{i}").exists():
        i += 1
    return f"{base}{i}"


def ensure_dirs(ds_root: Path) -> None:
    for sub in (
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
    ):
        (ds_root / sub).mkdir(parents=True, exist_ok=True)


def write_data_yaml(ds_root: Path) -> Path:
    data = {
        "path": str(ds_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "disc", 1: "cup"},
    }
    p = ds_root / "data.yaml"
    # Ultralytics accepts YAML; keep it YAML for readability
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


def fmt_yolo_line(cls_id: int, xc: float, yc: float, w: float, h: float) -> str:
    # Clamp to [0,1] for safety
    xc = float(min(1.0, max(0.0, xc)))
    yc = float(min(1.0, max(0.0, yc)))
    w = float(min(1.0, max(0.0, w)))
    h = float(min(1.0, max(0.0, h)))
    return f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def has_gt_boxes(img: Image) -> bool:
    return (img.gt_disc_box is not None) and (img.gt_cup_box is not None)


def has_pseudo_boxes(img: Image) -> bool:
    return (img.inter_pred_disc_box is not None) and (img.inter_pred_cup_box is not None)


def pseudo_conf_ok(img: Image, disc_min: float, cup_min: float) -> bool:
    # If confs are missing, treat as not OK (forces you to be explicit/consistent)
    if img.yolo_disc_conf is None or img.yolo_cup_conf is None:
        return False
    return (float(img.yolo_disc_conf) >= disc_min) and (float(img.yolo_cup_conf) >= cup_min)


def label_lines_for_image(
    img: Image,
    *,
    source: str,  # "gt" or "pseudo"
) -> List[str]:
    """
    Create YOLO label lines for this image.
    Enforces both classes: disc (0) and cup (1), if present in the chosen source.
    """
    if source == "gt":
        disc = img.gt_disc_box
        cup = img.gt_cup_box
    elif source == "pseudo":
        disc = img.inter_pred_disc_box
        cup = img.inter_pred_cup_box
    else:
        raise ValueError(f"Unknown label source: {source}")

    lines: List[str] = []
    if disc is not None:
        xc, yc, w, h = disc.as_tuple()
        lines.append(fmt_yolo_line(0, xc, yc, w, h))
    if cup is not None:
        xc, yc, w, h = cup.as_tuple()
        lines.append(fmt_yolo_line(1, xc, yc, w, h))
    return lines


def materialize_image_file(img: Image, dst_img_path: Path) -> None:
    """
    Write the image file without re-encoding:
      - Prefer embedded bytes (img.image_ref.packed) when present
      - Else copy from img.image_path
    """
    dst_img_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer embedded bytes if available
    ref = img.image_ref
    if ref is not None and getattr(ref, "packed", None) is not None:
        dst_img_path.write_bytes(ref.packed)  # exact file bytes (best fidelity)
        return

    # Else copy from disk path
    src = Path(img.image_path)
    if not src.exists():
        raise FileNotFoundError(f"Image path missing and no embedded bytes available: {src}")
    shutil.copy2(src, dst_img_path)


def split_groups(
    group_to_uids: Dict[str, List[str]],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[set[str], set[str], set[str]]:
    s = train_ratio + val_ratio + test_ratio
    if not math.isclose(s, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"train+val+test must sum to 1.0 (got {s})")

    groups = list(group_to_uids.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(math.floor(train_ratio * n))
    n_val = int(math.floor(val_ratio * n))
    # remainder to test
    train_g = set(groups[:n_train])
    val_g = set(groups[n_train : n_train + n_val])
    test_g = set(groups[n_train + n_val :])
    return train_g, val_g, test_g


@dataclass(frozen=True)
class ChosenLabel:
    label_source: str  # "gt" or "pseudo"
    split: str         # "train" | "val" | "test"
    uid: str


# -------------------------
# Main dataset builder
# -------------------------
def build_semi_supervised_yolo_dataset(
    images: List[Image],
    *,
    out_dir: Path,
    seed: int,
    group_by: str,  # "patient_id" or "uid"
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    val_test_gt_only: bool,
    pseudo_disc_conf_min: float,
    pseudo_cup_conf_min: float,
    include_pseudo_in_train: bool,
    max_pseudo_train: Optional[int],
) -> Dict[str, object]:
    """
    Strategy:
      - Split ONLY the GT-labeled groups into train/val/test
      - Add pseudo-labeled samples to TRAIN only (unless include_pseudo_in_train=False)

    This avoids "val/test on pseudo labels" by default.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_dirs(out_dir)

    # Index images by uid
    uid_to_img: Dict[str, Image] = {}
    for im in images:
        uid_to_img[im.uid] = im

    # Partition into eligible GT and eligible pseudo
    gt_uids: List[str] = []
    pseudo_uids: List[str] = []

    skipped_no_labels = 0
    skipped_bad_pseudo = 0

    for im in images:
        if has_gt_boxes(im):
            gt_uids.append(im.uid)
            continue

        # pseudo candidate
        if has_pseudo_boxes(im) and pseudo_conf_ok(im, pseudo_disc_conf_min, pseudo_cup_conf_min):
            pseudo_uids.append(im.uid)
        else:
            # unlabeled (or low quality pseudo)
            if has_pseudo_boxes(im) and not pseudo_conf_ok(im, pseudo_disc_conf_min, pseudo_cup_conf_min):
                skipped_bad_pseudo += 1
            else:
                skipped_no_labels += 1

    # Create groups for GT split
    if group_by == "patient_id":
        def group_key(im: Image) -> str:
            return f"{im.dataset}::{im.patient_id}"
    elif group_by == "uid":
        def group_key(im: Image) -> str:
            return im.uid
    else:
        raise ValueError("--group-by must be one of: patient_id, uid")

    group_to_uids: Dict[str, List[str]] = {}
    for uid in gt_uids:
        im = uid_to_img[uid]
        g = group_key(im)
        group_to_uids.setdefault(g, []).append(uid)

    train_g, val_g, test_g = split_groups(
        group_to_uids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    chosen: List[ChosenLabel] = []

    # Assign GT samples to splits
    for g, uids in group_to_uids.items():
        if g in train_g:
            split = "train"
        elif g in val_g:
            split = "val"
        else:
            split = "test"
        for uid in uids:
            chosen.append(ChosenLabel(label_source="gt", split=split, uid=uid))

    # Add pseudo to train only (common SSL approach)
    if include_pseudo_in_train:
        # deterministic shuffle of pseudo uids for reproducible subsampling
        rng = random.Random(seed)
        pseudo_uids_shuf = pseudo_uids[:]
        rng.shuffle(pseudo_uids_shuf)

        if max_pseudo_train is not None:
            pseudo_uids_shuf = pseudo_uids_shuf[: int(max_pseudo_train)]

        for uid in pseudo_uids_shuf:
            chosen.append(ChosenLabel(label_source="pseudo", split="train", uid=uid))

    # Materialize files
    stats = {
        "train": {"gt": 0, "pseudo": 0},
        "val": {"gt": 0, "pseudo": 0},
        "test": {"gt": 0, "pseudo": 0},
    }

    # Use uid-based filenames to avoid collisions across datasets
    for item in chosen:
        im = uid_to_img[item.uid]
        # decide extension: keep original to avoid re-encoding
        ext = (im.image_path.suffix or "").lower()
        if not ext:
            # If bytes are embedded, try ref.ext; else default png
            ext = getattr(im.image_ref, "ext", None) or ".png"
        if not ext.startswith("."):
            ext = "." + ext

        dst_img = out_dir / "images" / item.split / f"{im.uid}{ext}"
        dst_lbl = out_dir / "labels" / item.split / f"{im.uid}.txt"

        materialize_image_file(im, dst_img)

        lines = label_lines_for_image(im, source=item.label_source)
        # YOLO expects label file to exist; empty is valid, but we do not create empty here.
        if not lines:
            # If a record somehow has no boxes in chosen source, skip it (should be rare)
            continue
        dst_lbl.write_text("\n".join(lines), encoding="utf-8")

        stats[item.split][item.label_source] += 1

    data_yaml = write_data_yaml(out_dir)

    meta = {
        "created_at_utc": __import__("datetime").datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seed": seed,
        "group_by": group_by,
        "ratios_gt_groups": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "val_test_gt_only": bool(val_test_gt_only),
        "pseudo_thresholds": {"disc_min": pseudo_disc_conf_min, "cup_min": pseudo_cup_conf_min},
        "include_pseudo_in_train": bool(include_pseudo_in_train),
        "max_pseudo_train": max_pseudo_train,
        "skipped_no_labels": skipped_no_labels,
        "skipped_bad_pseudo": skipped_bad_pseudo,
        "counts_written": stats,
        "data_yaml": str(data_yaml),
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"data_yaml": data_yaml, "meta": meta}


# -------------------------
# Train
# -------------------------
def run_train(
    *,
    data_yaml: Path,
    runs_root: Path,
    init_weights: Optional[Path],
    base_model: Optional[str],
    cfg: Optional[Path],
    device: str,
    epochs: int,
    batch: int,
    imgsz: int,
    workers: int,
    freeze: int,
) -> Path:
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    run_name = get_next_run_name(runs_root, base="ss")
    run_root = runs_root / run_name
    run_root.mkdir(parents=True, exist_ok=False)

    # Construct model
    if init_weights is not None:
        model = YOLO(str(init_weights))
    else:
        if base_model is None:
            raise ValueError("Provide --init-weights to finetune OR --base-model to train from scratch.")
        model = YOLO(base_model)

    train_kwargs = dict(
        data=str(data_yaml),
        project=str(run_root),
        name="Train",
        device=device,
        epochs=int(epochs),
        batch=int(batch),
        imgsz=int(imgsz),
        workers=int(workers),
        exist_ok=False,
        resume=False,
        freeze=int(freeze) if freeze > 0 else 0,
    )
    if cfg is not None:
        train_kwargs["cfg"] = str(cfg)

    model.train(**train_kwargs)

    # Resolve weights
    weights_dir = run_root / "Train" / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    if best_pt.exists():
        return best_pt
    if last_pt.exists():
        return last_pt
    return weights_dir


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a semi-supervised YOLO dataset from Image Parquet and train/finetune.")

    # IO
    p.add_argument("--images-parquet", type=Path, required=True, help="Filtered Image Parquet (your Image schema).")
    p.add_argument("--out-yolo-ds", type=Path, required=True, help="Output YOLO dataset dir (created/overwritten contents).")
    p.add_argument("--runs-root", type=Path, required=True, help="Output runs root (new ss run created each time).")

    # Model init
    p.add_argument("--init-weights", type=Path, default=None, help="Pretrained YOLO weights (.pt) to finetune.")
    p.add_argument("--base-model", type=str, default=None, help="Base model checkpoint/name to train from scratch (e.g., yolo12x.pt).")
    p.add_argument("--cfg", type=Path, default=None, help="Ultralytics train config yaml (optional).")

    # Training
    p.add_argument("--device", type=str, default="0", help='Device string, e.g. "0" or "0,1,2,3" or "cpu".')
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--freeze", type=int, default=0, help="Freeze first N layers (0 disables).")

    # Splits (applied to GT groups only)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--group-by", type=str, default="patient_id", choices=["patient_id", "uid"])
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)

    # Pseudo-label controls
    p.add_argument("--pseudo-disc-conf-min", type=float, default=0.50)
    p.add_argument("--pseudo-cup-conf-min", type=float, default=0.50)
    p.add_argument("--no-pseudo-train", action="store_true", help="Disable adding pseudo labels to train (GT-only).")
    p.add_argument("--max-pseudo-train", type=int, default=None, help="Limit number of pseudo samples added to train.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    if args.out_yolo_ds.exists():
        # Safe behavior: require empty or non-existent unless user intentionally points to a new folder
        # (You can delete if you prefer overwrite; this is conservative.)
        # We only create subfolders; we do not delete existing contents.
        pass

    print("[INFO] Loading Image Parquet:", args.images_parquet)
    images = Image.load_parquet(args.images_parquet)
    print(f"[INFO] Loaded {len(images)} Image records")

    include_pseudo_in_train = not args.no_pseudo_train

    build_out = build_semi_supervised_yolo_dataset(
        images,
        out_dir=args.out_yolo_ds,
        seed=args.seed,
        group_by=args.group_by,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        val_test_gt_only=True,
        pseudo_disc_conf_min=float(args.pseudo_disc_conf_min),
        pseudo_cup_conf_min=float(args.pseudo_cup_conf_min),
        include_pseudo_in_train=include_pseudo_in_train,
        max_pseudo_train=args.max_pseudo_train,
    )

    data_yaml: Path = build_out["data_yaml"]  # type: ignore[assignment]
    meta: dict = build_out["meta"]            # type: ignore[assignment]

    print("[INFO] Dataset created:", args.out_yolo_ds)
    print("[INFO] data.yaml:", data_yaml)
    print("[INFO] counts_written:", json.dumps(meta["counts_written"], indent=2))

    best = run_train(
        data_yaml=data_yaml,
        runs_root=args.runs_root,
        init_weights=args.init_weights,
        base_model=args.base_model,
        cfg=args.cfg,
        device=args.device,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        freeze=args.freeze,
    )
    print("[OK] Training done. Weights/artifacts at:", best)


if __name__ == "__main__":
    main()