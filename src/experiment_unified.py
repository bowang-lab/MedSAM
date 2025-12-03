#!/usr/bin/env python3
# File: src/experiment_unified.py
"""
Unified Experiment Script for MedSAM/YOLO Pipeline.

Modes:
1. Standard (Random Split): Scans raw data folders, splits by ratio, trains YOLO.
   Usage: python -m src.experiment_unified --train-yolo

2. Reproducible (Parquet Split): Builds dataset from a pre-split Parquet file.
   Usage: python -m src.experiment_unified --train-yolo --splits-parquet /path/to/data.parquet

3. Finetune Only: Uses existing weights to finetune MedSAM.
   Usage: python -m src.experiment_unified --finetune-medsam --yolo-weights /path/to/best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# PyArrow imports for efficient Parquet filtering
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.model.yolo import (
    YOLORunner,
    set_global_seed,
)
from src.scripts.finetune_medsam import (
    TrainConfig as MedSAMTrainConfig,
    run_finetune as run_medsam_finetune,
)
from src.imgpipe.yolo_splits import create_yolo_dataset_from_parquet

# CPU thread control for cluster environments
_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "8")))
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(_cpus))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_cpus))

SEED = 42

# Default split ratios
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1


# =========================
# Environment Configs
# =========================
@dataclass(frozen=True)
class EnvPaths:
    data_root: Path
    out_dir: Path
    run_dir: Path
    cfg: Path
    model: Path
    device: str
    medsam_ckpt: Path


@dataclass(frozen=True)
class RunConfig:
    epochs: int
    batch: int
    workers: int
    imgsz: int
    conf: float
    iou: float
    do_train: bool = True
    do_finetune: bool = True
    chain_test_after_train: bool = True


# Shared Hyperparameters
BASE_YOLO_HP = dict(imgsz=640, conf=0.01, iou=0.70)

# --- Local Env ---
LOCAL_REPO_DIR = Path(os.getcwd())  # Assumes running from repo root
LOCAL_ENV = EnvPaths(
    data_root=Path("/Volumes/ResearchUSB/All_Datasets_Organized"),
    out_dir=LOCAL_REPO_DIR / "YOLO_DS_P",
    run_dir=LOCAL_REPO_DIR / "runs",
    cfg=LOCAL_REPO_DIR / "src/configs/train_custom.yaml",
    model=LOCAL_REPO_DIR / "weights/yolo12n.pt",  # Adjusted path assumption
    device="mps",
    medsam_ckpt=LOCAL_REPO_DIR / "weights/medsam_vit_b.pth",
)
LOCAL_RUN = RunConfig(
    epochs=1, batch=8, workers=0, **BASE_YOLO_HP,
    do_train=False, do_finetune=True, chain_test_after_train=False
)

# --- HPC Env ---
HPC_REPO_DIR = Path("/scratch/st-ipor-1/cperez/MedSAM")
HPC_ENV = EnvPaths(
    data_root=Path("/arc/project/st-ipor-1/carlosp/fundus_data/All_Datasets_Organized"),
    out_dir=HPC_REPO_DIR / "YOLO_DS_P",
    run_dir=HPC_REPO_DIR / "runs",
    cfg=HPC_REPO_DIR / "src/configs/train_custom.yaml",
    model=HPC_REPO_DIR / "weights/yolo12x.pt",
    device="0",
    medsam_ckpt=HPC_REPO_DIR / "weights/medsam_vit_b.pth",
)
HPC_RUN = RunConfig(
    epochs=400, batch=16, workers=6, **BASE_YOLO_HP,
    do_train=True, do_finetune=False, chain_test_after_train=True
)


# =========================
# Parquet Filtering Logic (Ported from experiment_2.py)
# =========================
def _filter_parquet_require_both_gt_boxes(
        in_parquet: Path,
        out_parquet: Path,
        *,
        batch_size: int,
        disc_box_col: str = "gt_disc_box",
        cup_box_col: str = "gt_cup_box",
        disc_mask_col: str = "gt_disc_mask",
        cup_mask_col: str = "gt_cup_mask",
) -> tuple[int, int, int]:
    """
    Filters rows to ensure they have valid GT boxes and masks for both Disc and Cup.
    """
    in_parquet = Path(in_parquet)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    dataset = ds.dataset(str(in_parquet), format="parquet")
    schema = dataset.schema

    # 1. Validation
    need_cols = (disc_box_col, cup_box_col, disc_mask_col, cup_mask_col)
    missing = [c for c in need_cols if c not in schema.names]
    if missing:
        raise ValueError(f"Parquet missing columns: {missing}")

    # 2. Helpers for PyArrow Compute
    def _box_ok_mask(rb: pa.RecordBatch, colname: str) -> pa.Array:
        col = rb.column(schema.get_field_index(colname))
        ok = pc.is_valid(col)
        # Check struct validity (basic check)
        return ok

    def _mask_ref_nonempty(rb: pa.RecordBatch, colname: str) -> pa.Array:
        col = rb.column(schema.get_field_index(colname))
        ok = pc.is_valid(col)
        # Check if struct is not null
        return ok

    # 3. Streaming Filter
    scanner = dataset.scanner(batch_size=int(batch_size))
    writer: pq.ParquetWriter | None = None
    total = kept = dropped = 0

    try:
        for rb in scanner.to_batches():
            n = rb.num_rows
            if n == 0: continue
            total += n

            # Filter: Must have valid box AND valid mask ref for both cup and disc
            m = pc.and_(
                pc.and_(_box_ok_mask(rb, disc_box_col), _box_ok_mask(rb, cup_box_col)),
                pc.and_(_mask_ref_nonempty(rb, disc_mask_col), _mask_ref_nonempty(rb, cup_mask_col)),
            )

            tbl = pa.Table.from_batches([rb], schema=schema)
            ftbl = tbl.filter(m)

            kept += ftbl.num_rows
            dropped += (tbl.num_rows - ftbl.num_rows)

            if writer is None:
                writer = pq.ParquetWriter(str(out_parquet), schema=schema)

            if ftbl.num_rows > 0:
                writer.write_table(ftbl.cast(schema, safe=False))

        if writer is None:  # Write empty table if nothing kept to ensure file exists
            pq.write_table(pa.Table.from_batches([], schema=schema), str(out_parquet))

    finally:
        if writer: writer.close()

    return kept, dropped, total


def ensure_yolo_dataset_from_splits_parquet(
        splits_parquet: Path,
        yolo_ds_root: Path,
        batch_size: int,
        require_both_gt_boxes: bool,
) -> None:
    """
    Builds the YOLO dataset directory from a Parquet file.
    Optionally filters the parquet first to ensure clean GT data.
    """
    if (yolo_ds_root / "data.yaml").exists():
        print(f"[INFO] Using existing YOLO dataset at: {yolo_ds_root}")
        return

    src_parquet = splits_parquet

    # Optional: Filter Parquet
    if require_both_gt_boxes:
        filtered_parquet = yolo_ds_root / "_filtered_gt.parquet"
        if not filtered_parquet.exists():
            print(f"[INFO] Filtering parquet (Require Both GT Boxes) -> {filtered_parquet}")
            kept, dropped, total = _filter_parquet_require_both_gt_boxes(
                in_parquet=splits_parquet,
                out_parquet=filtered_parquet,
                batch_size=batch_size
            )
            print(f"[INFO] Filter Stats: Total={total}, Kept={kept}, Dropped={dropped}")
        src_parquet = filtered_parquet

    # Materialize
    print(f"[INFO] Materializing YOLO dataset from: {src_parquet}")
    create_yolo_dataset_from_parquet(
        src_parquet,
        out_dir=yolo_ds_root,
        batch_size=batch_size,
        use_gt=True,
        save_images_parquet=True
    )


# =========================
# Core Logic
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified MedSAM/YOLO Experiment Runner")

    # Environment
    parser.add_argument("--local", action="store_true", help="Use local paths")

    # Dataset Configuration
    parser.add_argument("--yolo-ds", type=Path, default=None, help="Path to existing YOLO dataset (data.yaml folder)")
    parser.add_argument("--yolo-out-dir", type=Path, default=None, help="Destination for new YOLO dataset")
    parser.add_argument("--splits-parquet", type=Path, default=None, help="Source Parquet file with 'split' column")
    parser.add_argument("--require-both-gt-boxes", action="store_true", default=True, help="Drop rows missing GT boxes")
    parser.add_argument("--splits-batch-size", type=int, default=2048)
    parser.add_argument("--papila", type=float, help="Fraction of PAPILA to include (0=None, 1=All)")

    # Split Ratios (only used if NOT using parquet)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)

    # Model / Training
    parser.add_argument("--yolo-weights", type=Path, default=None, help="Weights for testing/finetuning")
    parser.add_argument("--yolo-device", type=str, default=None, help="Override device (e.g. '0' or 'cpu')")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)

    # Actions
    parser.add_argument("--create-ds", action="store_true")
    parser.add_argument("--train-yolo", action="store_true")
    parser.add_argument("--test-yolo", action="store_true")
    parser.add_argument("--finetune-medsam", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(SEED)

    # 1. Setup Environment
    if args.local:
        print("[INFO] Mode: LOCAL")
        env, cfg = LOCAL_ENV, LOCAL_RUN
    else:
        print("[INFO] Mode: HPC")
        env, cfg = HPC_ENV, HPC_RUN

    # Overrides
    if args.epochs: cfg = dataclasses.replace(cfg, epochs=args.epochs)
    if args.batch: cfg = dataclasses.replace(cfg, batch=args.batch)

    # Determine Stages
    do_create = args.create_ds
    do_train = args.train_yolo
    do_test = args.test_yolo
    do_finetune = args.finetune_medsam

    # Default behavior if no flags passed
    if not any([do_create, do_train, do_test, do_finetune]):
        do_create = True
        do_train = cfg.do_train
        do_test = cfg.chain_test_after_train
        do_finetune = cfg.do_finetune

    # 2. Resolve Paths
    yolo_ds_root = args.yolo_ds or args.yolo_out_dir or env.out_dir
    effective_device = args.yolo_device or env.device

    # run naming
    i = 1
    while (env.run_dir / f"run{i}").exists(): i += 1
    run_name = f"run{i}"
    run_root = env.run_dir / run_name

    # 3. Prepare Dataset
    # Strategy: If parquet provided, build from parquet. Else, build from raw folders.
    if do_create or do_train:
        if args.splits_parquet:
            print(f"[INFO] Building Dataset from Parquet: {args.splits_parquet}")
            ensure_yolo_dataset_from_splits_parquet(
                splits_parquet=args.splits_parquet,
                yolo_ds_root=yolo_ds_root,
                batch_size=args.splits_batch_size,
                require_both_gt_boxes=args.require_both_gt_boxes
            )
        elif not (yolo_ds_root / "data.yaml").exists():
            print(f"[INFO] Building Dataset from Raw Folders (Standard)")
            # We use YOLORunner to wrap the creation logic
            temp_runner = YOLORunner(
                data_root=env.data_root, out_dir=yolo_ds_root, run_dir=env.run_dir,
                cfg=env.cfg, model=str(env.model), device="cpu",  # device doesn't matter for DS creation
                epochs=1, batch=1, imgsz=640, conf=0.1, iou=0.1,
                papila=args.papila
            )
            temp_runner.ensure_data(args.train_ratio, args.val_ratio, args.test_ratio)
        else:
            print(f"[INFO] Dataset already exists at {yolo_ds_root}")

    # 4. Train YOLO
    trained_weights = None
    if do_train:
        print(f"[INFO] Training YOLO on {yolo_ds_root}")
        runner = YOLORunner(
            data_root=env.data_root, out_dir=yolo_ds_root, run_dir=env.run_dir,
            cfg=env.cfg, model=str(env.model), device=effective_device,
            epochs=cfg.epochs, batch=cfg.batch, imgsz=cfg.imgsz,
            conf=cfg.conf, iou=cfg.iou,
            run_name=run_name,
            yolo_ds=yolo_ds_root,  # Explicitly use the DS we just prepared
            papila=args.papila
        )
        trained_weights = runner.train()

    # 5. Test YOLO
    if do_test:
        test_weights = trained_weights or args.yolo_weights
        if not test_weights:
            print("[WARN] Skipping test: no weights found.")
        else:
            print(f"[INFO] Testing YOLO with {test_weights}")
            runner = YOLORunner(
                data_root=env.data_root, out_dir=yolo_ds_root, run_dir=env.run_dir,
                cfg=env.cfg, model=str(env.model), device=effective_device,
                epochs=1, batch=cfg.batch, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou,
                run_name=run_name, yolo_ds=yolo_ds_root, papila=args.papila
            )
            runner.test(test_weights)

    # 6. Finetune MedSAM
    if do_finetune:
        fundu_weights = trained_weights or args.yolo_weights
        if not fundu_weights:
            print("[WARN] Skipping MedSAM finetune: no YOLO weights found.")
        else:
            print(f"[INFO] Finetuning MedSAM using prompts from {fundu_weights}")

            # Setup Fundu paths
            fundu_root = run_root / "Fundu" if trained_weights else env.run_dir / "Fundu_Inference"
            fundu_root.mkdir(parents=True, exist_ok=True)
            j = 1
            while (fundu_root / f"Fundu{j}").exists(): j += 1
            fundu_run_name = f"Fundu{j}"

            med_cfg = MedSAMTrainConfig(
                data_root=env.data_root,
                yolo_ds=yolo_ds_root,
                out_dir=fundu_root / fundu_run_name,
                run_dir=fundu_root,
                run_name=fundu_run_name,
                yolo_weights=fundu_weights,
                exclude_datasets=None,
                yolo_device=effective_device,
                yolo_imgsz=cfg.imgsz,
                yolo_conf=cfg.conf,
                yolo_iou=cfg.iou,
                ckpt=env.medsam_ckpt,
                epochs=cfg.epochs,
                batch=max(1, cfg.batch // 2),
                workers=cfg.workers,
                do_train=True, do_test=True, test_prompt="det",
                seed=SEED
            )
            run_medsam_finetune(med_cfg)


import dataclasses  # needed for replace

if __name__ == "__main__":
    main()