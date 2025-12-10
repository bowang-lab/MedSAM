#!/usr/bin/env python3
# File: src/experiment.py
"""
Unified Experiment Script for MedSAM/YOLO Pipeline.
...
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

from src.model.yolo import (
    YOLORunner,
    set_global_seed,
)
from src.scripts.finetune_medsam import (
    TrainConfig as MedSAMTrainConfig,
    run_finetune as run_medsam_finetune,
)
from src.imgpipe.yolo_splits import create_yolo_dataset_from_parquet

# CPU thread control
_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "8")))
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(_cpus))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_cpus))

SEED = 42

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
LOCAL_REPO_DIR = Path(os.getcwd())
LOCAL_ENV = EnvPaths(
    data_root=Path("/Volumes/ResearchUSB/All_Datasets_Organized"),
    out_dir=LOCAL_REPO_DIR / "YOLO_DS_P",
    run_dir=LOCAL_REPO_DIR / "runs",
    cfg=LOCAL_REPO_DIR / "src/configs/train_custom.yaml",
    model=LOCAL_REPO_DIR / "weights/yolo12n.pt",
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
# Dataset Logic
# =========================

def ensure_yolo_dataset_from_splits_parquet(
        splits_parquet: Path,
        yolo_ds_root: Path,
        batch_size: int,
        exclude_datasets: list[str] | None = None,  # <--- NEW ARGUMENT
) -> None:
    """
    Builds the YOLO dataset directory from a Parquet file.
    """
    if (yolo_ds_root / "data.yaml").exists():
        print(f"[INFO] Using existing YOLO dataset at: {yolo_ds_root}")
        return

    print(f"[INFO] Materializing YOLO dataset from: {splits_parquet}")
    create_yolo_dataset_from_parquet(
        splits_parquet,
        out_dir=yolo_ds_root,
        batch_size=batch_size,
        use_gt=True,
        save_images_parquet=True,
        exclude_datasets=exclude_datasets,
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
    parser.add_argument("--splits-batch-size", type=int, default=2048)
    parser.add_argument("--exclude-ds", nargs="*", default=None, help="Datasets to exclude (substring match)") # <--- NEW CLI ARG

    # Split Ratios
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)

    # Model / Training
    parser.add_argument("--yolo-weights", type=Path, default=None, help="Weights for testing/finetuning")
    parser.add_argument("--yolo-device", type=str, default=None, help="Override device")
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

    if args.epochs: cfg = dataclasses.replace(cfg, epochs=args.epochs)
    if args.batch: cfg = dataclasses.replace(cfg, batch=args.batch)

    do_create = args.create_ds
    do_train = args.train_yolo
    do_test = args.test_yolo
    do_finetune = args.finetune_medsam

    if not any([do_create, do_train, do_test, do_finetune]):
        do_create = True
        do_train = cfg.do_train
        do_test = cfg.chain_test_after_train
        do_finetune = cfg.do_finetune

    # 2. Resolve Paths
    yolo_ds_root = args.yolo_ds or args.yolo_out_dir or env.out_dir
    effective_device = args.yolo_device or env.device

    i = 1
    while (env.run_dir / f"run{i}").exists(): i += 1
    run_name = f"run{i}"
    run_root = env.run_dir / run_name

    # 3. Prepare Dataset
    if do_create or do_train:
        if args.splits_parquet:
            print(f"[INFO] Building Dataset from Parquet: {args.splits_parquet}")
            ensure_yolo_dataset_from_splits_parquet(
                splits_parquet=args.splits_parquet,
                yolo_ds_root=yolo_ds_root,
                batch_size=args.splits_batch_size,
                exclude_datasets=args.exclude_ds  # <--- PASS ARG
            )
        elif not (yolo_ds_root / "data.yaml").exists():
            print(f"[INFO] Building Dataset from Raw Folders (Standard)")
            temp_runner = YOLORunner(
                data_root=env.data_root, out_dir=yolo_ds_root, run_dir=env.run_dir,
                cfg=env.cfg, model=str(env.model), device="cpu",
                epochs=1, batch=1, imgsz=640, conf=0.1, iou=0.1,
                exclude_datasets=args.exclude_ds  # <--- PASS ARG
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
            yolo_ds=yolo_ds_root,
            exclude_datasets=args.exclude_ds  # <--- PASS ARG (Just for consistency, mostly used in ensure_data)
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
                run_name=run_name, yolo_ds=yolo_ds_root,
                exclude_datasets=args.exclude_ds # <--- PASS ARG
            )
            runner.test(test_weights)

    # 6. Finetune MedSAM
    if do_finetune:
        fundu_weights = trained_weights or args.yolo_weights
        if not fundu_weights:
            print("[WARN] Skipping MedSAM finetune: no YOLO weights found.")
        else:
            print(f"[INFO] Finetuning MedSAM using prompts from {fundu_weights}")

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
                exclude_datasets=args.exclude_ds,  # <--- PASSED HERE
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


if __name__ == "__main__":
    main()