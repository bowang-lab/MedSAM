#!/usr/bin/env python3
# File: src/experiment.py

from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

from src.model.yolo import (
    YOLORunner,
    set_global_seed,
    finetune_yolo_from_parquet,
)
from src.scripts.finetune_medsam import (
    TrainConfig as MedSAMTrainConfig,
    run_finetune as run_medsam_finetune,
)
from src.scripts.predict_summarize import (
    PredictConfig,
    run_predictions,
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
        exclude_datasets: list[str] | None = None,
) -> None:
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
    parser = argparse.ArgumentParser(
        description="Unified MedSAM/YOLO Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Environment
    parser.add_argument("--local", action="store_true", help="Use local paths")

    # Dataset Configuration
    parser.add_argument("--yolo-ds", type=Path, default=None, help="Path to existing YOLO dataset")
    parser.add_argument("--yolo-out-dir", type=Path, default=None, help="Destination for new YOLO dataset")
    parser.add_argument("--splits-parquet", type=Path, default=None, help="Source Parquet file")
    parser.add_argument("--splits-batch-size", type=int, default=2048)
    parser.add_argument("--exclude-ds", nargs="*", default=None, help="Datasets to exclude")

    # Split Ratios
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)

    # Model / Training
    parser.add_argument("--yolo-weights", type=Path, default=None, help="Weights for testing/finetuning")
    parser.add_argument("--medsam-ckpt", type=Path, default=None, help="MedSAM checkpoint")
    parser.add_argument("--yolo-device", type=str, default=None, help="Override device")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None, help="Image size")
    parser.add_argument("--workers", type=int, default=None, help="Number of data workers")

    # MedSAM Finetuning Performance Flags (ADDED)
    parser.add_argument("--grad-checkpointing", action="store_true", help="Enable gradient checkpointing (saves VRAM)")
    parser.add_argument("--eval-every", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--compile-model", action="store_true", help="Compile model with torch.compile")

    # YOLO Finetuning Specific Arguments
    parser.add_argument("--init-weights", type=Path, default=None)
    parser.add_argument("--images-parquet", type=Path, default=None)
    parser.add_argument("--skip-materialization", action="store_true")
    parser.add_argument("--finetune-runs-root", type=Path, default=None)

    # Prediction Arguments
    parser.add_argument("--predict-parquet", type=Path, default=None)
    parser.add_argument("--predict-dir", type=Path, default=None)
    parser.add_argument("--predict-out-dir", type=Path, default=None)
    parser.add_argument("--predict-conf", type=float, default=0.001)
    parser.add_argument("--predict-batch-size", type=int, default=128)
    parser.add_argument("--predict-resume", action="store_true")
    parser.add_argument("--predict-splits", type=str, default=None)
    parser.add_argument("--predict-sam-amp", action="store_true", default=True)
    parser.add_argument("--predict-no-sam-amp", action="store_true")

    # Actions
    parser.add_argument("--create-ds", action="store_true")
    parser.add_argument("--train-yolo", action="store_true")
    parser.add_argument("--finetune-yolo", action="store_true")
    parser.add_argument("--test-yolo", action="store_true")
    parser.add_argument("--finetune-medsam", action="store_true")
    parser.add_argument("--predict", action="store_true")

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
    if args.imgsz: cfg = dataclasses.replace(cfg, imgsz=args.imgsz)
    if args.workers is not None: cfg = dataclasses.replace(cfg, workers=args.workers)

    do_create = args.create_ds
    do_train = args.train_yolo
    do_finetune_yolo = args.finetune_yolo
    do_test = args.test_yolo
    do_finetune_medsam = args.finetune_medsam
    do_predict = args.predict

    # Default behavior if no actions specified
    if not any([do_create, do_train, do_finetune_yolo, do_test, do_finetune_medsam, do_predict]):
        do_create = True
        do_train = cfg.do_train
        do_test = cfg.chain_test_after_train
        do_finetune_medsam = cfg.do_finetune

    # 2. Resolve Paths
    yolo_ds_root = args.yolo_ds or args.yolo_out_dir or env.out_dir
    # For yolo_device, if a comma-separated list is provided (e.g., "0,1,2,3"), 
    # extract the first device for YOLO inference (which runs on rank-0 only)
    if args.yolo_device:
        # Handle comma-separated device strings (take first device)
        effective_device = args.yolo_device.split(",")[0].strip()
    else:
        effective_device = env.device
    effective_medsam_ckpt = args.medsam_ckpt or env.medsam_ckpt

    i = 1
    while (env.run_dir / f"run{i}").exists(): i += 1
    run_name = f"run{i}"
    run_root = env.run_dir / run_name

    trained_weights = None
    finetuned_weights = None

    # 3. Prepare Dataset
    if do_create or do_train:
        if args.splits_parquet:
            print(f"[INFO] Building Dataset from Parquet: {args.splits_parquet}")
            ensure_yolo_dataset_from_splits_parquet(
                splits_parquet=args.splits_parquet,
                yolo_ds_root=yolo_ds_root,
                batch_size=args.splits_batch_size,
                exclude_datasets=args.exclude_ds,
            )
        elif not (yolo_ds_root / "data.yaml").exists():
            print(f"[INFO] Building Dataset from Raw Folders (Standard)")
            temp_runner = YOLORunner(
                data_root=env.data_root, out_dir=yolo_ds_root, run_dir=env.run_dir,
                cfg=env.cfg, model=str(env.model), device="cpu",
                epochs=1, batch=1, imgsz=640, conf=0.1, iou=0.1,
                exclude_datasets=args.exclude_ds,
            )
            temp_runner.ensure_data(args.train_ratio, args.val_ratio, args.test_ratio)

    # 4. Train YOLO
    if do_train:
        print(f"[INFO] Training YOLO on {yolo_ds_root}")
        runner = YOLORunner(
            data_root=env.data_root, out_dir=yolo_ds_root, run_dir=env.run_dir,
            cfg=env.cfg, model=str(env.model), device=effective_device,
            epochs=cfg.epochs, batch=cfg.batch, imgsz=cfg.imgsz,
            conf=cfg.conf, iou=cfg.iou,
            run_name=run_name,
            yolo_ds=yolo_ds_root,
            exclude_datasets=args.exclude_ds,
        )
        trained_weights = runner.train()

    # 5. Finetune YOLO
    if do_finetune_yolo:
        finetune_runs_root = args.finetune_runs_root or (env.run_dir / "finetune")
        finetune_ds_root = args.yolo_out_dir or yolo_ds_root

        print(f"[INFO] Finetuning YOLO from {args.init_weights}")

        finetuned_weights = finetune_yolo_from_parquet(
            images_parquet=args.images_parquet,
            out_yolo_ds=finetune_ds_root,
            runs_root=finetune_runs_root,
            init_weights=args.init_weights,
            cfg=env.cfg,
            device=effective_device,
            epochs=cfg.epochs,
            batch=cfg.batch,
            imgsz=cfg.imgsz,
            workers=cfg.workers,
            seed=SEED,
            skip_materialization=args.skip_materialization,
            run_name_base="ss",
        )
        yolo_ds_root = finetune_ds_root

    # 6. Test YOLO
    if do_test:
        test_weights = finetuned_weights or trained_weights or args.yolo_weights
        if test_weights:
            print(f"[INFO] Testing YOLO with {test_weights}")
            runner = YOLORunner(
                data_root=env.data_root, out_dir=yolo_ds_root, run_dir=env.run_dir,
                cfg=env.cfg, model=str(env.model), device=effective_device,
                epochs=1, batch=cfg.batch, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou,
                run_name=run_name, yolo_ds=yolo_ds_root,
                exclude_datasets=args.exclude_ds,
            )
            runner.test(test_weights)

    # 7. Finetune MedSAM
    if do_finetune_medsam:
        fundu_weights = finetuned_weights or trained_weights or args.yolo_weights
        if not fundu_weights:
            print("[WARN] Skipping MedSAM finetune: no YOLO weights found.")
        else:
            print(f"[INFO] Finetuning MedSAM using prompts from {fundu_weights}")

            # Use args.yolo_out_dir if provided for explicit output location
            if args.yolo_out_dir:
                fundu_root = args.yolo_out_dir
            elif trained_weights or finetuned_weights:
                fundu_root = run_root / "Fundu"
            else:
                fundu_root = env.run_dir / "Fundu_Inference"

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
                exclude_datasets=args.exclude_ds,
                yolo_device=effective_device,
                yolo_imgsz=cfg.imgsz,
                yolo_conf=cfg.conf,
                yolo_iou=cfg.iou,
                ckpt=effective_medsam_ckpt,
                epochs=cfg.epochs,
                # For MedSAM, use batch size as-is (already set appropriately for multi-GPU)
                # Only divide by 2 if batch is very large (>16) to save memory
                batch=cfg.batch if cfg.batch <= 16 else max(1, cfg.batch // 2),
                workers=cfg.workers,
                do_train=True, do_test=True, test_prompt="det",
                seed=SEED,
                # Pass new performance args
                grad_checkpointing=args.grad_checkpointing,
                eval_every=args.eval_every,
                compile_model=args.compile_model
            )
            run_medsam_finetune(med_cfg)

    # 8. Run Predictions
    if do_predict:
        predict_weights = finetuned_weights or trained_weights or args.yolo_weights
        if predict_weights:
            predict_input_parquet = args.predict_parquet
            predict_input_dir = args.predict_dir

            if not predict_input_parquet and not predict_input_dir:
                yolo_ds_images_parquet = yolo_ds_root / "saved_images.parquet"
                if yolo_ds_images_parquet.exists():
                    predict_input_parquet = yolo_ds_images_parquet

            if predict_input_parquet or predict_input_dir:
                predict_out = args.predict_out_dir
                if not predict_out:
                    if trained_weights or finetuned_weights:
                        predict_out = run_root / "predictions"
                    else:
                        predict_out = env.run_dir / "predictions_standalone"

                sam_amp = True
                if args.predict_no_sam_amp:
                    sam_amp = False
                elif args.predict_sam_amp:
                    sam_amp = True

                pred_config = PredictConfig(
                    out_dir=predict_out,
                    yolo_weights=predict_weights,
                    medsam_checkpoint=effective_medsam_ckpt,
                    images_parquet=predict_input_parquet,
                    images_dir=predict_input_dir,
                    device=effective_device,
                    conf=args.predict_conf,
                    iou=cfg.iou,
                    imgsz=cfg.imgsz,
                    sam_amp=sam_amp,
                    predict_batch_size=args.predict_batch_size,
                    resume=args.predict_resume,
                    splits=args.predict_splits,
                )
                run_predictions(pred_config)

    print("[INFO] Experiment completed.")


if __name__ == "__main__":
    main()