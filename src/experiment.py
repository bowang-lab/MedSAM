#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

SEED = 42

# Default split ratios (can be overridden via CLI)
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1


# =========================
# Environment configs
# =========================
@dataclass(frozen=True)
class EnvPaths:
    # YOLO + dataset
    data_root: Path
    out_dir: Path        # default YOLO dataset root (contains data.yaml if created here)
    run_dir: Path        # YOLO runs root
    cfg: Path            # YOLO train config
    model: Path          # YOLO base model weights
    device: str          # YOLO device string (e.g. "mps", "0,1")

    # MedSAM / Fundu finetuning
    medsam_ckpt: Path    # MedSAM checkpoint (.pth)


@dataclass(frozen=True)
class RunConfig:
    """Non-path parameters for the pipeline."""
    epochs: int
    batch: int
    workers: int
    imgsz: int
    conf: float
    iou: float
    do_train: bool = True                  # default YOLO training
    do_finetune: bool = True              # default Fundu/MedSAM finetuning
    chain_test_after_train: bool = True   # default YOLO test after YOLO train


# Shared YOLO hyperparameters (single source for defaults)
BASE_YOLO_HP = dict(
    imgsz=640,
    conf=0.01,
    iou=0.70,
)

# -------------------------
# LOCAL ENV (your laptop)
# -------------------------
LOCAL_REPO_DIR = Path("/Users/carlosperez/PycharmProjects/MedSAM")

LOCAL_ENV = EnvPaths(
    data_root=Path(
        "/Users/carlosperez/Library/CloudStorage/OneDrive-UBC/"
        "Ipek_Carlos/GlaucomaDatasets/All_Datasets_Organized"
    ),
    out_dir=LOCAL_REPO_DIR / "YOLO_DS_P",
    run_dir=LOCAL_REPO_DIR / "runs",
    cfg=LOCAL_REPO_DIR / "src/configs/train_custom.yaml",
    model=LOCAL_REPO_DIR / "work_dir/YOLO/yolo12n.pt",
    device="mps",
    medsam_ckpt=LOCAL_REPO_DIR / "work_dir/MedSAM/medsam_vit_b.pth",
)

LOCAL_RUN = RunConfig(
    epochs=1,
    batch=16,
    workers=0,
    **BASE_YOLO_HP,
    do_train=False,
    do_finetune=False,
    chain_test_after_train=False,
)

# -------------------------
# HPC ENV (Sockeye etc.)
# -------------------------
HPC_REPO_DIR = Path("/scratch/st-ipor-1/cperez/MedSAM")

HPC_ENV = EnvPaths(
    data_root=Path("/arc/project/st-ipor-1/carlosp/fundus_data/All_Datasets_Organized"),
    out_dir=HPC_REPO_DIR / "YOLO_DS_P",
    run_dir=HPC_REPO_DIR / "runs",
    cfg=HPC_REPO_DIR / "src/configs/train_custom.yaml",
    model=HPC_REPO_DIR / "weights/yolo12x.pt",
    device="0",  # multi-GPU on Sockeye
    medsam_ckpt=HPC_REPO_DIR / "weights/medsam_vit_b.pth",
)

HPC_RUN = RunConfig(
    epochs=400,
    batch=16,
    workers=6,
    **BASE_YOLO_HP,
    do_train=True,
    do_finetune=False,
    chain_test_after_train=True,
)


# =========================
# Arg parsing
# =========================
def parse_args() -> argparse.Namespace:
    """
    User-facing flags:

      --local           -> use LOCAL env configs (default: HPC)
      --yolo-ds         -> path to pre-made YOLO dataset (data.yaml root)
      --yolo-weights    -> path to YOLO weights (.pt) for test/finetune

      Stage flags (optional; if none are given, defaults from RunConfig are used):
      --create-ds       -> create/ensure YOLO dataset
      --train-yolo      -> train YOLO
      --test-yolo       -> test YOLO
      --finetune-medsam -> finetune MedSAM/Fundu

      Split ratios:
      --train-ratio     -> train split ratio (default 0.7)
      --val-ratio       -> validation split ratio (default 0.15)
      --test-ratio      -> test split ratio (default 0.15)
    """
    parser = argparse.ArgumentParser(
        description="Thin wrapper to run YOLO and/or Fundu/MedSAM with local or HPC presets, in modular stages."
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local environment paths instead of HPC paths.",
    )
    parser.add_argument(
        "--papila",
        type=float,
        help="Use papila in training ds. 0 = none, 1 = all, (0,1) is fraction",
    )
    parser.add_argument(
        "--yolo-ds",
        type=Path,
        default=None,
        help="Optional path to pre-made YOLO dataset root (contains data.yaml).",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=None,
        help=(
            "Optional path to pre-trained YOLO weights (.pt). "
            "Used for YOLO testing and MedSAM finetuning when YOLO is not trained in this run."
        ),
    )
    parser.add_argument(
        "--yolo-out-dir",
        type=Path,
        default=None,
        help=(
            "Override default YOLO dataset output root when creating a new dataset. "
            "If omitted, use the environment default."
        ),
    )

    # Stage controls
    parser.add_argument(
        "--create-ds",
        action="store_true",
        help="Create/ensure YOLO dataset (data.yaml).",
    )
    parser.add_argument(
        "--train-yolo",
        action="store_true",
        help="Train YOLO on the dataset.",
    )
    parser.add_argument(
        "--test-yolo",
        action="store_true",
        help="Run YOLO evaluation on the test set.",
    )
    parser.add_argument(
        "--finetune-medsam",
        action="store_true",
        help="Run Fundu/MedSAM finetuning using YOLO boxes as prompts.",
    )

    # Split ratios
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Train split ratio (default {DEFAULT_TRAIN_RATIO}).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help=f"Validation split ratio (default {DEFAULT_VAL_RATIO}).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help=f"Test split ratio (default {DEFAULT_TEST_RATIO}).",
    )

    return parser.parse_args()


# =========================
# Small helpers
# =========================
def get_next_run_name(run_root: Path, base: str = "run") -> str:
    """
    Find the next available run name under run_root, e.g.:

        base1, base2, ...

    such that run_root / f"{base}{k}" does not already exist.
    """
    i = 1
    while (run_root / f"{base}{i}").exists():
        i += 1
    return f"{base}{i}"


def select_environment(args: argparse.Namespace) -> tuple[EnvPaths, RunConfig]:
    """Pick LOCAL or HPC environment + run config."""
    if args.local:
        print("[INFO] Using LOCAL environment configuration.")
        return LOCAL_ENV, LOCAL_RUN
    else:
        print("[INFO] Using HPC environment configuration.")
        return HPC_ENV, HPC_RUN


def resolve_stages(
    args: argparse.Namespace,
    run_cfg: RunConfig,
) -> tuple[bool, bool, bool, bool]:
    """
    Determine which stages to execute:
    - create_ds
    - train_yolo
    - test_yolo
    - finetune_medsam
    """
    any_stage_flag = any(
        [args.create_ds, args.train_yolo, args.test_yolo, args.finetune_medsam]
    )

    if any_stage_flag:
        do_create_ds = args.create_ds
        do_train_yolo = args.train_yolo
        do_test_yolo = args.test_yolo
        do_finetune = args.finetune_medsam
    else:
        # Default pipeline behaviour when no explicit stage flags are given
        do_train_yolo = run_cfg.do_train
        do_finetune = run_cfg.do_finetune
        do_create_ds = True  # we always ensured the dataset before
        do_test_yolo = run_cfg.chain_test_after_train and do_train_yolo

    return do_create_ds, do_train_yolo, do_test_yolo, do_finetune


def prepare_run_root(run_dir: Path) -> tuple[str, Path]:
    """Ensure the base runs directory exists and choose a unique run name."""
    run_dir.mkdir(parents=True, exist_ok=True)
    run_name = get_next_run_name(run_dir, base="run")
    run_root = run_dir / run_name
    return run_name, run_root


def resolve_yolo_ds_root(args: argparse.Namespace, env: EnvPaths) -> Path:
    """
    Decide which YOLO dataset root to use:
    - If --yolo-ds is provided, treat it as a pre-made dataset (root with data.yaml)
    - Otherwise, use --yolo-out-dir if given, else the environment default env.out_dir
    """
    if args.yolo_ds is not None:
        return args.yolo_ds
    if args.yolo_out_dir is not None:
        return args.yolo_out_dir
    return env.out_dir


def print_config(
    env: EnvPaths,
    run_name: str,
    run_root: Path,
    run_cfg: RunConfig,
    args: argparse.Namespace,
    do_create_ds: bool,
    do_train_yolo: bool,
    do_test_yolo: bool,
    do_finetune: bool,
    yolo_ds_root: Path,
) -> None:
    """Log the effective configuration."""
    print(f"[INFO] DATA_ROOT       = {env.data_root}")
    print(f"[INFO] YOLO_OUT_DIR    = {env.out_dir}")
    print(f"[INFO] YOLO_RUN_DIR    = {env.run_dir}")
    print(f"[INFO] YOLO_RUN_NAME   = {run_name}  (run root: {run_root})")
    print(f"[INFO] YOLO_CFG        = {env.cfg}")
    print(f"[INFO] YOLO_MODEL      = {env.model}")
    print(f"[INFO] YOLO_DEVICE     = {env.device}")
    print(f"[INFO] YOLO_DS_ROOT    = {yolo_ds_root}")

    print(f"[INFO] EPOCHS          = {run_cfg.epochs}")
    print(f"[INFO] BATCH           = {run_cfg.batch}")
    print(f"[INFO] WORKERS         = {run_cfg.workers}")
    print(f"[INFO] IMGSZ           = {run_cfg.imgsz}")
    print(f"[INFO] CONF/IOU        = {run_cfg.conf} / {run_cfg.iou}")
    print(f"[INFO] SPLIT RATIOS    = train={args.train_ratio}, "
          f"val={args.val_ratio}, test={args.test_ratio}")
    print(
        "[INFO] STAGES          = "
        f"create_ds={do_create_ds}, "
        f"train_yolo={do_train_yolo}, "
        f"test_yolo={do_test_yolo}, "
        f"finetune_medsam={do_finetune}"
    )
    print(f"[INFO] MEDSAM_CKPT     = {env.medsam_ckpt}")
    if args.yolo_ds is not None:
        print(f"[INFO] YOLO_DS         = {args.yolo_ds}")
    if args.yolo_weights is not None:
        print(f"[INFO] YOLO_WEIGHTS    = {args.yolo_weights} (will be used for test/finetune if YOLO not trained)")


def build_yolo_runner(
    env: EnvPaths,
    run_cfg: RunConfig,
    args: argparse.Namespace,
    yolo_ds_root: Path,
    run_name: str,
) -> YOLORunner:
    """Construct the YOLO runner with chosen environment and run parameters."""
    return YOLORunner(
        data_root=env.data_root,
        out_dir=yolo_ds_root,
        run_dir=env.run_dir,  # base runs dir; runner will append run_name
        cfg=env.cfg,
        model=str(env.model),
        device=env.device,
        epochs=run_cfg.epochs,
        batch=run_cfg.batch,
        imgsz=run_cfg.imgsz,
        conf=run_cfg.conf,
        iou=run_cfg.iou,
        yolo_ds=args.yolo_ds,  # if provided, treat as pre-made DS
        run_name=run_name,
        resume=False,
        seed=SEED,
        single_top_per_class=True,
        papila=args.papila,
    )


def maybe_ensure_yolo_dataset(
    runner: YOLORunner,
    do_create_ds: bool,
    do_train_yolo: bool,
    do_test_yolo: bool,
    do_finetune: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    """Create / ensure YOLO dataset if any stage requires it."""
    if do_create_ds or do_train_yolo or do_test_yolo or do_finetune:
        # Basic sanity check
        total = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total <= 1.01):
            print(
                f"[WARN] Split ratios sum to {total:.3f}, not 1.0. "
                "Proceeding anyway; ensure this is intentional."
            )
        data_yaml = runner.ensure_data(train_ratio, val_ratio, test_ratio)
        print(f"[INFO] YOLO data.yaml  = {data_yaml}")
    else:
        print("[INFO] No stages require a YOLO dataset; skipping dataset creation.")


def maybe_train_yolo(
    runner: YOLORunner,
    do_train_yolo: bool,
    yolo_weights_arg: Path | None,
) -> Path | None:
    """Run YOLO training if requested and return trained weights path (or None)."""
    if not do_train_yolo:
        print("[INFO] YOLO training disabled for this run.")
        return None

    if yolo_weights_arg is not None:
        print(
            "[WARN] Both --train-yolo and --yolo-weights provided. "
            "Training will run, and freshly trained weights will be used for downstream stages."
        )

    trained_weights = runner.train()
    return trained_weights


def choose_yolo_test_weights(
    trained_weights: Path | None,
    yolo_weights_arg: Path | None,
) -> Path | None:
    """Pick weights for YOLO test: prefer freshly trained, else CLI-provided."""
    if trained_weights is not None:
        print(f"[INFO] Testing YOLO with freshly trained weights: {trained_weights}")
        return trained_weights

    if yolo_weights_arg is not None:
        if not yolo_weights_arg.exists():
            raise FileNotFoundError(f"--yolo-weights does not exist: {yolo_weights_arg}")
        print(f"[INFO] Testing YOLO with provided weights: {yolo_weights_arg}")
        return yolo_weights_arg

    print(
        "[WARN] --test-yolo set, but neither trained weights nor --yolo-weights are available. "
        "Skipping YOLO test."
    )
    return None


def maybe_test_yolo(
    runner: YOLORunner,
    do_test_yolo: bool,
    trained_weights: Path | None,
    yolo_weights_arg: Path | None,
    run_root: Path,
) -> None:
    """Run YOLO test if requested."""
    if not do_test_yolo:
        print("[INFO] YOLO testing disabled for this run.")
        return

    test_weights = choose_yolo_test_weights(trained_weights, yolo_weights_arg)
    if test_weights is None:
        return

    runner.test(test_weights)
    print(f"[INFO] YOLO artifacts for this run are under: {run_root}")
    print(f"        Train: {run_root / 'Train'}")
    print(f"        Test : {run_root / 'Test'}")


def choose_fundu_yolo_weights(
    trained_weights: Path | None,
    yolo_weights_arg: Path | None,
) -> Path | None:
    """
    Choose YOLO weights for MedSAM:
      1) Prefer freshly trained YOLO weights
      2) Otherwise use --yolo-weights
    """
    if trained_weights is not None:
        print(f"[INFO] MedSAM will use freshly trained YOLO weights: {trained_weights}")
        return trained_weights

    if yolo_weights_arg is not None:
        if not yolo_weights_arg.exists():
            raise FileNotFoundError(f"--yolo-weights does not exist: {yolo_weights_arg}")
        print(f"[INFO] MedSAM will use provided YOLO weights: {yolo_weights_arg}")
        return yolo_weights_arg

    print(
        "[WARN] --finetune-medsam set, but no YOLO weights are available. "
        "Enable --train-yolo or pass --yolo-weights."
    )
    return None


def infer_yolo_run_root(
    trained_weights: Path | None,
    fundu_yolo_weights: Path,
    run_root: Path,
) -> Path:
    """
    Infer the YOLO run root directory:

      - If YOLO trained in this invocation -> reuse run_root
      - Otherwise, infer from weights path using Ultralytics layout:

            .../runs/runX/Train/Train/weights/best.pt
    """
    if trained_weights is not None:
        return run_root

    p = fundu_yolo_weights.resolve()
    try:
        # parents[0] = weights/, [1] = Train, [2] = Train, [3] = runX, [4] = runs
        inferred_root = p.parents[3]
    except IndexError:
        # Fallback: go up two levels (weights -> Train)
        inferred_root = p.parent.parent
    print(f"[INFO] Inferred YOLO run root for MedSAM: {inferred_root}")
    return inferred_root


def prepare_fundu_paths(yolo_run_root: Path) -> tuple[str, Path, Path]:
    """
    Prepare Fundu layout:

      yolo_run_root/
          Fundu/
              Fundu1/
              Fundu2/
              ...

    Returns:
      fundu_run_name, fundu_run_dir, fundu_out_dir
    """
    fundu_root = yolo_run_root / "Fundu"
    fundu_root.mkdir(parents=True, exist_ok=True)

    fundu_run_name = get_next_run_name(fundu_root, base="Fundu")
    fundu_run_dir = fundu_root         # Trainer.run_dir
    fundu_out_dir = fundu_root / fundu_run_name  # Trainer.out_dir

    print(f"[INFO] Fundu root       = {fundu_root}")
    print(f"[INFO] Fundu run_dir    = {fundu_run_dir}")
    print(f"[INFO] Fundu out_dir    = {fundu_out_dir}")

    return fundu_run_name, fundu_run_dir, fundu_out_dir


def build_medsam_config(
    env: EnvPaths,
    run_cfg: RunConfig,
    fundu_run_name: str,
    fundu_run_dir: Path,
    fundu_out_dir: Path,
    yolo_ds_root: Path,
    fundu_yolo_weights: Path,
    args: argparse.Namespace,
) -> MedSAMTrainConfig:
    """Map YOLO run configuration -> MedSAM TrainConfig."""
    return MedSAMTrainConfig(
        # Data / IO
        data_root=env.data_root,
        yolo_ds=yolo_ds_root,
        out_dir=fundu_out_dir,
        run_dir=fundu_run_dir,
        run_name=fundu_run_name,
        yolo_weights=fundu_yolo_weights,

        # Optional dataset exclusions
        exclude_datasets=None,

        # Detector prompts: reuse YOLO settings
        yolo_device=env.device,
        yolo_imgsz=run_cfg.imgsz,
        yolo_conf=run_cfg.conf,
        yolo_iou=run_cfg.iou,
        det_cache=None,

        # Model/opt
        ckpt=env.medsam_ckpt,
        epochs=run_cfg.epochs,
        batch=max(1, run_cfg.batch // 2),      # slightly smaller batch for segmentation
        workers=0 if args.local else run_cfg.workers,
        seed=SEED,

        # Modes
        do_train=True,
        do_test=True,
        test_prompt="det",
        test_weights=None,
        resume=None,
    )


def maybe_finetune_medsam(
    do_finetune: bool,
    env: EnvPaths,
    run_cfg: RunConfig,
    yolo_ds_root: Path,
    trained_weights: Path | None,
    yolo_weights_arg: Path | None,
    run_root: Path,
    args: argparse.Namespace,
) -> None:
    """Run Fundu / MedSAM finetuning if requested."""
    if not do_finetune:
        print("[INFO] MedSAM finetuning disabled for this run.")
        return

    # Choose YOLO weights to drive Fundu
    fundu_yolo_weights = choose_fundu_yolo_weights(trained_weights, yolo_weights_arg)
    if fundu_yolo_weights is None:
        return

    # Infer YOLO run root for MedSAM outputs
    yolo_run_root = infer_yolo_run_root(trained_weights, fundu_yolo_weights, run_root)

    # Prepare Fundu paths and naming
    fundu_run_name, fundu_run_dir, fundu_out_dir = prepare_fundu_paths(yolo_run_root)

    print(f"[INFO] Starting Fundu/MedSAM finetuning with run name: {fundu_run_name}")
    print(f"[INFO] Fundu YOLO weights = {fundu_yolo_weights}")
    print(f"[INFO] Fundu YOLO dataset = {yolo_ds_root}")

    # Build MedSAM config and run finetuning
    med_cfg = build_medsam_config(
        env=env,
        run_cfg=run_cfg,
        fundu_run_name=fundu_run_name,
        fundu_run_dir=fundu_run_dir,
        fundu_out_dir=fundu_out_dir,
        yolo_ds_root=yolo_ds_root,
        fundu_yolo_weights=fundu_yolo_weights,
        args=args,
    )

    med_result = run_medsam_finetune(med_cfg)
    print("[INFO] Fundu/MedSAM finetuning completed.")
    print(f"[INFO] MedSAM best val Dice: {med_result.get('best_val_dice')}")
    if "test" in med_result:
        print(f"[INFO] MedSAM test summary: {med_result['test']}")
    print(f"[INFO] Fundu/MedSAM artifacts are under: {fundu_out_dir}")


def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    set_global_seed(SEED)

    args = parse_args()
    env, run_cfg = select_environment(args)
    do_create_ds, do_train_yolo, do_test_yolo, do_finetune = resolve_stages(args, run_cfg)

    # Prepare YOLO run naming and paths
    run_name, run_root = prepare_run_root(env.run_dir)
    yolo_ds_root = resolve_yolo_ds_root(args, env)

    print_config(
        env,
        run_name,
        run_root,
        run_cfg,
        args,
        do_create_ds,
        do_train_yolo,
        do_test_yolo,
        do_finetune,
        yolo_ds_root,
    )

    # YOLO pipeline
    runner = build_yolo_runner(env, run_cfg, args, yolo_ds_root, run_name)
    maybe_ensure_yolo_dataset(
        runner,
        do_create_ds,
        do_train_yolo,
        do_test_yolo,
        do_finetune,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )

    trained_weights = maybe_train_yolo(runner, do_train_yolo, args.yolo_weights)
    maybe_test_yolo(runner, do_test_yolo, trained_weights, args.yolo_weights, run_root)

    # Fundu / MedSAM finetuning
    maybe_finetune_medsam(
        do_finetune=do_finetune,
        env=env,
        run_cfg=run_cfg,
        yolo_ds_root=yolo_ds_root,
        trained_weights=trained_weights,
        yolo_weights_arg=args.yolo_weights,
        run_root=run_root,
        args=args,
    )


if __name__ == "__main__":
    main()