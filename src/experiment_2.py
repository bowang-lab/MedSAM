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

from src.imgpipe.yolo_splits import create_yolo_dataset_from_parquet

_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "8")))
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(_cpus))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_cpus))

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
    out_dir: Path  # default YOLO dataset root (contains data.yaml if created here)
    run_dir: Path  # YOLO runs root
    cfg: Path  # YOLO train config
    model: Path  # YOLO base model weights
    device: str  # default YOLO device string (e.g. "mps", "0"); can be overridden by --yolo-device

    # MedSAM / Fundu finetuning
    medsam_ckpt: Path  # MedSAM checkpoint (.pth)


@dataclass(frozen=True)
class RunConfig:
    """Non-path parameters for the pipeline."""
    epochs: int
    batch: int
    workers: int
    imgsz: int
    conf: float
    iou: float
    do_train: bool = True                 # default YOLO training
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
    data_root=Path("/Volumes/ResearchUSB/All_Datasets_Organized"),
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
    do_finetune=True,
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
    model=Path("/home/cperez67/medsam/runs/run12/Train/Train/weights"),
    device="0",  # default; overridden by --yolo-device from Slurm script
    medsam_ckpt=HPC_REPO_DIR / "weights/medsam_vit_b.pth",
)

HPC_RUN = RunConfig(
    epochs=20,
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
    parser = argparse.ArgumentParser(
        description="Thin wrapper to run YOLO and/or Fundu/MedSAM with local or HPC presets, in modular stages."
    )
    parser.add_argument("--local", action="store_true", help="Use local environment paths instead of HPC paths.")
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

    # Build YOLO dataset from an Image parquet that already has split assigned
    parser.add_argument(
        "--splits-parquet",
        type=Path,
        default=None,
        help=(
            "Parquet of Image records with split already assigned (train/val/test). "
            "If provided, this script will create a YOLO dataset directory using that split assignment."
        ),
    )
    parser.add_argument(
        "--splits-batch-size",
        type=int,
        default=2048,
        help="Streaming batch size used when building a YOLO dataset from --splits-parquet.",
    )

    # NEW: filter parquet rows to require BOTH disc + cup GT boxes
    parser.add_argument(
        "--require-both-gt-boxes",
        dest="require_both_gt_boxes",
        action="store_true",
        default=True,
        help="(default) When using --splits-parquet, drop rows missing gt_disc_box or gt_cup_box (or invalid/zero-area).",
    )
    parser.add_argument(
        "--allow-missing-gt-boxes",
        dest="require_both_gt_boxes",
        action="store_false",
        help="When using --splits-parquet, do NOT drop rows missing GT disc/cup boxes.",
    )

    # Stage controls
    parser.add_argument("--create-ds", action="store_true", help="Create/ensure YOLO dataset (data.yaml).")
    parser.add_argument("--train-yolo", action="store_true", help="Train YOLO on the dataset.")
    parser.add_argument("--test-yolo", action="store_true", help="Run YOLO evaluation on the test set.")
    parser.add_argument("--finetune-medsam", action="store_true", help="Run Fundu/MedSAM finetuning using YOLO boxes.")
    parser.add_argument("--yolo-device", type=str, default=None, help='Override YOLO device string, e.g. "0,1,2,3"')

    # Split ratios (used only in the non-parquet dataset creation mode)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)

    return parser.parse_args()


# =========================
# Small helpers
# =========================
def get_next_run_name(run_root: Path, base: str = "run") -> str:
    i = 1
    while (run_root / f"{base}{i}").exists():
        i += 1
    return f"{base}{i}"


def select_environment(args: argparse.Namespace) -> tuple[EnvPaths, RunConfig]:
    if args.local:
        print("[INFO] Using LOCAL environment configuration.")
        return LOCAL_ENV, LOCAL_RUN
    print("[INFO] Using HPC environment configuration.")
    return HPC_ENV, HPC_RUN


def resolve_stages(args: argparse.Namespace, run_cfg: RunConfig) -> tuple[bool, bool, bool, bool]:
    any_stage_flag = any([args.create_ds, args.train_yolo, args.test_yolo, args.finetune_medsam])

    if any_stage_flag:
        do_create_ds = args.create_ds
        do_train_yolo = args.train_yolo
        do_test_yolo = args.test_yolo
        do_finetune = args.finetune_medsam
    else:
        do_train_yolo = run_cfg.do_train
        do_finetune = run_cfg.do_finetune
        do_create_ds = True
        do_test_yolo = run_cfg.chain_test_after_train and do_train_yolo

    return do_create_ds, do_train_yolo, do_test_yolo, do_finetune


def prepare_run_root(run_dir: Path) -> tuple[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_name = get_next_run_name(run_dir, base="run")
    run_root = run_dir / run_name
    return run_name, run_root


def resolve_yolo_ds_root(args: argparse.Namespace, env: EnvPaths) -> Path:
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
    effective_yolo_device: str,
) -> None:
    print(f"[INFO] DATA_ROOT       = {env.data_root}")
    print(f"[INFO] YOLO_OUT_DIR    = {env.out_dir}")
    print(f"[INFO] YOLO_RUN_DIR    = {env.run_dir}")
    print(f"[INFO] YOLO_RUN_NAME   = {run_name}  (run root: {run_root})")
    print(f"[INFO] YOLO_CFG        = {env.cfg}")
    print(f"[INFO] YOLO_MODEL      = {env.model}")
    print(f"[INFO] YOLO_DEVICE     = {effective_yolo_device}")
    print(f"[INFO] YOLO_DS_ROOT    = {yolo_ds_root}")

    if args.splits_parquet is not None:
        print(f"[INFO] SPLITS_PARQUET  = {args.splits_parquet}")
        print(f"[INFO] SPLITS_BATCH    = {int(args.splits_batch_size)}")
        print(f"[INFO] REQ_BOTH_GT_BOX  = {bool(args.require_both_gt_boxes)}")
    else:
        print(
            f"[INFO] SPLIT RATIOS    = train={args.train_ratio}, "
            f"val={args.val_ratio}, test={args.test_ratio}"
        )

    print(f"[INFO] EPOCHS          = {run_cfg.epochs}")
    print(f"[INFO] BATCH           = {run_cfg.batch}")
    print(f"[INFO] WORKERS         = {run_cfg.workers}")
    print(f"[INFO] IMGSZ           = {run_cfg.imgsz}")
    print(f"[INFO] CONF/IOU        = {run_cfg.conf} / {run_cfg.iou}")
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
        print(f"[INFO] YOLO_WEIGHTS    = {args.yolo_weights}")


def build_yolo_runner(
    env: EnvPaths,
    run_cfg: RunConfig,
    args: argparse.Namespace,
    yolo_ds_root: Path,
    run_name: str,
    effective_yolo_device: str,
    *,
    yolo_ds_override: Path | None,
) -> YOLORunner:
    return YOLORunner(
        data_root=env.data_root,
        out_dir=yolo_ds_root,
        run_dir=env.run_dir,
        cfg=env.cfg,
        model=str(env.model),
        device=effective_yolo_device,
        epochs=run_cfg.epochs,
        batch=run_cfg.batch,
        imgsz=run_cfg.imgsz,
        conf=run_cfg.conf,
        iou=run_cfg.iou,
        yolo_ds=yolo_ds_override,
        run_name=run_name,
        resume=False,
        seed=SEED,
        single_top_per_class=True,
        papila=args.papila,
    )


# =========================
# NEW: parquet-level filtering for required cup+disc boxes
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
    Creates out_parquet with only rows having BOTH gt_disc_box & gt_cup_box
    AND BOTH gt_disc_mask & gt_cup_mask that are NOT "empty refs".

    Operates at Arrow level to preserve nested structs/bytes exactly.

    Returns: (kept, dropped, total)
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    in_parquet = Path(in_parquet)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    dataset = ds.dataset(str(in_parquet), format="parquet")
    schema = dataset.schema

    need_cols = (disc_box_col, cup_box_col, disc_mask_col, cup_mask_col)
    missing = [c for c in need_cols if c not in schema.names]
    if missing:
        raise ValueError(
            f"Cannot filter parquet; missing columns: {missing}. "
            f"Available columns include: {schema.names[:40]}{'...' if len(schema.names) > 40 else ''}"
        )

    def _box_ok_mask(rb: pa.RecordBatch, colname: str) -> pa.Array:
        col = rb.column(schema.get_field_index(colname))
        ok = pc.is_valid(col)

        if pa.types.is_struct(col.type):
            child_names = {f.name for f in col.type}
            # Try common box representations
            if {"x1", "y1", "x2", "y2"}.issubset(child_names):
                x1 = pc.struct_field(col, "x1")
                y1 = pc.struct_field(col, "y1")
                x2 = pc.struct_field(col, "x2")
                y2 = pc.struct_field(col, "y2")
                ok = pc.and_(ok, pc.and_(pc.is_valid(x1), pc.is_valid(y1)))
                ok = pc.and_(ok, pc.and_(pc.is_valid(x2), pc.is_valid(y2)))
                ok = pc.and_(ok, pc.and_(pc.greater(x2, x1), pc.greater(y2, y1)))
            elif {"cx", "cy", "w", "h"}.issubset(child_names):
                w = pc.struct_field(col, "w")
                h = pc.struct_field(col, "h")
                ok = pc.and_(ok, pc.and_(pc.is_valid(w), pc.is_valid(h)))
                ok = pc.and_(ok, pc.and_(pc.greater(w, 0), pc.greater(h, 0)))

        return ok

    def _mask_ref_nonempty(rb: pa.RecordBatch, colname: str) -> pa.Array:
        """
        True if mask ref exists AND has any usable backing:
          - packed/bytes field is valid, OR
          - path field is valid AND non-empty, OR
          - array/arr field is valid
        Field names vary; we probe what exists in the schema.
        """
        col = rb.column(schema.get_field_index(colname))
        ok = pc.is_valid(col)

        if not pa.types.is_struct(col.type):
            # If it's something else (rare), just require non-null
            return ok

        child_names = {f.name for f in col.type}

        # Common possibilities across versions
        candidates = []
        for nm in ("packed", "bytes", "data", "buf"):
            if nm in child_names:
                candidates.append(("valid", nm))
        for nm in ("path", "mask_path", "file", "filepath"):
            if nm in child_names:
                candidates.append(("path", nm))
        for nm in ("array", "arr", "mask", "ndarray"):
            if nm in child_names:
                candidates.append(("valid", nm))

        if not candidates:
            # Unknown struct layout; safest is just require non-null struct
            return ok

        any_backing = None
        for kind, nm in candidates:
            fld = pc.struct_field(col, nm)
            if kind == "path":
                # valid AND non-empty string
                cond = pc.and_(pc.is_valid(fld), pc.greater(pc.utf8_length(fld), 0))
            else:
                cond = pc.is_valid(fld)
            any_backing = cond if any_backing is None else pc.or_(any_backing, cond)

        return pc.and_(ok, any_backing)

    scanner = dataset.scanner(batch_size=int(batch_size))
    writer: pq.ParquetWriter | None = None

    total = kept = dropped = 0

    try:
        for rb in scanner.to_batches():
            n = rb.num_rows
            if n == 0:
                continue

            total += n

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

        if writer is None:
            pq.write_table(pa.Table.from_batches([], schema=schema), str(out_parquet))
    finally:
        if writer is not None:
            writer.close()

    return kept, dropped, total


def ensure_yolo_dataset_from_splits_parquet(
    splits_parquet: Path,
    yolo_ds_root: Path,
    *,
    batch_size: int,
    require_both_gt_boxes: bool,
) -> Path:
    splits_parquet = Path(splits_parquet)
    if not splits_parquet.exists():
        raise FileNotFoundError(f"--splits-parquet does not exist: {splits_parquet}")

    data_yaml = yolo_ds_root / "data.yaml"
    if data_yaml.exists():
        print(f"[INFO] Found existing YOLO dataset at: {yolo_ds_root} (data.yaml present).")
        return yolo_ds_root

    src_parquet = splits_parquet
    if require_both_gt_boxes:
        filtered_parquet = yolo_ds_root / "_splits_filtered_require_both_gt_boxes.parquet"
        if not filtered_parquet.exists():
            print(f"[INFO] Filtering splits parquet to require both GT boxes -> {filtered_parquet}")
            kept, dropped, total = _filter_parquet_require_both_gt_boxes(
                in_parquet=splits_parquet,
                out_parquet=filtered_parquet,
                batch_size=int(batch_size),
            )
            print(f"[INFO] Filter summary: total={total} kept={kept} dropped={dropped}")
        else:
            print(f"[INFO] Using existing filtered parquet: {filtered_parquet}")
        src_parquet = filtered_parquet
    else:
        print("[INFO] Not filtering parquet for GT box completeness (--allow-missing-gt-boxes set).")

    print(f"[INFO] Creating YOLO dataset from existing splits parquet -> {yolo_ds_root}")
    print(f"[INFO] Source parquet used for dataset creation: {src_parquet}")
    create_yolo_dataset_from_parquet(
        src_parquet,
        out_dir=yolo_ds_root,
        batch_size=int(batch_size),
        use_gt=True,
        save_images_parquet=True,
    )
    if not data_yaml.exists():
        raise RuntimeError(f"Expected data.yaml to be created at: {data_yaml}")
    return yolo_ds_root


def maybe_ensure_yolo_dataset(
    runner: YOLORunner,
    do_create_ds: bool,
    do_train_yolo: bool,
    do_test_yolo: bool,
    do_finetune: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    splits_parquet: Path | None,
    splits_batch_size: int,
    yolo_ds_root: Path,
    require_both_gt_boxes: bool,
) -> None:
    needs_ds = do_create_ds or do_train_yolo or do_test_yolo or do_finetune
    if not needs_ds:
        print("[INFO] No stages require a YOLO dataset; skipping dataset creation.")
        return

    if splits_parquet is not None:
        ensure_yolo_dataset_from_splits_parquet(
            splits_parquet=splits_parquet,
            yolo_ds_root=yolo_ds_root,
            batch_size=int(splits_batch_size),
            require_both_gt_boxes=require_both_gt_boxes,
        )
        print(f"[INFO] YOLO data.yaml  = {yolo_ds_root / 'data.yaml'}")
        return

    total = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total <= 1.01):
        print(f"[WARN] Split ratios sum to {total:.3f}, not 1.0. Proceeding anyway.")
    data_yaml = runner.ensure_data(train_ratio, val_ratio, test_ratio)
    print(f"[INFO] YOLO data.yaml  = {data_yaml}")


def maybe_train_yolo(runner: YOLORunner, do_train_yolo: bool, yolo_weights_arg: Path | None) -> Path | None:
    if not do_train_yolo:
        print("[INFO] YOLO training disabled for this run.")
        return None

    if yolo_weights_arg is not None:
        print(
            "[WARN] Both --train-yolo and --yolo-weights provided. "
            "Training will run, and freshly trained weights will be used for downstream stages."
        )
    return runner.train()


def choose_yolo_test_weights(trained_weights: Path | None, yolo_weights_arg: Path | None) -> Path | None:
    if trained_weights is not None:
        print(f"[INFO] Testing YOLO with freshly trained weights: {trained_weights}")
        return trained_weights

    if yolo_weights_arg is not None:
        if not yolo_weights_arg.exists():
            raise FileNotFoundError(f"--yolo-weights does not exist: {yolo_weights_arg}")
        print(f"[INFO] Testing YOLO with provided weights: {yolo_weights_arg}")
        return yolo_weights_arg

    print("[WARN] --test-yolo set, but neither trained weights nor --yolo-weights are available. Skipping YOLO test.")
    return None


def maybe_test_yolo(
    runner: YOLORunner,
    do_test_yolo: bool,
    trained_weights: Path | None,
    yolo_weights_arg: Path | None,
    run_root: Path,
) -> None:
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


def choose_fundu_yolo_weights(trained_weights: Path | None, yolo_weights_arg: Path | None) -> Path | None:
    if trained_weights is not None:
        print(f"[INFO] MedSAM will use freshly trained YOLO weights: {trained_weights}")
        return trained_weights

    if yolo_weights_arg is not None:
        if not yolo_weights_arg.exists():
            raise FileNotFoundError(f"--yolo-weights does not exist: {yolo_weights_arg}")
        print(f"[INFO] MedSAM will use provided YOLO weights: {yolo_weights_arg}")
        return yolo_weights_arg

    print("[WARN] --finetune-medsam set, but no YOLO weights are available. Enable --train-yolo or pass --yolo-weights.")
    return None


def infer_yolo_run_root(trained_weights: Path | None, fundu_yolo_weights: Path, run_root: Path) -> Path:
    if trained_weights is not None:
        return run_root

    p = fundu_yolo_weights.resolve()
    try:
        inferred_root = p.parents[3]
    except IndexError:
        inferred_root = p.parent.parent
    print(f"[INFO] Inferred YOLO run root for MedSAM: {inferred_root}")
    return inferred_root


def prepare_fundu_paths(yolo_run_root: Path) -> tuple[str, Path, Path]:
    fundu_root = yolo_run_root / "Fundu"
    fundu_root.mkdir(parents=True, exist_ok=True)

    fundu_run_name = get_next_run_name(fundu_root, base="Fundu")
    fundu_run_dir = fundu_root
    fundu_out_dir = fundu_root / fundu_run_name

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
    yolo_device: str,
) -> MedSAMTrainConfig:
    return MedSAMTrainConfig(
        data_root=env.data_root,
        yolo_ds=yolo_ds_root,
        out_dir=fundu_out_dir,
        run_dir=fundu_run_dir,
        run_name=fundu_run_name,
        yolo_weights=fundu_yolo_weights,
        exclude_datasets=None,
        yolo_device=yolo_device,
        yolo_imgsz=run_cfg.imgsz,
        yolo_conf=run_cfg.conf,
        yolo_iou=run_cfg.iou,
        det_cache=None,
        ckpt=env.medsam_ckpt,
        epochs=run_cfg.epochs,
        batch=max(1, run_cfg.batch // 2),
        workers=0 if args.local else run_cfg.workers,
        seed=SEED,
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
    effective_yolo_device: str,
) -> None:
    if not do_finetune:
        print("[INFO] MedSAM finetuning disabled for this run.")
        return

    fundu_yolo_weights = choose_fundu_yolo_weights(trained_weights, yolo_weights_arg)
    if fundu_yolo_weights is None:
        return

    yolo_run_root = infer_yolo_run_root(trained_weights, fundu_yolo_weights, run_root)
    fundu_run_name, fundu_run_dir, fundu_out_dir = prepare_fundu_paths(yolo_run_root)

    print(f"[INFO] Starting Fundu/MedSAM finetuning with run name: {fundu_run_name}")
    print(f"[INFO] Fundu YOLO weights = {fundu_yolo_weights}")
    print(f"[INFO] Fundu YOLO dataset = {yolo_ds_root}")

    med_cfg = build_medsam_config(
        env=env,
        run_cfg=run_cfg,
        fundu_run_name=fundu_run_name,
        fundu_run_dir=fundu_run_dir,
        fundu_out_dir=fundu_out_dir,
        yolo_ds_root=yolo_ds_root,
        fundu_yolo_weights=fundu_yolo_weights,
        args=args,
        yolo_device=effective_yolo_device,
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

    if args.splits_parquet is not None and args.yolo_ds is not None:
        raise ValueError("Use either --splits-parquet OR --yolo-ds, not both.")

    env, run_cfg = select_environment(args)
    do_create_ds, do_train_yolo, do_test_yolo, do_finetune = resolve_stages(args, run_cfg)

    effective_yolo_device = args.yolo_device if args.yolo_device is not None else env.device

    run_name, run_root = prepare_run_root(env.run_dir)
    yolo_ds_root = resolve_yolo_ds_root(args, env)

    yolo_ds_override: Path | None = None
    if args.splits_parquet is not None:
        yolo_ds_override = yolo_ds_root
    elif args.yolo_ds is not None:
        yolo_ds_override = args.yolo_ds

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
        effective_yolo_device,
    )

    runner = build_yolo_runner(
        env,
        run_cfg,
        args,
        yolo_ds_root,
        run_name,
        effective_yolo_device,
        yolo_ds_override=yolo_ds_override,
    )

    maybe_ensure_yolo_dataset(
        runner,
        do_create_ds,
        do_train_yolo,
        do_test_yolo,
        do_finetune,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        splits_parquet=args.splits_parquet,
        splits_batch_size=int(args.splits_batch_size),
        yolo_ds_root=yolo_ds_root,
        require_both_gt_boxes=bool(args.require_both_gt_boxes),
    )

    trained_weights = maybe_train_yolo(runner, do_train_yolo, args.yolo_weights)
    maybe_test_yolo(runner, do_test_yolo, trained_weights, args.yolo_weights, run_root)

    maybe_finetune_medsam(
        do_finetune=do_finetune,
        env=env,
        run_cfg=run_cfg,
        yolo_ds_root=yolo_ds_root,
        trained_weights=trained_weights,
        yolo_weights_arg=args.yolo_weights,
        run_root=run_root,
        args=args,
        effective_yolo_device=effective_yolo_device,
    )


if __name__ == "__main__":
    main()