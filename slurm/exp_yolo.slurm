#!/bin/bash
#SBATCH --job-name=Experiment
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --gpus=4
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

############################
# 1. Environment Setup
############################
REPO_DIR="/scratch/st-ipor-1/cperez/MedSAM"
export REPO_DIR
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

source "${REPO_DIR}/slurm/setup.sh"
setup_env

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/hpc_logs"

############################
# 2. Device Configuration
############################
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  YOLO_DEVICES="${CUDA_VISIBLE_DEVICES}"
else
  # Fallback logic
  YOLO_DEVICES="0,1,2,3"
fi
echo "YOLO_DEVICES=${YOLO_DEVICES}"

############################
# 3. Argument Selection Logic
############################
# Get the mode from command line arg 1, default to "1" if not provided
MODE="${1:-1}"

echo "========================================"
echo "RUNNING IN MODE: ${MODE}"
echo "========================================"

case $MODE in

  # ----------------------------------------------------------------
  # CONFIG 1: Standard / Stage Two (Raw Folders)
  # ----------------------------------------------------------------
  1)
    OUT_DIR="/scratch/st-ipor-1/cperez/MedSAM/DS/Stage_Two_YOLO_DS"

    ARGS=(
      --yolo-out-dir "${OUT_DIR}"
      --yolo-device "${YOLO_DEVICES}"
      # Add default actions if needed, otherwise experiment_unified defaults apply
      --train-yolo
      --test-yolo
    )
    ;;

  # ----------------------------------------------------------------
  # CONFIG 2: Parquet Splits (Reproducible)
  # ----------------------------------------------------------------
  2)
    SPLITS_PARQUET="/home/cperez67/medsam/src/image_data/parquet/combined.parquet"
    OUT_DIR="/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS_CONF_0.85_FROM_PARQUET"

    # Sanity check specific to this mode
    if [[ ! -f "${SPLITS_PARQUET}" ]]; then
      echo "[ERROR] Splits parquet not found: ${SPLITS_PARQUET}" >&2
      exit 2
    fi

    echo "[INFO] Source Parquet: ${SPLITS_PARQUET}"

    ARGS=(
      # Data Source
      --splits-parquet "${SPLITS_PARQUET}"
      --yolo-out-dir "${OUT_DIR}"
      --splits-batch-size 2048
      --require-both-gt-boxes       # Filter out rows missing GT boxes

      # Model Config
      --yolo-device "${YOLO_DEVICES}"
      --epochs 800
      --batch 16                    # Batch size per GPU (effective batch = 16 * 4 = 64)

      # Workflow Stages
      --create-ds
      --train-yolo
      --test-yolo
    )
    ;;

  # ----------------------------------------------------------------
  # Fallback
  # ----------------------------------------------------------------
  *)
    echo "Invalid mode selected: $MODE"
    echo "Usage: sbatch script.slurm [1|2]"
    exit 1
    ;;
esac

############################
# 4. Execution
############################
echo "[INFO] Output Directory: ${OUT_DIR}"
echo "[INFO] Launching Experiment..."

srun --gpu-bind=closest --cpu-bind=cores -u \
  python -m src.experiment_unified "${ARGS[@]}"

echo "[INFO] Job Finished"