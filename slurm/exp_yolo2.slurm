#!/bin/bash
#SBATCH --job-name=YOLO_TRAIN_PARQUET
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
# Determine GPU indices for Ultralytics (e.g., "0,1,2,3") based on Slurm allocation.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # If Slurm sets visibility, we just pass that list.
  YOLO_DEVICES="${CUDA_VISIBLE_DEVICES}"
else
  # Fallback if not explicitly set (rare in proper Slurm config)
  YOLO_DEVICES="0,1,2,3"
fi
echo "YOLO_DEVICES=${YOLO_DEVICES}"

############################
# 3. Dataset Paths
############################
# Input: Parquet file containing 'split' column (train/val/test)
SPLITS_PARQUET="/scratch/st-ipor-1/cperez/MedSAM/data/combined.parquet"

# Output: Where to materialize the YOLO directory (images/labels/data.yaml)
YOLO_DS_OUT="/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS_V4 _FROM_PARQUET"

############################
# 4. Experiment Arguments
############################
# We call the unified script with explicit stage flags.
ARGS=(
  # Data Source
  --splits-parquet "${SPLITS_PARQUET}"
  --yolo-out-dir "${YOLO_DS_OUT}"
  --splits-batch-size 2048
  --require-both-gt-boxes       # Filter out rows missing GT boxes

  # Model Config
  --yolo-device "${YOLO_DEVICES}"
  --epochs 400
  --batch 16                    # Batch size per GPU (effective batch = 16 * 4 = 64)
  --imgsz 640

  # Workflow Stages (Explicitly enabled)
  --create-ds
  --train-yolo
  --test-yolo
)

############################
# 5. Execution
############################
if [[ ! -f "${SPLITS_PARQUET}" ]]; then
  echo "[ERROR] Splits parquet not found: ${SPLITS_PARQUET}" >&2
  exit 2
fi

echo "[INFO] Starting YOLO Training Pipeline from Parquet..."
echo "[INFO] Source: ${SPLITS_PARQUET}"
echo "[INFO] Destination: ${YOLO_DS_OUT}"

srun --gpu-bind=closest --cpu-bind=cores -u \
  python -m src.experiment_unified "${ARGS[@]}"

echo "[INFO] Job Finished"