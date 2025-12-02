#!/bin/bash
#SBATCH --job-name=YOLO_TRAIN_multi_GPU_from_parquet
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --gpus=4
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%A.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%A.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

############################
# Repo / env
############################
REPO_DIR="/scratch/st-ipor-1/cperez/MedSAM"
export REPO_DIR
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

source "${REPO_DIR}/slurm/setup.sh"
setup_env

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/hpc_logs"

############################
# FIX: define YOLO_DEVICES safely (do NOT rely on it being pre-set)
############################
# Preferred source is CUDA_VISIBLE_DEVICES, which Slurm typically sets per job/step.
# Ultralytics accepts comma-separated GPU indices, e.g. "0,1,2,3".
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  YOLO_DEVICES="${CUDA_VISIBLE_DEVICES}"
else
  YOLO_DEVICES="0,1,2,3"
fi
export YOLO_DEVICES
echo "YOLO_DEVICES=${YOLO_DEVICES}"

############################
# NEW: Parquet with existing splits (train/val/test already assigned)
############################
# Point this at your saved_images.parquet (or similar) where Image.split is set.
SPLITS_PARQUET="/scratch/st-ipor-1/cperez/MedSAM/data/combined_new.parquet"

# Where to CREATE the YOLO dataset from the parquet (will contain data.yaml).
YOLO_DS_OUT="/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS_V3_FROM_PARQUET"

############################
# Experiment args
############################
ARGS=(
  --papila 0
  --splits-parquet "${SPLITS_PARQUET}"
  --yolo-out-dir "${YOLO_DS_OUT}"
  --splits-batch-size 2048
  --yolo-device "${YOLO_DEVICES}"

  # Optional: explicitly run stages (otherwise your script's defaults apply)
  # --create-ds
  # --train-yolo
  # --test-yolo
)

############################
# Sanity checks
############################
if [[ ! -f "${SPLITS_PARQUET}" ]]; then
  echo "[ERROR] splits parquet not found: ${SPLITS_PARQUET}" >&2
  exit 2
fi

############################
# Run
############################
srun --gpu-bind=closest --cpu-bind=cores -u \
  python -m src.experiment_2 "${ARGS[@]}"