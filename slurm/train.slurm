#!/bin/bash
#SBATCH --job-name=YOLO
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%A_%a.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%A_%a.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

############################
# User-configurable paths  #
############################
REPO_DIR="/scratch/st-ipor-1/cperez/MedSAM"
OUT_DIR="/scratch/st-ipor-1/cperez/MedSAM/PAPILA_TRAINING_DS"
DATA_DIR="/arc/project/st-ipor-1/carlosp/fundus_data/All_Datasets_Organized"
CONFIG="${REPO_DIR}/src/configs/train_custom.yaml"
WEIGHTS_LOCAL="${REPO_DIR}/weights/yolo12x.pt"

DEVICE="0"
BATCH=8

# Python module entrypoint
PYTHON_MODULE="src.scripts.train_yolo"

# Make project importable as `src.*`
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Optional: GPU/CPU binding hints
SRUN_BIND="--gpu-bind=closest --cpu-bind=cores"

source "${REPO_DIR}/slurm/setup.sh"
setup_env

mkdir -p "${REPO_DIR}/logs" /scratch/st-ipor-1/cperez/MedSAM/hpc_logs

########################################
# Parse modes from CLI and map to array
########################################
# Usage example to launch all three at once:
#   sbatch --array=0-2 ./slurm/train.slurm --modes=train,test,test
#
#   sbatch --array=0-1 ./slurm/train.slurm --modes=train,tune
#
# If you submit without --array, the script will run only the first mode.

MODES="train,tune,test"
if [[ $# -gt 0 ]]; then
  for arg in "$@"; do
    case "$arg" in
      --modes=*)
        MODES="${arg#*=}"
        shift
        ;;
      *)
        ;; # ignore unrecognized args (Slurm may append its own)
    esac
  done
fi
IFS=',' read -r -a MODE_ARRAY <<< "${MODES}"

# Derive which mode this array task should run
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODE="${MODE_ARRAY[$TASK_ID]:-train}"

echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}  ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-N/A}"
echo "[INFO] MODES=${MODES}"
echo "[INFO] Selected MODE=${MODE}"

########################################
# Common argument blocks (per mode)
########################################
# Notes:
# - Pass device "0" since this job requests --gpus=1. If you increase GPUs, update device accordingly.
# - You can override any flag at call time by adding it at the end of the srun line.

#  --model "${WEIGHTS_LOCAL}"
#  --model "${REPO_DIR}/runs/MultiClassLarge/weights/last.pt"


ARGS_TRAIN=(
  --train
  --run-dir "${REPO_DIR}/Nov12"
  --data-root "${DATA_DIR}"
  --yolo-ds "${OUT_DIR}"
  --run-name "With10%Papila"
  --model "${WEIGHTS_LOCAL}"
  --out-dir "${OUT_DIR}"
  --cfg "${CONFIG}"
  --device "${DEVICE}"
  --epochs 400
  --batch "${BATCH}"
  --imgsz 640
)

ARGS_TRAIN_TEST=(
  --run-dir "${REPO_DIR}/Nov12"
  --data-root "${DATA_DIR}"
  --yolo-ds "${OUT_DIR}"
  --run-name "With10%Papila"
  --model "${WEIGHTS_LOCAL}"
  --out-dir "${OUT_DIR}"
  --cfg "${CONFIG}"
  --device "${DEVICE}"
  --epochs 400
  --batch "${BATCH}"
  --imgsz 640
)

ARGS_TEST=(
  --run-dir "${REPO_DIR}/runs"
  --yolo-ds "${OUT_DIR}"
  --run-name "Test"
  --out-dir "${REPO_DIR}/runs"
  --test-weights "/scratch/st-ipor-1/cperez/MedSAM/runs/MultiClassV2-6616729-0/weights/best.pt"
  --device "${DEVICE}"
  --imgsz 640
  --batch "${BATCH}"
)

########################################
# Dispatch by MODE
########################################
cd "${REPO_DIR}"

case "${MODE}" in
  train)
    srun ${SRUN_BIND} -u python -m "${PYTHON_MODULE}" "${ARGS_TRAIN[@]}"
    ;;
  tune)
    srun ${SRUN_BIND} -u python -m "${PYTHON_MODULE}" "${ARGS_TUNE[@]}"
    ;;
  train_test)
    srun ${SRUN_BIND} -u python -m "${PYTHON_MODULE}" "${ARGS_TRAIN_TEST[@]}"
    ;;
  test)
    srun ${SRUN_BIND} -u python -m "${PYTHON_MODULE}" "${ARGS_TEST[@]}"
    ;;
  *)
    echo "[ERROR] Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac