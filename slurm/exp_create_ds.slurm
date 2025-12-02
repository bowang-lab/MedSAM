#!/bin/bash
#SBATCH --job-name=Create_Train_YOLO
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=80:00:00
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

BATCH=10

# Python module entrypoint
PYTHON_MODULE="src.experiment"

# Make project importable as `src.*`
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Optional: GPU/CPU binding hints
SRUN_BIND="--gpu-bind=closest --cpu-bind=cores"

source "${REPO_DIR}/slurm/setup.sh"
setup_env

mkdir -p "${REPO_DIR}/logs" /scratch/st-ipor-1/cperez/MedSAM/hpc_logs

# ==========================
# Argument configurations
# ==========================
ARGS_1=(
  --create-ds
  --papila 1
  --train-ratio 0
  --val-ratio 0
  --test-ratio 1
  --yolo-out-dir "/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS_ALL_PAPILA"
)

ARGS_2=(
  --papila 0
  --train-ratio 0.8
  --val-ratio 0.1
  --test-ratio 0.1
  --yolo-out-dir "/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS_P0"
)

ARGS_3=(
  --papila 0.1
  --train-ratio 0.8
  --val-ratio 0.1
  --test-ratio 0.1
  --yolo-out-dir "/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS_P10"
)

ARGS_4=(
  --papila 0.25
  --train-ratio 0.8
  --val-ratio 0.1
  --test-ratio 0.1
  --yolo-out-dir "/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS_P25"
)

########################################
# Select config by array task ID
########################################
cd "${REPO_DIR}"


srun ${SRUN_BIND} -u python -m "${PYTHON_MODULE}" "${ARGS_2[@]}"