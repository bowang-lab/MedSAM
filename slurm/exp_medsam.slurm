#!/bin/bash
#SBATCH --job-name=TRAIN_P_30
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --gpus=2
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%A_%a.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%A_%a.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

############################
# User-configurable paths  #
############################
REPO_DIR="/scratch/st-ipor-1/cperez/MedSAM"

BATCH=8

# Python module entrypoint
PYTHON_MODULE="src.experiment"

# Make project importable as `src.*`
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Optional: GPU/CPU binding hints
SRUN_BIND="--gpu-bind=closest --cpu-bind=cores"

source "${REPO_DIR}/slurm/setup.sh"
setup_env

mkdir -p "${REPO_DIR}/logs" /scratch/st-ipor-1/cperez/MedSAM/hpc_logs

ARGS_1=(
  --yolo-ds "/scratch/st-ipor-1/cperez/MedSAM/YOLO_DS_No_P"
  --yolo-weights "/scratch/st-ipor-1/cperez/MedSAM/runs/run3/Train/Train/weights/best.pt"
)
ARGS_2=(
  --yolo-ds "/scratch/st-ipor-1/cperez/MedSAM/YOLO_DS_P"
  --yolo-weights "/scratch/st-ipor-1/cperez/MedSAM/runs/run1/Train/Train/weights/best.pt"
)
ARGS_3=(
--papila
--yolo-out-dir "/scratch/st-ipor-1/cperez/MedSAM/YOLO_DS_P_30"
)
ARGS_4=(
--yolo-out-dir "/scratch/st-ipor-1/cperez/MedSAM/YOLO_DS_No_P"
)

########################################
# Dispatch by MODE
########################################
cd "${REPO_DIR}"

srun ${SRUN_BIND} -u python -m "${PYTHON_MODULE}" "${ARGS_3[@]}"
