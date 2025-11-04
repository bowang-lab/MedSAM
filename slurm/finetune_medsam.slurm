#!/bin/bash
#SBATCH --job-name=finetune_medsam
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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
OUT_DIR="/scratch/st-ipor-1/cperez/MedSAM/TRAINING_DS"
DATA_DIR="/arc/project/st-ipor-1/carlosp/fundus_data/All_Datasets_Organized"
CKPT="${REPO_DIR}/weights/medsam_vit_b.pth"


DEVICE="cuda:0"
BATCH=8

# Python module entrypoint
PYTHON_MODULE="src.scripts.finetune_medsam"

# Make project importable as `src.*`
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Optional: GPU/CPU binding hints
SRUN_BIND="--gpu-bind=closest --cpu-bind=cores"

source "${REPO_DIR}/slurm/setup.sh"
setup_env

mkdir -p "${REPO_DIR}/logs" /scratch/st-ipor-1/cperez/MedSAM/hpc_logs


########################################
# Dispatch by MODE
########################################
cd "${REPO_DIR}"


srun -u python -m ${PYTHON_MODULE} \
  --train \
  --device "${DEVICE}" \
  --data-root "${DATA_DIR}" \
  --ckpt "${CKPT}" \
  --out-dir "${OUT_DIR}" \
  --run-dir "${RUN_DIR}" \
  --run-name "${RUN_NAME}" \
  --epochs 50 \
  --batch 8 \
  --imgsz 1024 \
  --pad-jitter 0.3
