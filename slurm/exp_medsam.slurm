#!/bin/bash
#SBATCH --job-name=MEDSAM_ONLY
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=100:00:00
#SBATCH --gpus-per-node=2
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%A.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%A.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

############################
# Paths / environment
############################
REPO_DIR="/scratch/st-ipor-1/cperez/MedSAM"
DATA_ROOT="/arc/project/st-ipor-1/carlosp/fundus_data/All_Datasets_Organized"

# Pre-built YOLO dataset + trained YOLO weights
YOLO_DS="/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS"
YOLO_WTS="/scratch/st-ipor-1/cperez/MedSAM/runs/run8/Train/Train/weights/best.pt"

LOG_DIR="${REPO_DIR}/hpc_logs"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

source "${REPO_DIR}/slurm/setup.sh"
setup_env

cd "${REPO_DIR}"

############################
# Experiment args
############################
# We only want to run the MedSAM / Fundu finetuning stage, using:
#   - existing YOLO dataset   (--yolo-ds)
#   - existing YOLO weights   (--yolo-weights)
# No YOLO training or testing is requested in this job.
ARGS=(
  --finetune-medsam
  --yolo-ds "${YOLO_DS}"
  --yolo-weights "${YOLO_WTS}"
)

############################
# Launch (multi-GPU via torchrun)
############################
# SLURM_GPUS_ON_NODE is set by the scheduler; we use one process per GPU.
srun --gpu-bind=closest --cpu-bind=cores \
  torchrun --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE}" \
    -m src.experiment "${ARGS[@]}"