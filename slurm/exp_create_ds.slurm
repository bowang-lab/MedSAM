#!/bin/bash
#SBATCH --job-name=YOLO_CREATE_DS
#SBATCH --account=st-ipor-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=04:00:00
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
# Device Configuration (CPU Only)
############################
# Force the experiment script to use CPU.
# This prevents it from looking for GPUs or occupying GPU resources.
YOLO_DEVICES="cpu"
echo "YOLO_DEVICES=${YOLO_DEVICES}"

############################
# Experiment args
############################
ARGS=(
  --papila 0
  --yolo-out-dir "/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS"
  --yolo-device "${YOLO_DEVICES}"
  --create-ds
)

############################
# Run
############################
# Removed --gpu-bind=closest since we are running on CPU nodes.
echo "Starting Dataset Creation on CPU..."

srun --cpu-bind=cores -u \
  python -m src.experiment "${ARGS[@]}"

echo "Job Finished"