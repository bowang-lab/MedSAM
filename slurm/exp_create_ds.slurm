#!/bin/bash
#SBATCH --job-name=YOLO_CREATE_DS
#SBATCH --account=st-ipor-1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=4:00:00
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
  # If Slurm didn't set it for some reason, fall back to "0,1,2,3" for 4 GPUs.
  # (Not a resource change; this just prevents -u from crashing.)
  YOLO_DEVICES="0,1,2,3"
fi
export YOLO_DEVICES
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
srun --gpu-bind=closest --cpu-bind=cores -u \
  python -m src.experiment "${ARGS[@]}"