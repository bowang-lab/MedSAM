#!/bin/bash
#SBATCH --job-name=Predict_Summarize_MGPU
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%A.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%A.err

set -euo pipefail

REPO_DIR="/scratch/st-ipor-1/cperez/MedSAM"
export REPO_DIR
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

source "${REPO_DIR}/slurm/setup.sh"
setup_env

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/hpc_logs"

# Threads: avoid oversubscription
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# CUDA allocator fragmentation mitigation (matches your OOM hint)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARGS=(
  --images-parquet "/scratch/st-ipor-1/cperez/MedSAM/src/image_data/images_dev.parquet"
  --yolo-weights "/scratch/st-ipor-1/cperez/MedSAM/runs/run13/Train/Train/weights/best.pt"
  --medsam-checkpoint "/scratch/st-ipor-1/cperez/MedSAM/weights/medsam_vit_b.pth"
  --device cuda
  --out-dir "/scratch/st-ipor-1/cperez/MedSAM/PREDICT/DEC03/dev_set_predictions_run13"

  --resume
  --sam-amp
  --sam-resize-backend cv2
  --predict-batch-size 128
  --mask-store parquet
)

srun --cpu-bind=cores --gpu-bind=single:1 -u \
  python -m src.scripts.predict_summarize "${ARGS[@]}"