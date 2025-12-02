#!/bin/bash
#SBATCH --job-name=create_images
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=60G
#SBATCH --time=5:00:00
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
# Experiment args
############################
# YOLO_DEVICES is set by setup_env based on Slurm's GPU allocation
ARGS=(
  --root "/arc/project/st-ipor-1/carlosp/fundus_data/All_Datasets_Organized"
  --csv "/scratch/st-ipor-1/cperez/MedSAM/src/image_data/Cleaned_Metadata.csv"
  --out-parquet "/scratch/st-ipor-1/cperez/MedSAM/src/image_data/images.parquet"
)

ARGS2=(
  /arc/project/st-ipor-1/Fundus/Data/EYEPACS
  ${REPO_DIR}/data/EYEPACS
)

############################
# Run
############################
#srun --gpu-bind=closest --cpu-bind=cores -u \
#  python -m src.tools.jpg_to_png "${ARGS[@]}"

srun --gpu-bind=closest --cpu-bind=cores -u \
  python -m src.tools.create_images "${ARGS[@]}"