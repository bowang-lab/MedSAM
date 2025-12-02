#!/bin/bash
#SBATCH --job-name=merge_parquets
#SBATCH --account=st-ipor-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

# Load your standard environment setup
source "${REPO_DIR}/slurm/setup.sh"
setup_env

cd "${REPO_DIR}"

############################
# Configuration
############################
# Adjust these input/output paths as needed
INPUT_A_PRIMARY="/scratch/st-ipor-1/cperez/MedSAM/src/image_data/images_dev.parquet"
INPUT_B_SECONDARY="/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS/saved_images.parquet"
OUTPUT_FILE="/scratch/st-ipor-1/cperez/MedSAM/src/image_data/merged_images_dev.parquet"

ARGS=(
  --in-a "${INPUT_A_PRIMARY}"
  --in-b "${INPUT_B_SECONDARY}"
  --out "${OUTPUT_FILE}"
  --key "uid"
  --batch-size 4096
  --write-batch 1024
  --compression "zstd"
)

############################
# Run
############################
echo "[INFO] Starting Smart Merge on $(hostname)"
echo "[INFO] Primary: ${INPUT_A_PRIMARY}"
echo "[INFO] Secondary: ${INPUT_B_SECONDARY}"
echo "[INFO] Output: ${OUTPUT_FILE}"

# Using srun with cpu binding (no gpu binding needed)
srun --cpu-bind=cores -u \
  python -m src.tools.parquet_processing.merge_smart "${ARGS[@]}"

echo "[INFO] Done."