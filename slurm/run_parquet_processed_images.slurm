#!/bin/bash
#SBATCH --job-name=merge_all_parquets
#SBATCH --account=st-ipor-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

############################
# 1. Environment Setup
############################
REPO_DIR="/scratch/st-ipor-1/cperez/MedSAM"
export REPO_DIR
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

source "${REPO_DIR}/slurm/setup.sh"
setup_env

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/hpc_logs"

############################
# 2. Configuration
############################
# Adjust these paths if your filenames differ
DEV_SPLIT_PARQUET="/home/cperez67/medsam/src/image_data/parquet/images_dev_yolo_split.parquet"
DEV_BASE_PARQUET="/home/cperez67/medsam/src/image_data/parquet/og_images/images_dev.parquet"
EYEPACS_PARQUET="/home/cperez67/medsam/src/image_data/parquet/og_images/images_eyepacs.parquet"

# Output path
OUTPUT_PARQUET="/home/cperez67/medsam/src/image_data/parquet/og_images/combined_dataset.parquet"

############################
# 3. Execution
############################
echo "[INFO] Starting Parquet Merge Pipeline..."
echo "  - Dev Split (Primary): ${DEV_SPLIT_PARQUET}"
echo "  - Dev Base (Fill):     ${DEV_BASE_PARQUET}"
echo "  - Eyepacs (Append):    ${EYEPACS_PARQUET}"
echo "  - Output:              ${OUTPUT_PARQUET}"

srun --cpu-bind=cores -u \
  python -m src.tools.parquet_processing.run_parquet_processing \
    --dev-split "${DEV_SPLIT_PARQUET}" \
    --dev-base "${DEV_BASE_PARQUET}" \
    --eyepacs "${EYEPACS_PARQUET}" \
    --out "${OUTPUT_PARQUET}"

echo "[INFO] Job Finished"