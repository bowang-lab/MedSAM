#!/bin/bash
#SBATCH --job-name=process_parquet_pipeline
#SBATCH --account=st-ipor-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
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
# 2. Parameter Parsing
############################
# Default values
CONF_THRESH=0.75
DUP_FACTOR=2

# Parse args like: sbatch script.slurm --conf=0.85 --dup=3
for arg in "$@"; do
  case "$arg" in
    --conf=*)
      CONF_THRESH="${arg#*=}"
      ;;
    --dup=*)
      DUP_FACTOR="${arg#*=}"
      ;;
  esac
done

############################
# 3. Input Configuration
############################
# Predictions from previous runs
PRED_DEV="${REPO_DIR}/src/image_data/parquet/predictions/predictions_dev.parquet"
PRED_EYEPACS="${REPO_DIR}/src/image_data/parquet/predictions/predictions_eyepacs.parquet"

# The "Real" Ground Truth dataset (with splits already assigned)
YOLO_SPLIT_PARQUET="${REPO_DIR}/src/image_data/parquet/images_dev_yolo_split.parquet"

# Where to save the final semi-supervised dataset
OUT_DIR="${REPO_DIR}/src/image_data/parquet/processed_semi_supervised"

############################
# 4. Execution
############################
echo "[INFO] Starting Parquet Processing Pipeline..."
echo "  - Dev Preds:    $PRED_DEV"
echo "  - Eyepacs Preds:$PRED_EYEPACS"
echo "  - Real GT Split:$YOLO_SPLIT_PARQUET"
echo "  - Output Dir:   $OUT_DIR"
echo "  - Confidence:   $CONF_THRESH"
echo "  - Dup Factor:   $DUP_FACTOR"

# Ensure output directory exists
mkdir -p "$OUT_DIR"

srun --cpu-bind=cores -u \
  python -m src.tools.parquet_processing.process_predictions \
    --pred-dev "${PRED_DEV}" \
    --pred-eyepacs "${PRED_EYEPACS}" \
    --yolo-split "${YOLO_SPLIT_PARQUET}" \
    --out-dir "${OUT_DIR}" \
    --conf "${CONF_THRESH}" \
    --duplication-factor "${DUP_FACTOR}"

echo "[INFO] Job Finished."