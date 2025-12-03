#!/bin/bash
#SBATCH --job-name=YOLO_SEMI_TRAIN
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gpus=1
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
# Input: The final parquet file produced by run_process_predictions.slurm
# (Update filename if you used a specific confidence, e.g. semi_supervised_dataset_0.75.parquet)
INPUT_PARQUET="/home/cperez67/medsam/src/image_data/parquet/processed_semi_supervised/final_semi_supervised_dataset_0.88.parquet"

# Output: Where to materialize the training images/labels
OUT_YOLO_DS="${REPO_DIR}/DS/YOLO_SEMI_88"

# Output: Where training runs (logs/weights) will be saved
RUNS_ROOT="${REPO_DIR}/runs/semi_supervised_final_88"

# Weights to Fine-tune (e.g., your best previous model or a base model)
# Example: "${REPO_DIR}/weights/yolo12x.pt" or "${REPO_DIR}/runs/run1/weights/best.pt"
INIT_WEIGHTS="/home/cperez67/medsam/runs/run12/Train/Train/weights/best.pt"

############################
# 3. Training Parameters
############################
EPOCHS=200
BATCH=16
IMGSZ=640
WORKERS=8

############################
# 4. Execution
############################
echo "[INFO] Starting Semi-Supervised Training Job"
echo "  - Input Parquet: $INPUT_PARQUET"
echo "  - Dataset Dir:   $OUT_YOLO_DS"
echo "  - Init Weights:  $INIT_WEIGHTS"

if [[ ! -f "$INPUT_PARQUET" ]]; then
    echo "[ERROR] Input parquet not found: $INPUT_PARQUET"
    exit 1
fi

# Clean up previous dataset dir if retrying (optional, ensures fresh materialization)
# rm -rf "$OUT_YOLO_DS"

srun --gpu-bind=closest --cpu-bind=cores -u \
  python -m src.scripts.train_yolo_semi \
    --images-parquet "${INPUT_PARQUET}" \
    --out-yolo-ds "${OUT_YOLO_DS}" \
    --runs-root "${RUNS_ROOT}" \
    --init-weights "${INIT_WEIGHTS}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --imgsz "${IMGSZ}" \
    --workers "${WORKERS}" \
    --device "0"

echo "[INFO] Job Finished"