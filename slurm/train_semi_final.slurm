#!/bin/bash
#SBATCH --job-name=YOLO_SEMI_TRAIN_MULTI
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --gpus=2
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
# Input: The final parquet file from the previous processing step
INPUT_PARQUET="${REPO_DIR}/src/image_data/processed_semi_supervised/final_semi_supervised_dataset.parquet"

# Output: Where to materialize the dataset
OUT_YOLO_DS="${REPO_DIR}/DS/YOLO_SEMI_FINAL_V1"

# Output: Where training runs will be saved
RUNS_ROOT="${REPO_DIR}/runs/semi_supervised_final"

# Weights to Fine-tune
INIT_WEIGHTS="${REPO_DIR}/weights/yolo12x.pt"

# Determine GPU configuration automatically from Slurm
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  YOLO_DEVICES="${CUDA_VISIBLE_DEVICES}"
else
  # Fallback (should be set by SBATCH --gpus=4)
  YOLO_DEVICES="0,1,2,3"
fi

############################
# 3. Training Parameters
############################
EPOCHS=300
# Note: Batch size is per-GPU in Ultralytics DDP.
# 16 * 4 GPUs = 64 total batch size. Adjust based on GPU VRAM.
BATCH=16
IMGSZ=640
WORKERS=16

############################
# 4. Execution
############################
echo "[INFO] Starting Multi-GPU Semi-Supervised Training"
echo "  - Input Parquet: $INPUT_PARQUET"
echo "  - Dataset Dir:   $OUT_YOLO_DS"
echo "  - Init Weights:  $INIT_WEIGHTS"
echo "  - Devices:       $YOLO_DEVICES"

if [[ ! -f "$INPUT_PARQUET" ]]; then
    echo "[ERROR] Input parquet not found: $INPUT_PARQUET"
    exit 1
fi

# We rely on the python script's auto-detection or pass explicitly
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
    --device "${YOLO_DEVICES}"

echo "[INFO] Job Finished"