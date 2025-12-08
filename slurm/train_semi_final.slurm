#!/bin/bash
#SBATCH --job-name=YOLO_SEMI_TRAIN_RERUN
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=72:00:00
#SBATCH --gpus=4
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

REPO_DIR="/scratch/st-ipor-1/cperez/MedSAM"
export REPO_DIR
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
source "${REPO_DIR}/slurm/setup.sh"
setup_env
cd "${REPO_DIR}"

############################
# Config
############################
# NOTE: Input parquet path is still required by arg parser, even if skipping usage.
# Ensure this points to a valid path (e.g. the one used to create the DS)
INPUT_PARQUET="${REPO_DIR}/src/image_data/processed_semi_supervised/final_semi_supervised_dataset_85.parquet"

OUT_YOLO_DS="${REPO_DIR}/DS/YOLO_SEMI_FINAL_CONF85"
RUNS_ROOT="${REPO_DIR}/runs/semi_supervised_conf85_2"

# --- CORRECT WEIGHTS HERE ---
INIT_WEIGHTS="/home/cperez67/medsam/runs/run12/Train/Train/weights/best.pt"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  YOLO_DEVICES="${CUDA_VISIBLE_DEVICES}"
else
  YOLO_DEVICES="0,1,2,3"
fi

############################
# Execution
############################
echo "[INFO] Starting Rerun with Skip Materialization"
echo "  - DS Dir: $OUT_YOLO_DS"
echo "  - Weights: $INIT_WEIGHTS"

srun --gpu-bind=closest --cpu-bind=cores -u \
  python -m src.scripts.train_yolo_semi \
    --images-parquet "${INPUT_PARQUET}" \
    --out-yolo-ds "${OUT_YOLO_DS}" \
    --runs-root "${RUNS_ROOT}" \
    --init-weights "${INIT_WEIGHTS}" \
    --device "${YOLO_DEVICES}" \
    --epochs 400 \
    --batch 16 \
    --workers 16 \
    --skip-materialization

echo "[INFO] Job Finished"