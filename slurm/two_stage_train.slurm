#!/bin/bash
#SBATCH --job-name=two-stage
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=7:00:00
#SBATCH --gpus=1
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

# --------------------------- Paths (HPC) ---------------------------
PROJECT_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"


BB_DIR="${PROJECT_DIR}/bounding_box"
RUNS_DIR="${PROJECT_DIR}/runs/detect"

# Clean split, augmented split, and prebuilt ROI split (from preprocess step)
DATA_ROOT="${PROJECT_DIR}/data/yolo_split"       # clean (val/test)
AUG_ROOT="${PROJECT_DIR}/data/yolo_split_aug"    # augmented (train)
ROI_OUT="${PROJECT_DIR}/data/yolo_split_cupROI"  # prebuilt (used by Stage B)
# Prebuilt disc-only dataset (built by preprocess 'disc_only' stage)
DISC_ROOT="${PROJECT_DIR}/data/yolo_split_disc_only"

# Optional output for two-stage inference
OUT_JSONL="${PROJECT_DIR}/data/two_stage_boxes.jsonl"

# Experiment names (match script defaults)
STAGEA_NAME="stageA_disc_only"


# Optional: avoid cache warnings on compute nodes
export XDG_CACHE_HOME="/scratch/st-ipor-1/cperez/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TMPDIR="${PROJECT_DIR}/tmp"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"

mkdir -p "${PROJECT_DIR}/logs" "${RUNS_DIR}" "${PROJECT_DIR}/data"
cd "${PROJECT_DIR}"

# --------------------------- Env -----------------------------------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

echo "SLURM Job ID : ${SLURM_JOB_ID}"
echo "Node list    : ${SLURM_JOB_NODELIST}"
echo "Working dir  : $(pwd)"
echo "Runs dir     : ${RUNS_DIR}"

nvidia-smi -L || echo "nvidia-smi not available"

python - <<'PY'
import torch, ultralytics
print(f"[ENV] torch {torch.__version__} | cuda={torch.cuda.is_available()} | gpus={torch.cuda.device_count()}")
print(f"[ENV] ultralytics {ultralytics.__version__}")
PY

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"



# --------------------------- Helpers -------------------------------
die() { echo "[ERROR] $*" >&2; exit 1; }
check_file() { [[ -f "$1" ]] || die "$2 not found: $1"; }
check_dir()  { [[ -d "$1" ]] || die "$2 not found: $1"; }

# --------------------------- Stage A -------------------------------
run_stageA() {
  echo "[STAGE A] Train disc-only detector using PREBUILT disc-only dataset…"
  check_dir  "${DISC_ROOT}" "Prebuilt disc-only root"
  check_file "${DISC_ROOT}/data.yaml" "Disc-only YAML (od_only.yaml)"

  # Minimal CLI: just point --data_root at the disc-only root; no --aug_root needed.
  python "${BB_DIR}/train/train_stageA_disc_only.py" \
    --project_dir "${PROJECT_DIR}" \
    --data_root   "${DISC_ROOT}" \
    --model       "${BB_DIR}/weights/yolov8n.pt"

  local best="${RUNS_DIR}/${STAGEA_NAME}/weights/best.pt"
  check_file "${best}" "Stage A best checkpoint"
  echo "[STAGE A] Best checkpoint: ${best}"
}


# --------------------------- Run -----------------------------------
run_stageA

echo "[DONE] ]training completed at $(date)"