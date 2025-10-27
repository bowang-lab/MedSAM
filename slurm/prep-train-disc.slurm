#!/bin/bash
#SBATCH --job-name=prep-train-disc
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --gpus=4
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=START,FAIL,END

set -euo pipefail

# ====================== PATHS / CONFIG ======================
PROJECT_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"

# Preprocess pipeline
PIPELINE="${PROJECT_DIR}/bounding_box/preprocess/preprocess_pipeline.py"
CONFIG="${PROJECT_DIR}/bounding_box/preprocess/preprocess_pipeline_hpc.yaml"
AUG_YAML="${PROJECT_DIR}/bounding_box/preprocess/augment_config.yaml"  # only if your CONFIG references it

# Training paths
BB_DIR="${PROJECT_DIR}/bounding_box"
RUNS_DIR="${PROJECT_DIR}/bounding_box/runs/detect"

# Data roots produced by preprocess (defaults used by your training scripts)
DATA_ROOT="${PROJECT_DIR}/data/yolo_split"             # clean split
AUG_ROOT="${PROJECT_DIR}/data/yolo_split_aug"          # augmented train
ROI_OUT="${PROJECT_DIR}/data/yolo_split_cupROI"        # ROI crops (if built)
DISC_ROOT="${PROJECT_DIR}/data/yolo_split_disc_only"   # prebuilt disc-only (from preprocess 'disc_only')

YOLO_WEIGHTS="${BB_DIR}/weights/yolov8n.pt"
STAGEA_NAME="stageA_disc_only"

# Optional output for later inference
OUT_JSONL="${PROJECT_DIR}/data/two_stage_boxes.jsonl"

# ====================== PREP ENV & DIRS ======================
mkdir -p "${PROJECT_DIR}/logs" "${RUNS_DIR}" "${PROJECT_DIR}/data"
mkdir -p "$(dirname "${CONFIG}")"

# cache/tmp on scratch to avoid home warnings
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TMPDIR="${PROJECT_DIR}/tmp"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"

# CPU threading (avoid oversubscription)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# Conda env
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

# ====================== SANITY CHECKS ======================
die() { echo "[ERROR] $*" >&2; exit 1; }
check_file() { [[ -f "$1" ]] || die "$2 not found: $1"; }
check_dir()  { [[ -d "$1" ]] || die "$2 not found: $1"; }

check_file "${PIPELINE}" "Preprocess pipeline"
check_file "${CONFIG}"   "Pipeline config"
if [[ -f "${AUG_YAML}" ]]; then
  echo "[INFO] Found augment YAML: ${AUG_YAML}"
else
  echo "[WARN] Augment YAML not found at ${AUG_YAML} (only needed if CONFIG references it)."
fi

# ====================== RUN PREPROCESS ======================
echo "[INFO] Running preprocess with config: ${CONFIG}"
srun --kill-on-bad-exit=1 -u python "${PIPELINE}" --config "${CONFIG}"

# ====================== TRAIN STAGE A (DISC-ONLY) ======================
run_stageA() {
  echo "[STAGE A] Train disc-only detector using prebuilt disc-only dataset…"
  check_dir  "${DISC_ROOT}" "Prebuilt disc-only root"
  check_file "${DISC_ROOT}/data.yaml" "Disc-only dataset YAML (od_only.yaml)"
  check_file "${YOLO_WEIGHTS}" "YOLO base weights"

  srun --kill-on-bad-exit=1 -u python "${BB_DIR}/train/train_stageA_disc_only.py" \
    --project_dir "${PROJECT_DIR}" \
    --data_root   "${DISC_ROOT}" \
    --model       "${YOLO_WEIGHTS}"

  local best="${RUNS_DIR}/${STAGEA_NAME}/weights/best.pt"
  check_file "${best}" "Stage A best checkpoint"
  echo "[STAGE A] Best checkpoint: ${best}"
}

run_stageA

echo "[DONE] Training completed at $(date)"