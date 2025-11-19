#!/bin/bash
#SBATCH --job-name=finetune_medsam_unfreeze_encoders
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gpus=1
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%A_%a.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%A_%a.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

############################
# User-configurable paths  #
############################
REPO_DIR="/scratch/st-ipor-1/cperez/MedSAM"

# Outputs (metrics, preds, etc.)
OUT_DIR="/scratch/st-ipor-1/cperez/MedSAM/OUT/MEDSAM_TRAIN_P"
RUN_DIR="${REPO_DIR}/runs"
RUN_NAME="FineTune_MedSAM_${SLURM_JOB_ID:-local}"

# Root where ImageFactory can find images + masks
DATA_DIR="/arc/project/st-ipor-1/carlosp/fundus_data/All_Datasets_Organized"

# MedSAM base checkpoint (ViT-B)
CKPT="${REPO_DIR}/weights/medsam_vit_b.pth"

# If you have a YOLO dataset already built, point to the directory containing data.yaml.
# finetune_medsam.py REQUIRES --yolo-ds (train/val/test come from here).
YOLO_DS_DIR="/scratch/st-ipor-1/cperez/MedSAM/YOLO_DS_P"

# Pretrained YOLO weights (two-class OD/OC detector)
YOLO_WEIGHTS="/scratch/st-ipor-1/cperez/MedSAM/runs/run2/Train/Train/weights/best.pt"

# Optional resume (trainable parts only, i.e., decoder)
#RESUME="${REPO_DIR}/runs/FineTune_MedSAM/weights/last.pth"

############################
# Runtime configuration    #
############################
DEVICE="cuda:0"
BATCH=8
EPOCHS=100
IMGSZ=1024

# Box-mixing and robustness
USE_DET_PROB=0.50        # probability to use detector boxes during TRAIN
PAD_JITTER=0.30          # random padding fraction
BOX_JITTER_TR=0.05       # translation jitter
BOX_JITTER_SC=0.10       # scale jitter

# YOLO inference settings for box prompts
YOLO_DEVICE="${DEVICE}"
YOLO_IMGSZ=640
YOLO_CONF=0.25
YOLO_IOU=0.50

# Testing configuration
TEST_AFTER_TRAIN=1       # set 0 to skip test stage
TEST_PROMPT="det"        # "det" or "gt"
TEST_WEIGHTS=""          # optional: path to a specific checkpoint for testing

# Full model finetuning toggle (default is decoder-only)
FULL_FINETUNE=0          # set 1 to pass --unfreeze-encoders

# Python module entrypoint
PYTHON_MODULE="src.scripts.finetune_decoder"

############################
# Environment / caches     #
############################
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export MPLCONFIGDIR="/scratch/st-ipor-1/cperez/MedSAM/.mplcache"
export TORCH_HOME="/scratch/st-ipor-1/cperez/MedSAM/.torch"
export HF_HOME="/scratch/st-ipor-1/cperez/MedSAM/.hf"
mkdir -p "${MPLCONFIGDIR}" "${TORCH_HOME}" "${HF_HOME}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Optional: GPU/CPU binding hints
SRUN_BIND=(--gpu-bind=closest --cpu-bind=cores)

# Activate environment
source "${REPO_DIR}/slurm/setup.sh"
setup_env

mkdir -p "${REPO_DIR}/logs" "/scratch/st-ipor-1/cperez/MedSAM/hpc_logs" "${OUT_DIR}" "${RUN_DIR}"

############################
# Sanity checks            #
############################
if [[ ! -f "${CKPT}" ]]; then
  echo "[FATAL] MedSAM checkpoint not found: ${CKPT}" >&2
  exit 2
fi
if [[ -z "${YOLO_DS_DIR}" || ! -f "${YOLO_DS_DIR}/data.yaml" ]]; then
  echo "[FATAL] YOLO dataset with data.yaml is required: ${YOLO_DS_DIR}" >&2
  exit 3
fi
if [[ ! -f "${YOLO_WEIGHTS}" ]]; then
  echo "[FATAL] YOLO weights not found: ${YOLO_WEIGHTS}" >&2
  exit 4
fi

########################################
# Build argument list (matches finetune_medsam.py)
########################################
ARGS=(
  --train
  --test
  --device "${DEVICE}"
  --data-root "${DATA_DIR}"
  --yolo-ds "${YOLO_DS_DIR}"
  --yolo-weights "${YOLO_WEIGHTS}"
  --yolo-device "${YOLO_DEVICE}"
  --yolo-imgsz "${YOLO_IMGSZ}"
  --yolo-conf "${YOLO_CONF}"
  --yolo-iou "${YOLO_IOU}"
  --ckpt "${CKPT}"
  --out-dir "${OUT_DIR}"
  --run-dir "${RUN_DIR}"
  --run-name "${RUN_NAME}"
  --epochs "${EPOCHS}"
  --batch "${BATCH}"
  --imgsz "${IMGSZ}"
  --use-det-prob "${USE_DET_PROB}"
  --pad-jitter "${PAD_JITTER}"
  --box-jitter-tr "${BOX_JITTER_TR}"
  --box-jitter-sc "${BOX_JITTER_SC}"
  --test-prompt "${TEST_PROMPT}"
#  --unfreeze-encoders
)

# Optional resume
if [[ "${RESUME:-}" != "" && -f "${RESUME}" ]]; then
  echo "[INFO] Resuming from: ${RESUME}"
  ARGS+=( --resume "${RESUME}" )
fi

# Optional: test with a specific checkpoint (overrides best/last selection)
if [[ "${TEST_WEIGHTS}" != "" && -f "${TEST_WEIGHTS}" ]]; then
  echo "[INFO] Using specified test checkpoint: ${TEST_WEIGHTS}"
  ARGS+=( --test-weights "${TEST_WEIGHTS}" )
fi

# Toggle full finetune (unfreeze encoders)
if [[ "${FULL_FINETUNE}" -eq 1 ]]; then
  echo "[INFO] Full finetune enabled (--unfreeze-encoders)."
  ARGS+=( --unfreeze-encoders )
fi

# Optionally skip test
if [[ "${TEST_AFTER_TRAIN}" -eq 0 ]]; then
  # Remove --test if present
  TMP=()
  for a in "${ARGS[@]}"; dot
    [[ "$a" == "--test" ]] && continue
    TMP+=("$a")
  done
  ARGS=("${TMP[@]}")
fi

########################################
# Launch
########################################
cd "${REPO_DIR}"
echo "[INFO] Running: python -m ${PYTHON_MODULE} ${ARGS[*]}"
srun -u "${SRUN_BIND[@]}" python -m ${PYTHON_MODULE} "${ARGS[@]}"