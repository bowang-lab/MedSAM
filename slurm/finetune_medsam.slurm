#!/bin/bash
#SBATCH --job-name=finetune_medsam
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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
OUT_DIR="/scratch/st-ipor-1/cperez/MedSAM/TRAINING_DS"
DATA_DIR="/arc/project/st-ipor-1/carlosp/fundus_data/All_Datasets_Organized"
CKPT="${REPO_DIR}/weights/medsam_vit_b.pth"
RESUME="${REPO_DIR}/runs/FineTune_MedSAM/weights/last.pth"   # optional, used if exists

# Optional: YOLO weights for imperfect-box training (leave empty to disable)
YOLO_WEIGHTS="/scratch/st-ipor-1/cperez/MedSAM/runs/MultiClassLarge/weights/best.pt"

############################
# Runtime configuration    #
############################
DEVICE="cuda:0"
BATCH=8
EPOCHS=50
IMGSZ=1024
RUN_DIR="${REPO_DIR}/runs"
RUN_NAME="FineTune_MedSAM_${SLURM_JOB_ID:-local}"

# Python module entrypoint
PYTHON_MODULE="src.scripts.finetune_medsam"

# Make project importable as `src.*`
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Mitigate cache permission errors on HPC headless nodes
export MPLCONFIGDIR="/scratch/st-ipor-1/cperez/MedSAM/.mplcache"
export TORCH_HOME="/scratch/st-ipor-1/cperez/MedSAM/.torch"
export HF_HOME="/scratch/st-ipor-1/cperez/MedSAM/.hf"
mkdir -p "${MPLCONFIGDIR}" "${TORCH_HOME}" "${HF_HOME}"

# Threading hints
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Optional: GPU/CPU binding hints
SRUN_BIND="--gpu-bind=closest --cpu-bind=cores"

# Activate environment
source "${REPO_DIR}/slurm/setup.sh"
setup_env

mkdir -p "${REPO_DIR}/logs" "/scratch/st-ipor-1/cperez/MedSAM/hpc_logs" "${OUT_DIR}" "${RUN_DIR}"

########################################
# Build argument list
########################################
ARGS=(
  --train
  --device "${DEVICE}"
  --data-root "${DATA_DIR}"
  --ckpt "${CKPT}"
  --out-dir "${OUT_DIR}"
  --run-dir "${RUN_DIR}"
  --run-name "${RUN_NAME}"
  --epochs "${EPOCHS}"
  --batch "${BATCH}"
  --imgsz "${IMGSZ}"
  # Box robustness (decoder-only finetune is default; do NOT pass --unfreeze-encoders)
  --pad-jitter 0.30
  --box-trans 0.05
  --box-scale 0.10
  # Penalize spill outside prompt box (helps when boxes are loose)
  --spill-lam 0.30
  --spill-dilate 0.05
  # Optionally add point prompts to stabilize decoder (uncomment to enable)
  # --use-points --neg-points 3
)

# Resume if a previous checkpoint is available
if [[ -f "${RESUME}" ]]; then
  ARGS+=( --resume "${RESUME}" )
fi

# If YOLO weights provided, train with imperfect detector boxes some of the time
if [[ -n "${YOLO_WEIGHTS}" && -f "${YOLO_WEIGHTS}" ]]; then
  ARGS+=( --yolo-weights "${YOLO_WEIGHTS}" --yolo-device "${DEVICE}" --yolo-imgsz 640 --use-det-prob 0.50 )
fi

########################################
# Launch
########################################
cd "${REPO_DIR}"

srun -u ${SRUN_BIND} python -m "${PYTHON_MODULE}" "${ARGS[@]}"