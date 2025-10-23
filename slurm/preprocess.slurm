#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail
# set -x  # uncomment to echo commands for debugging

# -------- Paths --------
PROJ_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"

# Make project importable as `src.*`
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

TRAIN_MOD="src.scripts.preprocess"
RUNS_DIR="${PROJ_DIR}/runs/detect"
WEIGHTS_LOCAL="${PROJ_DIR}/weights/yolo12x.pt"
EXP_NAME="preprocess"

CONFIG="${PROJ_DIR}/src/configs/preprocess_hpc.yaml"

mkdir -p "${PROJ_DIR}/logs" "${RUNS_DIR}"

# -------- Conda --------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# -------- Offline / Quiet --------
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export ULTRALYTICS_HUB=False
export ULTRALYTICS_ANALYTICS=False

# -------- Caches / Temp --------
export XDG_CACHE_HOME="${PROJ_DIR}/.cache"
export XDG_CONFIG_HOME="${PROJ_DIR}/.config"
export ULTRALYTICS_CONFIG_DIR="${XDG_CONFIG_HOME}/Ultralytics"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TMPDIR="${PROJ_DIR}/tmp"
mkdir -p "$XDG_CACHE_HOME" "$ULTRALYTICS_CONFIG_DIR/DDP" "$MPLCONFIGDIR" "$TMPDIR"

# -------- Threads & Allocator --------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"

# -------- NCCL (single node) --------
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=warn
export MASTER_PORT=$(( 10000 + (RANDOM % 50000) ))  # avoid port collisions on shared nodes

# -------- GPU visibility & Ultralytics multi-GPU hint --------
# Map requested GPUs to local ordinals 0..N-1 and expose to PyTorch
CUDA_LIST=$(python - <<'PY'
import os
n = int(os.environ.get("SLURM_GPUS", "1"))
print(",".join(str(i) for i in range(n)))
PY
)
export CUDA_VISIBLE_DEVICES="${CUDA_LIST}"
# Let your ultralytics_device_arg() pick these up for DDP ("0,1,2,3", etc.)
export YOLO_DEVICES="${CUDA_LIST}"

echo "GPUs requested: ${SLURM_GPUS:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "YOLO_DEVICES=${YOLO_DEVICES}"
nvidia-smi -L || true

# -------- Build args safely --------
ARGS=(
  --config "${CONFIG}"
)

cd "${PROJ_DIR}"
srun -u python -m "${TRAIN_MOD}" "${ARGS[@]}"

# -------- Post-run check --------
BEST="${RUNS_DIR}/${EXP_NAME}/weights/best.pt"
[[ -f "${BEST}" ]] || { echo "[ERR] best.pt not found at ${BEST}" >&2; exit 3; }

echo "COMPLETED YOLO SPLIT SAVED"