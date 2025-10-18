#!/bin/bash
#SBATCH --job-name=cup-roi-y12x
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gpus=4
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail
# set -x  # uncomment to echo commands for debugging

# -------- Paths --------
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"

# Make project importable as `src.*`
export PYTHONPATH="${SCRATCH_DIR}:${PYTHONPATH:-}"

BB_DIR="${SCRATCH_DIR}/bounding_box"
TRAIN_MOD="src.model.train_cup"                       # run as a module to avoid import issues
RUNS_DIR="${SCRATCH_DIR}/runs/detect"
ROI_ROOT="${BB_DIR}/data/yolo_split_cupROI"
WEIGHTS_LOCAL="${BB_DIR}/weights/yolo12x.pt"
EXP_NAME="stageB_cup_roi_y12x"

mkdir -p "${SCRATCH_DIR}/logs" "${RUNS_DIR}"

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
export XDG_CACHE_HOME="${SCRATCH_DIR}/.cache"
export XDG_CONFIG_HOME="${SCRATCH_DIR}/.config"
export ULTRALYTICS_CONFIG_DIR="${XDG_CONFIG_HOME}/Ultralytics"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TMPDIR="${SCRATCH_DIR}/tmp"
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

# -------- Sanity checks --------
[[ -d "${ROI_ROOT}"     ]] || { echo "[ERR] ROI root not found: ${ROI_ROOT}" >&2; exit 2; }
[[ -f "${WEIGHTS_LOCAL}" ]]|| { echo "[ERR] Local weights not found: ${WEIGHTS_LOCAL}" >&2; exit 2; }

# data.yaml convenience symlink if needed
if [[ -f "${ROI_ROOT}/cup_roi.yaml" && ! -f "${ROI_ROOT}/data.yaml" ]]; then
  ln -sf "${ROI_ROOT}/cup_roi.yaml" "${ROI_ROOT}/data.yaml"
fi
[[ -f "${ROI_ROOT}/data.yaml" ]] || { echo "[ERR] Missing data.yaml at ${ROI_ROOT}/data.yaml" >&2; exit 2; }

# -------- Show CLI help (first lines) --------
python -m "${TRAIN_MOD}" --help | sed -n '1,40p'

# -------- Build args safely --------
ARGS=(
  --data-root "${ROI_ROOT}"
  --runs-root "${RUNS_DIR}"
  --weights   "${WEIGHTS_LOCAL}"
  --family    auto
  --size      x
  --epochs    200
  --imgsz     512
  --batch     16
  --optimizer SGD
  --mosaic    0.0
  --freeze    5
  --amp       true
  --name      "${EXP_NAME}"
  --workers   "${SLURM_CPUS_PER_TASK}"
)

# -------- Train --------
cd "${SCRATCH_DIR}"
srun -u python -m "${TRAIN_MOD}" "${ARGS[@]}"

# -------- Post-run check --------
BEST="${RUNS_DIR}/${EXP_NAME}/weights/best.pt"
[[ -f "${BEST}" ]] || { echo "[ERR] best.pt not found at ${BEST}" >&2; exit 3; }

echo "[OK] best.pt → ${BEST}"