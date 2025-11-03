#!/bin/bash

setup_env() {
  # -------- Conda --------
  ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$ENV_PREFIX"

  # -------- Offline / Quiet --------
  export WANDB_DISABLED=true
  export HF_HUB_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export ULTRALYTICS_HUB=False
  export ULTRALYTICS_ANALYTICS=False

  # -------- Caches / Temp --------
  export XDG_CACHE_HOME="${REPO_DIR}/.cache"
  export XDG_CONFIG_HOME="${REPO_DIR}/.config"
  export ULTRALYTICS_CONFIG_DIR="${XDG_CONFIG_HOME}/Ultralytics"
  export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
  export TMPDIR="${REPO_DIR}/tmp"
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
#  nvidia-smi -L || true
}