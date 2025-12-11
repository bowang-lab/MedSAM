#!/usr/bin/env bash
# File: slurm/setup.sh

set -euo pipefail

setup_env() {
  : "${REPO_DIR:?REPO_DIR must be set before calling setup_env}"

  # ---- Conda env ----
  ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$ENV_PREFIX"

  # ---- Disable external telemetry/networky stuff ----
  export WANDB_DISABLED=true
  export HF_HUB_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export ULTRALYTICS_HUB=False
  export ULTRALYTICS_ANALYTICS=False

  # ---- Localize caches/tmp to project scratch ----
  export XDG_CACHE_HOME="${REPO_DIR}/.cache"
  export XDG_CONFIG_HOME="${REPO_DIR}/.config"
  export ULTRALYTICS_CONFIG_DIR="${XDG_CONFIG_HOME}/Ultralytics"
  export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
  export TMPDIR="${REPO_DIR}/tmp"
  mkdir -p \
    "$XDG_CACHE_HOME" \
    "$XDG_CONFIG_HOME" \
    "$ULTRALYTICS_CONFIG_DIR" \
    "$ULTRALYTICS_CONFIG_DIR/DDP" \
    "$MPLCONFIGDIR" \
    "$TMPDIR"

  # ---- Threads: one process per task, so just use cpus-per-task ----
  CPUS="${SLURM_CPUS_PER_TASK:-6}"
  export OMP_NUM_THREADS="${CPUS}"
  export MKL_NUM_THREADS="${CPUS}"
  export OPENBLAS_NUM_THREADS="${CPUS}"
  export NUMEXPR_NUM_THREADS="${CPUS}"

  export PYTHONUNBUFFERED=1

  # ---- CUDA allocator options (helps fragmentation) ----
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"

  # ---- NCCL notes ----
  # For your current launch style (srun spawning independent workers, no DDP comm),
  # NCCL settings don't matter much. These are safe defaults.
  export NCCL_DEBUG="${NCCL_DEBUG:-warn}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

  # ---- GPU visibility ----
  # With --gpus-per-task=1, Slurm usually sets CUDA_VISIBLE_DEVICES per task.
  # Only set it if Slurm explicitly provides SLURM_JOB_GPUS and CUDA_VISIBLE_DEVICES is unset.
if [[ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
      export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
  fi

  echo "=== setup_env summary ==="
  echo "REPO_DIR=${REPO_DIR}"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
  echo "SLURM_NODELIST=${SLURM_NODELIST:-unset}"
  echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"
  echo "SLURM_PROCID=${SLURM_PROCID:-unset}"
  echo "SLURM_LOCALID=${SLURM_LOCALID:-unset}"
  echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
  echo "SLURM_GPUS_PER_TASK=${SLURM_GPUS_PER_TASK:-unset}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
  echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
  echo "========================="
}