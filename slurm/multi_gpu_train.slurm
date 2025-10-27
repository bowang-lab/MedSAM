#!/bin/bash
#SBATCH --job-name=bb-train-flex
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # we'll launch our own srun steps
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --gpus=4                    # MUST be >= (A_GPUS + B_GPUS) in parallel mode
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

# ======================= CONFIG (edit these) =======================
# TRAIN_MODE: "parallel" (A & B at once), "A-only", "B-only"
TRAIN_MODE="parallel"

# Per-stage GPU counts for parallel mode:
A_GPUS=2            # Stage A (disc-only) GPUs
B_GPUS=2            # Stage B (cup-ROI) GPUs

# (Optional) Per-stage CPU shares (dataloader/threads). They should sum <= #SBATCH --cpus-per-task.
A_CPUS=12
B_CPUS=12

# Project / env paths
PROJECT_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"

# Datasets
BB_DIR="${PROJECT_DIR}/bounding_box"
RUNS_DIR="${BB_DIR}/runs/detect"
DATA_ROOT="${BB_DIR}/data/yolo_split"           # clean (val/test)
AUG_ROOT="${BB_DIR}/data/yolo_split_aug"        # augmented (train if needed)
ROI_OUT="${BB_DIR}/data/yolo_split_cupROI"      # prebuilt ROI (Stage B)
DISC_ROOT="${BB_DIR}/data/yolo_split_disc_only" # prebuilt disc-only (Stage A)

# Weights / names
YOLO_WEIGHTS="${BB_DIR}/weights/yolov8n.pt"
STAGEA_NAME="stageA_disc_only"
STAGEB_NAME="stageB_cup_roi"

# Optional two-stage inference output
OUT_JSONL="${BB_DIR}/data/two_stage_boxes.jsonl"
# ==================================================================

# ===== Export ALL config so srun subshells can see them =====
export TRAIN_MODE A_GPUS B_GPUS A_CPUS B_CPUS
export PROJECT_DIR ENV_PREFIX BB_DIR RUNS_DIR
export DATA_ROOT AUG_ROOT ROI_OUT DISC_ROOT
export YOLO_WEIGHTS STAGEA_NAME STAGEB_NAME OUT_JSONL

# ========================= Caches / temp =========================
export XDG_CACHE_HOME="/scratch/st-ipor-1/cperez/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TMPDIR="${PROJECT_DIR}/tmp"
export YOLO_CONFIG_DIR="${PROJECT_DIR}/.ultralytics"   # avoid user-home warning
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR" "$YOLO_CONFIG_DIR"
mkdir -p "${PROJECT_DIR}/logs" "${RUNS_DIR}" "${BB_DIR}/data"

cd "${PROJECT_DIR}"

# ========================= Env =========================
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

echo "SLURM Job ID : ${SLURM_JOB_ID}"
echo "Node list    : ${SLURM_JOB_NODELIST}"
echo "Working dir  : $(pwd)"
echo "Runs dir     : ${RUNS_DIR}"
echo "DATA_ROOT    : ${DATA_ROOT}"
echo "AUG_ROOT     : ${AUG_ROOT}"
echo "DISC_ROOT    : ${DISC_ROOT}"
echo "ROI_OUT      : ${ROI_OUT}"
nvidia-smi -L || echo "nvidia-smi not available"

python - <<'PY'
import torch, ultralytics
print(f"[ENV] torch {torch.__version__} | cuda={torch.cuda.is_available()} | gpus={torch.cuda.device_count()}")
print(f"[ENV] ultralytics {ultralytics.__version__}")
PY

export PYTHONUNBUFFERED=1

# ========================= Helpers =========================
die() { echo "[ERROR] $*" >&2; exit 1; }
check_file() { [[ -f "$1" ]] || die "$2 not found: $1"; }
check_dir()  { [[ -d "$1" ]] || die "$2 not found: $1"; }

# Count GPUs available to the job
alloc_gpu_count() {
  if [[ -n "${SLURM_GPUS:-}" ]]; then echo "${SLURM_GPUS}"; return; fi
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then echo "${SLURM_GPUS_ON_NODE}"; return; fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a arr <<< "${CUDA_VISIBLE_DEVICES}"; echo "${#arr[@]}"; return
  fi
  nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
}

ensure_parallel_capacity() {
  local need=$((A_GPUS + B_GPUS))
  local have; have="$(alloc_gpu_count)"
  if (( need <= 0 )); then die "A_GPUS + B_GPUS must be >= 1"; fi
  if (( have < need )); then
    die "Parallel mode needs ${need} GPUs but only ${have} are allocated. Increase #SBATCH --gpus."
  fi
  if (( A_CPUS + B_CPUS > SLURM_CPUS_PER_TASK )); then
    echo "[WARN] A_CPUS+B_CPUS > --cpus-per-task; steps will compete for CPU." >&2
  fi
}

# Create a "0,1,2,...,N-1" list for YOLO_DEVICES inside each task (relative to that task's visibility)
make_devices_csv() {
  local n="${1}"
  local out="0"
  for ((i=1;i<n;i++)); do out+=",${i}"; done
  echo "$out"
}

# ========================= Stage A =========================
run_stageA() {
  echo "[STAGE A] Train disc-only detector…"
  echo "[STAGE A] DISC_ROOT=${DISC_ROOT}"
  echo "[STAGE A] DATA_ROOT=${DATA_ROOT} | AUG_ROOT=${AUG_ROOT}"

  # auto-resume if a previous run exists
  local last_ckpt="${RUNS_DIR}/${STAGEA_NAME}/weights/last.pt"
  local RESUME_SWITCH=""
  if [[ -f "${last_ckpt}" ]]; then
    echo "[STAGE A] Found previous checkpoint → ${last_ckpt}"
    RESUME_SWITCH="--resume auto"
  fi

  if [[ -d "${DISC_ROOT}" && -f "${DISC_ROOT}/od_only.yaml" ]]; then
    echo "[STAGE A] Using PREBUILT disc-only dataset at ${DISC_ROOT}"
    python "${BB_DIR}/train/train_stageA_disc_only.py" \
      --project_dir "${PROJECT_DIR}" \
      --data_root   "${DISC_ROOT}" \
      --model       "${YOLO_WEIGHTS}" \
      --name        "${STAGEA_NAME}" \
      ${RESUME_SWITCH}
  else
    echo "[STAGE A] Prebuilt disc-only dataset NOT found; building from clean+aug."
    check_dir "${DATA_ROOT}" "Clean YOLO split"
    check_dir "${AUG_ROOT}"  "Augmented YOLO split"
    python "${BB_DIR}/train/train_stageA_disc_only.py" \
      --project_dir "${PROJECT_DIR}" \
      --data_root   "${DATA_ROOT}" \
      --aug_root    "${AUG_ROOT}" \
      --train_splits train \
      --model       "${YOLO_WEIGHTS}" \
      --name        "${STAGEA_NAME}" \
      ${RESUME_SWITCH}
  fi

  local best="${RUNS_DIR}/${STAGEA_NAME}/weights/best.pt"
  check_file "${best}" "Stage A best checkpoint"
  echo "[STAGE A] Best checkpoint: ${best}"
}

# ========================= Stage B =========================
run_stageB() {
  echo "[STAGE B] Train cup-only detector on prebuilt ROI dataset…"
  echo "[STAGE B] ROI_OUT=${ROI_OUT}"
  check_dir "${ROI_OUT}" "ROI dataset"

  local last_ckpt="${RUNS_DIR}/${STAGEB_NAME}/weights/last.pt"
  local RESUME_SWITCH=""
  if [[ -f "${last_ckpt}" ]]; then
    echo "[STAGE B] Found previous checkpoint → ${last_ckpt}"
    RESUME_SWITCH="--resume auto"
  fi

  python "${BB_DIR}/train/train_stageB_cup_roi.py" \
    --project_dir "${PROJECT_DIR}" \
    --name        "${STAGEB_NAME}" \
    ${RESUME_SWITCH}

  local best="${RUNS_DIR}/${STAGEB_NAME}/weights/best.pt"
  check_file "${best}" "Stage B best checkpoint"
  echo "[STAGE B] Best checkpoint: ${best}"
}

# ========================= Stage B =========================
run_stageB() {
  echo "[STAGE B] Train cup-only detector on prebuilt ROI dataset…"
  echo "[STAGE B] ROI_OUT=${ROI_OUT}"
  check_dir "${ROI_OUT}" "ROI dataset"

  python "${BB_DIR}/train/train_stageB_cup_roi.py" \
    --project_dir "${PROJECT_DIR}"

  local best="${RUNS_DIR}/${STAGEB_NAME}/weights/best.pt"
  check_file "${best}" "Stage B best checkpoint"
  echo "[STAGE B] Best checkpoint: ${best}"
}

# Export functions for srun steps
export -f run_stageA run_stageB die check_file check_dir

# For robust step handling
SRUN_COMMON="--kill-on-bad-exit=1 --exclusive -N1 -n1"

# ========================= Dispatcher =========================
case "${TRAIN_MODE}" in
  parallel)
    echo "[MODE] PARALLEL: Stage A and Stage B concurrently, each using multiple GPUs"
    ensure_parallel_capacity

    # Build relative device lists per step, e.g., "0,1" for 2 GPUs.
    A_DEV="$(make_devices_csv "${A_GPUS}")"
    B_DEV="$(make_devices_csv "${B_GPUS}")"

    # Launch A & B as separate srun steps in background, capture PIDs
    pids=()
    names=()

    srun ${SRUN_COMMON} \
         --gpus-per-task="${A_GPUS}" \
         --cpus-per-task="${A_CPUS}" \
         --export=ALL,YOLO_DEVICES="${A_DEV}",OMP_NUM_THREADS="${A_CPUS}" \
         bash -lc 'run_stageA' &
    pids+=($!)
    names+=("StageA")

    srun ${SRUN_COMMON} \
         --gpus-per-task="${B_GPUS}" \
         --cpus-per-task="${B_CPUS}" \
         --export=ALL,YOLO_DEVICES="${B_DEV}",OMP_NUM_THREADS="${B_CPUS}" \
         bash -lc 'run_stageB' &
    pids+=($!)
    names+=("StageB")

    # Wait for both; if any fails, kill the other and exit 1
    overall_rc=0
    for i in "${!pids[@]}"; do
      pid="${pids[$i]}"
      name="${names[$i]}"
      if ! wait "$pid"; then
        echo "[ERROR] ${name} failed (PID ${pid}). Terminating remaining steps…" >&2
        overall_rc=1
        # terminate the other(s)
        for j in "${!pids[@]}"; do
          if [[ $j -ne $i ]]; then
            kill -TERM "${pids[$j]}" 2>/dev/null || true
          fi
        done
      fi
    done

    if (( overall_rc != 0 )); then
      echo "[FATAL] One or more stages failed." >&2
      exit 1
    fi
    ;;

  A-only)
    echo "[MODE] Stage A only"
    export YOLO_DEVICES="$(make_devices_csv "${A_GPUS}")"
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
    srun ${SRUN_COMMON} --gpus-per-task="${A_GPUS}" bash -lc 'run_stageA'
    ;;

  B-only)
    echo "[MODE] Stage B only"
    export YOLO_DEVICES="$(make_devices_csv "${B_GPUS}")"
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
    srun ${SRUN_COMMON} --gpus-per-task="${B_GPUS}" bash -lc 'run_stageB'
    ;;

  *)
    die "Unknown TRAIN_MODE='${TRAIN_MODE}'. Use: parallel | A-only | B-only"
    ;;
esac

# ========================= (Optional) Inference =========================
run_infer_optional() {
  local test_images="${DATA_ROOT}/images/test"
  if [[ -d "${test_images}" ]]; then
    echo "[INFER] Two-stage inference on clean test images…"
    local bestA="${RUNS_DIR}/${STAGEA_NAME}/weights/best.pt"
    local bestB="${RUNS_DIR}/${STAGEB_NAME}/weights/best.pt"
    check_file "${bestA}" "Stage A best checkpoint"
    check_file "${bestB}" "Stage B best checkpoint"

    python "${BB_DIR}/train/infer_two_stage_to_boxes.py" \
      --stageA "${bestA}" \
      --stageB "${bestB}" \
      --images "${test_images}" \
      --out_jsonl "${OUT_JSONL}" \
      --od_pad_pct 0.08 \
      --confA 0.25 \
      --confB 0.10
    echo "[INFER] Wrote: ${OUT_JSONL}"
  else
    echo "[WARN] No clean test split at ${test_images}; skipping inference."
  fi
}

# Uncomment to run inference automatically
# run_infer_optional

echo "[DONE] Training finished at $(date)"