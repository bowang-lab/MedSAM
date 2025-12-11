#!/bin/bash
#SBATCH --job-name=Experiment
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --gpus=4
#SBATCH --constraint=gpu_mem_32
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

###############################################################################
# 1. Environment Setup
###############################################################################
REPO_DIR="/scratch/st-ipor-1/cperez/MedSAM"
export REPO_DIR
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Load environment (ensure setup.sh exists or conda is active)
if [[ -f "${REPO_DIR}/slurm/setup.sh" ]]; then
    source "${REPO_DIR}/slurm/setup.sh"
    setup_env
fi

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/hpc_logs"

###############################################################################
# 2. Device Configuration
###############################################################################
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  YOLO_DEVICES="${CUDA_VISIBLE_DEVICES}"
else
  YOLO_DEVICES="0,1,2,3"
fi
echo "[INFO] YOLO_DEVICES=${YOLO_DEVICES}"

###############################################################################
# 3. CLI Parsing & Validation
###############################################################################

# Check if Mode argument is provided
if [[ $# -lt 1 ]]; then
    echo "===================================================================="
    echo "[ERROR] No execution mode specified."
    echo "Usage: sbatch exp_yolo.slurm <MODE> [OPTIONS]"
    echo ""
    echo "Available Modes:"
    echo "  1  - Standard training from raw folders"
    echo "  2  - Training from parquet splits"
    echo "  3  - Semi-supervised YOLO finetuning"
    echo "  4  - Full pipeline (Semi-Supervised + MedSAM)"
    echo "  5  - Predictions only (Standalone)"
    echo "  6  - Training + Predictions pipeline"
    echo "  7  - Semi-supervised + Predictions pipeline"
    echo "  8  - MedSAM finetuning only"
    echo "===================================================================="
    exit 1
fi

MODE="$1"
shift # Remove mode from args so we can parse overrides

# Defaults for Overrides
CONF_THRESH="${CONF_THRESH:-0.85}"
DUP_FACTOR="${DUP_FACTOR:-2}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"

# Parse overrides from remaining arguments
for arg in "$@"; do
  case "$arg" in
    --conf=*)     CONF_THRESH="${arg#*=}" ;;
    --dup=*)      DUP_FACTOR="${arg#*=}" ;;
    --epochs=*)   FINETUNE_EPOCHS="${arg#*=}" ;;
  esac
done

# Calculate percentage for filenames (e.g., 0.85 -> 85)
CONF_PCT=$(python3 -c "print(int(round(${CONF_THRESH} * 100)))")

echo "========================================"
echo "RUNNING MODE: ${MODE}"
echo "CONFIDENCE:   ${CONF_THRESH} (${CONF_PCT}%)"
echo "DUP FACTOR:   ${DUP_FACTOR}"
echo "EPOCHS:       ${FINETUNE_EPOCHS}"
echo "========================================"

###############################################################################
# 4. Mode Definitions (Hardcoded Arguments)
###############################################################################

# --- MODE 1: Standard Training (Raw Folders) ---
ARG_1=(
    "--create-ds"
    "--train-yolo"
    "--test-yolo"
    "--yolo-out-dir" "/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS"
    "--finetune-runs-root" "/scratch/st-ipor-1/cperez/MedSAM/runs"
    "--epochs" "100"
    "--batch" "16"
    "--imgsz" "640"
    "--workers" "8"
    "--yolo-device" "${YOLO_DEVICES}"
)

# --- MODE 2: Training from Parquet Splits ---
ARG_2=(
    "--create-ds"
    "--train-yolo"
    "--test-yolo"
    "--splits-parquet" "/scratch/st-ipor-1/cperez/MedSAM/src/image_data/splits_final.parquet"
    "--yolo-out-dir" "/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS_PARQUET"
    "--finetune-runs-root" "/scratch/st-ipor-1/cperez/MedSAM/runs"
    "--epochs" "100"
    "--batch" "16"
    "--imgsz" "640"
    "--workers" "8"
    "--yolo-device" "${YOLO_DEVICES}"
)

# --- MODE 3: Semi-Supervised YOLO Finetuning ---
ARG_3=(
    "--finetune-yolo"
    "--images-parquet" "/scratch/st-ipor-1/cperez/MedSAM/src/image_data/processed_predictions_conf${CONF_PCT}_dup${DUP_FACTOR}.parquet"
    "--init-weights" "/scratch/st-ipor-1/cperez/MedSAM/runs/YOLO_DEV_RUN/Train/Train/weights/best.pt"
    "--yolo-out-dir" "/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS_SemiSupervised_conf${CONF_PCT}_dup${DUP_FACTOR}"
    "--finetune-runs-root" "/scratch/st-ipor-1/cperez/MedSAM/runs/SemiSupervised/conf${CONF_PCT}_dup${DUP_FACTOR}"
    "--epochs" "${FINETUNE_EPOCHS}"
    "--batch" "16"
    "--imgsz" "640"
    "--workers" "8"
    "--skip-materialization"
    "--yolo-device" "${YOLO_DEVICES}"
)

# --- MODE 4: Full Pipeline (Semi-Supervised + MedSAM) ---
ARG_4=(
    "--finetune-yolo"
    "--test-yolo"
    "--finetune-medsam"
    "--images-parquet" "/scratch/st-ipor-1/cperez/MedSAM/src/image_data/processed_predictions_conf${CONF_PCT}_dup${DUP_FACTOR}.parquet"
    "--init-weights" "/scratch/st-ipor-1/cperez/MedSAM/runs/YOLO_DEV_RUN/Train/Train/weights/best.pt"
    "--yolo-out-dir" "/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS_SemiSupervised_conf${CONF_PCT}_dup${DUP_FACTOR}"
    "--finetune-runs-root" "/scratch/st-ipor-1/cperez/MedSAM/runs/SemiSupervised/conf${CONF_PCT}_dup${DUP_FACTOR}"
    "--epochs" "${FINETUNE_EPOCHS}"
    "--batch" "16"
    "--imgsz" "640"
    "--workers" "8"
    "--skip-materialization"
    "--yolo-device" "${YOLO_DEVICES}"
)

# --- MODE 5: Predictions Only ---
ARG_5=(
    "--predict"
    "--yolo-weights" "/scratch/st-ipor-1/cperez/MedSAM/runs/run13/Train/Train/weights/best.pt"
    "--medsam-ckpt" "/scratch/st-ipor-1/cperez/MedSAM/weights/medsam_vit_b.pth"
    "--predict-parquet" "/scratch/st-ipor-1/cperez/MedSAM/src/image_data/images_dev.parquet"
    "--predict-out-dir" "/scratch/st-ipor-1/cperez/MedSAM/PREDICT/experiment_predictions"
    "--predict-splits" "test"
    "--predict-batch-size" "128"
    "--predict-sam-amp"
    "--yolo-device" "${YOLO_DEVICES}"
)

# --- MODE 6: Training + Predictions Pipeline ---
ARG_6=(
    "--create-ds"
    "--train-yolo"
    "--test-yolo"
    "--predict"
    "--splits-parquet" "/scratch/st-ipor-1/cperez/MedSAM/src/image_data/splits_final.parquet"
    "--yolo-out-dir" "/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS_Mode6"
    "--finetune-runs-root" "/scratch/st-ipor-1/cperez/MedSAM/runs"
    "--medsam-ckpt" "/scratch/st-ipor-1/cperez/MedSAM/weights/medsam_vit_b.pth"
    "--epochs" "100"
    "--batch" "16"
    "--imgsz" "640"
    "--workers" "8"
    "--predict-splits" "test"
    "--predict-batch-size" "128"
    "--predict-sam-amp"
    "--yolo-device" "${YOLO_DEVICES}"
)

# --- MODE 7: Semi-Supervised + Predictions Pipeline ---
ARG_7=(
    "--finetune-yolo"
    "--test-yolo"
    "--predict"
    "--images-parquet" "/scratch/st-ipor-1/cperez/MedSAM/src/image_data/processed_predictions_conf${CONF_PCT}_dup${DUP_FACTOR}.parquet"
    "--init-weights" "/scratch/st-ipor-1/cperez/MedSAM/runs/YOLO_DEV_RUN/Train/Train/weights/best.pt"
    "--yolo-out-dir" "/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS_SemiSupervised_conf${CONF_PCT}_dup${DUP_FACTOR}"
    "--finetune-runs-root" "/scratch/st-ipor-1/cperez/MedSAM/runs/SemiSupervised/conf${CONF_PCT}_dup${DUP_FACTOR}"
    "--medsam-ckpt" "/scratch/st-ipor-1/cperez/MedSAM/weights/medsam_vit_b.pth"
    "--epochs" "${FINETUNE_EPOCHS}"
    "--batch" "16"
    "--imgsz" "640"
    "--workers" "8"
    "--skip-materialization"
    "--predict-splits" "test"
    "--predict-batch-size" "128"
    "--predict-sam-amp"
    "--yolo-device" "${YOLO_DEVICES}"
)

# --- MODE 8: MedSAM Finetuning Only ---
ARG_8=(
    "--finetune-medsam"
    "--yolo-weights" "/scratch/st-ipor-1/cperez/MedSAM/runs/YOLO_DEV_RUN/Train/Train/weights/best.pt"
    "--yolo-ds" "/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS"
    "--medsam-ckpt" "/scratch/st-ipor-1/cperez/MedSAM/weights/medsam_vit_b.pth"
    "--yolo-out-dir" "/scratch/st-ipor-1/cperez/MedSAM/runs/Fundu"
    "--epochs" "100"
    "--batch" "4"
    "--imgsz" "1024"
    "--workers" "4"
    "--yolo-device" "${YOLO_DEVICES}"
    "--grad-checkpointing"
    "--eval-every" "5"
)

###############################################################################
# 5. Selection & Execution
###############################################################################

# Select argument array based on MODE
case $MODE in
  1) CMD_ARGS=("${ARG_1[@]}") ;;
  2) CMD_ARGS=("${ARG_2[@]}") ;;
  3) CMD_ARGS=("${ARG_3[@]}") ;;
  4) CMD_ARGS=("${ARG_4[@]}") ;;
  5) CMD_ARGS=("${ARG_5[@]}") ;;
  6) CMD_ARGS=("${ARG_6[@]}") ;;
  7) CMD_ARGS=("${ARG_7[@]}") ;;
  8) CMD_ARGS=("${ARG_8[@]}") ;;
  *)
    echo "[ERROR] Invalid Mode: $MODE. Valid modes are 1-8."
    exit 1
    ;;
esac

# Append any remaining overrides/args passed via command line
CMD_ARGS+=("$@")

echo "========================================================================"
echo "[INFO] Executing Python Command:"
echo "python -m src.experiment ${CMD_ARGS[*]}"
echo "========================================================================"

# Optimizing memory allocator for large models to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun --gpu-bind=closest --cpu-bind=cores -u \
  python -m src.experiment "${CMD_ARGS[@]}"

echo "[INFO] Job Finished Successfully"