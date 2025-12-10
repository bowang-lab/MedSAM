#!/bin/bash
#SBATCH --job-name=Experiment
#SBATCH --account=st-ipor-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --gpus=4
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

############################
# 1. Environment Setup
############################
REPO_DIR="/scratch/st-ipor-1/cperez/MedSAM"
export REPO_DIR
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

source "${REPO_DIR}/slurm/setup.sh"
setup_env

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/hpc_logs"

############################
# 2. Device Configuration
############################
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  YOLO_DEVICES="${CUDA_VISIBLE_DEVICES}"
else
  # Fallback logic
  YOLO_DEVICES="0,1,2,3"
fi
echo "YOLO_DEVICES=${YOLO_DEVICES}"

############################
# 3. Config Loading Helper
############################
# Hardcoded path to config files
CONFIG_DIR="${REPO_DIR}/slurm/configs"
LOAD_CONFIG_SCRIPT="${REPO_DIR}/slurm/load_config.py"

# Function to load config and build args
load_config_args() {
  local mode=$1
  shift  # Remove mode from args
  local override_args=("$@")
  
  # Build override string for Python script
  local override_str=""
  for arg in "${override_args[@]}"; do
    override_str="${override_str} ${arg}"
  done
  
  # Call Python script to load config and build args
  local json_output
  json_output=$(python3 "${LOAD_CONFIG_SCRIPT}" "${mode}" "${REPO_DIR}" "${YOLO_DEVICES}" ${override_str})
  
  if [[ $? -ne 0 ]]; then
    echo "[ERROR] Failed to load config for mode ${mode}" >&2
    exit 2
  fi
  
  # Extract args and paths from JSON using Python (always available)
  readarray -t ARGS < <(python3 -c "import json, sys; data=json.load(sys.stdin); [print(x) for x in data['args']]" <<< "${json_output}")
  
  # Extract paths using Python
  eval "$(python3 << 'PYTHON_EOF'
import json
import sys

data = json.load(sys.stdin)
paths = data.get('paths', {})

# Output bash variable assignments
for key, value in paths.items():
    if value:
        # Convert snake_case to UPPER_SNAKE_CASE
        var_name = key.upper()
        print(f'{var_name}="{value}"')
PYTHON_EOF
<<< "${json_output}"
)"
}

############################
# 4. Argument Selection Logic
############################
# Get the mode from command line arg 1, default to "1" if not provided
MODE="${1:-1}"

# Optional CLI overrides (these are passed to load_config_args)
# These can be passed as: sbatch script.slurm 3 --conf=0.85 --dup=3 --epochs=30
CONF_THRESH="${CONF_THRESH:-0.85}"
DUP_FACTOR="${DUP_FACTOR:-3}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-30}"

# Parse additional arguments and collect overrides
shift || true  # Skip first arg (MODE)
OVERRIDE_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --conf=*)
      CONF_THRESH="${arg#*=}"
      CONF_PCT=$(python3 -c "print(int(round(${CONF_THRESH} * 100)))")
      OVERRIDE_ARGS+=("CONF_PCT=${CONF_PCT}")
      ;;
    --dup=*)
      DUP_FACTOR="${arg#*=}"
      OVERRIDE_ARGS+=("DUP_FACTOR=${DUP_FACTOR}")
      ;;
    --epochs=*)
      FINETUNE_EPOCHS="${arg#*=}"
      OVERRIDE_ARGS+=("epochs=${FINETUNE_EPOCHS}")
      ;;
    --weights=*)
      # For mode 5: --weights maps to yolo_weights
      OVERRIDE_ARGS+=("yolo_weights=${arg#*=}")
      ;;
    --medsam=*)
      # For mode 5/8: --medsam maps to medsam_ckpt
      OVERRIDE_ARGS+=("medsam_ckpt=${arg#*=}")
      ;;
    --parquet=*)
      # For mode 5: --parquet maps to predict_parquet
      OVERRIDE_ARGS+=("predict_parquet=${arg#*=}")
      ;;
    --out=*)
      # For mode 5: --out maps to predict_out_dir
      OVERRIDE_ARGS+=("predict_out_dir=${arg#*=}")
      ;;
    --splits=*)
      # For mode 5: --splits maps to predict_splits
      OVERRIDE_ARGS+=("predict_splits=${arg#*=}")
      ;;
    --yolo-weights=*|--yolo-ds=*|--batch=*|--imgsz=*)
      # Convert --yolo-weights to yolo_weights, etc.
      key="${arg%%=*}"
      value="${arg#*=}"
      key="${key#--}"  # Remove --
      key="${key//-/_}"  # Convert - to _
      OVERRIDE_ARGS+=("${key}=${value}")
      ;;
  esac
done

# Calculate CONF_PCT if not already set
if [[ -z "${CONF_PCT:-}" ]]; then
  CONF_PCT=$(python3 -c "print(int(round(${CONF_THRESH} * 100)))")
fi

echo "========================================"
echo "RUNNING IN MODE: ${MODE}"
echo "========================================"

case $MODE in

  # ----------------------------------------------------------------
  # CONFIG 1: Standard / Stage Two (Raw Folders)
  # ----------------------------------------------------------------
  1)
    load_config_args 1 "${OVERRIDE_ARGS[@]}"
    ;;

  # ----------------------------------------------------------------
  # CONFIG 2: Parquet Splits (Reproducible)
  # ----------------------------------------------------------------
  2)
    load_config_args 2 "${OVERRIDE_ARGS[@]}"
    
    # Sanity check
    if [[ ! -f "${SPLITS_PARQUET}" ]]; then
      echo "[ERROR] Splits parquet not found: ${SPLITS_PARQUET}" >&2
      exit 2
    fi
    echo "[INFO] Source Parquet: ${SPLITS_PARQUET}"
    ;;

  # ----------------------------------------------------------------
  # CONFIG 3: Semi-Supervised YOLO Finetuning
  # Finetunes YOLO from pretrained weights using processed parquet
  # Usage: sbatch exp_yolo.slurm 3 [--conf=0.85] [--dup=3] [--epochs=30]
  # ----------------------------------------------------------------
  3)
    load_config_args 3 "${OVERRIDE_ARGS[@]}" "CONF_PCT=${CONF_PCT}" "DUP_FACTOR=${DUP_FACTOR}" "epochs=${FINETUNE_EPOCHS}"
    
    # Sanity checks
    if [[ ! -f "${IMAGES_PARQUET}" ]]; then
      echo "[ERROR] Processed parquet not found: ${IMAGES_PARQUET}" >&2
      echo "[INFO] Run the preprocessing pipeline first:"
      echo "       python -m src.tools.parquet_processing.run_process_predictions \\"
      echo "         --conf ${CONF_THRESH} --duplication-factor ${DUP_FACTOR}"
      exit 2
    fi

    if [[ ! -f "${INIT_WEIGHTS}" ]]; then
      echo "[ERROR] Init weights not found: ${INIT_WEIGHTS}" >&2
      exit 2
    fi

    echo "[INFO] Semi-Supervised YOLO Finetuning"
    echo "[INFO] Confidence Threshold: ${CONF_THRESH} (${CONF_PCT}%)"
    echo "[INFO] Duplication Factor: ${DUP_FACTOR}"
    echo "[INFO] Images Parquet: ${IMAGES_PARQUET}"
    echo "[INFO] Init Weights: ${INIT_WEIGHTS}"
    echo "[INFO] Output DS: ${OUT_DIR}"
    echo "[INFO] Runs Root: ${RUNS_ROOT}"
    ;;

  # ----------------------------------------------------------------
  # CONFIG 4: Semi-Supervised Finetuning + MedSAM Pipeline
  # Full pipeline: Finetune YOLO -> Test YOLO -> Finetune MedSAM
  # Usage: sbatch exp_yolo.slurm 4 [--conf=0.85] [--dup=3] [--epochs=30]
  # ----------------------------------------------------------------
  4)
    load_config_args 4 "${OVERRIDE_ARGS[@]}" "CONF_PCT=${CONF_PCT}" "DUP_FACTOR=${DUP_FACTOR}" "epochs=${FINETUNE_EPOCHS}"
    
    if [[ ! -f "${IMAGES_PARQUET}" ]]; then
      echo "[ERROR] Processed parquet not found: ${IMAGES_PARQUET}" >&2
      exit 2
    fi

    if [[ ! -f "${INIT_WEIGHTS}" ]]; then
      echo "[ERROR] Init weights not found: ${INIT_WEIGHTS}" >&2
      exit 2
    fi

    echo "[INFO] Full Semi-Supervised Pipeline (YOLO + MedSAM)"
    echo "[INFO] Confidence: ${CONF_THRESH} (${CONF_PCT}%)"
    echo "[INFO] Images Parquet: ${IMAGES_PARQUET}"
    echo "[INFO] Init Weights: ${INIT_WEIGHTS}"
    ;;

  # ----------------------------------------------------------------
  # CONFIG 5: Predictions Only (Standalone)
  # Run YOLO+MedSAM predictions on a dataset
  # Usage: sbatch exp_yolo.slurm 5 [--weights=/path/to/weights.pt] [--parquet=/path/to/images.parquet]
  # ----------------------------------------------------------------
  5)
    # Add environment variable overrides if not already in OVERRIDE_ARGS
    [[ -n "${PREDICT_WEIGHTS:-}" ]] && OVERRIDE_ARGS+=("yolo_weights=${PREDICT_WEIGHTS}")
    [[ -n "${MEDSAM_CKPT:-}" ]] && OVERRIDE_ARGS+=("medsam_ckpt=${MEDSAM_CKPT}")
    [[ -n "${PREDICT_PARQUET:-}" ]] && OVERRIDE_ARGS+=("predict_parquet=${PREDICT_PARQUET}")
    [[ -n "${PREDICT_OUT:-}" ]] && OVERRIDE_ARGS+=("predict_out_dir=${PREDICT_OUT}")
    [[ -n "${PREDICT_SPLITS:-}" ]] && OVERRIDE_ARGS+=("predict_splits=${PREDICT_SPLITS}")
    
    load_config_args 5 "${OVERRIDE_ARGS[@]}"
    
    # Use loaded paths or fallback to defaults
    YOLO_WEIGHTS="${YOLO_WEIGHTS:-/scratch/st-ipor-1/cperez/MedSAM/runs/run13/Train/Train/weights/best.pt}"
    MEDSAM_CKPT="${MEDSAM_CKPT:-/scratch/st-ipor-1/cperez/MedSAM/weights/medsam_vit_b.pth}"
    PREDICT_PARQUET="${PREDICT_PARQUET:-/scratch/st-ipor-1/cperez/MedSAM/src/image_data/images_dev.parquet}"
    PREDICT_OUT_DIR="${PREDICT_OUT_DIR:-/scratch/st-ipor-1/cperez/MedSAM/PREDICT/experiment_predictions}"
    OUT_DIR="${PREDICT_OUT_DIR}"
    
    # Sanity checks
    if [[ ! -f "${YOLO_WEIGHTS}" ]]; then
      echo "[ERROR] YOLO weights not found: ${YOLO_WEIGHTS}" >&2
      exit 2
    fi

    if [[ ! -f "${MEDSAM_CKPT}" ]]; then
      echo "[ERROR] MedSAM checkpoint not found: ${MEDSAM_CKPT}" >&2
      exit 2
    fi

    if [[ ! -f "${PREDICT_PARQUET}" ]]; then
      echo "[ERROR] Predict parquet not found: ${PREDICT_PARQUET}" >&2
      exit 2
    fi

    echo "[INFO] Prediction Mode (Standalone)"
    echo "[INFO] YOLO Weights: ${YOLO_WEIGHTS}"
    echo "[INFO] MedSAM Checkpoint: ${MEDSAM_CKPT}"
    echo "[INFO] Input Parquet: ${PREDICT_PARQUET}"
    echo "[INFO] Output Dir: ${PREDICT_OUT_DIR}"
    ;;

  # ----------------------------------------------------------------
  # CONFIG 6: Training + Predictions Pipeline
  # Train YOLO -> Test -> Run predictions on test set
  # Usage: sbatch exp_yolo.slurm 6
  # ----------------------------------------------------------------
  6)
    load_config_args 6 "${OVERRIDE_ARGS[@]}"
    echo "[INFO] Training + Predictions Pipeline"
    echo "[INFO] Output DS: ${OUT_DIR}"
    ;;

  # ----------------------------------------------------------------
  # CONFIG 7: Semi-Supervised + Predictions Pipeline
  # Finetune YOLO -> Test -> Run predictions
  # Usage: sbatch exp_yolo.slurm 7 [--conf=0.85] [--dup=3]
  # ----------------------------------------------------------------
  7)
    load_config_args 7 "${OVERRIDE_ARGS[@]}" "CONF_PCT=${CONF_PCT}" "DUP_FACTOR=${DUP_FACTOR}" "epochs=${FINETUNE_EPOCHS}"
    
    if [[ ! -f "${IMAGES_PARQUET}" ]]; then
      echo "[ERROR] Processed parquet not found: ${IMAGES_PARQUET}" >&2
      exit 2
    fi

    if [[ ! -f "${INIT_WEIGHTS}" ]]; then
      echo "[ERROR] Init weights not found: ${INIT_WEIGHTS}" >&2
      exit 2
    fi

    echo "[INFO] Semi-Supervised + Predictions Pipeline"
    echo "[INFO] Confidence: ${CONF_THRESH} (${CONF_PCT}%)"
    echo "[INFO] Images Parquet: ${IMAGES_PARQUET}"
    echo "[INFO] Init Weights: ${INIT_WEIGHTS}"
    ;;

  # ----------------------------------------------------------------
  # CONFIG 8: MedSAM Finetuning Only
  # Finetune MedSAM using existing YOLO weights and dataset
  # Usage: sbatch exp_yolo.slurm 8 [--yolo-weights=/path/to/weights.pt] [--yolo-ds=/path/to/dataset] [--medsam=/path/to/medsam.pth] [--epochs=50]
  # ----------------------------------------------------------------
  8)
    # Add environment variable overrides if not already in OVERRIDE_ARGS
    [[ -n "${YOLO_WEIGHTS:-}" ]] && OVERRIDE_ARGS+=("yolo_weights=${YOLO_WEIGHTS}")
    [[ -n "${YOLO_DS:-}" ]] && OVERRIDE_ARGS+=("yolo_ds=${YOLO_DS}")
    [[ -n "${MEDSAM_CKPT:-}" ]] && OVERRIDE_ARGS+=("medsam_ckpt=${MEDSAM_CKPT}")
    [[ -n "${MEDSAM_EPOCHS:-}" ]] && OVERRIDE_ARGS+=("epochs=${MEDSAM_EPOCHS}")
    [[ -n "${MEDSAM_BATCH:-}" ]] && OVERRIDE_ARGS+=("batch=${MEDSAM_BATCH}")
    [[ -n "${MEDSAM_IMGSZ:-}" ]] && OVERRIDE_ARGS+=("imgsz=${MEDSAM_IMGSZ}")
    
    load_config_args 8 "${OVERRIDE_ARGS[@]}"
    
    # Use loaded paths or fallback to defaults
    YOLO_WEIGHTS="${YOLO_WEIGHTS:-/scratch/st-ipor-1/cperez/MedSAM/runs/run13/Train/Train/weights/best.pt}"
    YOLO_DS="${YOLO_DS:-/scratch/st-ipor-1/cperez/MedSAM/DS/YOLO_DS}"
    MEDSAM_CKPT="${MEDSAM_CKPT:-/scratch/st-ipor-1/cperez/MedSAM/weights/medsam_vit_b.pth}"
    OUT_DIR="${OUT_DIR:-${REPO_DIR}/runs/Fundu}"

    # Sanity checks
    if [[ ! -f "${YOLO_WEIGHTS}" ]]; then
      echo "[ERROR] YOLO weights not found: ${YOLO_WEIGHTS}" >&2
      exit 2
    fi

    if [[ ! -f "${MEDSAM_CKPT}" ]]; then
      echo "[ERROR] MedSAM checkpoint not found: ${MEDSAM_CKPT}" >&2
      exit 2
    fi

    if [[ ! -d "${YOLO_DS}" ]] || [[ ! -f "${YOLO_DS}/data.yaml" ]]; then
      echo "[ERROR] YOLO dataset not found or invalid: ${YOLO_DS}" >&2
      echo "[INFO] Expected directory with data.yaml file"
      exit 2
    fi

    echo "[INFO] MedSAM Finetuning Mode (Standalone)"
    echo "[INFO] YOLO Weights: ${YOLO_WEIGHTS}"
    echo "[INFO] YOLO Dataset: ${YOLO_DS}"
    echo "[INFO] MedSAM Checkpoint: ${MEDSAM_CKPT}"
    echo "[INFO] Output will be in: ${OUT_DIR}"
    ;;

  # ----------------------------------------------------------------
  # Fallback
  # ----------------------------------------------------------------
  *)
    echo "Invalid mode selected: $MODE"
    echo ""
    echo "Usage: sbatch exp_yolo.slurm [MODE] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  1  - Standard training from raw folders (default)"
    echo "  2  - Training from parquet splits"
    echo "  3  - Semi-supervised YOLO finetuning"
    echo "  4  - Full semi-supervised pipeline (YOLO + MedSAM)"
    echo "  5  - Predictions only (standalone)"
    echo "  6  - Training + predictions pipeline"
    echo "  7  - Semi-supervised + predictions pipeline"
    echo "  8  - MedSAM finetuning only (requires existing YOLO weights/dataset)"
    echo ""
    echo "Options for modes 3, 4, 7:"
    echo "  --conf=X      Confidence threshold (default: 0.85)"
    echo "  --dup=X       Duplication factor (default: 3)"
    echo "  --epochs=X    Training epochs (default: 30)"
    echo ""
    echo "Options for mode 5:"
    echo "  --weights=X   Path to YOLO weights"
    echo "  --medsam=X    Path to MedSAM checkpoint"
    echo "  --parquet=X   Path to input parquet"
    echo "  --out=X       Output directory"
    echo "  --splits=X    Comma-separated splits (default: test)"
    echo ""
    echo "Options for mode 8:"
    echo "  --yolo-weights=X   Path to YOLO weights (required)"
    echo "  --yolo-ds=X        Path to YOLO dataset directory (required)"
    echo "  --medsam=X         Path to MedSAM checkpoint (default: weights/medsam_vit_b.pth)"
    echo "  --epochs=X         Training epochs (default: 50)"
    echo "  --batch=X          Batch size (default: 8)"
    echo "  --imgsz=X          Image size (default: 1024)"
    echo ""
    echo "Examples:"
    echo "  sbatch exp_yolo.slurm 1                    # Standard training"
    echo "  sbatch exp_yolo.slurm 3                    # Semi-supervised finetuning"
    echo "  sbatch exp_yolo.slurm 3 --conf=0.88 --epochs=50"
    echo "  sbatch exp_yolo.slurm 4 --conf=0.85        # Full pipeline with MedSAM"
    echo "  sbatch exp_yolo.slurm 5 --weights=/path/to/best.pt --medsam=/path/to/medsam.pth --parquet=/path/to/images.parquet"
    echo "  sbatch exp_yolo.slurm 6                    # Train + predict"
    echo "  sbatch exp_yolo.slurm 7 --conf=0.85        # Semi-supervised + predict"
    echo "  sbatch exp_yolo.slurm 8 --yolo-weights=/path/to/best.pt --yolo-ds=/path/to/dataset --epochs=50"
    exit 1
    ;;
esac

############################
# 5. Execution
############################
if [[ -n "${OUT_DIR:-}" ]]; then
  echo "[INFO] Output Directory: ${OUT_DIR}"
fi
echo "[INFO] Launching Experiment..."
echo "[INFO] Arguments: ${ARGS[*]}"

srun --gpu-bind=closest --cpu-bind=cores -u \
  python -m src.experiment "${ARGS[@]}"

echo "[INFO] Job Finished"