#!/bin/bash
#SBATCH --job-name=create_parquet_images
#SBATCH --account=st-ipor-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

############################
# 1. Environment & Setup
############################
REPO_DIR="/scratch/st-ipor-1/cperez/MedSAM"
export REPO_DIR
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

source "${REPO_DIR}/slurm/setup.sh"
setup_env

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/hpc_logs"

############################
# 2. CPU Threading Configuration
############################
# Optimized for Python Multiprocessing (prevent oversubscription)
# We give 1 thread to OMP/BLAS so each Python worker process is lightweight.
export NUMEXPR_MAX_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

############################
# 3. Mode Selection
############################
# Usage: sbatch slurm/create_datasets.slurm --mode=eyepacs
# Default: dev
MODE="dev"

for arg in "$@"; do
  case "$arg" in
    --mode=*)
      MODE="${arg#*=}"
      ;;
  esac
done

echo "[INFO] Running in MODE: ${MODE}"
echo "[INFO] CPUs allocated: ${SLURM_CPUS_PER_TASK}"

############################
# 4. Execution Logic
############################

case "${MODE}" in

  # ------------------------------------------------------------------
  # MODE: dev
  # Description: Creates standard image parquet from general CSV.
  # Script: src.tools.create_images
  # ------------------------------------------------------------------
  dev)
ARGS_DEV=(
      --root "/arc/project/st-ipor-1/carlosp/fundus_data/All_Datasets_Organized"
      --csv "${REPO_DIR}/src/image_data/Cleaned_Metadata.csv"
      --out-parquet "${REPO_DIR}/src/image_data/images.parquet"
      --compression "zstd"
      --no-embed-images
    )

    echo "[INFO] Launching create_images (Standard Dev Set)..."
    srun --cpu-bind=cores -u \
      python -m src.tools.create_images "${ARGS_DEV[@]}"
    ;;

  # ------------------------------------------------------------------
  # MODE: eyepacs
  # Description: Creates EYEPACS parquet using highly parallelized script.
  # Script: src.tools.build_eyepacs_images
  # ------------------------------------------------------------------
  eyepacs)
    # Using 28 workers leaves 4 cores free for system/IO overhead
    WORKERS=28

    ARGS_EYEPACS=(
      --csv "${REPO_DIR}/src/image_data/UBC_data_concatonatedV2.csv"
      --root-dir "/scratch/st-ipor-1/cperez/MedSAM/data/EYEPACS_fundus_png2"
      --dataset-name "EYEPACS"
      --out-parquet "${REPO_DIR}/src/image_data/eyepacs_images.parquet"
      --mp-workers "${WORKERS}"
      --write-batch 8192
      --compression "zstd"
      --log-every 5000
    )

    echo "[INFO] Launching build_eyepacs_images with ${WORKERS} workers..."
    srun --cpu-bind=cores -u \
      python -m src.tools.build_eyepacs_images "${ARGS_EYEPACS[@]}"
    ;;

  *)
    echo "[ERROR] Unknown mode: ${MODE}"
    echo "Usage: sbatch slurm/create_datasets.slurm --mode=[dev|eyepacs]"
    exit 1
    ;;
esac

echo "[INFO] Job ${SLURM_JOB_ID} finished successfully."