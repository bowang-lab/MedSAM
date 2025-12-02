#!/bin/bash
#SBATCH --job-name=create_eyepacs_parquet
#SBATCH --account=st-ipor-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=5:00:00
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%A.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/hpc_logs/%x-%A.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=FAIL,END

set -euo pipefail

############################
# Repo / env
############################
REPO_DIR="/scratch/st-ipor-1/cperez/MedSAM"
export REPO_DIR
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

source "${REPO_DIR}/slurm/setup.sh"
setup_env

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/hpc_logs"

############################
# Thread / CPU sanity (avoid oversubscription + NumExpr thread errors)
############################
export NUMEXPR_MAX_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

############################
# Args (images are directly in root-dir, so NO --fundus-subdir)
############################
ARGS=(
  --csv "/scratch/st-ipor-1/cperez/MedSAM/src/image_data/UBC_data_concatonatedV2.csv"
  --root-dir "/scratch/st-ipor-1/cperez/MedSAM/data/EYEPACS_fundus_png2"
  --dataset-name "EYEPACS"
  --out-parquet "/scratch/st-ipor-1/cperez/MedSAM/src/image_data/eyepacs_images.parquet"
  --mp-workers 16
  --write-batch 8192
  --compression "zstd"
  --log-level "INFO"
  --log-every 10000
)

############################
# Run
############################
# No GPU needed. Drop gpu-bind flags.
srun --cpu-bind=cores -u \
  python -m src.tools.build_eyepacs_images "${ARGS[@]}"