#!/bin/bash
#SBATCH --job-name=convert_eyepacs_png
#SBATCH --account=st-ipor-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=16:00:00
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

# your existing helper sets up python/venv/modules
source "${REPO_DIR}/slurm/setup.sh"
setup_env

cd "${REPO_DIR}"
mkdir -p "${REPO_DIR}/hpc_logs"

############################
# Paths / args
############################
INPUT_DIR="/arc/project/st-ipor-1/Fundus/Data/EYEPACS"
OUTPUT_DIR="/scratch/st-ipor-1/cperez/MedSAM/data/EYEPACS_fundus_png"

# Use all allocated CPUs for parallel conversion
WORKERS="${SLURM_CPUS_PER_TASK}"

ARGS=(
  "${INPUT_DIR}"
  "${OUTPUT_DIR}"
  --workers "${WORKERS}"
  --compress-level 1
  --log-level "INFO"
  # --force-rgb
  # --preserve-metadata
  # --verify
)

############################
# Run
############################
srun --cpu-bind=cores -u \
  python -m src.tools.modify_eyepacs "${ARGS[@]}"