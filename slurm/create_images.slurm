#!/bin/bash
#SBATCH --job-name=create_images
#SBATCH --account=st-ipor-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
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
# Experiment args
############################
# Note: Adjusted paths to match your scratch directory structure
ARGS=(
  --root "/arc/project/st-ipor-1/carlosp/fundus_data/All_Datasets_Organized"
  --csv "${REPO_DIR}/src/image_data/Cleaned_Metadata_dev.csv"
  --out-parquet "${REPO_DIR}/src/image_data/images_dev.parquet"
  --compression "zstd"
)

############################
# Run
############################
# Using srun with cpu binding. No GPU requested or bound.
echo "Creating Parquet dataset on $(hostname)..."

srun --cpu-bind=cores -u \
  python -m src.tools.create_images "${ARGS[@]}"

echo "Done."