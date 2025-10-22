#!/bin/bash
#SBATCH --job-name=preprocess_oop_test
#SBATCH --account=st-ipor-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/cperez/MedSAM/logs/%x-%j.err
#SBATCH --mail-user=cperez67@student.ubc.ca
#SBATCH --mail-type=START,END,FAIL

set -euo pipefail

# --- Paths (HPC) ---
SCRATCH_DIR="/scratch/st-ipor-1/cperez/MedSAM"
ENV_PREFIX="/arc/project/st-ipor-1/carlosp/envs/medsam"
PIPELINE="${SCRATCH_DIR}/src/scripts/preprocess_pipeline_oop.py"
CONFIG="${SCRATCH_DIR}/src/configs/pipeline_hpc.yaml"

# Create dirs (fix logs path)
mkdir -p "$(dirname "${CONFIG}")" \
       "${SCRATCH_DIR}/data3" \
       "${SCRATCH_DIR}/logs"

# --- Conda env ---
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Optional: avoid cache warnings on compute nodes
export XDG_CACHE_HOME="/scratch/st-ipor-1/cperez/.cache"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export TMPDIR="${SCRATCH_DIR}/tmp"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"

# Threading (avoid oversubscription)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1


# --- Sanity checks ---
if [[ ! -f "$PIPELINE" ]]; then
echo "[ERR] Pipeline not found: $PIPELINE" >&2
exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
echo "[ERR] Config not found: $CONFIG" >&2
exit 1
fi

# --- Run the full preprocessing pipeline via YAML config ---
echo "[INFO] Using config: ${CONFIG}"
# srun -u python "${PIPELINE}" --config "${CONFIG}"

cd /scratch/st-ipor-1/cperez/MedSAM
python -m src.scripts.preprocess_pipeline_oop --config ./src/configs/pipeline_hpc.yaml