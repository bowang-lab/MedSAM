#!/bin/bash

# Check if running inside a SLURM allocation
if [ -z "$SLURM_JOB_ID" ]; then
    salloc -J medsamGUI --partition=gpuq --nodes=1 --ntasks=1 --cpus-per-task=5 --gres=gpu:1 --mem-per-cpu=10000Mb --time=13:00:00 --x11 bash "$0"
    exit $?
fi

module load mpi
module load cuda11.7
module load cudnn8.5-cuda11.7

source /home/${USER}/.bashrc
conda activate /scratch/glastonbury/conda_envs/MedSAM

if [ -z "$DISPLAY" ]; then
    echo "X11 forwarding is not enabled. Please connect with ssh -X or -Y."
    exit 1
fi

export QT_XCB_GL_INTEGRATION=none

echo "Starting MedSAM GUI..."
cd /group/glastonbury/soumick/codebase/MedSAM/
python gui.py