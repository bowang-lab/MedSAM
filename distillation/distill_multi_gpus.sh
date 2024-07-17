#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=Distill_LiteMedSAM
#SBATCH --mem=200GB
#SBATCH --gres=gpu:4
#SBATCH --partition=a100
#SBATCH --output=logs/relite_%x-%j.out
#SBATCH --error=logs/relite_%x-%j.err
#SBATCH --time=3-00:00:00

set -x -e

# log the sbatch environment
echo "start time: $(date)"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR

# Training setup
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE

## Master node setup
MAIN_HOST=`hostname -s`
export MASTER_ADDR=$MAIN_HOST

# Get a free port using python
export MASTER_PORT=$(python - <<EOF
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 0))  # OS will allocate a free port
free_port = sock.getsockname()[1]
sock.close()
print(free_port)
EOF
)

export NNODES=$SLURM_NNODES
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES)) # M nodes x N GPUs

echo "nnodes: ${NNODES}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1

export NCCL_DEBUG=INFO

echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR
echo SLURM_NTASKS=$SLURM_NTASKS

train_npy_dir=""
medsam_checkpoint=""
resume="" ## For later

module load cuda-11.7 

for (( i=0; i < $SLURM_NTASKS; ++i ))
do
    /opt/slurm/bin/srun -lN1 --mem=200G --gres=gpu:4 -c $SLURM_CPUS_ON_NODE -N 1 -n 1 -r $i bash -c \
    "python distill_multi_gpu.py \
        -i $train_npy_dir \
        -task_name MedSAM-Lite \
        -medsam_checkpoint ${medsam_checkpoint} \
        -work_dir ./work_dir_lite \
        -batch_size 16 \
        -num_workers 8 \
        -world_size ${WORLD_SIZE} \
        -node_rank ${i} \
        -init_method tcp://${MASTER_ADDR}:${MASTER_PORT}" >> ./logs/log_for_${SLURM_JOB_ID}_node_${i}.log 2>&1 &
        #-resume $resume \
done
wait ## Wait for the tasks to finish

echo "END TIME: $(date)"
