#!/bin/bash

# Training setup
GPUS_PER_NODE=1

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

export NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES)) # M nodes x N GPUs

echo "nnodes: ${NNODES}"

export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1

export NCCL_DEBUG=INFO

dataroot="./data/MedSAM_train/CT_Abd"
pretrained_checkpoint="medsam_lite.pth"

python train_multi_gpus.py \
    -i ${dataroot} \
    -task_name MedSAM-Lite-Box \
    -pretrained_checkpoint ${pretrained_checkpoint} \
    -work_dir ./work_dir_medsam_lite \
    -batch_size 16 \
    -num_workers 8 \
    -lr 0.0005 \
    --data_aug \
    -world_size ${WORLD_SIZE} \
    -node_rank ${NODE_RANK} \
    -init_method tcp://${MASTER_ADDR}:${MASTER_PORT}

echo "END TIME: $(date)"
