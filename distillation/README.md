# LiteMedSAM Distillation

## Prerequisites
Set up the environment and perform data preprocessing as instructed in the [LiteMedSAM README](https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM)

The codebase contains a pretrained Tiny-ViT pretrained weights `tinyvit_pretrained.pth` (download [here](https://drive.google.com/file/d/1WeS_vuHmkLk3Hb5EpRbU6z66o56J0qvN/view?usp=sharing)) to serve as a starting point for distillation.

## Single GPU Distillation

```bash
python distill_multi_gpu.py \
    -i path/to/preprocessed/npy/directory \
    -task_name MedSAM-Lite \
    -medsam_checkpoint path/to/medsam_vit_b.pth \
    -tinyvit_checkpoint path/to/tinyvit_pretrained.pth \
    -work_dir ./work_dir \
    -batch_size 16 \
    -num_workers 8
```

To resume interrupted distillation from a checkpoint, add `-resume path/to/medsam_lite_encoder_latest.pth` to the command line arguments.

## Multi-GPU Distillation

To distill LiteMedSAM on multiple GPUs:
```bash
python distill_multi_gpu.py \
    -i path/to/preprocessed/npy/directory \
    -task_name MedSAM-Lite \
    -medsam_checkpoint path/to/MedSAM_checkpoint \
    -tinyvit_checkpoint path/to/tinyvit_pretrained.pth \
    -work_dir ./work_dir_lite \
    -batch_size 16 \
    -num_workers 8 \
    -world_size <number of gpus in total> \
    -node_rank <current machine rank; 0 if training on a single machine with multiple gpus> \
    -init_method tcp://<master address>:<master port>
```
To obtain the master address:
```bash
hostname -s
```
To find an available port as the master port:
```python
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 0))  # OS will allocate a free port
free_port = sock.getsockname()[1]
sock.close()
print(free_port)
```

Additionally, an out-of-box shell script for performing multi-GPU distillation on SLURM clusters is provided in `distill_multi_gpus.sh`. Please feel free to modify the script based on your cluster configuration.

To resume interrupted distillation from a checkpoint, add `-resume <your_work_dir>` to the command line arguments instead of the checkpoint path for multi-GPU distillation; the script will automatically find the latest checkpoint in the work directory. For additional command line arguments, see `python distill_multi_gpu.py -h`.
