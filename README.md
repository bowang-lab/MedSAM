# LiteMedSAM for Scribble prompts


## Installation

The codebase is tested with: `Ubuntu 20.04` | Python `3.10` | `CUDA 11.8` | `Pytorch 2.1.2`

1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone -b LiteMedSAMScribble https://github.com/bowang-lab/MedSAM/`
4. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`


## Model Training

The data preprocessing is the same as the settings in [LiteMedSAM](https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM#data-preprocessing). 


```bash
python train_one_gpu_scribble.py \
    -data_root data/MedSAM_train \
    -pretrained_checkpoint lite_medsam.pth \
    -work_dir work_dir \
    -num_workers 4 \
    -batch_size 4 \
    -num_epochs 1000
```

For additional command line arguments, see `python train_one_gpu_scribble.py -h`.


## Acknowledgements
We thank the authors of [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), [TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT), and [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once) for making their source code publicly available.

