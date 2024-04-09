# LiteMedSAM

A lightweight version of MedSAM for fast training and inference. The model was trained with the following two states:

- Stage 1. Distill a lightweight image encoder `TinyViT` from the MedSAM image encoder `ViT` by imposing the image embedding outputs to be the same
- State 2. Replace the MedSAM image encoder `ViT` with `TinyViT` and fine-tune the whole pipeline


## Installation

The codebase is tested with: `Ubuntu 20.04` | Python `3.10` | `CUDA 11.8` | `Pytorch 2.1.2`

1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone -b LiteMedSAM https://github.com/bowang-lab/MedSAM/`
4. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`


## Quick tutorial on making submissions to CVPR 2024 MedSAM on Laptop Challenge

### Sanity test

- Download the LiteMedSAM checkpoint here and put it in `work_dir/LiteMedSAM`.
- Download the demo data `demo_scribble.zip` [here](https://drive.google.com/drive/folders/1t3Rs9QbfGSEv2fIFlk8vi7jc0SclD1cq?usp=sharing)
- Run the following command for a sanity test

```bash
python CVPR24_LiteMedSAM_infer.py -i test_demo/imgs/ -o test_demo/segs
```


### Build Docker

```bash
docker build -f Dockerfile -t litemedsam .
```

> Note: don't forget the `.` in the end

Run the docker on the testing demo images

```bash
docker container run -m 8G --name litemedsam --rm -v $PWD/test_demo/imgs/:/workspace/inputs/ -v $PWD/test_demo/litemedsam-seg/:/workspace/outputs/ litemedsam:latest /bin/bash -c "sh predict.sh"
```

> Note: please run `chmod -R 777 ./*` if you run into `Permission denied` error.

Save docker 

```bash
docker save litemedsam | gzip -c > litemedsam.tar.gz
```

### Compute Metrics

```bash
python evaluation/compute_metrics.py -s test_demo/litemedsam-seg -g test_demo/gts -csv_dir ./metrics.csv
```


## Model Training

### Data preprocessing
1. Download the Lite-MedSAM [checkpoint](https://drive.google.com/file/d/18Zed-TUTsmr2zc5CHUWd5Tu13nb6vq6z/view?usp=sharing) and put it under the current directory.
2. Download the [demo dataset](https://zenodo.org/records/7860267). This tutorial assumes it is unzipped it to `data/FLARE22Train/`.
3. Run the pre-processing script to convert the dataset to `npz` format:
```bash
python pre_CT_MR.py \
    -img_path data/FLARE22Train/images \ ## path to training images
    -img_name_suffix _0000.nii.gz \ ## extension of training images
    -gt_path data/FLARE22Train/labels \ ## path to training labels
    -gt_name_suffix .nii.gz \ ## extension of training labels
    -output_path data \ ## path to save the preprocessed data
    -num_workers 4 \ ## number of workers for preprocessing
    -modality CT \ ## modality of the preprocessed data
    -anatomy Abd \ ## anatomy of the preprocessed data
    -window_level 40 \ ## window level for CT
    -window_width 400 \ ## window width for CT
    --save_nii ## Also save the preprocessed data in nii.gz format for visual inspection in other software
```
* Split dataset: first 40 cases of the demo dataset for training, saved in `MedSAM_train`, the last 10 for testing, saved in `MedSAM_test`.
* For detailed usage of the script, see `python pre_CT_MR.py -h`.

4. Convert the training `npz` to `npy` format for training:
```bash
python npz_to_npy.py \
    -npz_dir data/MedSAM_train \ ## path to the preprocessed npz training data
    -npy_dir data/npy \ ## path to save the converted npy data for training
    -num_workers 4 ## number of workers for conversion in parallel
```

### Fine-tune pretrained Lite-MedSAM

> The training pipeline requires about 10GB GPU memory with a batch size of 4


#### Single GPU

To train Lite-MedSAM on a single GPU, run:
```bash
python train_one_gpu.py \
    -data_root data/MedSAM_train \
    -pretrained_checkpoint lite_medsam.pth \
    -work_dir work_dir \
    -num_workers 4 \
    -batch_size 4 \
    -num_epochs 10
```

To resume interrupted training from a checkpoint, run:
```bash
python train_one_gpu.py \
    -data_root data/MedSAM_train \
    -resume work_dir/medsam_lite_latest.pth \
    -work_dir work_dir \
    -num_workers 4 \
    -batch_size 4 \
    -num_epochs 10
```

For additional command line arguments, see `python train_one_gpu.py -h`.

#### Multi-GPU
To fine-tune Lite-MedSAM on multiple GPUs, run:
```bash
python train_multi_gpus.py \
    -i data/npy \ ## path to the training dataset
    -task_name MedSAM-Lite-Box \
    -pretrained_checkpoint lite_medsam.pth \
    -work_dir ./work_dir_ddp \
    -batch_size 16 \
    -num_workers 8 \
    -lr 0.0005 \
    --data_aug \ ## use data augmentation
    -world_size <WORLD_SIZE> \ ## Total number of GPUs will be used
    -node_rank 0 \ ## if training on a single machine, set to 0
    -init_method tcp://<MASTER_ADDR>:<MASTER_PORT>
```
Alternatively, you can use the provided `train_multi_gpus.sh` script to train on multiple GPUs. To resume interrupted training from a checkpoint, add `-resume <your_work_dir>` to the command line arguments instead of the checkpoint path for multi-GPU training;
the script will automatically find the latest checkpoint in the work directory. For additional command line arguments, see `python train_multi_gpus.py -h`.



## Inference (sanity test)
The inference script assumes the testing data have been converted to `npz` format.
To run inference on the 3D CT FLARE22 dataset, run:

```bash
python inference_3D.py \
    -data_root data/npz/MedSAM_test/CT_Abd \ ## preprocessed npz data
    -pred_save_dir ./preds/CT_Abd \
    -medsam_lite_checkpoint_path work_dir/medsam_lite_latest.pth \
    -num_workers 4 \
    --save_overlay \ ## save segmentation overlay on the input image
    -png_save_dir ./preds/CT_Abd_overlay \ ## only used when --save_overlay is set
    --overwrite ## overwrite existing predictions, default continue from existing predictions
```

For additional command line arguments, see `python inference_3D.py -h`.


We also provide a script to run inference on the 2D images `inference_2D.py`, whose usage is the same as the 3D script.

## Frequently Asked Questions (FAQ)
### What is the difference between the preprocessed npz and npy data?
* The `npz` format is used to store both 2D and 3D images (focusing on the ROI), along with their corresponding ground truth masks. In the case of 3D images, spacings are also included. All these data elements are compactly packed into a single `npz` file. This format is primarily used for distributing our training and validation datasets. Notably, for the validation dataset, bounding boxes are provided in place of ground truth label masks.

* In contrast, the `npy` format stores each 2D image or individual slice of a 3D image along with its label mask in separate files. This format is utilized when loading data for training purposes.

### I'm having trouble loading my trained model's checkpoint for inference. What should I do?
If you encounter difficulties loading a trained model's checkpoint for inference, we recommend users first try using the `extract_weights.py` script located under `MedSAM/utils/`. This script is for extracting weights from your existing checkpoint and save them into a new checkpoint file. 

To use this script, execute the following command in your terminal:

```bash
python extract_weights.py \
    -from_pth <YOUR_CHECKPOINT_PATH> \
    -to_pth <NEW_CHECKPOINT_PATH>
```
Replace `<YOUR_CHECKPOINT_PATH>` with the path to your saved trained model checkpoint, and `<NEW_CHECKPOINT_PATH>` with the desired path for the new checkpoint file. Once you have executed this command and created the new checkpoint, it should be ready for use in inference tasks.

## Acknowledgements
We thank the authors of [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) and [TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT) for making their source code publicly available.

