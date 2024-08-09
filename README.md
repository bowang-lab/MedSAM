# MedSAM2-Segment Anything In Medical Images and Videos: Benchmark and Deployment

This is the official repository for benchmarking and fine-tuning SAM2 on medical images. Welcome to join our [mailing list](https://forms.gle/hk4Efp6uWnhjUHFP6) to get updates.


[[`Paper`](https://arxiv.org/abs/2408.03322)] [[Online Demo](https://huggingface.co/spaces/junma/MedSAM2)] [[`Gradio API`](./app.py)] [[`3D Slicer Plugin`](https://github.com/bowang-lab/MedSAMSlicer/tree/SAM2)] 





https://github.com/user-attachments/assets/48d32ef4-1f15-469e-993a-bdf8854ec88c





## Installation

Environment Requirements: `Ubuntu 20.04` | Python `3.10` | `CUDA 12.1+` | `Pytorch 2.3.1`

1. Create a virtual environment `conda create -n sam2_in_med python=3.10 -y` and activate it `conda activate sam2_in_med`
2. Install [Pytorch 2.3.1+](https://pytorch.org/get-started/locally/)
3. git clone -b MedSAM2 https://github.com/bowang-lab/MedSAM/
4. Set `CUDA_HOME` environment variable to the path of your CUDA installation. For example, `export CUDA_HOME=/usr/local/cuda-12.1`
5. Enter the MedSAM2 folder `cd MedSAM2` and run `pip install -e .`
> If one enconters error in building wheels, please refer to [common installation issues](https://github.com/facebookresearch/segment-anything-2/blob/main/INSTALL.md#common-installation-issues).

## Gradio API

1. Install dependencies 

```bash
pip install gradio==3.38.0
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt-get update
sudo apt-get install ffmpeg
```

2. Run `python app.py`

## Fine-tune SAM2 on the Abdomen CT Dataset

1. Download [SAM2 checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt) and place it at `./checkpoints/sam2_hiera_tiny.pt` .

2. Download the [demo dataset](https://zenodo.org/records/7860267). This tutorial assumes it is unzipped it to `data/FLARE22Train/`.
Or directly run `pip install zenodo_get` and `zenodo_get7860267`

### Data preparation and preprocessing

This dataset contains 50 abdomen CT scans and each scan contains an annotation mask with 13 organs. The names of the organ label are available at [MICCAI FLARE2022](https://flare22.grand-challenge.org/).

1. Run the pre-processing script to convert the dataset to `npz` format:
```bash
python pre_CT_MR.py \
    -img_path data/FLARE22Train/images \
    -img_name_suffix _0000.nii.gz \
    -gt_path data/FLARE22Train/labels \
    -gt_name_suffix .nii.gz \
    -output_path data \
    -num_workers 4 \
    -modality CT \
    -anatomy Abd \
    -window_level 40 \
    -window_width 400 \
    --save_nii
```
- Split dataset: 80% for training and 20% for testing
- Adjust CT scans to [soft tissue](https://radiopaedia.org/articles/windowing-ct) window level (40) and width (400)
- Save the pre-processed images and labels as `npz` files
- For detailed usage of the script, see `python pre_CT_MR.py -h`.

2. Convert the training `npz` to `npy` format for training:
```bash
python npz_to_npy.py \
    -npz_dir data/npz_train/CT_Abd \
    -npy_dir data/npy \
    -num_workers 4
```
> For more efficient fine-tuning, the ground truth `npy` masks are resampled to `[256, 256]`.

### Model Fine-tuning

> The fine-tuning pipeline requires about 42GB GPU memory with a batch size of 16 for the Tiny model on a single A6000 GPU.

To fine-tune SAM2, run:
```bash
python finetune_sam2_img.py \
    -i ./data/npy \
    -task_name MedSAM2-Tiny-Flare22 \
    -work_dir ./work_dir \
    -batch_size 16 \
    -pretrain_model_path ./checkpoints/sam2_hiera_tiny.pt \
    -model_cfg sam2_hiera_t.yaml
```

To resume interrupted finetuning from a checkpoint, run:
```bash
python finetune_sam2_img.py \
    -i ./data/npy \
    -task_name MedSAM2-Tiny-Flare22 \
    -work_dir ./work_dir \
    -batch_size 16 \
    -pretrain_model_path ./checkpoints/sam2_hiera_tiny.pt \
    -model_cfg sam2_hiera_t.yaml \
    -resume ./work_dir/<task_name>-<date>-<time>/medsam2_model_latest.pth
```

For additional command line arguments, see `python finetune_sam2_img.py -h`.

## Inference
The inference script assumes the testing data have been converted to `npz` format.
To run inference on the 3D CT FLARE22 dataset with the fine-tuned model, run:
```bash
python infer_medsam2_flare22.py \
    -data_root data/npz_test/CT_Abd \
    -pred_save_dir segs/medsam2 \
    -sam2_checkpoint checkpoints/sam2_hiera_tiny.pt \
    -medsam2_checkpoint ./work_dir/medsam2_t_flare22.pth \
    -model_cfg sam2_hiera_t.yaml \
    -bbox_shift 5 \
    -num_workers 10 \ ## number of workers for inference in parallel
    --visualize ## Save segmentation, ground truth volume, and images in .nii.gz for visualization
```

Similarly, to run inference with the vanilla SAM2 model as described in the paper, run:
```bash
python infer_sam2_flare22.py \
    -data_root data/npz_test/CT_Abd \
    -pred_save_dir segs/sam2 \
    -sam2_checkpoint checkpoints/sam2_hiera_tiny.pt \
    -model_cfg sam2_hiera_t.yaml \
    -bbox_shift 5 \
    -num_workers 10
```

## Evaluation

The evaluation script to compute the Dice and NSD scores are provided under the `./metrics` folder. To evaluate the segmentation results, run:
```bash
python compute_metrics_flare22.py \
    -s ../segs/medsam2 \
    -g ../data/npz_test/CT_Abd \
    -csv_dir ./medsam2
```


###  To-do list
- support multi-gpu training
- provide video tutorial 


## Acknowledgements
- We highly appreciate all the dataset owners for providing the public dataset to the community.
- We thank Meta AI for making the source code of [SAM2](https://github.com/facebookresearch/segment-anything-2) publicly available.
- We thank 3D Slicer and Gradio team for providing the user-friendly platforms


## Reference

```
@article{MedSAM2-Eval-Deploy,
    title={Segment Anything in Medical Images and Videos: Benchmark and Deployment},
    author={Ma, Jun and Kim, Sumin and Li, Feifei and Baharoon, Mohammed and Askereh, Reza and Lyu, Hongwei and Wang, Bo},
    journal={arXiv preprint arXiv:2408.03322},
    year={2024}
}
      
```

