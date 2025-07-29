# MedSAM with point prompts 

## Inference

Please try this out-of-the-box demo: [(colab)](https://colab.research.google.com/drive/1cCBw_IhdPiWE4sN7QwqKJPgAFlWsKgkm?usp=sharing)

## Training

This training script shows how to train MedSAM with point prompts on the [MICCAI FLARE 2022](https://flare22.grand-challenge.org/) dataset, and assume that the dataset has been preprocessed into the format used by MedSAM as described [here](https://github.com/bowang-lab/MedSAM#data-preprocessing).

The training script `train_point_prompt.py` takes the following arguments:
* `-i`, `--tr_npy_path`: Path to the preprocessed npy data in MedSAM's format
* `-medsam_checkpoint`: Path to the MedSAM checkpoint
* `-work_dir`: Path to the directory where the model checkpoints will be saved
* `-resume`: Path to the checkpoint to resume training from
* `-batch_size`: Batch size

For example, assume that the preprocessed data is stored in directory `npy_data`, the MedSAM model is stored in `MedSAM/work_dir/MedSAM/medsam_vit_b.pth`, and the model checkpoints should be saved in `train_point_prompt`. To train the model with a batch size of 16, run the following command:
```
python train_point_prompt.py \
    -i npy_data \
    -medsam_checkpoint MedSAM/work_dir/MedSAM/medsam_vit_b.pth \
    -work_dir ./train_point_prompt
```

To resume an interrupted training, simply add the `-resume` argument:
```
python train_point_prompt.py \
    -i npy_data \
    -medsam_checkpoint MedSAM/work_dir/MedSAM/medsam_vit_b.pth \
    -work_dir ./train_point_prompt \
    -resume ./train_point_prompt/medsam_point_prompt_latest.pth
```
