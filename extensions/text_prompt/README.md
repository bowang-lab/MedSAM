# MedSAM with text prompts 


## Requirements
The text prompt training uses the CLIP model from [Huggingface transformers](https://huggingface.co/docs/transformers/index). To install Huggingface transformers:
```
pip install transformers
```

## Inference

Please try this out-of-the-box demo: [colab](https://colab.research.google.com/drive/1wexPLewVMI-9EMiplfyoEtGGayYDH3tt?usp=sharing)

## Training

This training script demonstrates how to train MedSAM with text prompts on the [MICCAI FLARE 2022](https://flare22.grand-challenge.org/) dataset, and assume that the dataset has been preprocessed into the format used by MedSAM as described [here](https://github.com/bowang-lab/MedSAM#data-preprocessing).

The training script `train_text_prompt.py` takes the following arguments:
* `-i`, `--tr_npy_path`: Path to the preprocessed npy data in MedSAM's format
* `-medsam_checkpoint`: Path to the MedSAM checkpoint
* `-work_dir`: Path to the directory where the model checkpoints will be saved
* `-resume`: Path to the checkpoint to resume training from
* `-batch_size`: Batch size

For example, assume that the preprocessed data is stored in directory `npy_data`, the MedSAM model is stored in `MedSAM/work_dir/MedSAM/medsam_vit_b.pth`, and the model checkpoints should be saved in `train_text_prompt`. To train the model with a batch size of 16, run the following command:
```
python train_text_prompt.py \
    -i npy_data \
    -medsam_checkpoint MedSAM/work_dir/MedSAM/medsam_vit_b.pth \
    -work_dir ./train_text_prompt
```

To resume an interrupted training, simply add the `-resume` argument:
```
python train_text_prompt.py \
    -i npy_data \
    -medsam_checkpoint MedSAM/work_dir/MedSAM/medsam_vit_b.pth \
    -work_dir ./train_text_prompt \
    -resume ./train_text_prompt/medsam_text_prompt_latest.pt
```

## Train on your own dataset
To train MedSAM with text prompts on your own dataset, you need to modify the `label_dict` in `NpyDataset` in the training script based on the label values and the corresponding text prompts. For example, if your dataset has two labels, `1` and `2`, and you want to use the text prompts `normal` and `abnormal`, then the `label_dict` should be:
```
class NpyDataset(Dataset): 
    def __init__(self, data_root, image_size=1024, data_aug=True):
        ...
        self.label_dict = {
            1: ['normal'], ## need to be a list
            2: ['abnormal']
        }
```
