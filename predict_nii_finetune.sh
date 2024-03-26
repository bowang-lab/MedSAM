#!/bin/bash

# Directory containing ground truth .nii.gz files used to build bounding boxes.
nii_gts_dir="./data/labelsTs/"

# Directory containing image *_0000.nii.gz files for input images.
nii_img_dir="./data/imagesTs/"

# LiteMedSAM model checkpoint directory. It can be either the original lite_medsam.pth or a fine-tuned version.
medsam_lite_checkpoint_path="./work_dir/LiteMedSAM/lite_medsam.pth"
# Example of fine-tuned model checkpoint path:
# medsam_lite_checkpoint_path="./work_dir/LiteMedSAM/MedSAM-Lite-hnc-ct-20240131-0209/medsam_lite_best.pth"

# Directory to save PNG files for prediction sanity check.
png_save_dir="./data/png/pred_finetune/"

# Directory to save predicted segmentation masks in *.nii.gz format.
nii_pred_dir="./data/nii/pred_finetune/"

# Directory to save bounding boxes in *.nii.gz format.
nii_box_dir="./data/nii_box/pred_finetune_box/"

# Dictionary to specify different name suffixes for preprocessing, default is CT with *_0000.nii.gz.
modality_suffix_dict='{"CT":"_0000.nii.gz"}'
# Example with four modalities:
# modality_suffix_dict='{"CT":"_0000.nii.gz","PET":"_0001.nii.gz","T1":"_0002.nii.gz","T2":"_0003.nii.gz"}'

# Run Python script for 3D inference with the provided arguments.
python inference_3D_finetune_nii.py \
    -nii_gts_dir ${nii_gts_dir} \
    -medsam_lite_checkpoint_path ${medsam_lite_checkpoint_path} \
    -nii_img_dir ${nii_img_dir} \
    -num_workers 8 \
    -png_save_dir ${png_save_dir}  \
    -nii_pred_dir ${nii_pred_dir}  \
    -nii_box_dir ${nii_box_dir}  \
    -modality_suffix_dict ${modality_suffix_dict} \
    --save_overlay \
    --overwrite \
    --zero_shot
    # If you want to predict on a fine-tuned model, please remove the zero_shot argument.

echo "END TIME: $(date)"

