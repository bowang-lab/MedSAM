# Introduction
This folder contains the code for data organization, splitting, preprocessing, and checkpoint converting. 


## Data Organization
Since the orginal data formats and folder structures vary greatly across different dataset, we need to organize them as unified structures, allowing to use the same functions for data preprocessing.
The expected folder structures are as follows:

3D nii data
```
----dataset_name
--------images
------------xxxx_0000.nii.gz
------------xxxx_0000.nii.gz
--------labels
------------xxxx.nii.gz
------------xxxx.nii.gz
```
Note: you can also use different suffix for images and labels. Please change them in the following preprocessing scripts as well. 

2D data
```
----dataset_name
--------images
------------xxxx.jpg/png
------------xxxx.jpg/png
--------labels
------------xxxx.png
------------xxxx.jpg/png
```

video data
```
----dataset_name
--------images
------------video1
----------------xxxx.png
------------video2
----------------xxxx.png
--------labels
------------video1
----------------xxxx.png
------------video2
----------------xxxx.png
```

Unfortunately, it is impossible to have one script to finish all the data organization. We manually organized with commonly used data format converting functions, including `dcm2nii`, `mhd2nii`, `nii2nii`, `nrrd2nii`, `jpg2png`, `tif2png`, `rle_decode`. These functions are available at `format_convert.py`

## Data Splitting
For common 2D images (e.g., skin cancer demoscopy, chest X-Ray), they can be directly separated into 80%/10%/10% for training, parameter tuning, and internal validation, respectively. For 3D images (e.g., all the MRI/CT scans) and video data, they should be split in the case/video level rather than 2D slice/frame level. For 2D whole-slide images, the splitting is in the whole-slide level. Since they cannot be directly sent to the model because of the high resolution, we divided them into patches with a fixed size of `1024x1024` after data splitting. 

After finishing the data organization, the data splitting can be easily done by running
```bash
python split.py
```
Please set the proper data path in the script. The expected folder structures (e.g., 3D images) are

```
----dataset_name
--------images
------------xxxx_0000.nii.gz
------------xxxx_0000.nii.gz
--------labels
------------xxxx.nii.gz
------------xxxx.nii.gz
--------validation
------------images
----------------xxxx_0000.nii.gz
----------------xxxx_0000.nii.gz
------------labels
----------------xxxx.nii.gz
----------------xxxx.nii.gz
--------testing
------------images
----------------xxxx_0000.nii.gz
----------------xxxx_0000.nii.gz
------------labels
----------------xxxx.nii.gz
----------------xxxx.nii.gz
```

## Data Preprocessing and Ensembling

All the images will be preprocessed as `npy` files. There are two main reasons for choosing this format. First, it allows fast data loading (main reason). We learned this point from [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Second, numpy file is a universal data interface to unify all the different data formats. For the convenience of debugging and inference, we also saved the original images and labels as `npz` files. Spacing information is also saved for CT and MR images. 

The following steps are applied to all images
- max-min normalization
- resample image size to 1024x2014
- save the pre-processed images and labels as npy files

Different modalities also have their own additional pre-process steps based on the data features. 

For CT images, we fist adjust the window level and width following the [common practice](https://radiopaedia.org/articles/windowing-ct). 
- Soft tissue window level (40) and width (400)
- Chest window level (-600) and width (1500)
- Brain window level (40) and width (80)

For MR and ultrasound, mammography, and Optical Coherence Tomography (OCT) images (i.e., ultrasound), we apply intensity cut-off with 0.5 and 99.5 percentiles of the foreground voxels. Regarding RGB images (e.g., endoscopy, dermoscopy, fundus, and pathology images), if they are already within the expected intensity range of [0, 255], their intensities remained unchanged. However, if they fell outside this range, max-min normalization was applited to rescale the intensity values to [0, 255].

Preprocess for CT/MR images: 
```bash
python pre_CT_MR.py
```

Preprocess for grey and RGB images: 
```bash
python pre_grey_rgb.py
```

Note: Please set the corresponding folder path and molidaty information. We provided an example in the script. 

Data ensembling of different training datasets is very simple. Since all the training data are converted into `npy` files during preprocessing, you just need to merge them into one folder. 


## Checkpoint Converting
If the model is trained with multiple GPUs, please use the script `ckpt_convert.py` to convert the format since users only use one GPU for model inference in real practice. 

Set the path to `sam_ckpt_path`, `medsam_ckpt_path`, and `save_path` and run 

```bash
python ckpt_convert.py
```

