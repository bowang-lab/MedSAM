# Using SAM2 to Segment 2D&3D Medical Images and Videos

Access the preprocessed data from: https://www.codabench.org/competitions/1847/

## Inference
The following scripts are available for the inference of each data.

Please specify the following:
```bash
--imgs_path: Path to image folder
--gts_path: Path to ground truth folder
--pred_save_dir: Path to save segmentation results
--checkpoints: Path to the SAM2 checkpoint
--cfg: The configuration for the model to be used
--save_overlay: Please specify this flag if you want to save the overlay results of the ground truth and the segmentation results
--png_save_dir: Path to the overlays
```

For inference_3D.py, you can also specify the following to save the segmentation results in nifti format.
```bash
--save_nifti: Please specify this flag to save the results in nifti format
--nifti_path: Path to save nifti files.
```

Inference of the 2D images:
```bash
python inference_2D.py
```

```bash
python inference_3D.py 
```

Inference of Endoscopy video:
```bash
python inference_video_Endo.py
```

Inference of US video:
```bash
python inference_video_US.py
```
