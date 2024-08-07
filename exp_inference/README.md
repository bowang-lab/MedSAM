# Using SAM2 to Segment 2D&3D Medical Images and Videos

The data consists of `imgs`, `gts`, and `videos` folder. The default folder for the inference results is `results`. The following scripts are available for the inference of each data.

Inference of the 2D images:
```bash
python inference_2D.py
```

Inference of the 3D scans:
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