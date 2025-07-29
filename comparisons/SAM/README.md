# Introduction

The pre-trained [SAM](https://github.com/facebookresearch/segment-anything) model was used as a baseline in this study. SAM has three model types: ViT-H, ViT-L, and ViT-B. We used ViT-B model since it has a good trade off between segmentation accuracy and performance based on their ablation study (Fig.13). 


1. Download the [checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth). 

Note: We assuem that the preprocessing steps have been done. 

2. For 2D images, run

```bash
python infer_SAM_2D_npz.py -i input_path -o output_path -m model_path
```


3. For 3D images, run

```bash
python infer_SAM_3D_npz.py -i input_path -o output_path -m model_path
```
