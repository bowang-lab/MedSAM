# MedSAM
This the official repository for MedSAM: Segment Anything in Medical Images.


## Installation 
1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/bowang-lab/MedSAM`
4. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`


## Fine-tune SAM on customized dataset

We provide a step-by-step tutorial with a small dataset to help you quickly start the training process.

### Data preparation and preprocessing

Download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it at `work_dir/SAM/sam_vit_b_01ec64.pth` .

Download the demo [dataset](https://zenodo.org/record/7860267) and unzip.

This dataset contains 50 abdomen CT scans and each scan contain an annotation mask with 13 organs. The names of the organ label are available at [MICCAI FLARE2022](https://flare22.grand-challenge.org/).
In this tutorial, we will fine-tune SAM for gallbladder segmentation.

Run pre-processing

```bash
python pre_CT.py -i path_to_image_folder -gt path_to_gt_folder -o path_to_output
```

- split dataset: 80% for training and 20% for testing
- image normalization
- pre-compute image embedding
- save the normalized images, ground truth masks, and image embedding as a `npz` file

> Note: Medical images have various data formats. Thus, it's impossible that one script can handle all these different formats. Here, we provide two typical examples for CT and non-CT (e.g., various MR sequences, PET images) image preprocessing. You can adapt the preprocessing code to your own datasets.

### Model Training

Please check the step-by-step tutorial: `finetune_and_inference_tutorial.py`.

You can also train the model on the whole dataset. 
Download the training set ([GoogleDrive](https://drive.google.com/drive/folders/1pwpAkWPe6czxkATG9SmVV0TP62NZiKld?usp=share_link))

> Note: For the convenience of file sharing, we compress each image and mask pair in a `npz` file. The pre-computed image embedding is too large (require ~1 TB space). You can generate it with the following command


```bash
python utils/precompute_img_embed.py -i path_to_train_folder -o ./data/Tr_emb
```


Train the model

```bash
python train -i ./data/Tr_emb --task_name SAM-ViT-B --num_epochs 1000 --batch_size 8 --lr 1e-5
```

If you find this dataset valuable in your research, kindly acknowledge and credit the original data sources: [AMOS](https://zenodo.org/record/7262581), [BraTS2021](http://braintumorsegmentation.org/), [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/), [M\&Ms](https://www.ub.edu/mnms/), [PROMISE12](https://promise12.grand-challenge.org/) [ABCs](https://abcs.mgh.harvard.edu/), [AbdomenCT-1K](https://ieeexplore.ieee.org/document/9497733), [MSD](http://medicaldecathlon.com/), [KiTS19](https://kits19.grand-challenge.org/), [LiTS](https://competitions.codalab.org/competitions/17094), [COVID-19 CT-Seg](https://github.com/JunMa11/COVID-19-CT-Seg-Benchmark), [HECKTOR](https://www.sciencedirect.com/science/article/pii/S1361841521003819) [DRIVE](https://drive.grand-challenge.org/), [Colon gland](https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation), [polyp](https://www.nature.com/articles/s41597-023-01981-y), [instruments](https://www.synapse.org/#!Synapse:syn22427422), [Abdomen Ultrasound](https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm), [Breast Ultrasound](https://www.sciencedirect.com/science/article/pii/S2352340919312181), [JSRT](http://imgcom.jsrt.or.jp/minijsrtdb/)


## Inference

Download the model checkpoint ([GoogleDrive](https://drive.google.com/drive/folders/1bWv_Zs5oYLpGMAvbotnlNXJPq7ltRUvF?usp=share_link)) and testing data ([GoogleDrive](https://drive.google.com/drive/folders/1Qx-4EM0MoarzAfvSIp9fkpk8UBrWM6EP?usp=share_link)) and put them to `data/Test` and `work_dir/MedSAM` respectively. 

Run

```bash
python MedSAM_Inference.py -i ./data/Test -o ./ -chk work_dir/MedSAM/medsam_20230423_vit_b_0.0.1.pth
```

The segmentation results are available at [here](https://drive.google.com/drive/folders/1I8sgCRi30QtMix8DbDBIBTGDM_1FmSaO?usp=sharing).


The implementation code of DSC and NSD can be obtained [here](http://medicaldecathlon.com/files/Surface_distance_based_measures.ipynb).


## To-do-list

- [ ] Train the ViT-H model
- [ ] Explore other fine-tuning methods, e.g., fine-tune the image encoder as well, lora fine-tuning
- [ ] Support scribble prompts
- [ ] Support IoU/DSC regression
- [ ] Enlarge the dataset
- [ ] 3D slicer and napari support

We are excited about the potential of segmentation foundation models in the medical image domain. However, training such models requires extensive computing resources. Therefore, we have made all the pre-processed training and images publicly available for research purposes. To prevent duplication of effort (e.g., conduct the same experiemnts), we encourage sharing of results and trained models on the discussion page. We look forward to working with the community to advance this exciting research area.


## Acknowledgements
- We highly appreciate all the challenge organizers and dataset owners for providing the public dataset to the community. 
- We thank Meta AI for making the source code of [segment anything](https://github.com/facebookresearch/segment-anything) publicly available.
- We also thank Alexandre Bonnet for sharing this great [blog](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/)


## Reference

```
@article{MedSAM,
  title={Segment Anything in Medical Images},
  author={Ma, Jun and Wang, Bo},
  journal={arXiv preprint arXiv:2304.12306},
  year={2023}
}
```
