# Introduction



We conducted an annotation study to show how MedSAM can be used to assit annotation. Specifically, we used the recently released [adrenocortical carcinoma CT dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93257945), where the segmentation target (adrenal tumor) did not appear in the training set or existing validation sets. We randomly selected 10 cases with 733 tumor slices that need to be annotated. The data are available [here](https://drive.google.com/drive/folders/1QhD4vPDie-P2ddpur6lofRYWMHklYao3?usp=sharing). 


The annotation process contained three steps. 

1. Initial marker
Two radiologists independently draw the long and short tumor axes (initial marker), which is a common measure in clinical practice (e.g., RECIST). This process was conducted every 3-10 slices for the 3D tumor. 

2. For each annotated slice, a rectangle binary mask was generated based on the linear label that can completely cover the linear label. For the unlabeled slices, the rectangle binary masks were simulated by interploating the surrounding labeled slices.

```bash
python label_interpolate.py
```

3. We converted the binary masks to bounding boxes followed by feeding them to medsam together with images and generating segmentation results. 

```bash
python medsam_infer_3Dbox_adrenal.py
```
