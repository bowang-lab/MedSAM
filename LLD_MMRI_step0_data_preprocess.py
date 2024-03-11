"""
preprocessing code for the LLD-MMRI dataset
https://github.com/LMMMEng/LLD-MMRI-Dataset

1. add spacing and origin to the nii files
2. rename the files as caseID-catetory-MRI_phase + '_0000.nii.gz'


Jun Ma
"""

# %%
import glob
import os
join = os.path.join
listdir = os.listdir
isdir = os.path.isdir
makedirs = os.makedirs
dirname = os.path.dirname
basename = os.path.basename
from shutil import copyfile
import SimpleITK as sitk
from tqdm import tqdm
import json
from pprint import pprint
import pandas as pd
import re
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import multiprocessing as mp

# %%
ori_dir = "LLD-MMRI/images"
json_path = 'LLD-MMRI/labels/Annotation.json'
save_dir = "LLD-MMRI/images-nii"
os.makedirs(save_dir, exist_ok=True)
cases = sorted(listdir(ori_dir))

with open(json_path) as f:
    json_data = json.load(f)
case_names = list(json_data['Annotation_info'].keys())
# %%
# case_name = case_names[0]
def renames(case_name):
    case_phases = json_data['Annotation_info'][case_name]
    for case_phase in case_phases:
        phase = case_phase['phase']
        studyUID = case_phase['studyUID']
        seriesUID = case_phase['seriesUID']
        annotation = case_phase['annotation']
        pixel_spacing = case_phase['pixel_spacing']
        slice_spacing = case_phase['slice_spacing']
        slice_thickness = case_phase['slice_thickness']
        origin = case_phase['origin']
        num_targets = annotation['num_targets']
        catetory = annotation['lesion']['0']['category']
        
        sitk_img = sitk.ReadImage(join(ori_dir, case_name, studyUID, seriesUID + '.nii.gz'))
        # set origin
        sitk_img.SetOrigin(origin)
        # set spacing
        spacing_3d = (pixel_spacing[0], pixel_spacing[1], slice_spacing)
        sitk_img.SetSpacing(spacing_3d)
        # set direction
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        sitk_img.SetDirection(direction)

        nii_name = case_name + '_' + str(catetory) + '_' + phase.replace(" ","") + '_0000.nii.gz'
        sitk.WriteImage(sitk_img, join(save_dir, nii_name))


if __name__ == "__main__":
    with mp.Pool(6) as p:
        r = list(tqdm(p.imap(renames, cases), total=len(cases)))
