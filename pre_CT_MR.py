# -*- coding: utf-8 -*-
# %% import packages
# pip install connected-components-3d
import numpy as np

# import nibabel as nib
import SimpleITK as sitk
import os

join = os.path.join
from tqdm import tqdm
import cc3d

import multiprocessing as mp
import re

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-modality", type=str, default="CT", help="CT or MR, [default: CT]")
parser.add_argument("-anatomy", type=str, default="Abd",
                    help="Anaotmy name, [default: Abd]")
parser.add_argument("-img_name_suffix", type=str, default="_0000.nii.gz",
                    help="Suffix of the image name, [default: _0000.nii.gz]")
parser.add_argument("-gt_name_suffix", type=str, default=".nii.gz",
                    help="Suffix of the ground truth name, [default: .nii.gz]")
parser.add_argument("-prefix", type=str, default="CT_Abd_",
                    help="Prefix of the npz file name, [default: CT_Abd_]")
parser.add_argument("-img_path", type=str, default="data/FLARE22Train/images",
                    help="Path to the nii images, [default: data/FLARE22Train/images]")
parser.add_argument("-gt_path", type=str, default="data/FLARE22Train/labels",
                    help="Path to the ground truth, [default: data/FLARE22Train/labels]")
parser.add_argument("-output_path", type=str, default="data",
                    help="Path to save the npy files, [default: ./data]")
parser.add_argument("-num_workers", type=int, default=4,
                    help="Number of workers, [default: 4]")
parser.add_argument("-window_level", type=int, default=40,
                    help="CT window level, [default: 40]")
parser.add_argument("-window_width", type=int, default=400,
                    help="CT window width, [default: 400]")

args = parser.parse_args()

# convert nii image to npz files, including original image and corresponding masks
modality = args.modality  # CT or MR
anatomy = args.anatomy  # anantomy + dataset name
img_name_suffix = args.img_name_suffix  # "_0000.nii.gz"
gt_name_suffix = args.gt_name_suffix  # ".nii.gz"
prefix = modality + "_" + anatomy + "_"

nii_path = args.img_path  # path to the nii images
gt_path = args.gt_path  # path to the ground truth
output_path = args.output_path  # path to save the preprocessed files
npy_tr_path = join(output_path, "MedSAM_train", prefix[:-1])
os.makedirs(join(npy_tr_path, "gts"), exist_ok=True)
os.makedirs(join(npy_tr_path, "imgs"), exist_ok=True)
npy_ts_path = join(output_path, "MedSAM_test", prefix[:-1])
os.makedirs(join(npy_ts_path, "gts"), exist_ok=True)
os.makedirs(join(npy_ts_path, "imgs"), exist_ok=True)

num_workers = args.num_workers

voxel_num_thre2d = 100
voxel_num_thre3d = 1000

names = sorted(os.listdir(gt_path))
print(f"ori \# files {len(names)=}")
names = [
    name
    for name in names
    if os.path.exists(join(nii_path, name.split(gt_name_suffix)[0] + img_name_suffix))
]
print(f"after sanity check \# files {len(names)=}")

# set label ids that are excluded
remove_label_ids = [
    12
]  # remove deodenum since it is scattered in the image, which is hard to specify with the bounding box
tumor_id = None  # only set this when there are multiple tumors; convert semantic masks to instance masks
# set window level and width
# https://radiopaedia.org/articles/windowing-ct
WINDOW_LEVEL = args.window_level # only for CT images
WINDOW_WIDTH = args.window_width # only for CT images

# %% save preprocessed images and masks as npz files
def preprocess(name):
    re_case = re.compile(r"FLARE22_Tr_(\d+)")
    case_num = int(re_case.findall(name)[0])
    if case_num > 40:
        npy_path = npy_ts_path ## Last 10 cases are used for testing
    else:
        npy_path = npy_tr_path ## First 40 cases are used for fine-tuning
    image_name = name.split(gt_name_suffix)[0] + img_name_suffix
    gt_name = name
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
    # remove label ids
    for remove_label_id in remove_label_ids:
        gt_data_ori[gt_data_ori == remove_label_id] = 0
    # label tumor masks as instances and remove from gt_data_ori
    if tumor_id is not None:
        tumor_bw = np.uint8(gt_data_ori == tumor_id)
        gt_data_ori[tumor_bw > 0] = 0
        # label tumor masks as instances
        tumor_inst, tumor_n = cc3d.connected_components(
            tumor_bw, connectivity=26, return_N=True
        )
        # put the tumor instances back to gt_data_ori
        gt_data_ori[tumor_inst > 0] = (
            tumor_inst[tumor_inst > 0] + np.max(gt_data_ori) + 1
        )

    # exclude the objects with less than 1000 pixels in 3D
    gt_data_ori = cc3d.dust(
        gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
    )
    # remove small objects with less than 100 pixels in 2D slices

    for slice_i in range(gt_data_ori.shape[0]):
        gt_i = gt_data_ori[slice_i, :, :]
        # remove small objects with less than 100 pixels
        # reason: fro such small objects, the main challenge is detection rather than segmentation
        gt_data_ori[slice_i, :, :] = cc3d.dust(
            gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
        )
    # find non-zero slices
    z_index, _, _ = np.where(gt_data_ori > 0)
    z_index = np.unique(z_index)

    if len(z_index) > 0:
        # crop the ground truth with non-zero slices
        gt_roi = gt_data_ori[z_index, :, :]
        # load image and preprocess
        img_sitk = sitk.ReadImage(join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        # nii preprocess start
        if modality == "CT":
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
        else:
            lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
            image_data_pre[image_data == 0] = 0

        image_data_pre = np.uint8(image_data_pre)
        img_roi = image_data_pre[z_index, :, :]
        np.savez_compressed(join(npy_path, prefix + gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())
        # save the image and ground truth as nii files for sanity check;
        # they can be removed
        img_roi_sitk = sitk.GetImageFromArray(img_roi)
        gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
        sitk.WriteImage(
            img_roi_sitk,
            join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_img.nii.gz"),
        )
        sitk.WriteImage(
            gt_roi_sitk,
            join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_gt.nii.gz"),
        )
        # save the each CT image as npy file
        for i in range(img_roi.shape[0]):
            img_i = img_roi[i, :, :]
            img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)

            img_01 = (img_3c - img_3c.min()) / np.clip(
                img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)

            gt_i = gt_roi[i, :, :]

            gt_i = np.uint8(gt_i)
            assert img_01.shape[:2] == gt_i.shape
            np.save(
                join(
                    npy_path,
                    "imgs",
                    prefix
                    + gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(i).zfill(3)
                    + ".npy",
                ),
                img_01,
            )
            np.save(
                join(
                    npy_path,
                    "gts",
                    prefix
                    + gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(i).zfill(3)
                    + ".npy",
                ),
                gt_i,
            )

if __name__ == "__main__":
    with mp.Pool(num_workers) as p:
        with tqdm(total=len(names)) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(preprocess, names))):
                pbar.update()