# %%
import pandas as pd
import numpy as np
from os.path import join, exists
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F
from segment_anything import sam_model_registry
import SimpleITK as sitk
import random
import os
import cv2
from skimage import io, measure
from tqdm import tqdm
from collections import OrderedDict
import time

def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC.astype(np.uint8)

image_size = 1024
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# %%
# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def get_bbox(mask, bbox_shift=5):
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)
    bboxes = np.array([x_min, y_min, x_max, y_max])
    return bboxes

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)
    
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False,
        )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


# %% load model
model_path = 'medsam_vit_b.pth'
medsam_model = sam_model_registry['vit_b'](checkpoint=model_path)
medsam_model = medsam_model.to(device)
medsam_model.eval()

seg_info = OrderedDict()
seg_info['name'] = []
seg_info['running time'] = []


img_path = 'images'
marker_path = 'marker-expert1_interpolated'
seg_path = 'medsam_seg_expert1'
os.makedirs(seg_path, exist_ok=True)

# load data
names = sorted(os.listdir(marker_path))
for name in tqdm(names):
    start_time = time.time()
    img_name = name.split('.nii.gz')[0] + '_0000.nii.gz'
    img_sitk = sitk.ReadImage(join(img_path, img_name))
    image_data = sitk.GetArrayFromImage(img_sitk)
    # adjust window level and window width
    image_data_pre = image_data.astype(np.float32) # np.clip(image_data, -160.0, 240.0)
    image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
    image_data_pre = np.uint8(image_data_pre)
    seg_data = np.zeros_like(image_data_pre, dtype=np.uint8)
    marker_data = sitk.GetArrayFromImage(sitk.ReadImage(join(marker_path, name)))
    marker_data = np.uint8(marker_data)
    label_ids = np.unique(marker_data)[1:]
    print(f'label ids: {label_ids}')
    for label_id in label_ids:
        marker_data_id = (marker_data == label_id).astype(np.uint8)
        marker_zids, _, _ = np.where(marker_data_id > 0)
        marker_zids = np.sort(np.unique(marker_zids))
        print(f'z indices: {marker_zids}')
        bbox_dict = {} # key: z_index, value: bbox
        for z in marker_zids:
            # get bbox for each slice
            z_box = get_bbox(marker_data_id[z, :, :], bbox_shift=5)
            bbox_dict[z] = z_box
        # find largest bbox in bbox_dict
        bbox_areas = [np.prod(bbox_dict[z][2:] - bbox_dict[z][:2]) for z in bbox_dict.keys()]
        z_middle = list(bbox_dict.keys())[np.argmax(bbox_areas)] # middle slice
        z_min = min(bbox_dict.keys())
        z_max = max(bbox_dict.keys())
        z_middle_bbox = bbox_dict[z_middle]
        # sanity check
        # img_roi = image_data_pre[z_middle, z_middle_bbox[1]:z_middle_bbox[3], z_middle_bbox[0]:z_middle_bbox[2]]
        # io.imsave(name.split('.nii.gz')[0] + '_roi.png', img_roi)

        # infer from middle slice to the z_max
        print('infer', name, 'from middle slice to the z_max')
        for z in tqdm(range(z_middle, z_max+1)): # include z_max
            img_2d = image_data_pre[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape
            # img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
            img_1024 = cv2.resize(img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            # convert the shape to (3, H, W)
            img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = medsam_model.image_encoder(img_1024_tensor) # (1, 256, 64, 64)
            if z in bbox_dict.keys():
                box_1024 = bbox_dict[z] / np.array([W, H, W, H]) * 1024
            else:
                pre_seg = seg_data[z-1, :, :] # use the previous slice
                if np.max(pre_seg) > 0:
                    pre_seg1024 = cv2.resize(pre_seg, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    box_1024 = get_bbox(pre_seg1024)
                else:
                    # find the closest z index in bbox_dict
                    z_diff = [abs(z - z_id) for z_id in bbox_dict.keys()]
                    z_closest = list(bbox_dict.keys())[np.argmin(z_diff)]
                    box_1024 = bbox_dict[z_closest] / np.array([W, H, W, H]) * 1024
            bbox_dict[z] = box_1024 / 1024 * np.array([W, H, W, H])
            img_2d_seg = medsam_inference(medsam_model, image_embedding, box_1024[None,:], H, W)
            seg_data[z, img_2d_seg>0] = 1

        # infer from middle slice to the z_max
        print('infer', name, 'from middle slice to the z_min')
        for z in tqdm(range(z_middle-1, z_min-1, -1)):
            img_2d = image_data_pre[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape
            img_1024 = cv2.resize(img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            # convert the shape to (3, H, W)
            img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = medsam_model.image_encoder(img_1024_tensor) # (1, 256, 64, 64)

            if z in bbox_dict.keys():
                box_1024 = bbox_dict[z] / np.array([W, H, W, H]) * 1024
            else:
                pre_seg = seg_data[z+1, :, :]
                if np.max(pre_seg) > 0:
                    pre_seg1024 = cv2.resize(pre_seg, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    box_1024 = get_bbox(pre_seg1024.astype(np.uint8))
                else:
                    # find the closest z index in bbox_dict
                    z_diff = [abs(z - z_id) for z_id in bbox_dict.keys()]
                    z_closest = list(bbox_dict.keys())[np.argmin(z_diff)]
                    box_1024 = bbox_dict[z_closest] / np.array([W, H, W, H]) * 1024
            bbox_dict[z] = box_1024 / 1024 * np.array([W, H, W, H])
            img_2d_seg = medsam_inference(medsam_model, image_embedding, box_1024[None,:], H, W)
            seg_data[z, img_2d_seg>0] = 1

    seg_sitk = sitk.GetImageFromArray(seg_data)
    seg_sitk.CopyInformation(img_sitk)
    sitk.WriteImage(seg_sitk, join(seg_path, name))
    end_time = time.time()

    # save bounding box info
    seg_info['name'].append(name)
    seg_info['running time'].append(end_time - start_time)

# save bbox info
seg_df = pd.DataFrame(seg_info)
seg_df.to_csv(join(seg_path, 'seg_info.csv'), index=False)

