#%% set environment
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join
from tqdm import tqdm
from skimage import transform
import torch
from segment_anything import sam_model_registry, SamPredictor
import glob
import os
import argparse


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))     


# %%
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str, default="")
parser.add_argument('-o', '--seg_path', type=str, default="")
parser.add_argument('-m', '--model_path', type=str, default="path to sam_vit_b_01ec64")
args = parser.parse_args()
img_path = args.img_path
seg_path = args.seg_path
makedirs(seg_path, exist_ok=True)

SAM_MODEL_TYPE = "vit_b"
SAM_CKPT_PATH = args.model_path
device = torch.device("cuda:0")

gt_path_files = sorted(glob.glob(join(img_path, '**/*.npz'), recursive=True))
print('find {} files'.format(len(gt_path_files)))
image_size = 1024
bbox_shift = 20

# %% set up model
sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CKPT_PATH)
sam_model.to(device=device)
predictor = SamPredictor(sam_model)

#%% predict npz files and save results
for gt_path_file in tqdm(gt_path_files):
    npz_name = os.path.basename(gt_path_file)
    task_folder = gt_path_file.split('/')[-2]
    os.makedirs(join(seg_path, task_folder), exist_ok=True)
    if not os.path.isfile(join(seg_path, task_folder, npz_name)):
        npz_data = np.load(gt_path_file, 'r', allow_pickle=True) # (H, W, 3)
        img_3c = npz_data['imgs'] # (Num, H, W)
        gt = npz_data['gts']
        segs = np.zeros_like(gt, dtype=np.uint8) 

        if len(img_3c.shape) == 2:
            img_3c = np.repeat(img_3c[:,:, None], 3, axis=-1)

        resize_img_1024 = cv2.resize(img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        predictor.set_image(resize_img_1024.astype(np.uint8)) # conpute the image embedding only once

        gt_1024 = cv2.resize(gt, (1024, 1024), interpolation=cv2.INTER_NEAREST) # (1024, 1024)
        label_ids = np.unique(gt)[1:]
        for label_id in label_ids:
            gt_1024_label_id = np.uint8(gt_1024 == label_id) # only one label, (256, 256)
            y_indices, x_indices = np.where(gt_1024_label_id > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            H, W = gt_1024_label_id.shape
            x_min = max(0, x_min - bbox_shift)
            x_max = min(W, x_max + bbox_shift)
            y_min = max(0, y_min - bbox_shift)
            y_max = min(H, y_max + bbox_shift)
            bboxes1024 = np.array([x_min, y_min, x_max, y_max])
            
            sam_mask, _, _ = predictor.predict(point_coords=None, point_labels=None, box=bboxes1024[None, :], multimask_output=False) #1024x1024, bool
            sam_mask = transform.resize(sam_mask[0].astype(np.uint8), (gt.shape[-2], gt.shape[-1]), order=0, preserve_range=True, mode='constant', anti_aliasing=False) # (256, 256)
            segs[sam_mask > 0] = label_id

        np.savez_compressed(join(seg_path, task_folder, npz_name), segs=segs, gts=gt)
 