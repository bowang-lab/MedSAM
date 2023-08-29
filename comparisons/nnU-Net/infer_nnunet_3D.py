# %%
import os
import glob
import random
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir, basename
from tqdm import tqdm, trange
from copy import deepcopy
from time import time
import numpy as np
import torch
from torch._dynamo import OptimizedModule
from torch import multiprocessing as mp
from datetime import datetime

import cv2
from skimage import morphology
import torch.nn.functional as F

from matplotlib import pyplot as plt

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

import argparse

import timeit

torch.cuda.empty_cache()
os.environ['PYTHONHASHSEED']=str(2023)
random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-checkpoint',
    type=str,
    default='',
    help='Path to the model checkpoint directory in nnUNet_results',
    required=True
)
parser.add_argument(
    '-data_root',
    type=str,
    default='',
    help='Path to the validation data directory',
    required=True
)
parser.add_argument(
    '-pred_save_dir',
    type=str,
    default='',
    help='Path to the directory where the segmentation results will be saved in npz format'
)
parser.add_argument('--save_overlay', action='store_true', default=False, help="Whether to save segmentation overlay")
parser.add_argument(
    '-png_save_dir',
    type=str,
    default='',
    help='Path to the directory where the segmentation overlay will be saved in png format'
)
parser.add_argument(
    '-bbox_shift',
    type=int,
    default=10,
    help='Perturbation shift of bounding box prompt'
)
parser.add_argument(
    '-num_workers', type=int, default=1,
    help='number of workers for multiprocessing'
)

args = parser.parse_args()
checkpoint = args.checkpoint
data_root = args.data_root
pred_save_dir = args.pred_save_dir
png_save_dir = args.png_save_dir
makedirs(pred_save_dir, exist_ok=True)
save_overlay = args.save_overlay
if save_overlay:
    makedirs(png_save_dir, exist_ok=True)
num_workers = args.num_workers
data_root_files = listdir(data_root)
## Check if there are subfolders
has_task = isdir(join(data_root, data_root_files[0]))
if has_task:
    gt_path_files = sorted(glob.glob(join(data_root, '**/*.npz'), recursive=True))
else:
    gt_path_files = sorted(glob.glob(join(data_root, '*.npz'), recursive=True))
bbox_shift = args.bbox_shift
props = {'spacing': (999, 1, 1)}
# %%
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.45])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def dice_coefficient(preds, targets):
    smooth = 1.0
    assert preds.shape == targets.shape

    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=False, ## disable tta
    perform_everything_on_gpu=True,
    device=torch.device('cuda', 0),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=False
)
predictor.initialize_from_trained_model_folder(
    join(checkpoint, 'nnUNetTrainer__nnUNetPlans__2d'),
    use_folds='all',
    checkpoint_name='checkpoint_final.pth',
)

# %%
def preprocess_slice(img_2D, gt_2D, bbox_shift=10):
    """
    Append bounding box prompt channel to image
    """
    img_1c = img_2D[None, ...] ## (1, H, W)
    if gt_2D.shape[0] != img_1c.shape[1] or gt_2D.shape[1] != img_1c.shape[2]:
        gt_2D = cv2.resize(
            gt_2D, (img_1c.shape[2], img_1c.shape[1]),
            interpolation=cv2.INTER_NEAREST
        )
        gt_2D = np.uint8(gt_2D)
    else:
        gt_2D = gt_2D.astype(np.uint8)
    try:
        assert np.max(gt_2D) == 1 and np.min(gt_2D) == 0, 'ground truth should be 0, 1, got: ' + str(np.unique(gt_2D))
    except:
        assert np.max(gt_2D) == 0 and np.min(gt_2D) == 0, 'ground truth should be 0, 1, got: ' + str(np.unique(gt_2D))
        return None

    y_indices, x_indices = np.where(gt_2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = gt_2D.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    ## Append bbox prompt channel as the last channel
    img_2c = np.concatenate([img_1c, np.zeros((1, img_1c.shape[1], img_1c.shape[2]))], axis=0)
    img_2c[-1, y_min:y_max, x_min:x_max] = 1.0
    img_2c = img_2c[:, None, ...] ## (2, 1, H, W)

    return torch.tensor(img_2c).float()

# %%
def nnunet_infer_npz(gt_path_file):
    npz_name = basename(gt_path_file)
    if has_task:
        task_folder = gt_path_file.split('/')[-2]
        pred_save_dir_task = join(pred_save_dir, task_folder)
        png_save_dir_task = join(png_save_dir, task_folder)
        makedirs(pred_save_dir_task, exist_ok=True)
        makedirs(png_save_dir_task, exist_ok=True)
    else:
        pred_save_dir_task = pred_save_dir
        png_save_dir_task = png_save_dir
    if isfile(join(pred_save_dir_task, npz_name)):
        return

    npz = np.load(gt_path_file, 'r', allow_pickle=True)
    img_3D = npz['imgs'] # (Num, H, W)
    gt_3D = npz['gts'] # (Num, H, W)
    spacing = npz['spacing']
    seg_3D = np.zeros_like(gt_3D, dtype=np.uint8)

    for i in range(img_3D.shape[0]):
        img_2D = img_3D[i,:,:] # (H, W)
        gt = gt_3D[i,:,:] # (H, W)
        label_ids = np.unique(gt)[1:]
        for label_id in label_ids:
            gt2D = np.uint8(gt == label_id) # one label at a time
            img_2c = preprocess_slice(
                img_2D,
                gt2D,
                bbox_shift=bbox_shift
            )
            if img_2c is None:
                continue ## no label available for this slice
            seg_2D = predictor.predict_single_npy_array(
                input_image = img_2c,
                image_properties = props,
                segmentation_previous_stage = None,
                output_file_truncated = None,
                save_or_return_probabilities = False
            )
            seg_2D = seg_2D.squeeze()
            seg_2D = cv2.resize(
                seg_2D, (gt2D.shape[1], gt2D.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)

            seg_3D[i, seg_2D>0] = label_id

    np.savez_compressed(
        join(pred_save_dir_task, npz_name),
        segs=seg_3D,
        gts=gt_3D,
        spacing=spacing
    )

    # visualize image, mask and bounding box
    
    if save_overlay:
        idx = int(seg_3D.shape[0] / 2)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img_3D[idx], cmap='gray')
        ax[0].set_title("Image")
        ax[0].axis('off')
        ax[1].imshow(img_3D[idx], cmap='gray')
        show_mask(gt_3D[idx], ax[1])
        ax[1].axis('off')
        ax[1].set_title("Ground Truth")
        ax[2].imshow(img_3D[idx], cmap='gray')
        show_mask(seg_3D[idx], ax[2])
        ax[2].set_title("Segmentation")
        ax[2].axis('off')
        plt.savefig(
            join(png_save_dir_task, npz_name.split(".")[0] + '.png'),
            dpi=300
        )
        plt.close()

    print(f"Case {npz_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    num_workers = num_workers
    mp.set_start_method('spawn')
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=len(gt_path_files)) as pbar:
            pbar.set_description(f"[{basename(data_root)}]: ")
            for i, _ in tqdm(enumerate(pool.imap_unordered(nnunet_infer_npz, gt_path_files))):
                pbar.update()
        