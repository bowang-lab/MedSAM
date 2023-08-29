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

import cv2
from skimage import morphology, color
import torch.nn.functional as F

from matplotlib import pyplot as plt

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

import argparse
# %%
torch.cuda.empty_cache()
os.environ['PYTHONHASHSEED']=str(2023)
random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)

# %%
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
    '--grey',
    action='store_true',
    default=False,
    help="Whether the input image is in grey scale"
)
parser.add_argument(
    '-num_workers', type=int, default=1,
    help='number of workers for multiprocessing'
)

# %%
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

## Check whether there exist subfolders
has_task = isdir(join(data_root, data_root_files[0]))
if has_task:
    gt_path_files = sorted(glob.glob(join(data_root, '**/*.npz'), recursive=True))
else:
    gt_path_files = sorted(glob.glob(join(data_root, '*.npz'), recursive=True))
bbox_shift = args.bbox_shift
props = {'spacing': (999, 1, 1)}
is_grey = args.grey

# %%
def show_mask(mask, ax, random_color=True, alpha=0.45):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
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

# %%
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
def preprocess_image_rgb(img_3c, gt_2D, bbox_shift=10):
    """
    Append bounding box prompt channel to image
    """
    img_3c = np.transpose(img_3c, (2, 0, 1)) ## (3, H, W)
    if gt_2D.shape[0] != img_3c.shape[1] or gt_2D.shape[1] != img_3c.shape[2]:
        gt_2D = cv2.resize(
            gt_2D, (img_3c.shape[2], img_3c.shape[1]),
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
    img_4c = np.concatenate([img_3c, np.zeros((1, img_3c.shape[1], img_3c.shape[2]))], axis=0)
    img_4c[-1, y_min:y_max, x_min:x_max] = 1.0
    img_4c = img_4c[:, None, ...] ## (4, 1, H, W)

    return torch.tensor(img_4c).float()

def preprocess_image_grey(img_3c, gt_2D, bbox_shift=10):
    """
    Append bounding box prompt channel to image
    """
    if len(img_3c.shape) == 3:
        img_1c = np.uint8(color.rgb2gray(img_3c)*255.0)
    else:
        img_1c = img_3c
    img_1c = img_1c[None, ...] ## (1, H, W)
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
    imgs = npz['imgs'] # (H, W, C)
    gts = npz['gts'] # (H, W)
    segs = np.zeros_like(imgs[..., 0], dtype=np.uint8)

    label_ids = np.unique(gts)[1:]
    for label_id in label_ids:
        gt_2D = np.uint8(gts == label_id) # one label at a time
        if is_grey:
            img_bbox = preprocess_image_grey(
                imgs,
                gt_2D,
                bbox_shift=bbox_shift
            )
        else:
            img_bbox = preprocess_image_rgb(
                imgs,
                gt_2D,
                bbox_shift=bbox_shift
            )
        if img_bbox == None:
            continue ## No label available for the image
        seg_2D = predictor.predict_single_npy_array(
            input_image = img_bbox,
            image_properties = props,
            segmentation_previous_stage = None,
            output_file_truncated = None,
            save_or_return_probabilities = False
        )
        seg_2D = seg_2D.squeeze()
        seg_2D = cv2.resize(
            seg_2D, (imgs.shape[1], imgs.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)

        segs[seg_2D > 0] = label_id

    if gts.shape[0] != imgs.shape[0] or gts.shape[1] != imgs.shape[1]:
        gts = cv2.resize(
            gts,
            (imgs.shape[1], imgs.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    np.savez_compressed(
        join(pred_save_dir_task, npz_name),
        segs=segs,
        gts=gts
    )

    if save_overlay:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        if args.grey:
            ax[0].imshow(imgs, cmap='gray')
        else:
            ax[0].imshow(imgs)
        ax[0].set_title("Image")
        ax[0].axis('off')
        if args.grey:
            ax[1].imshow(imgs, cmap='gray')
        else:
            ax[1].imshow(imgs)
        show_mask(gts, ax[1])
        ax[1].axis('off')
        ax[1].set_title("Ground Truth")
        if args.grey:
            ax[2].imshow(imgs, cmap='gray')
        else:
            ax[2].imshow(imgs)
        show_mask(segs, ax[2])
        ax[2].set_title("Segmentation")
        ax[2].axis('off')
        plt.savefig(
            join(png_save_dir_task, npz_name.split(".")[0] + '.png'),
            dpi=300
        )
        plt.close()
# %%
if __name__ == '__main__':
    num_workers = num_workers
    mp.set_start_method('spawn')
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=len(gt_path_files)) as pbar:
            for i, _ in tqdm(enumerate(pool.imap_unordered(nnunet_infer_npz, gt_path_files))):
                pbar.update()