import torch
from sam2.build_sam import build_sam2
from glob import glob
from tqdm import tqdm
from time import time
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import os
from os.path import join, isfile, basename
import cv2
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
from datetime import datetime
import random
import argparse

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--checkpoint',
    type=str,
    default="./checkpoints/sam2_hiera_base_plus.pt",
    help='checkpoint path',
)
parser.add_argument(
    '--cfg',
    type=str,
    default="sam2_hiera_b+.yaml",
    help='model config',
)
parser.add_argument(
    '--png_save_dir',
    type=str,
    default="./results/overlay_base",
    help='GT and predicted masks will be saved here',
)
parser.add_argument(
    '--save_overlay',
    default=False,
    action='store_true',
    help='whether to save the overlay image'
)
parser.add_argument(
    '--imgs_path',
    type=str,
    default="./data/imgs",
    help='imgs path',
)
parser.add_argument(
    '--gts_path',
    type=str,
    default="./data/gts",
    help='gts path',
)
parser.add_argument(
    '--pred_save_dir',
    type=str,
    default="./results/segs_base",
    help='segs path',
)

args = parser.parse_args()
checkpoint = args.checkpoint
model_cfg = args.cfg
save_overlay = args.save_overlay
png_save_dir = args.png_save_dir
imgs_path = args.imgs_path
gts_path = args.gts_path
pred_save_dir = args.pred_save_dir
if save_overlay:
    os.makedirs(png_save_dir, exist_ok=True)
os.makedirs(pred_save_dir, exist_ok=True)

predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def get_bbox256(mask_256, bbox_shift=5):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates
    
    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask_256 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask_256.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)

    bboxes256 = np.array([x_min, y_min, x_max, y_max])

    return bboxes256

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     


@torch.inference_mode()
def infer(img_npz_file):
    print(f'infering {img_npz_file}')
    #npz = np.load(os.path.join(img_path, file), 'r', allow_pickle=True)
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3c = npz_data['imgs'] # (H, W, 3)
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    boxes = npz_data['boxes']
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint16)

    for idx, box in enumerate(boxes, start=1):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(img_3c)
            xmin, ymin, xmax, ymax = box
            box = box[None, ...] # (1, 4)
            masks, _, _ = predictor.predict(box=box, multimask_output=False)
            segs[masks[0]>0] = idx

    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )
    # visualize image, mask and bounding box with GT
    if save_overlay:
        npz_gts = np.load(join(gts_path, npz_name), 'r', allow_pickle=True)['gts']
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3c)
        ax[1].imshow(img_3c)
        ax[0].set_title("Image")
        ax[1].set_title("SAM2 Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box in enumerate(boxes):
            color = np.random.rand(3)
            box_viz = box
            show_box(box_viz, ax[1], edgecolor=color)
            show_mask((segs == i+1).astype(np.uint8), ax[1], mask_color=color)

            show_box(box_viz, ax[0], edgecolor=color)
            show_mask((npz_gts == i+1).astype(np.uint8), ax[0], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()


if __name__ == '__main__':
    img_npz_files = sorted(glob(join(imgs_path, '2D*.npz'), recursive=True))
    processed = os.listdir(pred_save_dir)
    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['time'] = []
    for img_npz_file in tqdm(img_npz_files):
        start_time = time()
        assert basename(img_npz_file).startswith('2D'), f'file name should start with 2D, but got {basename(img_npz_file)}'
        infer(img_npz_file)
        end_time = time()
        efficiency['case'].append(basename(img_npz_file))
        efficiency['time'].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(current_time, 'file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(join(pred_save_dir, 'efficiency.csv'), index=False)
