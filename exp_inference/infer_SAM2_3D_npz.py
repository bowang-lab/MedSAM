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
from sam2.build_sam import build_sam2_video_predictor_npz
import cv2
import SimpleITK as sitk
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
parser.add_argument(
    '--save_nifti',
    default=False,
    action='store_true',
    help='whether to save nifti'
)
parser.add_argument(
    '--nifti_path',
    type=str,
    default="./results/segs_nifti",
    help='segs nifti path',
)

args = parser.parse_args()
checkpoint = args.checkpoint
model_cfg = args.cfg
save_overlay = args.save_overlay
png_save_dir = args.png_save_dir
imgs_path = args.imgs_path
gts_path = args.gts_path
pred_save_dir = args.pred_save_dir
save_nifti = args.save_nifti
nifti_path = args.nifti_path

predictor_perslice = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)

os.makedirs(pred_save_dir, exist_ok=True)
if save_overlay:
    os.makedirs(png_save_dir, exist_ok=True)
if save_nifti:
    os.makedirs(nifti_path, exist_ok=True)


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


def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array


@torch.inference_mode()
def infer_3d(img_npz_file):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    gts = np.load(os.path.join(gts_path, npz_name), 'r', allow_pickle=True)['gts']
    img_3D = npz_data['imgs']  # (D, H, W)
    assert np.max(img_3D) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D)}'
    D, H, W = img_3D.shape
    segs_3D = np.zeros(img_3D.shape, dtype=np.uint8)
    boxes_3D = npz_data['boxes']  # (D, num_boxes, 4)
    video_height = img_3D.shape[1]
    video_width = img_3D.shape[2]
    img_resized = resize_grayscale_to_rgb_and_resize(img_3D, 1024) #d, 3, 1024, 1024
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()
    img_mean=(0.485, 0.456, 0.406)
    img_std=(0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std
    z_mids = []
    for idx, box in enumerate(boxes_3D, start=1):
        gt = (gts == (idx))
        indices = np.where(gt)
        z_indices = indices[0]
        if len(z_indices) == 0:
            print('no gt')
            continue
        z_min = z_indices.min() if z_indices.size > 0 else None
        z_max = z_indices.max() if z_indices.size > 0 else None
        box_2d = box[[0,1,3,4]]
        box_2d = box_2d[None, ...]  # (1, 4)
        img = img_resized[z_min:(z_max+1)]
        z_mid = int(img.shape[0]/2)
        z_mid_orig = z_mid + z_min
        z_mids.append(z_mid_orig)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            img_slice = img_3D[z_mid_orig]
            img_slice = np.repeat(img_slice[:, :, None], 3, axis=-1)
            predictor_perslice.set_image(img_slice)
            mks, _, _ = predictor_perslice.predict(box=box_2d, multimask_output=False)
            mask_prompt = mks[0].astype(np.uint8)

            inference_state = predictor.init_state(img, video_height, video_width)
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=z_mid, obj_id=1, mask=mask_prompt)
            segs_3D[z_mid_orig, ((masks[0] > 0.0).cpu().numpy())[0]] = idx
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx
            predictor.reset_state(inference_state)
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=z_mid, obj_id=1, mask=mask_prompt)
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
                segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx
            predictor.reset_state(inference_state)

    print(np.unique(segs_3D))
    np.savez_compressed(join(pred_save_dir, npz_name), segs=segs_3D)

    if save_nifti:
        sitk_image = sitk.GetImageFromArray(segs_3D)
        sitk.WriteImage(sitk_image, os.path.join(nifti_path, npz_name.replace('.npz', '.nii.gz')))

    if save_overlay:
        npz_gts = np.load(join(gts_path, npz_name), 'r', allow_pickle=True)
        gts = npz_gts['gts']
        idx = random.sample(z_mids,1)[0] 
        print('plot for idx ', idx)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3D[idx], cmap='gray')
        ax[1].imshow(img_3D[idx], cmap='gray')
        ax[0].set_title("Image")
        ax[1].set_title("SAM2 Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box3D in enumerate(boxes_3D, start=1):
            if np.sum(segs_3D[idx]==i) > 0:
                color = np.random.rand(3)
                x_min, y_min, z_min, x_max, y_max, z_max = box3D
                box_viz = np.array([x_min, y_min, x_max, y_max])
                if idx >= z_min and idx <= z_max:
                    show_box(box_viz, ax[1], edgecolor=color)
                show_mask(segs_3D[idx]==i, ax[1], mask_color=color)
                if idx >= z_min and idx <= z_max:
                    show_box(box_viz, ax[0], edgecolor=color)
                show_mask(gts[idx]==i, ax[0], mask_color=color)
            else:
                print('no mask')

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()


if __name__ == '__main__':
    img_npz_files = sorted(glob(join(imgs_path, '3D*.npz'), recursive=True))

    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['time'] = []
    for img_npz_file in tqdm(img_npz_files):
        start_time = time()
        infer_3d(img_npz_file)
        end_time = time()
        efficiency['case'].append(basename(img_npz_file))
        efficiency['time'].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(current_time, 'file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(join(pred_save_dir, 'efficiency.csv'), index=False)
