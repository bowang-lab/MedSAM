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
    default="./results/vidoes/US/overlay_base",
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
    default="./data/videos/US/CAMUS_preprocessed_test_relabel",
    help='imgs path',
)
parser.add_argument(
    '--gts_path',
    type=str,
    default="./data/videos/US/CAMUS_preprocessed_test_relabel",
    help='gts path',
)
parser.add_argument(
    '--pred_save_dir',
    type=str,
    default="./results/vidoes/US/segs_base",
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

predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)
predictor_perslice = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def mask2D_to_bbox(gt2D, shift=5):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = gt2D.shape
    bbox_shift = np.random.randint(0, shift + 1, 1)[0]
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    boxes = np.array([x_min, y_min, x_max, y_max])
    return boxes

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



def resize_rgb_video(array, image_size):
    """
    Resize a 4D RGB video NumPy array and resize each frame.
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    
    c, d, h, w = array.shape
    assert c == 3
    resized_array = np.zeros((d, c, image_size, image_size))
    
    for i in range(d):
        img_array = array[:, i, :, :].transpose(1, 2, 0)  # (h, w, 3)
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        img_resized = img_pil.resize((image_size, image_size))
        img_resized_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_resized_array
    
    return resized_array



@torch.inference_mode()
def infer_video(img_npz_file):
    print(f'infering {img_npz_file}')
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    gts = np.load(os.path.join(gts_path, npz_name), 'r', allow_pickle=True)['gts']
    print(np.unique(gts))
    img_3D = npz_data['imgs']  # (3, D, H, W)
    assert np.max(img_3D) < 256, f'input data should be in range [0, 255], but got {np.unique(img_3D)}'
    C, D, H, W = img_3D.shape
    segs_3D = np.zeros((D, H, W), dtype=np.uint8)
    video_height = img_3D.shape[2]
    video_width = img_3D.shape[3]
    img_resized = resize_rgb_video(img_3D, 1024) #d, 3, 1024, 1024
    img_3D = np.transpose(img_3D, (1, 0, 2, 3))
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()
    img_mean=(0.485, 0.456, 0.406)
    img_std=(0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std
    z_inits = []
    prompts = []
    gt_ids_3d = np.unique(gts)
    
    gt_ids_3d = gt_ids_3d[gt_ids_3d != 0]
    ulabs = np.unique(gts)
    
    for idx, lab in enumerate(ulabs, start=1):
        gt = (gts == (idx))
        indices = np.where(gt)
        z_indices = indices[0]
        if len(z_indices) == 0:
            print('no gt')
            continue
        z_min = z_indices.min() if z_indices.size > 0 else None
        z_max = z_indices.max() if z_indices.size > 0 else None
        z_init = 0 
        gt_roi = gt[z_min:(z_max+1)]
        box_2d = mask2D_to_bbox(gt_roi[z_init])
        box_2d = box_2d[None, ...]  # (1, 4)
        img = img_resized[z_min:(z_max+1)]
        z_init_orig = z_init + z_min
        z_inits.append(z_init_orig)
        prompts.append(box_2d)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            img_slice = img_3D[z_init_orig] #3, h, w
            img_slice = np.transpose(img_slice, (1, 2, 0))
            predictor_perslice.set_image(img_slice)
            mks, _, _ = predictor_perslice.predict(box=box_2d, multimask_output=False)
            mask_prompt = mks[0]
            inference_state = predictor.init_state(img, video_height, video_width)
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=z_init, obj_id=1, mask=mask_prompt)
            segs_3D[z_init_orig, ((masks[0] > 0.0).cpu().numpy())[0]] = idx
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                segs_3D[(z_min + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx
            predictor.reset_state(inference_state)

    print(np.unique(segs_3D))
    np.savez_compressed(join(pred_save_dir, npz_name), segs=segs_3D)

    if save_overlay:
        npz_gts = np.load(join(gts_path, npz_name), 'r', allow_pickle=True)
        gts = npz_gts['gts']
        idx = random.sample(z_inits,1)[0] 
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(np.transpose(img_3D[idx], (1,2,0))) #d, 3, h, w to hw3
        ax[1].imshow(np.transpose(img_3D[idx], (1,2,0)))
        ax[0].set_title("GT")
        ax[1].set_title("SAM2 Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box2d in enumerate(prompts, start=1):
            if np.sum(segs_3D[idx]==i) > 0:
                color = np.random.rand(3)
                print(box2d)
                x_min, y_min, x_max, y_max = box2d[0]
                box_viz = np.array([x_min, y_min, x_max, y_max])
                show_box(box_viz, ax[1], edgecolor=color)
                show_mask(segs_3D[idx]==i, ax[1], mask_color=color)
                show_box(box_viz, ax[0], edgecolor=color)
                show_mask(gts[idx]==i, ax[0], mask_color=color)
            else:
                print('no mask')

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()


if __name__ == '__main__':
    img_npz_files = sorted(glob(join(imgs_path, '*.npz'), recursive=True))
    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['time'] = []
    for img_npz_file in tqdm(img_npz_files):
        start_time = time()
        infer_video(img_npz_file)
        end_time = time()
        efficiency['case'].append(basename(img_npz_file))
        efficiency['time'].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(current_time, 'file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(join(pred_save_dir, 'efficiency.csv'), index=False)
