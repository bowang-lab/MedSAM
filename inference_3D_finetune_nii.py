from os import listdir, makedirs
from os.path import join, isfile, basename

from glob import glob
from tqdm import tqdm
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import cc3d
import torch.nn.functional as F

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
from matplotlib import pyplot as plt
import cv2
import torch.multiprocessing as mp
import SimpleITK as sitk
import argparse
import json

#%% set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)

parser = argparse.ArgumentParser()

parser.add_argument(
    '-nii_gts_dir',
    type=str,
    required=True,
    help='Root directory containing ground truth .nii.gz files used for bounding boxes.'
)

parser.add_argument(
    '-medsam_lite_checkpoint_path',
    type=str,
    default="workdir/lite_medsam.pth",
    help='Path to the checkpoint of MedSAM-Lite (default is "workdir/lite_medsam.pth").'
)

parser.add_argument(
    '-nii_img_dir',
    type=str,
    required=True,
    help='Directory containing the original .nii.gz image files.'
)

parser.add_argument(
    '-device',
    type=str,
    default="cuda:0",
    help='Device to run the inference (default is "cuda:0").'
)

parser.add_argument(
    '-num_workers',
    type=int,
    default=4,
    help='Number of workers for inference with multiprocessing (default is 4).'
)

parser.add_argument(
    '--save_overlay',
    action='store_true',
    help='Flag to save the overlay image (no additional argument needed).'
)

parser.add_argument(
    '-png_save_dir',
    type=str,
    default='./overlay/CT_Abd',
    help='Directory to save the overlay image (default is "./overlay/CT_Abd").'
)

parser.add_argument(
    '-nii_pred_dir',
    type=str,
    default='./nii_pred_dir/CT_Abd',
    help='Directory to save the predicted segmentation masks in .nii.gz format (default is "./nii_pred_dir/CT_Abd").'
)

parser.add_argument(
    '-nii_box_dir',
    type=str,
    default='./nii_box_dir/CT_Abd',
    help='Directory to save the bounding box masks in .nii.gz format (default is "./nii_box_dir/CT_Abd").'
)

parser.add_argument(
    '--overwrite',
    action='store_true',
    help='Flag to overwrite existing predictions (no additional argument needed).'
)

parser.add_argument(
    '--zero_shot',
    action='store_true',
    help='Flag for zero-shot prediction. (no additional argument needed)'
)

parser.add_argument("-window_level", type=int, default=40,
                    help="CT window level, [default: 40]")
parser.add_argument("-window_width", type=int, default=400,
                    help="CT window width, [default: 400]")

parser.add_argument("-modality_suffix_dict", type=json.loads, 
                    default='{"CT":"_0000.nii.gz"}',
                    help="Modality suffix mapping as a JSON string.")
#example for multi-modaility  inputs
#'{"CT":"_0000.nii.gz","PET":"_0001.nii.gz","T1":"_0002.nii.gz","T2":"_0003.nii.gz"}

#'repeat_first', 'combine_rgb', 'blending_b'
modality_fuse_method = 'combine_rgb'

args = parser.parse_args()

nii_gts_dir = args.nii_gts_dir
save_overlay = args.save_overlay
num_workers = args.num_workers
overwrite = args.overwrite
nii_img_dir = args.nii_img_dir
nii_pred_dir = args.nii_pred_dir
nii_box_dir = args.nii_box_dir
modality_suffix_dict = args.modality_suffix_dict
zero_shot =args.zero_shot
medsam_lite_checkpoint_path = args.medsam_lite_checkpoint_path

# set window level and width
# https://radiopaedia.org/articles/windowing-ct
WINDOW_LEVEL = args.window_level # only for CT images
WINDOW_WIDTH = args.window_width # only for CT images

tumor_id = 2  # only set this when there are multiple tumors; convert semantic masks to instance masks

voxel_num_thre2d = 20
voxel_num_thre3d = 500 # changed from 1000 to 500, some tumors are very small


if save_overlay:
    assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
    png_save_dir = args.png_save_dir
    makedirs(png_save_dir, exist_ok=True)
    
makedirs(nii_box_dir, exist_ok=True)
makedirs(nii_pred_dir, exist_ok=True)


bbox_shift = 5
device = torch.device(args.device)
gt_path_files = sorted(glob(join(nii_gts_dir, '*.nii.gz'), recursive=True))

image_size = 256

def resize_longest_side(image, target_length):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    long_side_length = target_length
    oldh, oldw = image.shape[0], image.shape[1]
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

def draw_bbox_on_slice(slice_data, bbox, label_id, thickness=3):
    """
    Draws a bounding box on a given slice.

    Parameters:
    slice_data : ndarray
        The 2D array representing the image slice.
    bbox : tuple
        The bounding box coordinates (x_min, y_min, x_max, y_max).
    label_id : int
        The label to assign to the bounding box lines.
    thickness : int
        The thickness of the bounding box lines.
    """
    x_min, y_min, x_max, y_max = bbox

    # Draw top and bottom lines
    slice_data[y_min:y_min+thickness, x_min:x_max] = label_id
    slice_data[y_max-thickness:y_max, x_min:x_max] = label_id

    # Draw left and right lines
    slice_data[y_min:y_max, x_min:x_min+thickness] = label_id
    slice_data[y_min:y_max, x_max-thickness:x_max] = label_id
    
class MedSAM_Lite(nn.Module):
    def __init__(
            self, 
            image_encoder, 
            mask_decoder,
            prompt_encoder
        ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, box_np):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box_np, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :] # (B, 1, 4)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_np,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing

        Parameters
        ----------
        masks : torch.Tensor
            masks predicted by the model
        new_size : tuple
            the shape of the image after resizing to the longest side of 256
        original_size : tuple
            the original shape of the image

        Returns
        -------
        torch.Tensor
            the upsampled mask to the original size
        """
        # Crop
        masks = masks[..., :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks


def show_mask(mask, ax, mask_color=None, alpha=0.5):
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     


def resize_box(box, new_size, original_size):
    """
    Revert box coordinates from scale at 256 to original scale

    Parameters
    ----------
    box : np.ndarray
        box coordinates at 256 scale
    new_size : tuple
        Image shape with the longest edge resized to 256
    original_size : tuple
        Original image shape

    Returns
    -------
    np.ndarray
        box coordinates at original scale
    """
    new_box = np.zeros_like(box)
    ratio = max(original_size) / max(new_size)
    for i in range(len(box)):
       new_box[i] = int(box[i] * ratio)

    return new_box


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_256, new_size, original_size):
    box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)
    
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points = None,
        boxes = box_torch,
        masks = None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False
    )

    low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
    low_res_pred = torch.sigmoid(low_res_pred)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg

def get_bbox(gt2D, bbox_shift=5):
    assert np.max(gt2D)==1 and np.min(gt2D)==0.0, f'ground truth should be 0, 1, but got {np.unique(gt2D)}'
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = gt2D.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)
    bboxes = np.array([x_min, y_min, x_max, y_max])

    return bboxes


medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[
        64, ## (64, 256, 256)
        128, ## (128, 128, 128)
        160, ## (160, 64, 64)
        320 ## (320, 64, 64) 
    ],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
)

medsam_lite_prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(256, 256),
    mask_in_chans=16
)

medsam_lite_mask_decoder = MaskDecoder(
    num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
)

medsam_lite_model = MedSAM_Lite(
    image_encoder = medsam_lite_image_encoder,
    mask_decoder = medsam_lite_mask_decoder,
    prompt_encoder = medsam_lite_prompt_encoder
)

medsam_lite_checkpoint = torch.load(medsam_lite_checkpoint_path, map_location='cpu')

if zero_shot:
    medsam_lite_model.load_state_dict(medsam_lite_checkpoint) # assuming 'model' key contains the state_dict #finetune may add additional keys.
else:
    medsam_lite_model.load_state_dict(medsam_lite_checkpoint['model']) # assuming 'model' key contains the state_dict #finetune may add additional keys.
    
medsam_lite_model.to(device)
medsam_lite_model.eval()

# nii preprocess start
def preprocess_img(image_data, modality):
    if modality == "CT":
        lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
        upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
    elif modality == "PET": #clip PET images to 0-6 SUV, then rescale to 0-255
        image_data_pre = np.clip(image_data, 0, 6)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
    else:  # clip other images to 0.05% to 99.5% intensity, then rescale to 0-255
        lower_bound, upper_bound = np.percentile(
            image_data[image_data > 0], 0.5
        ), np.percentile(image_data[image_data > 0], 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = image_data
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        image_data_pre[image_data == 0] = 0
        
    return image_data_pre

class ImageReader:
    def __init__(self, img_path, modalities):
        self.img_path = img_path
        self.modalities = modalities
        self.images = {}
        self.sitk_images = {}
        
    def read_images(self, name):
        for modality in self.modalities:
            if modality in modality_suffix_dict:
                img_name = name.split('.nii.gz')[0]  + modality_suffix_dict[modality]
                img_sitk = sitk.ReadImage(join(self.img_path, img_name))
                self.images[modality] = sitk.GetArrayFromImage(img_sitk)
                self.sitk_images[modality] = img_sitk
                
    def get_image_data(self, modality):
        return self.images.get(modality)  
      
    def get_sitk_data(self, modality):
        return self.sitk_images.get(modality)
    
def fuse_images(images, method='combine_rgb'):
    if method == 'repeat_first':
        return np.stack([images[0], images[0], images[0]], axis=-1)
    elif method == 'combine_rgb':
        return np.stack([images[0], images[1], images[2]], axis=-1)
    elif method == 'blending_b': 
        # use only if modality number is 4, the last two image modality will be blended. 
        
        alpha = 0.5  # Weight for the blend image
        # Function to rescale img1 to the scale of img2
        def rescale_img(img1, img2):
            min_img1, max_img1 = img1.min(), img1.max()
            min_img2, max_img2 = img2.min(), img2.max()
            # Rescale img1
            rescaled_img1 = (img1 - min_img1) / (max_img1 - min_img1) * (max_img2 - min_img2) + min_img2
            return rescaled_img1

        # Rename images for clarity
        img1_for_blend = rescale_img(images[2], images[3])
        blended_last_channel = alpha * img1_for_blend + (1 - alpha) * images[3]
    
        # Stacking the blended channels to create the RGB image
        return np.stack([images[0], images[1], blended_last_channel], axis=-1)
    else:
        raise ValueError("Unsupported fusion method.")
    
def MedSAM_infer_nii(nii_gts_dir):
    nii_name = basename(nii_gts_dir)
    patient_id = nii_name.split('.nii.gz')[0]
    
    # Read NIfTI ground truth
    nii_gts = sitk.ReadImage(nii_gts_dir)
    arr_gts = sitk.GetArrayFromImage(nii_gts)  # Convert to numpy array
    
    #check number of modalities
    modalities = list(modality_suffix_dict.keys())
    if len(modalities)==1:
        single_modality = True
        modality = list(modality_suffix_dict.keys())[0]
        nii_img_name = join(nii_img_dir, patient_id+modality_suffix_dict[modality]) 
        
        # Read NIfTI image 
        nii_img = sitk.ReadImage(nii_img_name)
        arr_img = sitk.GetArrayFromImage(nii_img)  # Convert to numpy array
    elif len(modalities)>1:
        single_modality = False
        reader = ImageReader(nii_img_dir, list(modality_suffix_dict.keys()))
        reader.read_images(nii_name)
        
        # Read NIfTI image, preprocess and fuse image
        img_list = [preprocess_img(reader.get_image_data(modality),modality) for modality in reader.modalities]
        fused_image = fuse_images(img_list, method = modality_fuse_method)
        arr_img = fused_image
        
    gt_data_ori = arr_gts
    
    if tumor_id is not None:
        tumor_bw = np.uint8(arr_gts == tumor_id)
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
    roi_slices, _, _ = np.where(gt_data_ori > 0)
    roi_slices = np.unique(roi_slices)
    
    
    # Placeholder for the segmentation/bbox result, initialize with zeros
    seg_3D = np.zeros_like(arr_gts, dtype=np.uint8)
    box_3D = np.zeros_like(arr_gts, dtype=np.uint8)
    
    if single_modality:
        img_3D = preprocess_img(arr_img, modality)  # if single modality, need to preprocess
    else:
        img_3D = arr_img
    
    pred_path = join(nii_pred_dir, patient_id + '.nii.gz')

    if not isfile(pred_path) or overwrite:
        box_list = [dict() for _ in range(img_3D.shape[0])]
        for i in roi_slices:
            if single_modality:
                img_2d = img_3D[i,:,:] # (H, W)
                H, W = img_2d.shape[:2]
                img_3c = np.repeat(img_2d[:,:, None], 3, axis=-1) # (H, W, 3) # repeat for single modality
            else:
                img_2d = img_3D[i,:,:,:] # (H, W)
                H, W = img_2d.shape[:2]
                img_3c = img_2d 
                
            ## MedSAM Lite preprocessing
            img_256 = resize_longest_side(img_3c, 256)
            newh, neww = img_256.shape[:2]
            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )
            img_256_padded = pad_image(img_256, 256)
            img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(img_256_tensor)

            gt = gt_data_ori[i,:,:] # (H, W)
            
            label_ids = np.unique(gt)[1:]
            
            for label_id in label_ids:
                gt2D = np.uint8(gt == label_id) # only one label, (H, W)
                if gt2D.shape != (newh, neww):
                    gt2D_resize = cv2.resize(
                        gt2D.astype(np.uint8), (neww, newh),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(np.uint8)
                else:
                    gt2D_resize = gt2D.astype(np.uint8)
                gt2D_padded = pad_image(gt2D_resize, 256) ## (256, 256)
                if np.sum(gt2D_padded) > 0:
                    box = get_bbox(gt2D_padded, bbox_shift) # (4,)
                    sam_mask = medsam_inference(medsam_lite_model, image_embedding, box, (newh, neww), (H, W))
                    seg_3D[i, sam_mask>0] = label_id
                    box_list[i][label_id] = box
                    
                    box_viz = resize_box(box, (newh, neww), (H, W))
                    draw_bbox_on_slice(box_3D[i, :, :], box_viz, label_id, thickness=2)
            
        label_ids = np.unique(arr_gts)[1:]

        # np.savez_compressed(
        #     join(pred_save_dir, task_folder, npz_name),
        #     segs=seg_3D, gts=arr_gts, spacing=spacing
        # )
        
        #seg_3D[seg_3D>1] = 2
        
        seg_sitk = sitk.GetImageFromArray(seg_3D)
        seg_sitk.CopyInformation(nii_gts)
        sitk.WriteImage(seg_sitk, join(nii_pred_dir, patient_id+'.nii.gz'))  

        box_sitk = sitk.GetImageFromArray(box_3D)
        box_sitk.CopyInformation(nii_gts)
        sitk.WriteImage(box_sitk, join(nii_box_dir, patient_id+'.nii.gz'))
    
        # visualize image, mask and bounding box
        if save_overlay:
            idx = int(seg_3D.shape[0] / 2)
            box_dict = box_list[idx]
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            if single_modality:
                ax[0].imshow(img_3D[idx], cmap='gray')
                ax[1].imshow(img_3D[idx], cmap='gray')
                ax[2].imshow(img_3D[idx], cmap='gray')
            else:
                ax[0].imshow(img_3D[idx].astype(int))
                ax[1].imshow(img_3D[idx].astype(int))
                ax[2].imshow(img_3D[idx].astype(int))
                
            ax[0].set_title("Image")
            ax[1].set_title("Ground Truth")
            ax[2].set_title(f"Segmentation")
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
            for label_id, box_256 in box_dict.items():
                print(label_id)
                if label_id == 1:
                    color = np.array([1,0,0]) 
                else:
                    color = np.array([0,0,1]) 
                #np.random.rand(3)
                box_viz = resize_box(box_256, (newh, neww), (H, W))
                show_mask(arr_gts[idx], ax[1], mask_color=color)
                show_box(box_viz, ax[1], edgecolor=color)
                show_mask(seg_3D[idx], ax[2], mask_color=color)
                show_box(box_viz, ax[2], edgecolor=color)
            plt.tight_layout()
            plt.savefig(join(png_save_dir, patient_id + '.png'), dpi=300)
            plt.close()

if __name__ == '__main__':
    num_workers = num_workers
    mp.set_start_method('spawn')
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=len(gt_path_files)) as pbar:
            for i, _ in tqdm(enumerate(pool.imap_unordered(MedSAM_infer_nii, gt_path_files))):
                pbar.update()
