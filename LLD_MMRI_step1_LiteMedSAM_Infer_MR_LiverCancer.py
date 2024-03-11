"""
Using MedSAM to generate tumor masks for the LLD_MMRI dataset based on bounding box prompts


Step 0. Preprocess: run LLD_MMRI_step0_data_preprocess.py
Step 1. Tumor segmentation with LiteMedSAM


Jun Ma
"""
from os import listdir, makedirs
from os.path import join, isfile, basename
from glob import glob
from tqdm import tqdm
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
import json
import torch.nn.functional as F

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
from matplotlib import pyplot as plt
import cv2
import torch.multiprocessing as mp
from skimage import morphology


#%% set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
np.random.seed(2023)

#%% set paths
img_path = "LLD-MMRI/images-nii"
save_dir = "LLD-MMRI/images-nii-tumor-seg-post"
json_path = 'LLD-MMRI/labels/Annotation.json'
medsam_lite_checkpoint_path = 'work_dir/lite_medsam.pth'
device = 'cuda:0'
makedirs(save_dir, exist_ok=True)

with open(json_path) as f:
    json_data = json.load(f)
case_names = list(json_data['Annotation_info'].keys())

bbox_shift = 5
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
medsam_lite_model.load_state_dict(medsam_lite_checkpoint)
medsam_lite_model.to(device)
medsam_lite_model.eval()




def infer_MR_liver_tumor(case_name):
    case_phases = json_data['Annotation_info'][case_name]
    for case_phase in case_phases:
        phase = case_phase['phase']
        annotation = case_phase['annotation']
        num_targets = annotation['num_targets']
        catetory = annotation['lesion']['0']['category']
        # load MR data
        nii_name = case_name + '_' + str(catetory) + '_' + phase.replace(" ","") + '_0000.nii.gz'
        if num_targets > 1:
            print(f"Case {case_name} has {num_targets} targets in phase {phase}.")
        img_sitk = sitk.ReadImage(join(img_path, nii_name))
        image_data = sitk.GetArrayFromImage(img_sitk) # (D, H, W)
        
        # preprocess MR data
        lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
        image_data_pre[image_data==0] = 0
        image_data_pre = np.uint8(image_data_pre)

        bbox_2d = annotation['lesion']['0']['bbox']['2D_box'] # list; each element is a dict: 'slice_idx', 'x_min', 'y_min', 'x_max', 'y_max', 'area'
        seg_data_3d = np.zeros_like(image_data, dtype=np.uint8)
        for bbox in bbox_2d:
            slice_idx = int(bbox['slice_idx'])
            img_2d = image_data_pre[slice_idx, :, :]
            H, W = img_2d.shape
            img_3c = np.repeat(img_2d[:,:, None], 3, axis=-1)
            img_256 = resize_longest_side(img_3c, 256)
            newh, neww = img_256.shape[:2]
            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )
            img_256_padded = pad_image(img_256, 256)
            img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(img_256_tensor)

            xmin, ymin, xmax, ymax = int(bbox['x_min']), int(bbox['y_min']), int(bbox['x_max']), int(bbox['y_max'])
            xmin = xmin / img_3c.shape[0] * 256
            xmax = xmax / img_3c.shape[0] * 256
            ymin = ymin / img_3c.shape[1] * 256
            ymax = ymax / img_3c.shape[1] * 256
            bboxes256 = np.array([xmin, ymin, xmax, ymax])

            seg_mask = medsam_inference(medsam_lite_model, image_embedding, bboxes256, (newh, neww), (H, W))
            # seg_mask = morphology.remove_small_holes(seg_mask>0)
            seg_data_3d[slice_idx, :, :] = np.uint8(morphology.remove_small_holes(seg_mask, 150))


        # save seg_data_3d as nii by sitk
        seg_data_3d_sitk = sitk.GetImageFromArray(seg_data_3d)
        seg_data_3d_sitk.SetSpacing(img_sitk.GetSpacing())
        seg_data_3d_sitk.SetOrigin(img_sitk.GetOrigin())
        seg_data_3d_sitk.SetDirection(img_sitk.GetDirection())
        sitk.WriteImage(seg_data_3d_sitk, join(save_dir, nii_name.replace('_0000.nii.gz', '.nii.gz')))

if __name__ == '__main__':
    num_workers = 4
    mp.set_start_method('spawn')
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=len(case_names)) as pbar:
            for i, _ in tqdm(enumerate(pool.imap_unordered(infer_MR_liver_tumor, case_names))):
                pbar.update()
