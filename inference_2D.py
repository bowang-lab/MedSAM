from glob import glob
from os import listdir, makedirs
from os.path import join, isfile, basename
from tqdm import tqdm, trange
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

from tiny_vit_sam import TinyViT
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from matplotlib import pyplot as plt
import cv2
import torch.multiprocessing as mp

import argparse

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument(
    '-data_root',
    type=str,
    default=None,
    help='root directory of the data',
    required=True
)
parser.add_argument(
    '-pred_save_dir',
    type=str,
    default=None,
    help='directory to save the prediction',
    required=True
)
parser.add_argument(
    '-medsam_lite_checkpoint_path',
    type=str,
    default="lite_medsam.pth",
    help='path to the checkpoint of MedSAM-Lite',
    required=True
)
parser.add_argument(
    '-device',
    type=str,
    default="cpu",
    help='device to run the inference',
)
parser.add_argument(
    '-num_workers',
    type=int,
    default=4,
    help='number of workers for inference with multiprocessing',
)
parser.add_argument(
    '--save_overlay',
    action='store_true',
    help='whether to save the overlay image'
)
parser.add_argument(
    '-png_save_dir',
    type=str,
    default='./overlay',
    help='directory to save the overlay image'
)
parser.add_argument(
    '--overwrite',
    action='store_true',
    help='whether to overwrite the existing prediction'
)

args = parser.parse_args()

data_root = args.data_root
pred_save_dir = args.pred_save_dir
save_overlay = args.save_overlay
num_workers = args.num_workers
overwrite = args.overwrite
medsam_lite_checkpoint_path = args.medsam_lite_checkpoint_path
if save_overlay:
    assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
    png_save_dir = args.png_save_dir
    makedirs(png_save_dir, exist_ok=True)
makedirs(pred_save_dir, exist_ok=True)
bbox_shift = 5
device = torch.device(args.device)
gt_path_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
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

# %%
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


def revert_box(box, new_size, original_size):
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
    box_torch = torch.as_tensor(box_256, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :] # (B, 1, 4)
    
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
# %%
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

# %%
medsam_lite_model = MedSAM_Lite(
    image_encoder = medsam_lite_image_encoder,
    mask_decoder = medsam_lite_mask_decoder,
    prompt_encoder = medsam_lite_prompt_encoder
)
medsam_lite_checkpoint = torch.load(medsam_lite_checkpoint_path, map_location='cpu')
medsam_lite_model.load_state_dict(medsam_lite_checkpoint)
medsam_lite_model.to(device)
medsam_lite_model.eval()

# %%
def MedSAM_infer_npz(gt_path_file):
    npz_name = basename(gt_path_file)
    task_folder = gt_path_file.split('/')[-2]
    makedirs(join(pred_save_dir, task_folder), exist_ok=True)
    if (not isfile(join(pred_save_dir, task_folder, npz_name))) or overwrite:
        npz_data = np.load(gt_path_file, 'r', allow_pickle=True) # (H, W, 3)
        img_3c = npz_data['imgs'] # (H, W, 3)
        H, W = img_3c.shape[:2]
        gts = npz_data['gts']
        if gts.shape != (H, W):
            gts = cv2.resize(
                gts.astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

        ## MedSAM Lite preprocessing
        img_256 = resize_longest_side(img_3c, 256)
        newh, neww = img_256.shape[:2]
        img_256_norm = (img_256 - img_256.min()) / np.clip(
            img_256.max() - img_256.min(), a_min=1e-8, a_max=None
        )
        img_256_padded = pad_image(img_256_norm, 256)
        img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = medsam_lite_model.image_encoder(img_256_tensor)

        label_ids = np.unique(gts)[1:]
        box_list = []
        for label_id in label_ids:
            gt2D = np.uint8(gts == label_id)
            if gt2D.shape != (newh, neww):
                gt2D_resize = cv2.resize(
                    gt2D.astype(np.uint8), (neww, newh),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.uint8)
            else:
                gt2D_resize = gt2D.astype(np.uint8)
            gt2D_padded = pad_image(gt2D_resize, 256)
            if np.sum(gt2D_padded) > 0:
                box = get_bbox(gt2D_padded, bbox_shift) # (4,)
                box = box[None, ...] # (1, 4)
                sam_mask = medsam_inference(medsam_lite_model, image_embedding, box, (newh, neww), (H, W))
                segs[sam_mask>0] = label_id
                box_list.append(box.squeeze())
        
        label_ids = np.unique(gts)[1:]

        np.savez_compressed(
            join(pred_save_dir, task_folder, npz_name),
            segs=segs, gts=gts
        )

        # visualize image, mask and bounding box
        if save_overlay:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(img_3c)
            ax[1].imshow(img_3c)
            ax[2].imshow(img_3c)
            ax[0].set_title("Image")
            ax[1].set_title("Ground Truth")
            ax[2].set_title(f"Segmentation")
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')

            for i, label_id in enumerate(label_ids):
                color = np.random.rand(3)
                box_viz = revert_box(box_list[i], (newh, neww), (H, W))
                show_box(box_viz, ax[1], edgecolor=color)
                show_mask((gts == label_id).astype(np.uint8), ax[1], mask_color=color)
                show_box(box_viz, ax[2], edgecolor=color)
                show_mask((segs == label_id).astype(np.uint8), ax[2], mask_color=color)

            plt.tight_layout()
            plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
            plt.close()

if __name__ == '__main__':
    num_workers = num_workers
    mp.set_start_method('spawn')
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=len(gt_path_files)) as pbar:
            for i, _ in tqdm(enumerate(pool.imap_unordered(MedSAM_infer_npz, gt_path_files))):
                pbar.update()

