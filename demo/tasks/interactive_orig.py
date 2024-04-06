# --------------------------------------------------------
# Adapted from 
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.visualizer import Visualizer
import matplotlib.pyplot as plt


import cv2
from PIL import Image

def show_mask(mask, color=None, alpha=0.45):
    if color is None:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([*color, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

def resize_longest_side(image,img_side_len=256):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    long_side_length = img_side_len
    oldh, oldw = image.shape[0], image.shape[1]
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image,img_side_len=256):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = img_side_len - h
    padw = img_side_len - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))
    return image_padded


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, scribble_256, new_size, original_size):
    scribble_torch = torch.as_tensor(scribble_256, dtype=torch.float, device=img_embed.device)
    # if len(box_torch.shape) == 2:
    #     box_torch = box_torch[:, None, :] # (B, 1, 4)
    
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points = None,
        boxes = None,
        masks = scribble_torch,
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



t = []
t.append(transforms.Resize((256,256), interpolation=Image.BICUBIC))
transform = transforms.Compose(t)

def interactive_infer_image(medsam_lite_model, image, img_side_len=256):

    image_ori = resize_longest_side(np.array(image['image']),img_side_len)
    image_ori = (image_ori - image_ori.min()) / np.clip(
            image_ori.max() - image_ori.min(), a_min=1e-8, a_max=None
            ).astype(np.float32)
    image_ori = pad_image(image_ori,img_side_len)
    mask_ori = image['mask']
    width = image_ori.shape[0]
    height = image_ori.shape[1]
    image_ori = np.asarray(image_ori)
    visual = Visualizer(image_ori*255, metadata=None)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()
    images = images[None,...]
    with torch.no_grad():
        image_embedding = medsam_lite_model.image_encoder(images)

    mask_ori = np.asarray(mask_ori)[:,:,0:1].copy()
    mask_ori = resize_longest_side(mask_ori,img_side_len)
    mask_ori = mask_ori[:, :, None]
    mask_ori = pad_image(mask_ori,img_side_len)
    mask_ori = torch.from_numpy(mask_ori).permute(2,0,1)[None,]
    mask_ori = mask_ori > 0
    mask_ori = mask_ori * 1

    sam_mask = medsam_inference(medsam_lite_model, image_embedding, mask_ori, (256, 256), (256, 256))
    sam_mask = sam_mask[None, ...]


    for idx, mask in enumerate(sam_mask): # 1,256,256
        demo = visual.draw_binary_mask(mask)
    res = demo.get_image()
    torch.cuda.empty_cache()
    return Image.fromarray(res)