import os
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
import torch.nn.functional as F
import torch.nn as nn
import argparse

import gradio as gr
import torch
import argparse
import sys
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_parent_directory = os.path.dirname(os.path.dirname(current_path))
if parent_parent_directory not in sys.path:
    sys.path.append(parent_parent_directory)
from demo.tasks import *


parser = argparse.ArgumentParser()


parser.add_argument(
    '-medsam_lite_checkpoint_path',
    type=str,
    default= './checkpoints/medsam_lite_scribble.pth', 
    help='path to the checkpoint of MedSAM-Lite',
    required=False
)
parser.add_argument(
    '-device',
    type=str,
    default="cuda:0",
    help='device to run the inference',
)


args = parser.parse_args()
medsam_lite_checkpoint_path = args.medsam_lite_checkpoint_path
device = torch.device(args.device)
image_size = 256

class MedSAM_Lite(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        
    def forward(self, image, masks):
        image_embedding = self.image_encoder(image) # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None, #boxes,
            masks=masks,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks, iou_predictions

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[:, :, :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks

# %%
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
medsam_lite_model.load_state_dict(medsam_lite_checkpoint, strict=True)
medsam_lite_model.to(device)
medsam_lite_model.eval()

# The following are adapted from 
# SEEM -- Segment Everything Everywhere All At Once
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)

@torch.no_grad()
def inference(image, *args, **kwargs):
    return interactive_infer_image(medsam_lite_model, image, *args, **kwargs)

class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)


title = "LiteMedSAM: stoke demo"
description = "MedSAM scribble demo"


inputs = [ImageMask(label="[Stroke] Draw on Image",type="pil")] 
gr.Interface(
    fn=inference,
    inputs=inputs,
    outputs=[
        gr.outputs.Image(
        type="pil",
        label="Segmentation Results"),
    ],
    title=title,
    description=description,
    allow_flagging='never',
    cache_examples=False,
).launch(share=True)
