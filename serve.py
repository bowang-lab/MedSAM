# -*- coding: utf-8 -*-
import sys
import time
import json

from tqdm import tqdm
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn
from numpysocket import NumpySocket

import numpy as np
from skimage import transform, io
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from segment_anything import sam_model_registry

# debug
import matplotlib.pyplot as plt

# 0. freeze seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, height, width):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


# settings and app states
SAM_MODEL_TYPE = "vit_b"
MedSAM_CKPT_PATH = "work_dir/MedSAM/medsam_vit_b.pth"
MEDSAM_IMG_INPUT_SIZE = 1024
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

H, W = None, None
image = None
embeddings = []

# load MedSAM model
print("Loading MedSAM model, a sec")
tic = time.perf_counter()
medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH, device=device).to(device)
medsam_model.eval()
print(f"MedSam loaded, took {time.perf_counter() - tic}")

app = FastAPI()

# calc embedding
# receive number of slices, for each slice, receive the slice then calc embedding


def get_image(wmin: int, wmax: int):
    global image
    global H
    global W
    with NumpySocket() as s:
        s.bind(("", 5556))
        s.listen()
        conn, addr = s.accept()

        with conn:
            arr = conn.recv()

        # windowlization
        arr = np.clip(arr, wmin, wmax)
        arr = (arr - wmin) / (wmax - wmin) * 255

        # arr = np.rot90(arr, k=3, axes=(1, 2))
        arr = np.transpose(arr, (0, 2, 1))

        # print(arr.shape)
        # TODO: add restrictions on image dimension
        # assert (
        #     len(arr.shape) == 2 or arr.shape[-1] == 3
        # ), f"Accept either 1 channel gray image or 3 channel rgb. Got image shape {arr.shape} "
        image = arr
        H, W = arr.shape[1:]  # TODO: make sure h, w not filpped

    for slice_idx in tqdm(range(image.shape[0])):
    # for slice_idx in tqdm(range(4)):
        slc = image[slice_idx]

        # plt.imshow(slc)
        # plt.savefig("out.png")

        print(slc.min(), slc.max())
        if len(slc.shape) == 2:
            img_3c = np.repeat(slc[:, :, None], 3, axis=-1)
        else:
            img_3c = slc

        img_1024 = transform.resize(
            img_3c,
            (1024, 1024),
            order=3,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

        embeddings.append(embedding)
    # print(len(embeddings))


def get_bbox1024(mask_1024, bbox_shift=3):
    y_indices, x_indices = np.where(mask_1024 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask_1024.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)
    bboxes1024 = np.array([x_min, y_min, x_max, y_max])
    return bboxes1024

class ImageParams(BaseModel):
    wmin: int
    wmax: int


@app.post("/setImage")
def set_image(params: ImageParams, background_tasks: BackgroundTasks):
    global image
    global embeddings
    image = None
    embeddings = []
    print(params.wmin, params.wmax)
    background_tasks.add_task(get_image, wmin=params.wmin, wmax=params.wmax)
    return


class InferenceParams(BaseModel):
    slice_idx: int
    bbox: list[int]  # (xmin, ymin, xmax, ymax), origional size
    zrange: list[int] # (zmin, zmax), inference will be performed in this slice range, including zmin and zmax


@app.post("/infer")
def infer(params: InferenceParams):
    print(params.slice_idx, params.bbox, params.zrange)
    zmin, zmax = params.zrange
    zmax += 1
    zmin -= 1
    bbox_1024_prev = np.array([params.bbox]) / np.array([W, H, W, H]) * 1024
    res = {}
    
    def inf(bbox_1024_prev):
        mask = medsam_inference(medsam_model, embeddings[csidx], bbox_1024_prev, H, W)
        mask_t = transform.resize(mask, (1024, 1024), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
        return mask.tolist(), [get_bbox1024(mask_t)]

    for csidx in range(params.slice_idx, zmax):
        res[csidx], bbox_1024_prev = inf(bbox_1024_prev)
        if csidx == params.slice_idx:
            bbox_1024_center_inf = bbox_1024_prev.copy()

    bbox_1024_prev = bbox_1024_center_inf
    for csidx in range(params.slice_idx-1, zmin, -1):
        res[csidx], bbox_1024_prev = inf(bbox_1024_prev)

    
    print(len(res))
    # plt.imshow(res)
    # plt.savefig("out.png")

    return json.dumps(res)


if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=5555, reload=False, workers=1)
