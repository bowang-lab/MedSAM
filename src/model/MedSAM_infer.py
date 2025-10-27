# MedSAM_infer.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io as skio, transform as sktf

# ---- MedSAM ----
from segment_anything import sam_model_registry

@dataclass
class MedSAMModel:
    model: torch.nn.Module
    device: torch.device

def pick_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        if device_arg.lower() == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if device_arg != "cpu" and torch.cuda.is_available():
            return torch.device(device_arg if device_arg != "cuda" else "cuda:0")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_medsam(checkpoint: Path, device: torch.device, variant: str = "vit_b") -> MedSAMModel:
    model = sam_model_registry[variant](checkpoint=str(checkpoint))
    model = model.to(device)
    model.eval()
    return MedSAMModel(model=model, device=device)

@torch.no_grad()
def embed_image_1024(msam: MedSAMModel, img_3c: np.ndarray) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
    H, W, _ = img_3c.shape
    img_1024 = sktf.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
    img_1024_t = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(msam.device)
    emb = msam.model.image_encoder(img_1024_t)  # (1,256,64,64)
    return emb, H, W, img_1024_t

@torch.no_grad()
def medsam_infer(msam: MedSAMModel,
                 img_embed: torch.Tensor,
                 box_xyxy: Tuple[int, int, int, int],
                 H: int, W: int) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    box_np = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    box_1024 = box_np / np.array([W, H, W, H], dtype=np.float32) * 1024.0

    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B,1,4)

    sparse_embeddings, dense_embeddings = msam.model.prompt_encoder(
        points=None, boxes=box_torch, masks=None
    )
    low_res_logits, _ = msam.model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=msam.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)  # (1,1,256,256)
    pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False)
    mask = (pred.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8)
    return mask
