# -*- coding: utf-8 -*-
import torch

# %% convert medsam model checkpoint to sam checkpoint format for convenient inference
sam_ckpt_path = ""
medsam_ckpt_path = ""
save_path = ""
multi_gpu_ckpt = True  # set as True if the model is trained with multi-gpu

sam_ckpt = torch.load(sam_ckpt_path)
medsam_ckpt = torch.load(medsam_ckpt_path)
sam_keys = sam_ckpt.keys()
for key in sam_keys:
    if not multi_gpu_ckpt:
        sam_ckpt[key] = medsam_ckpt["model"][key]
    else:
        sam_ckpt[key] = medsam_ckpt["model"]["module." + key]

torch.save(sam_ckpt, save_path)
