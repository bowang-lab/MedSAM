# %%
import os
import glob
import random
import argparse
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir
from tqdm import tqdm
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
from matplotlib import pyplot as plt
import cv2
import torch.nn.functional as F

from tiny_vit_sam import TinyViT

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--tr_npy_path', type=str,
                    default=None,
                    help='path to training npy files; two subfolders: gts and imgs')
parser.add_argument('-task_name', type=str, default='MedSAM-Lite')
parser.add_argument('-medsam_checkpoint', type=str,
                    default=None,
                    help='path to MedSAM encoder checkpoint')
parser.add_argument('-tinyvit_checkpoint', type=str,
                    default=None,
                    help='path to TinyViT encoder checkpoint (not required)')
parser.add_argument('-work_dir', type=str, default='./work_dir',
                    help='path to save the model checkpoints and logs')
parser.add_argument('--data_aug', type=bool, default=True, 
                    help='use data augmentation during training')
# train
parser.add_argument('-num_epochs', type=int, default=1000)
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-num_workers', type=int, default=8)
# Optimizer parameters
parser.add_argument('-weight_decay', type=float, default=0.01,
                    help='weight decay (default: 0.01)')
parser.add_argument('-lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (absolute lr)')
parser.add_argument('-resume', type = str,
                    default = None,
                    help="Resuming training from saved checkpoint (only required when resuming training)")

args = parser.parse_args()

# %%
data_root = args.tr_npy_path
checkpoint = args.medsam_checkpoint
medsam_tiny_encoder_ckpt_init = args.tinyvit_checkpoint
ckpt_dir = args.work_dir
log_dir = args.work_dir
max_epochs = args.num_epochs
batch_size = args.batch_size
num_workers = args.batch_size
device = "cuda:0"
lr = args.lr
makedirs(ckpt_dir, exist_ok=True)
makedirs(log_dir, exist_ok=True)

# %%
class Logger(object):
    def __init__(self):
        self.logging = {
            'train_losses': [],
            'lrs': [],
            'epoch_start_timestamps': [],
            'epoch_end_timestamps': []
        }

    def log(self, key, value):
        self.logging[key].append(value)
    
    def plot_progress_png(self, output_folder):
        fig, ax_all = plt.subplots(2, 1, figsize=(10, 8))
        x_values = [i for i in range(1, len(self.logging['train_losses']) + 1)]
        ax_all[0].plot(
            x_values,
            self.logging['train_losses'],
            #color='b',
            ls='-',
            label="loss_tr",
        )
        ax_all[0].set_ylabel("loss", fontsize=10)
        ax_all[0].legend(loc=(0, 1))

        ax_all[1].plot(x_values, [i - j for i, j in zip(self.logging['epoch_end_timestamps'],
                                                 self.logging['epoch_start_timestamps'])],
                                                 #color='b',
                ls='-',
                label="epoch duration",
                )
        ax_all[1].set_xlabel("epoch", fontsize=10)
        ax_all[1].set_ylabel("time [s]", fontsize=10)
        ax_all[1].legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.logging

    def load_checkpoint(self, checkpoint: dict):
        self.logging = checkpoint

# %%
class NpyDataset(Dataset): 
    def __init__(self, data_root, data_aug=True):
        self.data_root = data_root
        self.gt_path = join(data_root, 'gts')
        self.img_path = join(data_root, 'imgs')
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, '**/*.npy'), recursive=True))
        self.gt_path_files = [file for file in self.gt_path_files if os.path.isfile(join(self.img_path, os.path.basename(file)))]
        self.data_aug = data_aug
        print(f'number of images: {len(self.gt_path_files)}')
    
    def __len__(self):
        return len(self.gt_path_files)
    
    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_3c = np.load(join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
        resize_img_skimg = cv2.resize(
            img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC
        )
        resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(resize_img_skimg_01, (2, 0, 1))
        assert np.max(img_1024)<=1.0 and np.min(img_1024)>=0.0, 'image should be normalized to [0, 1]'

        img_256 = cv2.resize(
            img_3c, (256, 256), interpolation=cv2.INTER_AREA
        )
        img_256 = (img_256 - img_256.min()) / np.clip(
            img_256.max() - img_256.min(), a_min=1e-8, a_max=None
        )
        img_256 = np.transpose(img_256, (2, 0, 1))
        assert np.max(img_256)<=1.0 and np.min(img_256)>=0.0, 'image should be normalized to [0, 1]'

        if self.data_aug:
            if random.random() > 0.5:
                img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-1))
                img_256 = np.ascontiguousarray(np.flip(img_256, axis=-1))
                # print('DA with flip left right')
            if random.random() > 0.5:
                img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-2))
                img_256 = np.ascontiguousarray(np.flip(img_256, axis=-2))
                # print('DA with flip up down')

        return torch.tensor(img_1024).float(), torch.tensor(img_256).float(), img_name

medsam_model = sam_model_registry["vit_b"](checkpoint=checkpoint)
teacher_model = deepcopy(medsam_model.image_encoder)
teacher_model = teacher_model.to(device)
teacher_model.eval()
del medsam_model
torch.cuda.empty_cache()
for param in teacher_model.parameters():
    param.requires_grad = False ## freeze teacher model
# %%
student_model = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[
        64,
        128,
        160,
        320 
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
) ## TinyViT setup

# %%
pretrained_student_ckpt = torch.load(
    medsam_tiny_encoder_ckpt_init,
    map_location="cpu"
)
student_model.load_state_dict(pretrained_student_ckpt)
student_model = student_model.to(device)
# %%
print(f"MedSAM encoder size: {sum(p.numel() for p in teacher_model.parameters())}")
print(f"TinyViT encoder size: {sum(p.numel() for p in student_model.parameters())}")

logger = Logger()
optimizer = optim.AdamW(
    student_model.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    eps=1e-06,
    weight_decay=0.001
)
# %%
if args.resume is not None:
    checkpoint = torch.load(args.resume)
    student_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    best_loss = checkpoint["best_loss"]
    start_epoch = checkpoint["epoch"] + 1
    if "logger" in checkpoint:
        logger.load_checkpoint(checkpoint["logger"])
else:
    best_loss = 1e10
    start_epoch = 0
# %%
train_dataset = NpyDataset(data_root=data_root, data_aug=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
# %%
for epoch in range(start_epoch, max_epochs):
    epoch_loss = 0
    logger.log('lrs', optimizer.param_groups[0]['lr'])
    logger.log('epoch_start_timestamps', time())
    pbar = tqdm(train_loader)
    for step, (teacher_input, student_input, image_name) in enumerate(pbar):
        teacher_input = teacher_input.to(device)
        student_input = student_input.to(device)
        with torch.no_grad():
            teacher_output = teacher_model(teacher_input)
        student_output = student_model(student_input)
        target = teacher_output.detach().clone()
        loss = F.mse_loss(student_output, target)
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch}, loss {loss.item():.4f}")
    logger.log('epoch_end_timestamps', time())
    epoch_loss /= (step + 1)
    logger.log('train_losses', epoch_loss)
    checkpoint = {
        "epoch": epoch,
        "model": student_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
        "logger": logger.get_checkpoint()
    }
    torch.save(
        checkpoint,
        join(ckpt_dir, f"medsam_lite_encoder_latest.pth")
    )
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(
            checkpoint,
            join(ckpt_dir, f"medsam_lite_encoder_best.pth")
        )
    logger.plot_progress_png(log_dir)