# %%
import os
import glob
import random
import monai
from os import makedirs
from os.path import join
from tqdm import tqdm
from copy import deepcopy
from time import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

import cv2
import argparse
from matplotlib import pyplot as plt
import segmentation_models_pytorch as smp
import monai
# %%
parser = argparse.ArgumentParser(description='Train DeepLabV3Plus')
parser.add_argument('-i', '--data_root', type=str, default='', help='Two subfolders for training data: imgs and gts')
parser.add_argument('-o', '--ckpt_dir', type=str, default='', help='Checkpoint save directory')
parser.add_argument('-b', '--batch_size', type=int, default=600, help='batch size')
parser.add_argument('--num_workers', type=int, default=30, help='number of workers for dataloader')
parser.add_argument("--max_epochs", type=int, default=500, help="number of epochs")
parser.add_argument('--compile', action='store_true', help='compile model')
args = parser.parse_args()

model_compile = args.compile
num_epochs = args.max_epochs
resume = None
device = torch.device("cuda:0")
data_root = args.data_root
ckpt_dir = args.ckpt_dir
batch_size = args.batch_size
num_workers = args.num_workers
makedirs(ckpt_dir, exist_ok=True)

# %%
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.45])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))


# %%
class NpyDataset(Dataset): 
    def __init__(self, data_root, image_size=224, bbox_shift=5, data_aug=False):
        self.data_root = data_root
        self.gt_path = join(data_root, 'gts')
        self.img_path = join(data_root, 'imgs')
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, '**/*.npy'), recursive=True))
        self.gt_path_files = [file for file in self.gt_path_files if os.path.isfile(join(self.img_path, os.path.basename(file)))]
        self.image_size = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug
        print(f'number of images: {len(self.gt_path_files)}')
    
    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        assert img_name == os.path.basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]
        img_3c = np.load(join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
        resize_img_cv2 = cv2.resize(
            img_3c,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA
        )
        resize_img_cv2_01 = (resize_img_cv2 - resize_img_cv2.min()) / np.clip(resize_img_cv2.max() - resize_img_cv2.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3)
        # convert the shape to (3, H, W)
        resize_img = np.transpose(resize_img_cv2_01, (2, 0, 1))
        assert np.max(resize_img)<=1.0 and np.min(resize_img)>=0.0, 'image should be normalized to [0, 1]'
        gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True) # multiple labels [0, 1,4,5...]
        if gt.shape[0] != self.image_size or gt.shape[1] != self.image_size:
            gt_resize = cv2.resize(
                gt, (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST
            )
            gt_resize = np.uint8(gt_resize)
        else:
            gt_resize = gt
        label_ids = np.unique(gt_resize)[1:]
        label_id = random.choice(label_ids.tolist())
        gt2D = np.uint8(gt_resize == label_id) # only one label
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0, 'ground truth should be 0, 1, got: ' + str(np.unique(gt2D))
        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                resize_img = np.ascontiguousarray(np.flip(resize_img, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                # print('DA with flip left right')
            if random.random() > 0.5:
                resize_img = np.ascontiguousarray(np.flip(resize_img, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                # print('DA with flip up down')
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        ## Append bbox prompt channel
        resize_img_bbox = np.concatenate([resize_img, np.zeros((1, self.image_size, self.image_size))], axis=0)
        resize_img_bbox[-1, y_min:y_max, x_min:x_max] = 1.0
        # print(img_name, resize_img_bbox.shape, gt2D.shape) 
        return torch.tensor(resize_img_bbox).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(bboxes).float(), img_name


# %%
model = smp.DeepLabV3Plus(
    encoder_name="resnet50",        # encoder model type
    encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
    in_channels=4,                  # Additional channel for bounding box prompt
    classes=1,                      # model output channels (number of classes in your dataset)
    activation=None                 # Output logits
)
model.to(device)
if model_compile:
    print("Compiling model...")
    model = torch.compile(model)

# %%
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=4e-5,
)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.94)
# %%
train_dataset = NpyDataset(data_root=data_root, data_aug=False, bbox_shift=5, image_size=224)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# %%
# loss function
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean', to_onehot_y=False)
# %%
# training
if resume is not None:
    checkpoint = torch.load(resume)
    model._orig_mod.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_loss = checkpoint['best_loss']
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch} with best loss {best_loss:.4f}")
else:
    best_loss = 1e10
    best_epoch = 0
    start_epoch = 0

for epoch in range(start_epoch, num_epochs):
    epoch_loss = [1e10 for _ in range(len(train_dataloader))]
    pbar = tqdm(train_dataloader)
    for step, (image, gt2D, boxes, img_names) in enumerate(pbar):
        optimizer.zero_grad()
        boxes_np = boxes.detach().cpu().numpy()
        image, gt2D = image.to(device), gt2D.to(device)
        pred = model(image)
        loss = seg_loss(pred, gt2D)
        epoch_loss[step] = loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

    epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
    model_weights = model._orig_mod.state_dict()
    checkpoint = {
        "model": model_weights,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "loss": epoch_loss_reduced,
        "best_loss": best_loss,
    }
    torch.save(checkpoint, join(ckpt_dir, "deeplabv3plus_latest.pt"))
    if epoch_loss_reduced < best_loss:
        print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
        best_loss = epoch_loss_reduced
        torch.save(checkpoint, join(ckpt_dir, "deeplabv3plus_best.pt"))

    epoch_loss_reduced = 1e10
    lr_scheduler.step()
