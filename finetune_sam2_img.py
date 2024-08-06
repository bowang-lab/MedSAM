# -*- coding: utf-8 -*-
"""
finetune sam2 model on medical image data
only finetune the image encoder and mask decoder
freeze the prompt encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
from sam2.build_sam import build_sam2
from typing import List, Optional, Tuple
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms
import cv2

# set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(2024)

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "*.npy"), recursive=True)
        )
        self.gt_path_files = self.gt_path_files
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        self._transform = SAM2Transforms(resolution=1024, mask_threshold=0)
        print(f"number of images: {len(self.gt_path_files)}")


    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = self._transform(img.copy())
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        assert gt.shape == (256, 256), "ground truth should be 256x256"
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))

        bboxes = np.array([x_min, y_min, x_max, y_max])*4 ## scale bbox from 256 to 1024

        return (
            img_1024, ## [3, 1024, 1024]
            torch.tensor(gt2D[None, :, :]).long(), ## [1, 256, 256]
            torch.tensor(bboxes).float(), 
            img_name,
        )


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default=None,
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM2-Tiny-Flare22")
parser.add_argument(
    "-model_cfg", type=str, default="sam2_hiera_t.yaml", help="model config file"
)
parser.add_argument("-pretrain_model_path",
                    type=str,
                    default=None,
)
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=1000)
parser.add_argument("-batch_size", type=int, default=16)
parser.add_argument("-bbox_shift", type=int, default=5)
parser.add_argument("-num_workers", type=int, default=0)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float,
    default=6e-5,
    metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-resume", type=str,
    default=None,
    help="Resuming training from checkpoint"
)
parser.add_argument("-device", type=str, default="cuda:0")
args, unknown = parser.parse_known_args()

# sanity test of dataset class
tr_dataset = NpyDataset(args.tr_npy_path, bbox_shift=args.bbox_shift)
tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
images, gts, bboxes, names_temp = next(iter(tr_dataloader))
idx = random.randint(0, images.shape[0]-1)
inv_sam2_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Normalize(mean=[0, 0, 0], std=[1 / i for i in tr_dataset._transform.std]),
        torchvision.transforms.Normalize(mean=[-1 * i for i in tr_dataset._transform.mean], std=[1, 1, 1]),
    ]
)
_, axs = plt.subplots(1, 2, figsize=(25, 25))
axs[0].imshow(
    inv_sam2_transform(images[idx].clone()).permute(1, 2, 0).numpy()
)
show_mask(
    cv2.resize(
        gts[idx].squeeze(0).numpy(),
        (1024, 1024),
        interpolation=cv2.INTER_NEAREST
    ),
    axs[0]
)
show_box(bboxes[idx].numpy(), axs[0])
axs[0].axis("off")
axs[0].set_title(names_temp[idx])
idx = random.randint(0, images.shape[0]-1)
axs[1].imshow(
    inv_sam2_transform(images[idx].clone()).permute(1, 2, 0).numpy()
)
show_mask(
    cv2.resize(
        gts[idx].clone().squeeze(0).numpy(),
        (1024, 1024),
        interpolation=cv2.INTER_NEAREST
    ),
    axs[1]
)
show_box(bboxes[idx].numpy(), axs[1])
axs[1].axis("off")
axs[1].set_title(names_temp[idx])
plt.subplots_adjust(wspace=0.01, hspace=0)
plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
plt.close()

run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)

class MedSAM2(nn.Module):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.sam2_model = model
        # freeze prompt encoder
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
        

    def forward(self, image, box):
        """
        image: (B, 3, 1024, 1024)
        box: (B, 2, 2)
        """
        _features = self._image_encoder(image)
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_coords = box_torch.reshape(-1, 2, 2) # (B, 4) to (B, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=image.device)
                box_labels = box_labels.repeat(box_torch.size(0), 1)
            concat_points = (box_coords, box_labels)

            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=None,
            )
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed, # (B, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return low_res_masks_logits
    
    def _image_encoder(self, input_image):
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features


# %%
def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    model_cfg = args.model_cfg
    sam2_checkpoint = args.pretrain_model_path
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=args.device, apply_postprocessing=True)
    medsam_model = MedSAM2(model=sam2_model)
    medsam_model.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    ) 
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  

    img_mask_encdec_params = list(medsam_model.sam2_model.image_encoder.parameters()) + list(
        medsam_model.sam2_model.sam_mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )  
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    num_epochs = args.num_epochs
    losses = []
    best_loss = 1e10
    train_dataset = NpyDataset(args.tr_npy_path, bbox_shift=args.bbox_shift)

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            
            medsam_pred = medsam_model(image, boxes_np)
            loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        epoch_loss /= step
        losses.append(epoch_loss)
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        ## save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

        # %% plot loss
        plt.plot(losses)
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "train_loss.png"))
        plt.close()


if __name__ == "__main__":
    main()