import os
import glob
import random
import monai
from os import makedirs
from os.path import join, basename
from tqdm import tqdm
from copy import deepcopy
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from segment_anything import sam_model_registry
from segment_anything.modeling import PromptEncoder
import cv2
from matplotlib import pyplot as plt
from transformers import CLIPTokenizer, CLIPTextModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--tr_npy_path',
    type=str,
    help="Path to the data root directory.",
    required=True
)
parser.add_argument(
    '-medsam_checkpoint',
    type=str,
    help="Path to the MedSAM checkpoint.",
    required=True
)
parser.add_argument(
    '-work_dir',
    type=str,
    default="finetune_text_prompt",
    help="Path to where the checkpoints and logs are saved."
)
parser.add_argument(
    '-max_epochs',
    type=int,
    default=1000,
    help="Maximum number of epochs."
)
parser.add_argument(
    '-batch_size',
    type=int,
    default=16,
    help="Batch size."
)
parser.add_argument(
    '-num_workers',
    type=int,
    default=8,
    help="Number of data loader workers."
)
parser.add_argument(
    '-resume',
    type=str,
    default=None,
    help="Path to the checkpoint to resume from."
)
parser.add_argument(
    '-lr',
    type=float,
    default=0.00005,
    help="learning rate (absolute lr)"
)
parser.add_argument(
    '-weight_decay',
    type=float,
    default=0.01,
    help="Weight decay."
)
parser.add_argument(
    '-seed',
    type=int,
    default=2023,
    help="Random seed for reproducibility."
)
parser.add_argument(
    '--disable_aug',
    action='store_true',
    help="Disable data augmentation."
)

args = parser.parse_args()

data_root = args.tr_npy_path
work_dir = args.work_dir
num_epochs = args.max_epochs
batch_size = args.batch_size
num_workers = args.num_workers
medsam_checkpoint = args.medsam_checkpoint
data_aug = not args.disable_aug
seed = args.seed
device = "cuda:0"
makedirs(work_dir, exist_ok=True)

torch.cuda.empty_cache()
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

## Dataset class
class NpyDataset(Dataset): 
    def __init__(self, data_root, image_size=1024, data_aug=True):
        self.data_root = data_root
        self.gt_path = join(data_root, 'gts')
        self.img_path = join(data_root, 'imgs')
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, '**/*.npy'), recursive=True))
        self.gt_path_files = [file for file in self.gt_path_files if os.path.isfile(join(self.img_path, os.path.basename(file)))]
        self.image_size = image_size
        self.data_aug = data_aug
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")   
        self.label_dict = {
            1: ["Liver", "liver"],
            2: ["Right Kidney", "right kidney", "kidney"],
            3: ["Spleen", "spleen"],
            4: ["Pancreas", "pancreas"],
            5: ["Aorta", "aorta"],
            6: ["Inferior Vena Cava", "IVC", "inferior vena cava", "ivc", "vena cava", "vena", "cava"],
            7: ["Right Adrenal Gland", "RAG", "right adrenal gland", "rag", "adrenal gland", "adrenal"],
            8: ["Left Adrenal Gland", "LAG", "left adrenal gland", "lag", "adrenal gland", "adrenal"],
            9: ["Gallbladder", "gallbladder"],
            10: ["Esophagus", "esophagus"],
            11: ["Stomach", "stomach"],
            12: ["Duodenum", "duodenum"],
            13: ["Left Kidney", "left kidney", "kidney"],
        }
    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = basename(self.gt_path_files[index])
        assert img_name == basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]
        img_1024 = np.load(join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1)) # (3, 256, 256)
        assert np.max(img_1024)<=1.0 and np.min(img_1024)>=0.0, 'image should be normalized to [0, 1]'
        gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True) # multiple labels [0, 1,4,5...], (256,256)
        if gt.shape[0] != 256 or gt.shape[1] != 256:
            ## To match the shape of low_res_masks
            gt_resize = cv2.resize(
                gt,
                (256, 256),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        else:
            gt_resize = gt.astype(np.uint8)
        label_ids = np.unique(gt_resize)[1:]
        label_id = random.choice(label_ids.tolist())
        try:
            gt2D = np.uint8(gt_resize == label_id) # only one label, (256, 256)
        except:
            label_id = np.max(gt)
            gt2D = np.uint8(gt_resize == label_id) # only one label, (256, 256)
        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
            if random.random() > 0.5:
                img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
        gt2D = np.uint8(gt2D > 0)

        ## Ramdonly select a synonum of the label
        caption = random.choice(self.label_dict[label_id])
        text_token = self.tokenize_text(caption)

        return {
            "image": torch.tensor(img_1024).float(),
            "gt2D": torch.tensor(gt2D[None, :,:]).long(),
            "text": [caption],
            "token": text_token,
            "image_name": img_name
        }

    def tokenize_text(self, text):
        """
        Tokenize text using CLIP tokenizer
        """
        return self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt" 
        ).input_ids.squeeze(0)

# Text Prompt Encoder class
class TextPromptEncoder(PromptEncoder):
    def __init__(
        self,
        embed_dim = 256,
        image_embedding_size = (64, 64),
        input_image_size = (1024, 1024),
        mask_in_chans = 1,
        activation = nn.GELU,
        ) -> None:
        super().__init__(embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        text_encoder.requires_grad_(False)
        self.text_encoder = text_encoder
        self.text_encoder_head = nn.Linear(512, embed_dim)

    def forward(
        self, points,
        boxes,
        masks,
        tokens,
    ):
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          tokens (torch.Tensor or none): text tokens to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks, tokens)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if tokens is not None:
            with torch.no_grad():
                encoder_hidden_states = self.text_encoder(tokens)[0]
            text_embeddings = self.text_encoder_head(encoder_hidden_states)
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings
    
    def _get_batch_size(self, points, boxes, masks, tokens):
        """
        Returns the batch size of the inputs.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif tokens is not None:
            return tokens.shape[0]
        else:
            return 1

# MedSAM model class
class MedSAM(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                freeze_image_encoder=True,
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder except for text_encoder_head
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.text_encoder_head.parameters():
            param.requires_grad = True
        
        self.freeze_image_encoder = freeze_image_encoder
        if self.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def forward(self, image, tokens):
        # do not compute gradients for image encoder
        with torch.no_grad():
            image_embedding = self.image_encoder(image) # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            tokens=tokens,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding, # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False,
          ) # (B, 1, 256, 256)

        return low_res_masks

sam_model = sam_model_registry["vit_b"](checkpoint=medsam_checkpoint)
text_prompt_encoder = TextPromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size = (1024, 1024),
    mask_in_chans=1,
    activation=nn.GELU,
)
medsam_prompt_encoder_state_dict = sam_model.prompt_encoder.state_dict()
## Load pretrained weights from MedSAM's prompt encoder except for the text encoder
for keys in text_prompt_encoder.state_dict().keys():
    if keys in medsam_prompt_encoder_state_dict.keys():
        text_prompt_encoder.state_dict()[keys] = deepcopy(medsam_prompt_encoder_state_dict[keys])
    else:
        assert keys.startswith("text_encoder")
print(f"Text Prompt Encoder size: {sum(p.numel() for p in text_prompt_encoder.parameters())}")
medsam_model = MedSAM(
    image_encoder = sam_model.image_encoder,
    mask_decoder = deepcopy(sam_model.mask_decoder),
    prompt_encoder = text_prompt_encoder,
    freeze_image_encoder = True
)
medsam_model = medsam_model.to(device)
medsam_model.train()
print(f"MedSAM size: {sum(p.numel() for p in medsam_model.parameters())}")

optim_params = list(
        medsam_model.prompt_encoder.text_encoder_head.parameters()
    ) + list(
        medsam_model.mask_decoder.parameters()
    )
optimizer = optim.AdamW(
    optim_params,
    lr = args.lr,
    betas = (0.9, 0.999),
    eps = 1e-08,
    weight_decay = args.weight_decay
)
print('Number of parameters to update: ', sum(p.numel() for p in medsam_model.parameters() if p.requires_grad))

seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

train_dataset = NpyDataset(data_root=data_root, data_aug=data_aug)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

resume = args.resume
if resume:
    checkpoint = torch.load(resume)
    medsam_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
    print(f"Loaded checkpoint from epoch {start_epoch}, best loss: {best_loss:.4f}")
else:
    start_epoch = 0
    best_loss = 1e10

losses = []
epoch_time = []
for epoch in range(start_epoch, num_epochs):
    epoch_loss = [1e10 for _ in range(len(train_loader))]
    epoch_start_time = time()
    pbar = tqdm(train_loader)
    for step, batch in enumerate(pbar):
        optimizer.zero_grad()
        image, gt2D = batch["image"].to(device), batch["gt2D"].to(device)
        tokens = batch["token"].to(device)
        medsam_pred = medsam_model(image, tokens)
        loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
        epoch_loss[step] = loss.item()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

    epoch_end_time = time()
    epoch_time.append(epoch_end_time - epoch_start_time)
    epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
    losses.append(epoch_loss_reduced)
    model_weights = medsam_model.state_dict()
    checkpoint = {
        "model": model_weights,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "loss": epoch_loss_reduced,
        "best_loss": best_loss
    }
    if epoch_loss_reduced < best_loss:
        print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
        best_loss = epoch_loss_reduced
        checkpoint["best_loss"] = best_loss
        torch.save(checkpoint, join(work_dir, "medsam_text_prompt_best.pth"))

    torch.save(checkpoint, join(work_dir, "medsam_text_prompt_latest.pth"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1.plot(losses)
    ax1.set_title("Dice + Cross Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.plot(epoch_time)
    ax2.set_title("Epoch Running Time")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Time (s)")
    fig.savefig(join(work_dir, "medsam_text_prompt_loss_time.png"))

    epoch_loss_reduced = 1e10
