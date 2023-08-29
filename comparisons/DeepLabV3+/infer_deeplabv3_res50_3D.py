# %%
import os
import glob
import random
from os import listdir, makedirs
from os.path import join, isfile, isdir, basename
from tqdm import tqdm
import numpy as np
import torch
from torch._dynamo import OptimizedModule
from torch import multiprocessing as mp
from datetime import datetime
import cv2
from skimage import morphology
import torch.nn.functional as F
from matplotlib import pyplot as plt
import segmentation_models_pytorch as smp
import argparse

torch.cuda.empty_cache()
os.environ['PYTHONHASHSEED']=str(2023)
random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-checkpoint',
    type=str,
    default='',
    help='Path to the model checkpoint',
    required=True
)
parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    '-data_root',
    type=str,
    default='',
    help='Path to the validation data directory',
    required=True
)
parser.add_argument(
    '-pred_save_dir',
    type=str,
    default='segs',
    help='Path to the directory where the segmentation results will be saved in npz format'
)
parser.add_argument('--save_overlay', action='store_true', default=False, help="Whether to save segmentation overlay")
parser.add_argument(
    '-png_save_dir',
    type=str,
    default='png',
    help='Path to the directory where the segmentation overlay will be saved in png format'
)
parser.add_argument('-num_workers', type=int, default=1)

args = parser.parse_args()
checkpoint = args.checkpoint
device = args.device
data_root = args.data_root
pred_save_dir = args.pred_save_dir
png_save_dir = args.png_save_dir
makedirs(pred_save_dir, exist_ok=True)
save_overlay = args.save_overlay
if save_overlay:
    makedirs(png_save_dir, exist_ok=True)
num_workers = args.num_workers
data_root_files = listdir(data_root)

## Check whether the data root has subfolders
has_task = isdir(join(data_root, data_root_files[0]))
if has_task:
    gt_path_files = sorted(glob.glob(join(data_root, '**/*.npz'), recursive=True))
else:
    gt_path_files = sorted(glob.glob(join(data_root, '*.npz'), recursive=True))
image_size = 224
bbox_shift = 5
# %%
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

def dice_coefficient(preds, targets):
    smooth = 1.0
    assert preds.shape == targets.shape

    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice

def get_bbox(mask, bbox_shift=5):
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)
    bboxes = np.array([x_min, y_min, x_max, y_max])

    return bboxes


def preprocess_slice(img_2D, gt_2D, image_size=224, bbox_shift=5):
    img_3c = np.repeat(img_2D[..., None], 3, axis=-1) # (H, W, 3)
    resize_img_cv2 = cv2.resize(
        img_3c,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA
    )
    resize_img_cv2_01 = (resize_img_cv2 - resize_img_cv2.min()) / np.clip(resize_img_cv2.max() - resize_img_cv2.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    resize_img = np.transpose(resize_img_cv2_01, (2, 0, 1))
    assert np.max(resize_img)<=1.0 and np.min(resize_img)>=0.0, 'image should be normalized to [0, 1]'
    if gt_2D.shape[0] != image_size or gt_2D.shape[1] != image_size:
        gt_2D = cv2.resize(
            gt_2D, (image_size, image_size),
            interpolation=cv2.INTER_NEAREST
        )
        gt_2D = np.uint8(gt_2D)
    else:
        gt_2D = gt_2D.astype(np.uint8)
    try:
        assert np.max(gt_2D) == 1 and np.min(gt_2D) == 0, 'ground truth should be 0, 1, got: ' + str(np.unique(gt_2D))
    except:
        assert np.max(gt_2D) == 0 and np.min(gt_2D) == 0, 'ground truth should be 0, 1, got: ' + str(np.unique(gt_2D))
        return None

    y_indices, x_indices = np.where(gt_2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = gt_2D.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    ## Append bbox prompt channel
    resize_img_bbox = np.concatenate([resize_img, np.zeros((1, image_size, image_size))], axis=0)
    resize_img_bbox[-1, y_min:y_max, x_min:x_max] = 1.0
    resize_img_bbox = resize_img_bbox[None, ...]

    return torch.tensor(resize_img_bbox).float()

model = smp.DeepLabV3Plus(
    encoder_name="resnet50",        # encoder model type
    encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
    in_channels=4,                  # Additional channel for bounding box prompt
    classes=1,                      # model output channels (number of classes in your dataset)
    activation=None                 # Output logits
)
checkpoint = torch.load(checkpoint)
weights = checkpoint['model']
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

def deeplabv3plus_infer_npz(gt_path_file):
    npz_name = basename(gt_path_file)
    if has_task:
        task_folder = gt_path_file.split('/')[-2]
        pred_save_dir_task = join(pred_save_dir, task_folder)
        png_save_dir_task = join(png_save_dir, task_folder)
        makedirs(pred_save_dir_task, exist_ok=True)
        makedirs(png_save_dir_task, exist_ok=True)
    else:
        pred_save_dir_task = pred_save_dir
        png_save_dir_task = png_save_dir
    if isfile(join(pred_save_dir_task, npz_name)):
        return
    npz = np.load(gt_path_file, 'r', allow_pickle=True)
    img_3D = npz['imgs'] # (Num, H, W)
    gt_3D = npz['gts'] # (Num, 256, 256)
    spacing = npz['spacing']
    seg_3D = np.zeros_like(gt_3D, dtype=np.uint8) # (Num, 256, 256)

    for i in range(img_3D.shape[0]):
        img_2D = img_3D[i,:,:] # (H, W)
        gt = gt_3D[i,:,:] # (H, W)
        label_ids = np.unique(gt)[1:]
        for label_id in label_ids:
            gt2D = np.uint8(gt == label_id) # only one label, (256, 256)
            img_4c = preprocess_slice(
                img_2D,
                gt2D,
                image_size=image_size,
                bbox_shift=bbox_shift
            )
            if img_4c is None:
                continue
            img_4c = img_4c.to(device)
            with torch.no_grad():
                seg_logits = model(img_4c)
                seg_logits = F.interpolate(
                    seg_logits,
                    size=(gt2D.shape[0], gt2D.shape[1]),
                    mode='bilinear',
                    align_corners=False
                )
                seg_probs = torch.sigmoid(seg_logits)
                seg_probs_np = seg_probs.detach().cpu().numpy().squeeze()
                torch.cuda.empty_cache()
            seg_2D = np.uint8(seg_probs_np > 0.5)
            seg_3D[i, seg_2D>0] = label_id

    np.savez_compressed(
        join(pred_save_dir_task, npz_name),
        segs=seg_3D,
        gts=gt_3D,
        spacing=spacing
    )

    if save_overlay:
        idx = int(seg_3D.shape[0] / 2)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img_3D[idx], cmap='gray')
        ax[0].set_title("Image")
        ax[0].axis('off')
        ax[1].imshow(img_3D[idx], cmap='gray')
        show_mask(gt_3D[idx], ax[1])
        ax[1].axis('off')
        ax[1].set_title("Ground Truth")
        ax[2].imshow(img_3D[idx], cmap='gray')
        show_mask(seg_3D[idx], ax[2])
        ax[2].set_title("Segmentation")
        ax[2].axis('off')
        plt.savefig(
            join(png_save_dir_task, npz_name.split(".")[0] + '.png'),
            dpi=300
        )
        plt.close()

    print(f"Case {npz_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# %%
if __name__ == '__main__':
    num_workers = num_workers
    try:
        mp.set_start_method('spawn', force=True)
        print("spawned")
    except RuntimeError:
        pass
    with mp.Pool(processes=num_workers) as pool:
        with tqdm(total=len(gt_path_files)) as pbar:
            for i, _ in tqdm(enumerate(pool.imap_unordered(deeplabv3plus_infer_npz, gt_path_files))):
                pbar.update()
