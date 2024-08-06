import numpy as np
from os.path import join
from os import makedirs, listdir
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import SimpleITK as sitk
import cv2
from skimage import measure
from tqdm import tqdm
import argparse

from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms

from torch import multiprocessing as mp

#%% set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

label_dict = {
    1: 'Liver',
    2: 'Right Kidney',
    3: 'Spleen',
    4: 'Pancreas',
    5: 'Aorta',
    6: 'Inferior Vena Cava', # IVC
    7: 'Right Adrenal Gland', # RAG
    8: 'Left Adrenal Gland', # LAG
    9: 'Gallbladder',
    10: 'Esophagus',
    11: 'Stomach',
    13: 'Left Kidney'
}

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
            image_embeddings=img_embed, # (1, 256, 64, 64)
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
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]

        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        return _features


def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC.astype(np.uint8)

image_size = 1024
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# visualization functions
# source: https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# change color to avoid red and green
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

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

@torch.no_grad()
def medsam_inference(
    medsam_model,
    features,
    box_1024,
    H, W
    ):
    img_embed, high_res_features = features["image_embed"], features["high_res_feats"]
    box_torch = torch.as_tensor(box_1024, dtype=torch.float32, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_coords = box_torch.reshape(-1, 2, 2) # (B, 4) to (B, 2, 2)
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=img_embed.device)
        box_labels = box_labels.repeat(box_torch.size(0), 1)
    concat_points = (box_coords, box_labels)

    sparse_embeddings, dense_embeddings = medsam_model.sam2_model.sam_prompt_encoder(
        points=concat_points,
        boxes=None,
        masks=None,
    )
    low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = medsam_model.sam2_model.sam_mask_decoder(
        image_embeddings=img_embed, # (1, 256, 64, 64)
        image_pe=medsam_model.sam2_model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )

    low_res_pred = torch.sigmoid(low_res_masks_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg

# %% load model

parser = argparse.ArgumentParser(
    description="Run inference on validation set with MedSAM2"
)
parser.add_argument(
    "-data_root",
    type=str,
    default=None,
    help="Path to the data folder",
)
parser.add_argument(
    "-pred_save_dir",
    type=str,
    default=None,
    help="Path to save the segmentation results",
)
parser.add_argument(
    "-sam2_checkpoint",
    type=str,
    default='./checkpoints/sam2_hiera_tiny.pt',
    help="SAM2 pretrained model checkpoint",
)
parser.add_argument(
    "-model_cfg",
    type=str,
    default="sam2_hiera_t.yaml",
    help="Model config file"
)
parser.add_argument(
    "-medsam2_checkpoint",
    type=str,
    default="./work_dir/medsam2_t_flare22.pth",
    help="Path to the finetuned MedSAM2 model",
)
parser.add_argument("-device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-bbox_shift",
    type=int,
    default=5,
    help="Bounding box perturbation",
)
parser.add_argument(
    "-num_workers",
    type=int,
    default=16,
    help="Number of workers for multiprocessing",
)
parser.add_argument("--visualize", action="store_true", help="Save the .nii.gz segmentation results")
args = parser.parse_args()

visualize = args.visualize
data_root = args.data_root
pred_save_dir = args.pred_save_dir
makedirs(pred_save_dir, exist_ok=True)
bbox_shift = args.bbox_shift

device = args.device
model_cfg = args.model_cfg
sam2_checkpoint = args.sam2_checkpoint
medsam2_checkpoint = args.medsam2_checkpoint
num_workers = args.num_workers

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, mode="eval", apply_postprocessing=True)
medsam2_checkpoint = torch.load(medsam2_checkpoint, map_location="cpu")
medsam_model = MedSAM2(model=sam2_model)
medsam_model.load_state_dict(medsam2_checkpoint["model"], strict=True)
medsam_model.eval()
sam2_transforms = SAM2Transforms(resolution=1024, mask_threshold=0)

# load data
_names = sorted(listdir(data_root))
names = [name for name in _names if name.endswith('.npz')]
def main(name):
    npz = np.load(join(data_root, name), allow_pickle=True)
    img_3D = npz['imgs']

    segs_dict = {}
    gt_3D = npz['gts']
    label_ids = np.unique(gt_3D)[1:]
    ## Simulate 3D box for each organ
    for label_id in label_ids:
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8)
        marker_data_id = (gt_3D == label_id).astype(np.uint8)
        marker_zids, _, _ = np.where(marker_data_id > 0)
        marker_zids = np.sort(np.unique(marker_zids))
        bbox_dict = {} # key: z_index, value: bbox
        for z in marker_zids:
            z_box = get_bbox(marker_data_id[z, :, :])
            bbox_dict[z] = z_box
        # find largest bbox in bbox_dict
        bbox_areas = [np.prod(bbox_dict[z][2:] - bbox_dict[z][:2]) for z in bbox_dict.keys()]
        z_max_area = list(bbox_dict.keys())[np.argmax(bbox_areas)]
        z_min = min(bbox_dict.keys())
        z_max = max(bbox_dict.keys())
        z_max_area_bbox = mid_slice_bbox_2d = bbox_dict[z_max_area]

        z_middle = int((z_max - z_min)/2 + z_min)
    
        z_max = min(z_max+1, img_3D.shape[0])
        for z in range(z_middle, z_max):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape
            
            # convert the shape to (3, H, W)
            img_1024_tensor = sam2_transforms(img_3c)[None, ...].to(device)
            # get the image embedding
            with torch.no_grad():
                _features = medsam_model._image_encoder(img_1024_tensor) # (1, 256, 64, 64)
            if z == z_middle:
                box_1024 = mid_slice_bbox_2d / np.array([W, H, W, H]) * 1024
            else:
                pre_seg = segs_3d_temp[z-1, :, :]
                pre_seg1024 = cv2.resize(pre_seg, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                if np.max(pre_seg1024) > 0:
                    box_1024 = get_bbox(pre_seg1024)
                else:
                    box_1024 = mid_slice_bbox_2d / np.array([W, H, W, H]) * 1024
            img_2d_seg = medsam_inference(medsam_model, _features, box_1024[None,:], H, W)
            segs_3d_temp[z, img_2d_seg>0] = 1
        
        # infer from middle slice to the z_min
        z_min = max(-1, z_min-1)
        for z in range(z_middle-1, z_min, -1):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_1024_tensor = sam2_transforms(img_3c)[None, ...].to(device)
            # get the image embedding
            with torch.no_grad():
                _features = medsam_model._image_encoder(img_1024_tensor) # (1, 256, 64, 64)

            pre_seg = segs_3d_temp[z+1, :, :]
            pre_seg1024 = cv2.resize(pre_seg, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            if np.max(pre_seg1024) > 0:
                box_1024 = get_bbox(pre_seg1024)
            else:
                box_1024 = mid_slice_bbox_2d / np.array([W, H, W, H]) * 1024
            img_2d_seg = medsam_inference(medsam_model, _features, box_1024[None,:], H, W)
            segs_3d_temp[z, img_2d_seg>0] = 1
        segs_dict[label_id] = segs_3d_temp.copy() ## save the segmentation result in one-hot format

    np.savez_compressed(
        join(pred_save_dir, name),
        **{label_dict[label_id]: segs_dict[label_id] for label_id in label_ids},
    )

    if visualize:
        for label_id in label_ids:
            seg_sitk = sitk.GetImageFromArray(segs_dict[label_id])
            seg_sitk.SetSpacing(npz['spacing'])
            sitk.WriteImage(seg_sitk, join(pred_save_dir, name.replace('.npz', f'_{label_dict[label_id]}.nii.gz')))

    
        img_sitk = sitk.GetImageFromArray(img_3D)
        img_sitk.SetSpacing(npz['spacing'])
        sitk.WriteImage(img_sitk, join(pred_save_dir, name.replace('.npz', '_0000.nii.gz')))

        gts_sitk = sitk.GetImageFromArray(gt_3D)
        gts_sitk.SetSpacing(npz['spacing'])
        sitk.WriteImage(gts_sitk, join(pred_save_dir, name.replace('.npz', '_gt.nii.gz')))


if __name__ == '__main__':
    mp.set_start_method('spawn')
    with mp.Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(main, names), total=len(names)))