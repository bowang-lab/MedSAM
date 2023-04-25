#%% import packages
# precompute image embeddings and save them to disk for model training

import numpy as np
import os
join = os.path.join 
from skimage import io, segmentation
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse

#%% parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str, default='data/Tr_Release_Part1', help='path to the image npz folder')
parser.add_argument('-o', '--save_path', type=str, default='data/Tr_emb', help='path to save the image embeddings')
parser.add_argument('--model_type', type=str, default='vit_b', help='model type')
parser.add_argument('--checkpoint', type=str, default='../work_dir/SAM/sam_vit_b_01ec64.pth', help='path to the pre-trained SAM model')
args = parser.parse_args()

pre_img_path = args.img_path # and also Tr_Release_Part2 when part1 is done
save_img_emb_path = args.save_path
os.makedirs(save_img_emb_path, exist_ok=True)
npz_files = sorted(os.listdir(pre_img_path))
#%% set up the model
sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to('cuda:0')
sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)

# compute image embeddings
for name in tqdm(npz_files):
    img = np.load(join(pre_img_path, name))['img'] # (256, 256, 3)
    gt = np.load(join(pre_img_path, name))['gt']
    resize_img = sam_transform.apply_image(img)
    resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to('cuda:0')
    # model input: (1, 3, 1024, 1024)
    input_image = sam_model.preprocess(resize_img_tensor[None,:,:,:]) # (1, 3, 1024, 1024)
    assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'
    with torch.no_grad():
        embedding = sam_model.image_encoder(input_image)
    np.savez(join(save_img_emb_path, name), img=img, gt=gt, img_embedding=embedding.cpu().numpy()[0])
    
    # sanity check
    img_idx = img.copy()
    bd = segmentation.find_boundaries(gt, mode='inner')
    img_idx[bd, :] = [255, 0, 0]
    io.imsave(save_img_emb_path + '.png', img_idx)