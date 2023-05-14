# %% import packages
import numpy as np
import os
from glob import glob
import pandas as pd

join = os.path.join
from skimage import transform, io, segmentation
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import argparse

# set up the parser
parser = argparse.ArgumentParser(description="preprocess grey and RGB images")

# add arguments to the parser
parser.add_argument(
    "-i",
    "--img_path",
    type=str,
    default="data/MedSAMDemo_2D/train/images",
    help="path to the images",
)
parser.add_argument(
    "-gt",
    "--gt_path",
    type=str,
    default="data/MedSAMDemo_2D/train/labels",
    help="path to the ground truth (gt)",
)

parser.add_argument(
    "--csv",
    type=str,
    default=None,
    help="path to the csv file",
)

parser.add_argument(
    "-o",
    "--npz_path",
    type=str,
    default="data/demo2D",
    help="path to save the npz files",
)
parser.add_argument(
    "--data_name",
    type=str,
    default="demo2d",
    help="dataset name; used to name the final npz file, e.g., demo2d.npz",
)
parser.add_argument("--image_size", type=int, default=256, help="image size")
parser.add_argument(
    "--img_name_suffix", type=str, default=".png", help="image name suffix"
)
parser.add_argument("--label_id", type=int, default=255, help="label id")
parser.add_argument("--model_type", type=str, default="vit_b", help="model type")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="work_dir/SAM/sam_vit_b_01ec64.pth",
    help="checkpoint",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument("--seed", type=int, default=2023, help="random seed")

# parse the arguments
args = parser.parse_args()

# convert 2d grey or rgb images to npz file
imgs = []
gts = []
img_embeddings = []

# set up the model
# get the model from sam_model_registry using the model_type argument
# and load it with checkpoint argument
# download save the SAM checkpoint.
# [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](VIT-B SAM model)

sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(
    args.device
)


def process(gt_name: str, image_name: str):
    if image_name == None:
        image_name = gt_name.split(".")[0] + args.img_name_suffix
    gt_data = io.imread(join(args.gt_path, gt_name))
    # if it is rgb, select the first channel
    if len(gt_data.shape) == 3:
        gt_data = gt_data[:, :, 0]
    assert len(gt_data.shape) == 2, "ground truth should be 2D"

    # resize ground truch image
    gt_data = transform.resize(
        gt_data == args.label_id,
        (args.image_size, args.image_size),
        order=0,
        preserve_range=True,
        mode="constant",
    )
    # convert to uint8
    gt_data = np.uint8(gt_data)

    if np.sum(gt_data) > 100:  # exclude tiny objects
        """Optional binary thresholding can be added"""
        assert (
            np.max(gt_data) == 1 and np.unique(gt_data).shape[0] == 2
        ), "ground truth should be binary"

        image_data = io.imread(join(args.img_path, image_name))
        # Remove any alpha channel if present.
        if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
            image_data = image_data[:, :, :3]
        # If image is grayscale, then repeat the last channel to convert to rgb
        if len(image_data.shape) == 2:
            image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
        # nii preprocess start
        lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(
            image_data, 99.5
        )
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        # min-max normalize and scale
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        image_data_pre[image_data == 0] = 0

        image_data_pre = transform.resize(
            image_data_pre,
            (args.image_size, args.image_size),
            order=3,
            preserve_range=True,
            mode="constant",
            anti_aliasing=True,
        )
        image_data_pre = np.uint8(image_data_pre)

        imgs.append(image_data_pre)

        assert np.sum(gt_data) > 100, "ground truth should have more than 100 pixels"

        gts.append(gt_data)

        # resize image to 3*1024*1024
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image_data_pre)
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(
            args.device
        )
        input_image = sam_model.preprocess(
            resize_img_tensor[None, :, :, :]
        )  # (1, 3, 1024, 1024)
        assert input_image.shape == (
            1,
            3,
            sam_model.image_encoder.img_size,
            sam_model.image_encoder.img_size,
        ), "input image should be resized to 1024*1024"
        # pre-compute the image embedding
        with torch.no_grad():
            embedding = sam_model.image_encoder(input_image)
            img_embeddings.append(embedding.cpu().numpy()[0])


if args.csv != None:
    # if data is presented in csv format
    # columns must be named image_filename and mask_filename respectively
    try:
        os.path.exists(args.csv)
    except FileNotFoundError as e:
        print(f"File {args.csv} not found!!")

    df = pd.read_csv(args.csv)
    bar = tqdm(df.iterrows(), total=len(df))
    for idx, row in bar:
        process(row.mask_filename, row.image_filename)

else:
    # get all the names of the images in the ground truth folder
    names = sorted(os.listdir(args.gt_path))
    # print the number of images found in the ground truth folder
    print("image number:", len(names))
    for gt_name in tqdm(names):
        process(gt_name, None)

# create a directory to save the npz files
save_path = args.npz_path + "_" + args.model_type
os.makedirs(save_path, exist_ok=True)


# save all 2D images as one npz file: ori_imgs, ori_gts, img_embeddings
# stack the list to array
print("Num. of images:", len(imgs))
if len(imgs) > 1:
    imgs = np.stack(imgs, axis=0)  # (n, 256, 256, 3)
    gts = np.stack(gts, axis=0)  # (n, 256, 256)
    img_embeddings = np.stack(img_embeddings, axis=0)  # (n, 1, 256, 64, 64)
    np.savez_compressed(
        join(save_path, args.data_name + ".npz"),
        imgs=imgs,
        gts=gts,
        img_embeddings=img_embeddings,
    )
    # save an example image for sanity check
    idx = np.random.randint(imgs.shape[0])
    img_idx = imgs[idx, :, :, :]
    gt_idx = gts[idx, :, :]
    bd = segmentation.find_boundaries(gt_idx, mode="inner")
    img_idx[bd, :] = [255, 0, 0]
    io.imsave(save_path + ".png", img_idx, check_contrast=False)
else:
    print(
        "Do not find image and ground-truth pairs. Please check your dataset and argument settings"
    )
