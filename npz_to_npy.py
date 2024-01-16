"""
Convert the preprocessed .npz files to .npy files for training
"""
# %% import packages
import numpy as np
import os
join = os.path.join
listdir = os.listdir
makedirs = os.makedirs
from tqdm import tqdm
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-npz_dir", type=str, default="data/npz/MedSAM_train/CT_Abd",
                    help="Path to the directory containing preprocessed .npz data, [default: data/npz/MedSAM_train/CT_Abd]")
parser.add_argument("-npy_dir", type=str, default="data/npy",
                    help="Path to the directory where the .npy files for training will be saved, [default: ./data/npy]")
parser.add_argument("-num_workers", type=int, default=4,
                    help="Number of workers to convert npz to npy in parallel, [default: 4]")
args = parser.parse_args()
# %%
npz_dir = args.npz_dir
npy_dir = args.npy_dir
makedirs(join(npy_dir, "imgs"), exist_ok=True)
makedirs(join(npy_dir, "gts"), exist_ok=True)
npz_names = [f for f in listdir(npz_dir) if f.endswith(".npz")]
num_workers = args.num_workers
# %%
# convert npz files to npy files
def convert_npz_to_npy(npz_name):
    """
    Convert npz files to npy files for training

    Parameters
    ----------
    npz_name : str
        Name of the npz file to be converted
    """
    name = npz_name.split(".npz")[0]
    npz_path = join(npz_dir, npz_name)
    npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
    imgs = npz["imgs"]
    gts = npz["gts"]
    if len(gts.shape) > 2: ## 3D image
        for i in range(imgs.shape[0]):
            img_i = imgs[i, :, :]
            img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)

            img_01 = (img_3c - img_3c.min()) / np.clip(
                img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)

            gt_i = gts[i, :, :]

            gt_i = np.uint8(gt_i)
            assert img_01.shape[:2] == gt_i.shape
            np.save(join(npy_dir, "imgs", name + "-" + str(i).zfill(3) + ".npy"), img_01)
            np.save(join(npy_dir, "gts", name + "-" + str(i).zfill(3) + ".npy"), gt_i)
    else: ## 2D image
        if len(imgs.shape) < 3:
            img_3c = np.repeat(imgs[:, :, None], 3, axis=-1)
        else:
            img_3c = imgs

        img_01 = (img_3c - img_3c.min()) / np.clip(
            img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None
        )  # normalize to [0, 1], (H, W, 3)
        assert img_01.shape[:2] == gts.shape

        np.save(join(npy_dir, "imgs", name + ".npy"), img_01)
        np.save(join(npy_dir, "gts", name + ".npy"), gts)
# %%
if __name__ == "__main__":
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(npz_names)) as pbar:
            pbar.set_description("Converting npz to npy")
            for i, _ in enumerate(pool.imap_unordered(convert_npz_to_npy, npz_names)):
                pbar.update()
