# %%
import os
import numpy as np
import copy
import glob
import argparse
import cv2

from visual_sampler.sampler_v2 import build_shape_sampler
from visual_sampler.config import cfg
# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root", type=str, default='/home/sumin2/Downloads/CVPR/scribble-train-demo',
    help="data root."
)
parser.add_argument(
    "--save_path", type=str, default='/home/sumin2/Downloads/CVPR/train_scribbles',
    help="save dir."
)

args = parser.parse_args()

def show_mask_cv2(mask, image, color=None, alpha=0.9):
    if color is None:
        color = np.random.randint(0, 255, 3)
        #color = [255,0,0]
    h, w = mask.shape[-2:]

    overlay = np.zeros_like(image)
    for i in range(3):
        overlay[:, :, i] = color[i]

    overlay = cv2.bitwise_and(overlay, overlay, mask=mask)
    combined = cv2.addWeighted(overlay, alpha, image, 1 , 0)

    return combined

# %%
shape_sampler = build_shape_sampler(cfg)
root = args.root 
save_dir = args.save_path
os.makedirs(save_dir, exist_ok=True)
files = glob.glob(os.path.join(root, '*/*.npz'))
files = sorted(files)
sancheck_path = os.path.join(save_dir, 'sancheck')
os.makedirs(sancheck_path, exist_ok=True)
for file in files:
    base = os.path.basename(file)
    print(f'start {file}')
    npz = np.load(file, allow_pickle=True, mmap_mode="r")
    imgs = npz["imgs"] #512,512,3
    gts = npz['gts'] #512,512
    if len(gts.shape) ==3: #3D data
        spacing = npz['spacing']
        z_index, _, _ = np.where(gts > 0)
        z_index = np.unique(z_index)
        scribbles_list = []
        for z in z_index:
            gts2D = gts[z, :, :].copy()
            gts2D = gts2D.astype(np.float32)
            instances = np.unique(gts2D)
            instances = instances[instances!=0]
            all_scribbles = np.zeros_like(gts2D)
            for instance in instances:
                mask = (gts2D == instance)
                scribble = (shape_sampler(mask) * instance).squeeze().numpy() #512,512
                all_scribbles[scribble == instance] = scribble[scribble == instance]
            # for background
            mask = (gts2D == 0)
            scribble = (shape_sampler.forward_background(mask)).squeeze().numpy() #512,512
            scribble = scribble * 1000
            all_scribbles[scribble == 1000] = scribble[scribble == 1000]
            base2 = base.split('.npz')[0] + "_" + str(z) + 'npz'
            scribbles_list.append(all_scribbles)
            #np.savez_compressed(os.path.join(save_dir, base2), imgs=imgs, scribbles=all_scribbles, gts = gts2D)
            # Scribble Overlay
            mask = (~(all_scribbles == 0)).astype(np.uint8)
            im = copy.deepcopy(imgs[z][..., np.newaxis])
            im = np.repeat(im, 3, axis=-1)
            im = show_mask_cv2(mask, im, color = [0, 255, 0], alpha=1)
            cv2.imwrite(os.path.join(sancheck_path, base2.split('.npz')[0]+'_overlay.png'), im)
            # GT overlay
            im = copy.deepcopy(imgs[z][..., np.newaxis])
            im = np.repeat(im, 3, axis=-1)
            instances = instances[instances != 1000]
            for instance_id in instances:
                mask = (gts2D == instance_id).astype(np.uint8)
                im = show_mask_cv2(mask, im)
            cv2.imwrite(os.path.join(sancheck_path, base2.split('.npz')[0]+'_gt.png'), im)
        scr = np.stack(scribbles_list)
        np.savez_compressed(os.path.join(save_dir, base), imgs=imgs, scribbles=scr, gts = gts, spacing=spacing)
    else:
        assert len(gts.shape) == 2
        gts = gts.astype(np.float32)
        instances = np.unique(gts)
        instances = instances[instances!=0]
        all_scribbles = np.zeros_like(gts)
        for instance in instances:
            mask = (gts == instance)
            scribble = (shape_sampler(mask) * instance).squeeze().numpy() #512,512
            all_scribbles[scribble == instance] = scribble[scribble == instance]
        # for background
        mask = (gts == 0)
        scribble = (shape_sampler.forward_background(mask)).squeeze().numpy() #512,512
        scribble = scribble * 1000
        all_scribbles[scribble == 1000] = scribble[scribble == 1000]
        np.savez_compressed(os.path.join(save_dir, base), imgs=imgs, scribbles=all_scribbles, gts = gts)

        instances = np.append(instances, [1000])
        im = copy.deepcopy(imgs)
        #Scribble Overlay
        mask = (~(all_scribbles == 0)).astype(np.uint8)
        im = show_mask_cv2(mask, im, color = [0, 255, 0], alpha=1)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(sancheck_path, base.split('.npz')[0]+'_overlay.png'), im)
        # GT overlay
        im = copy.deepcopy(imgs)
        instances = instances[instances != 1000]
        for instance_id in instances:
            mask = (gts == instance_id).astype(np.uint8)
            im = show_mask_cv2(mask, im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(sancheck_path, base.split('.npz')[0]+'_gt.png'), im)

