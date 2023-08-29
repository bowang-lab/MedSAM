#%% import packages
import numpy as np
import os
join = os.path.join 
from skimage import io, transform
from tqdm import tqdm

# convert 2D data to npy files, including images and corresponding masks
modality = 'dd' # e.g., 'Dermoscopy 
anatomy = 'dd'  # e.g., 'SkinCancer'
img_name_suffix = '.png' 
gt_name_suffix = '.png' 
prefix = modality + '_' + anatomy + '_'
save_suffix = '.npy' 
image_size = 1024
img_path = 'path to /images' # path to the images
gt_path = 'path to/labels' # path to the corresponding annotations
npy_path = 'path to/data/npy/' + prefix[:-1] # save npy path e.g., MedSAM/data/npy/; don't miss the `/`
os.makedirs(join(npy_path, "gts"), exist_ok=True)
os.makedirs(join(npy_path, "imgs"), exist_ok=True)
names = sorted(os.listdir(gt_path))
print(f'ori \# files {len(names)=}')

# set label ids that are excluded
remove_label_ids = [] 
tumor_id = None # only set this when there are multiple tumors in one image; convert semantic masks to instance masks
label_id_offset = 0
do_intensity_cutoff = False # True for grey images
#%% save preprocessed images and masks as npz files
for name in tqdm(names):
    image_name = name.split(gt_name_suffix)[0] + img_name_suffix
    gt_name = name
    npy_save_name = prefix + gt_name.split(gt_name_suffix)[0]+save_suffix
    gt_data_ori = np.uint8(io.imread(join(gt_path, gt_name)))
    # remove label ids
    for remove_label_id in remove_label_ids:
        gt_data_ori[gt_data_ori==remove_label_id] = 0
    # label tumor masks as instances and remove from gt_data_ori
    if tumor_id is not None:
        tumor_bw = np.uint8(gt_data_ori==tumor_id)
        gt_data_ori[tumor_bw>0] = 0
        # label tumor masks as instances
        tumor_inst, tumor_n = cc3d.connected_components(tumor_bw, connectivity=26, return_N=True)
        # put the tumor instances back to gt_data_ori
        gt_data_ori[tumor_inst>0] = tumor_inst[tumor_inst>0] + label_id_offset + 1
    
    # crop the ground truth with non-zero slices
    image_data = io.imread(join(img_path, image_name))
    if np.max(image_data) > 255.0:
        image_data = np.uint8((image_data-image_data.min()) / (np.max(image_data)-np.min(image_data))*255.0)
    if len(image_data.shape) == 2:
        image_data = np.repeat(np.expand_dims(image_data, -1), 3, -1)
    assert len(image_data.shape) == 3, 'image data is not three channels: img shape:' + str(image_data.shape) + image_name
    # convert three channel to one channel
    if image_data.shape[-1] > 3:
        image_data = image_data[:,:,:3]
    # image preprocess start
    if do_intensity_cutoff:
        lower_bound, upper_bound = np.percentile(image_data[image_data>0], 0.5), np.percentile(image_data[image_data>0], 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
        image_data_pre[image_data==0] = 0
        image_data_pre = np.uint8(image_data_pre)
    else:
        # print('no intensity cutoff')
        image_data_pre = image_data.copy()
    np.savez_compressed(join(npy_path, prefix + gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=image_data_pre, gts=gt_data_ori)   
    resize_img = transform.resize(image_data_pre, (image_size, image_size), order=3, mode='constant', preserve_range=True, anti_aliasing=True)
    resize_img01 = resize_img/255.0
    resize_gt = transform.resize(gt_data_ori, (image_size, image_size), order=0, mode='constant', preserve_range=True, anti_aliasing=False)
    # save resize img and gt as npy
    np.save(join(npy_path, "imgs", npy_save_name), resize_img01)
    np.save(join(npy_path, "gts", npy_save_name), resize_gt.astype(np.uint8))

