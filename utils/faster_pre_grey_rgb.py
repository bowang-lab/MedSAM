import numpy as np
import os
from skimage import io, transform
from tqdm import tqdm
import multiprocessing as mp

# Function to process each image and mask
def process_image(name):
    img_name_suffix = '.PNG'
    gt_name_suffix = '.png'
    prefix = modality + '_' + anatomy + '_'
    npy_save_name = prefix + name.split(gt_name_suffix)[0] + '.npy'
    gt_data_ori = np.uint8(io.imread(os.path.join(gt_path, name)))
    
    for remove_label_id in remove_label_ids:
        gt_data_ori[gt_data_ori == remove_label_id] = 0

    image_name = name.split(gt_name_suffix)[0] + img_name_suffix
    image_data = io.imread(os.path.join(img_path, image_name))
    if np.max(image_data) > 255.0:
        image_data = np.uint8((image_data - image_data.min()) / (np.max(image_data) - image_data.min()) * 255.0)
    if len(image_data.shape) == 2:
        image_data = np.repeat(np.expand_dims(image_data, -1), 3, -1)

    if do_intensity_cutoff:
        lower_bound, upper_bound = np.percentile(image_data[image_data > 0], 0.5), np.percentile(image_data[image_data > 0], 99.5)
        image_data = np.clip(image_data, lower_bound, upper_bound)
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255.0
        image_data[image_data == 0] = 0

    resize_img = transform.resize(image_data, (image_size, image_size), order=3, mode='constant', preserve_range=True, anti_aliasing=True) 
    resize_gt = transform.resize(gt_data_ori, (image_size, image_size), order=0, mode='constant', preserve_range=True, anti_aliasing=False)
    
    # reduce image size and save compressed npy
    np.savez_compressed(os.path.join(npy_path, "imgs", npy_save_name), resize_img.astype(np.uint8))
    np.savez_compressed(os.path.join(npy_path, "gts", npy_save_name), resize_gt.astype(np.uint8))

# Main script
if __name__ == '__main__':
    modality = 'Ultrasound'
    anatomy = 'femoralTriangle'
    image_size = 1024
    img_path = '/app/data/medsam_practice/images'
    gt_path = '/app/data/medsam_practice/labels'
    npy_path = '/app/data/medsam_practice/npy/' + modality + '_' + anatomy
    os.makedirs(os.path.join(npy_path, "gts"), exist_ok=True)
    os.makedirs(os.path.join(npy_path, "imgs"), exist_ok=True)
    names = sorted(os.listdir(gt_path))
    remove_label_ids = []
    do_intensity_cutoff = False

    # Create a pool of processes. Number of processes is set to the number of CPUs available.
    pool = mp.Pool(mp.cpu_count())
    
    # Process each file in parallel
    list(tqdm(pool.imap(process_image, names), total=len(names)))
    
    pool.close()
    pool.join()
