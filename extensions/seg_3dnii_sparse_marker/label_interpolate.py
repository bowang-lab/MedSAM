#!/usr/bin/env python

import os
join = os.path.join
import SimpleITK as sitk
import numpy as np
from collections import OrderedDict
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interp1d

def interpolate_labels(label_volume):
    depth, height, width = label_volume.shape
    
    # Create an array to hold the interpolated labels
    interpolated_labels = np.zeros((depth, height, width), dtype=label_volume.dtype)
    
    # Loop through each unique label in the label volume
    for label in np.unique(label_volume):
        if label == 0:  # Skip background
            continue
        
        # Create a binary mask for the current label
        binary_mask = (label_volume == label)
        
        # Extract slices with the current label
        labeled_slices = [i for i in range(depth) if np.any(binary_mask[i])]
        
        if len(labeled_slices) < 2:  # At least two slices are needed for interpolation
            continue
        
        # Initialize array to hold distances for the slices
        distances = np.zeros((depth, height, width))
        
        # Calculate distances from object border for each labeled slice
        for i in labeled_slices:
            distances[i] = distance_transform_edt(np.logical_not(binary_mask[i]))
        
        # Create an interpolating function
        f = interp1d(labeled_slices, [distances[i] for i in labeled_slices],
                     axis=0, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        # Interpolate
        for i, next_slice in zip(labeled_slices[:-1], labeled_slices[1:]):
            interpolated_slices = np.round(f(np.arange(i, next_slice + 1))).astype(int)
            
            # Update label
            interpolated_labels[i:next_slice + 1][interpolated_slices <= 0] = label
            
    return interpolated_labels

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


# Check input directories.
marker_dir = 'marker-expert1'
save_dir = marker_dir + '_interpolated'
os.makedirs(save_dir, exist_ok=True)
names = sorted(os.listdir(marker_dir))
names = [name for name in names if name.endswith('.nii.gz')]

for name in names:
    nii = sitk.ReadImage(join(marker_dir, name))
    marker_data = np.uint8(sitk.GetArrayFromImage(nii))
    # simulate bounding box based on marker
    box_data = np.zeros_like(marker_data, dtype=np.uint8)
    label_ids = np.unique(marker_data)[1:]
    print(f'label ids: {label_ids}')
    for label_id in label_ids:
        marker_data_id = (marker_data == label_id).astype(np.uint8)
        marker_zids, _, _ = np.where(marker_data_id > 0)
        marker_zids = np.sort(np.unique(marker_zids))
        print(f'z indices: {marker_zids}')
        # bbox_dict = {} # key: z_index, value: bbox
        for z in marker_zids:
            # get bbox for each slice
            z_box = get_bbox(marker_data_id[z, :, :], bbox_shift=5)
            box_data[z, z_box[1]:z_box[3], z_box[0]:z_box[2]] = label_id
    # interpolate labels
    interpolated_labels = interpolate_labels(box_data)
    # save interpolated labels
    save_name = name# .replace('.nii.gz', '_interpolated.nii.gz')
    save_path = join(save_dir, save_name)
    save_sitk = sitk.GetImageFromArray(interpolated_labels)
    # add meta information
    save_sitk.CopyInformation(nii)
    sitk.WriteImage(save_sitk, save_path)