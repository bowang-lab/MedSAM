# -*- coding: utf-8 -*-
import os
join = os.path.join
import random

path_nii = '' # please complete path; two subfolders: images and labels
path_video = None # or specify the path
path_2d = None # or specify the path

#%% split 3D nii data
if path_nii is not None:
    img_path = join(path_nii, 'images')
    gt_path = join(path_nii, 'labels')
    gt_names = sorted(os.listdir(gt_path))
    img_suffix = '_0000.nii.gz'
    gt_suffix = '.nii.gz'
    # split 20% data for validation and testing
    validation_path = join(path_nii, 'validation')
    os.makedirs(join(validation_path, 'images'), exist_ok=True)
    os.makedirs(join(validation_path, 'labels'), exist_ok=True)
    testing_path = join(path_nii, 'testing')
    os.makedirs(join(testing_path, 'images'), exist_ok=True)
    os.makedirs(join(testing_path, 'labels'), exist_ok=True)
    candidates = random.sample(gt_names, int(len(gt_names)*0.2))
    # split half of test names for validation
    validation_names = random.sample(candidates, int(len(candidates)*0.5))
    test_names = [name for name in candidates if name not in validation_names]
    # move validation and testing data to corresponding folders
    for name in validation_names:
        img_name = name.split(gt_suffix)[0] + img_suffix
        os.rename(join(img_path, img_name), join(validation_path, 'images', img_name))
        os.rename(join(gt_path, name), join(validation_path, 'labels', name))
    for name in test_names:
        img_name = name.split(gt_suffix)[0] + img_suffix
        os.rename(join(img_path, img_name), join(testing_path, 'images', img_name))
        os.rename(join(gt_path, name), join(testing_path, 'labels', name))


##% split 2D images
if path_2d is not None:
    img_path = join(path_2d, 'images')
    gt_path = join(path_2d, 'labels')
    gt_names = sorted(os.listdir(gt_path))
    img_suffix = '.png'
    gt_suffix = '.png'
    # split 20% data for validation and testing
    validation_path = join(path_2d, 'validation')
    os.makedirs(join(validation_path, 'images'), exist_ok=True)
    os.makedirs(join(validation_path, 'labels'), exist_ok=True)
    testing_path = join(path_2d, 'testing')
    os.makedirs(join(testing_path, 'images'), exist_ok=True)
    os.makedirs(join(testing_path, 'labels'), exist_ok=True)
    candidates = random.sample(gt_names, int(len(gt_names)*0.2))
    # split half of test names for validation
    validation_names = random.sample(candidates, int(len(candidates)*0.5))
    test_names = [name for name in candidates if name not in validation_names]
    # move validation and testing data to corresponding folders
    for name in validation_names:
        img_name = name.split(gt_suffix)[0] + img_suffix
        os.rename(join(img_path, img_name), join(validation_path, 'images', img_name))
        os.rename(join(gt_path, name), join(validation_path, 'labels', name))

    for name in test_names:
        img_name = name.split(gt_suffix)[0] + img_suffix
        os.rename(join(img_path, img_name), join(testing_path, 'images', img_name))
        os.rename(join(gt_path, name), join(testing_path, 'labels', name))

#%% split video data
if path_video is not None:
    img_path = join(path_video, 'images')
    gt_path = join(path_video, 'labels')
    gt_folders = sorted(os.listdir(gt_path))
    # split 20% videos for validation and testing
    validation_path = join(path_video, 'validation')
    os.makedirs(join(validation_path, 'images'), exist_ok=True)
    os.makedirs(join(validation_path, 'labels'), exist_ok=True)
    testing_path = join(path_video, 'testing')
    os.makedirs(join(testing_path, 'images'), exist_ok=True)
    os.makedirs(join(testing_path, 'labels'), exist_ok=True)
    candidates = random.sample(gt_folders, int(len(gt_folders)*0.2))
    # split half of test names for validation
    validation_names = random.sample(candidates, int(len(candidates)*0.5))
    test_names = [name for name in candidates if name not in validation_names]
    # move validation and testing data to corresponding folders
    for name in validation_names:
        os.rename(join(img_path, name), join(validation_path, 'images', name))
        os.rename(join(gt_path, name), join(validation_path, 'labels', name))
    for name in test_names:
        os.rename(join(img_path, name), join(testing_path, 'images', name))
        os.rename(join(gt_path, name), join(testing_path, 'labels', name))
