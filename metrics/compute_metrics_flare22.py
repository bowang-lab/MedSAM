# %% move files based on csv file
import numpy as np
from os import listdir, makedirs
from os.path import join
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from tqdm import tqdm
import multiprocessing as mp
import argparse

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
    11 : 'Stomach',
    13: 'Left Kidney'
}

def compute_multi_class_dsc(gt, npz_seg):
    dsc = {}
    for i in label_dict.keys():
        gt_i = gt == i
        if label_dict[i] in npz_seg.files:
            seg_i = npz_seg[label_dict[i]]
        else:
            seg_i = np.zeros_like(gt_i)
        if np.sum(gt_i)==0 and np.sum(seg_i)==0:
            dsc[label_dict[i]] = 1
        elif np.sum(gt_i)==0 and np.sum(seg_i)>0:
            dsc[label_dict[i]] = 0
        else:
            dsc[label_dict[i]] = compute_dice_coefficient(gt_i, seg_i)

    return dsc


def compute_multi_class_nsd(gt, npz_seg, spacing, tolerance=2.0):
    nsd = {}
    for i in label_dict.keys():
        gt_i = gt == i
        if label_dict[i] in npz_seg.files:
            seg_i = npz_seg[label_dict[i]]
        else:
            seg_i = np.zeros_like(gt_i)
        if np.sum(gt_i)==0 and np.sum(seg_i)==0:
            nsd[label_dict[i]] = 1
        elif np.sum(gt_i)==0 and np.sum(seg_i)>0:
            nsd[label_dict[i]] = 0
        else:
            surface_distance = compute_surface_distances(
                gt_i, seg_i, spacing_mm=spacing
            )
            nsd[label_dict[i]] = compute_surface_dice_at_tolerance(surface_distance, tolerance)

    return nsd


parser = argparse.ArgumentParser()

parser.add_argument('-s', '--seg_dir', default=None, type=str)
parser.add_argument('-g', '--gt_dir',  default=None, type=str)
parser.add_argument('-csv_dir', default='./', type=str)
parser.add_argument('-nw', '--num_workers', default=8, type=int)
parser.add_argument('-nsd', default=True, type=bool, help='set it to False to disable NSD computation and save time')
args = parser.parse_args()

seg_dir = args.seg_dir
gt_dir = args.gt_dir
num_workers = args.num_workers
compute_NSD = args.nsd
csv_dir = args.csv_dir
makedirs(csv_dir, exist_ok=True)

def compute_metrics(npz_name):
    """
    return
    npz_name: str
    dsc: float
    """
    metric_dict = {'dsc': -1.}
    if compute_NSD:
        metric_dict['nsd'] = -1.
    try:
        npz_seg = np.load(join(seg_dir, npz_name), allow_pickle=True, mmap_mode='r')
    except FileNotFoundError as e:
        print(e)
        raise FileNotFoundError(f'File {npz_name} is missing in submission')

    try:
        npz_gt = np.load(join(gt_dir, npz_name), allow_pickle=True, mmap_mode='r')
    except FileNotFoundError as e:
        print(e)
        raise FileNotFoundError(f'File {npz_name} is not a valid case')

    gts = npz_gt['gts']
    spacing = npz_gt['spacing']
    
    dsc = compute_multi_class_dsc(gts, npz_seg)
    if compute_NSD:
        nsd = compute_multi_class_nsd(gts, npz_seg, spacing)
    if compute_NSD:
        return npz_name, dsc, nsd
    else:
        return npz_name, dsc

if __name__ == '__main__':
    seg_metrics = OrderedDict()
    seg_metrics['case'] = []
    for k, v in label_dict.items():
        seg_metrics[f"{v}_DSC"] = []
    if compute_NSD:
        for k, v in label_dict.items():
            seg_metrics[f"{v}_NSD"] = []
    
    npz_names = listdir(gt_dir)
    npz_names = [npz_name for npz_name in npz_names if npz_name.endswith('.npz')]
    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(npz_names)) as pbar:
            if compute_NSD:
                for i, (npz_name, dsc, nsd) in enumerate(pool.imap_unordered(compute_metrics, npz_names)):
                    seg_metrics['case'].append(npz_name)
                    for k, v in dsc.items():
                        seg_metrics[f"{k}_DSC"].append(np.round(v, 4))
                    for k, v in nsd.items():
                        seg_metrics[f"{k}_NSD"].append(np.round(v, 4))
                    pbar.update()
            else:
                for i, (npz_name, dsc) in enumerate(pool.imap_unordered(compute_metrics, npz_names)):
                    for k, v in dsc.items():
                        seg_metrics[f"{k}_DSC"].append(np.round(v, 4))
                    pbar.update()

    df = pd.DataFrame(seg_metrics)
    df.to_csv(join(csv_dir, 'metrics.csv'), index=False)

    ## make summary csv
    df_dsc = df[["case"] + [f"{v}_DSC" for k, v in label_dict.items()]].copy()
    df_dsc_mean = df_dsc[[f"{v}_DSC" for k, v in label_dict.items()]].mean()
    df_dsc_mean.to_csv(join(csv_dir, 'dsc_summary.csv'))

    if compute_NSD:
        df_nsd = df[["case"] + [f"{v}_NSD" for k, v in label_dict.items()]].copy()
        df_nsd_mean = df_nsd[[f"{v}_NSD" for k, v in label_dict.items()]].mean()
        df_nsd_mean.to_csv(join(csv_dir, 'nsd_summary.csv'))