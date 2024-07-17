# %%
import os
import glob
import random
import shutil
import argparse
from os import listdir, makedirs
from os.path import join, exists, isfile, isdir
from tqdm import tqdm
from copy import deepcopy
from time import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry
import cv2
import torch.nn.functional as F
from tiny_vit_sam import TinyViT

# %%
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

# %% Dataset class
class NpyDataset(Dataset): 
    def __init__(self, data_root, data_aug=True):
        self.data_root = data_root
        self.gt_path = join(data_root, 'gts')
        self.img_path = join(data_root, 'imgs')
        self.gt_path_files = sorted(glob.glob(join(self.gt_path, '**/*.npy'), recursive=True))
        self.gt_path_files = [file for file in self.gt_path_files if os.path.isfile(join(self.img_path, os.path.basename(file)))]
        self.data_aug = data_aug
        print(f'number of images: {len(self.gt_path_files)}')
    
    def __len__(self):
        return len(self.gt_path_files)
    
    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_3c = np.load(join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)
        img_1024 = cv2.resize(
            img_3c, (1024, 1024), interpolation=cv2.INTER_CUBIC
        )
        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert np.max(img_1024)<=1.0 and np.min(img_1024)>=0.0, 'image should be normalized to [0, 1]'

        img_256 = cv2.resize(
            img_3c, (256, 256), interpolation=cv2.INTER_AREA
        )
        img_256 = (img_256 - img_256.min()) / np.clip(
            img_256.max() - img_256.min(), a_min=1e-8, a_max=None
        )
        # convert the shape to (3, H, W)
        img_256 = np.transpose(img_256, (2, 0, 1))
        assert np.max(img_256)<=1.0 and np.min(img_256)>=0.0, 'image should be normalized to [0, 1]'

        if self.data_aug:
            if random.random() > 0.5:
                img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-1))
                img_256 = np.ascontiguousarray(np.flip(img_256, axis=-1))
                # print('DA with flip left right')
            if random.random() > 0.5:
                img_1024 = np.ascontiguousarray(np.flip(img_1024, axis=-2))
                img_256 = np.ascontiguousarray(np.flip(img_256, axis=-2))
                # print('DA with flip up down')

        return torch.tensor(img_1024).float(), torch.tensor(img_256).float(), img_name

# %%
class Logger(object):
    def __init__(self):
        self.logging = {
            'train_losses': [],
            'lrs': [],
            'epoch_start_timestamps': [],
            'epoch_end_timestamps': []
        }

    def log(self, key, value):
        self.logging[key].append(value)
    
    def plot_progress_png(self, output_folder):
        fig, ax_all = plt.subplots(2, 1, figsize=(10, 8))
        x_values = [i for i in range(1, len(self.logging['train_losses']) + 1)]
        ax_all[0].plot(
            x_values,
            self.logging['train_losses'],
            #color='b',
            ls='-',
            label="loss_tr",
        )
        ax_all[0].set_ylabel("loss", fontsize=10)
        ax_all[0].legend(loc=(0, 1))

        ax_all[1].plot(x_values, [i - j for i, j in zip(self.logging['epoch_end_timestamps'],
                                                 self.logging['epoch_start_timestamps'])],
                                                 #color='b',
                ls='-',
                label="epoch duration",
                )
        ax_all[1].set_xlabel("epoch", fontsize=10)
        ax_all[1].set_ylabel("time [s]", fontsize=10)
        ax_all[1].legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()

    def get_checkpoint(self):
        return self.logging

    def load_checkpoint(self, checkpoint: dict):
        self.logging = checkpoint

# %% set up parser
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--tr_npy_path', type=str,
                        default='',
                        help='path to training npy files; two subfolders: gts and imgs')
    parser.add_argument('-task_name', type=str, default='MedSAM-Lite')
    parser.add_argument('-medsam_checkpoint', type=str, default=None,
                        help='path to MedSAM encoder checkpoint')
    parser.add_argument('-tinyvit_checkpoint', type=str, default=None,
                        help='path to TinyViT encoder checkpoint (not required)')
    parser.add_argument('-work_dir', type=str, default='./work_dir',
                        help='path to save the model checkpoints and logs')
    parser.add_argument('--data_aug', type=bool, default=True, 
                        help='use data augmentation during training')
    # train
    parser.add_argument('-num_epochs', type=int, default=1000)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-num_workers', type=int, default=8)
    # Optimizer parameters
    parser.add_argument('-weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    parser.add_argument('-lr', type=float, default=0.00005, metavar='LR',
                        help='learning rate (absolute lr)')
    ## Distributed training args
    parser.add_argument('-world_size', type=int, help='world size')
    parser.add_argument('-node_rank', type=int, help='Node rank')
    parser.add_argument('-bucket_cap_mb', type = int, default = 25,
                        help='The amount of memory in Mb that DDP will accumulate before firing off gradient communication for the bucket (need to tune)')
    parser.add_argument('-resume', type = str, default = None,
                        help="Resuming training from checkpoint folder (only required when resuming training)")
    parser.add_argument('-init_method', type = str, default = "env://")

    args = parser.parse_args()

    return args

# %%
def revert_sync_batchnorm(module: torch.nn.Module) -> torch.nn.Module:
    # Code adapted from https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547
    # Original author: Kapil Yedidi (@kapily)
    converted_module = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        # Unfortunately, SyncBatchNorm does not store the original class - if it did
        # we could return the one that was originally created.
        converted_module = nn.BatchNorm2d(
            module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats
        )
        if module.affine:
            with torch.no_grad():
                converted_module.weight = module.weight
                converted_module.bias = module.bias
        converted_module.running_mean = module.running_mean
        converted_module.running_var = module.running_var
        converted_module.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            converted_module.qconfig = module.qconfig
    for name, child in module.named_children():
        converted_module.add_module(name, revert_sync_batchnorm(child))
    del module

    return converted_module

# %%
def main(args):
    ngpus_per_node = torch.cuda.device_count()
    print("Spwaning processces")
    mp.spawn(main_worker, nprocs = ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    node_rank = int(args.node_rank)
    rank = node_rank * ngpus_per_node + gpu
    world_size = args.world_size
    print(f"[Rank {rank}]: Use GPU: {gpu} for training")
    is_main_host = rank == 0
    
    if is_main_host:
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        model_save_path = join(args.work_dir, args.task_name + '-' + run_id)
        os.makedirs(model_save_path, exist_ok=True)
        shutil.copyfile(__file__, join(model_save_path, run_id + '_' + os.path.basename(__file__)))
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(
        backend = "nccl",
        init_method = args.init_method,
        rank = rank,
        world_size = world_size
    )

    medsam_model = sam_model_registry["vit_b"](checkpoint=args.medsam_checkpoint)
    torch.distributed.barrier()

    teacher_model = deepcopy(medsam_model.image_encoder)
    teacher_model.eval()
    del medsam_model
    torch.cuda.empty_cache()
    for param in teacher_model.parameters():
        param.requires_grad = False ## freeze MedSAM teacher model
    teacher_model.to(gpu)

    student_model = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[
            64,
            128,
            160,
            320
        ],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8
    ) ## TinyViT setup

    if args.tinyvit_checkpoint is not None:
        pretrained_student_ckpt = torch.load(
            args.tinyvit_checkpoint,
            map_location="cpu"
        )
        student_model.load_state_dict(pretrained_student_ckpt)
        torch.distributed.barrier()

    student_model.to(gpu)
    #print("Compiling student model...")
    #student_model = torch.compile(student_model)

    for module in student_model.modules():
        cls_name = module.__class__.__name__
        if "BatchNorm" in cls_name:
            assert cls_name == "BatchNorm2d" ## Make sure there's only 2d BN layers, so that I can revert them properly

    student_model = nn.SyncBatchNorm.convert_sync_batchnorm(student_model)
    student_model = nn.parallel.DistributedDataParallel(
        student_model,
        device_ids = [gpu],
        output_device = gpu,
        gradient_as_bucket_view = True,
        find_unused_parameters = False,
        bucket_cap_mb = args.bucket_cap_mb
    )
    # %%
    print(f"MedSAM encoder size: {sum(p.numel() for p in teacher_model.parameters())}")
    print(f"TinyViT encoder size: {sum(p.numel() for p in student_model.parameters())}")
    # %%
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.weight_decay
    )

    grad_scaler = GradScaler()
    logger = Logger()

    if args.resume is not None:
        ckpt_folders = sorted(os.listdir(args.resume))
        ckpt_folders = [f for f in ckpt_folders if (f.startswith(args.task_name) and os.path.isfile(os.path.join(args.resume, f, 'medsam_lite_encoder_latest.pth')))]
        print('*'*20)
        print('existing ckpts in', args.resume, ckpt_folders)
        # find the latest ckpt folders
        time_strings = [f.split(args.task_name + '-')[-1] for f in ckpt_folders]
        dates = [datetime.strptime(f, '%Y%m%d-%H%M') for f in time_strings]
        latest_date = max(dates)
        latest_ckpt = os.path.join(args.work_dir, args.task_name + '-' + latest_date.strftime('%Y%m%d-%H%M'), 'medsam_lite_encoder_latest.pth')
        try:
            ## Map model to be loaded to specified single GPU
            # loc = 'cuda:{}'.format(gpu)
            # checkpoint = torch.load(latest_ckpt, map_location = loc)
            checkpoint = torch.load(latest_ckpt)
            student_model.module.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_loss = checkpoint["best_loss"]
            start_epoch = checkpoint["epoch"] + 1
            if "logger" in checkpoint:
                logger.load_checkpoint(checkpoint["logger"])
            if "grad_scaler" in checkpoint:
                grad_scaler.load_state_dict(checkpoint["grad_scaler"])

            # medsam_model.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print(rank, "=> loaded checkpoint '{}' (epoch {})".format(latest_ckpt, checkpoint['epoch']))
        except:
            print(rank, "=> no checkpoint found at '{}'".format(latest_ckpt))
            exit()
        torch.distributed.barrier()

    # if args.resume is not None:
    #     checkpoint = torch.load(args.resume)
    #     student_model.module.load_state_dict(checkpoint["model"])
    #     optimizer.load_state_dict(checkpoint["optimizer"])
    #     best_loss = checkpoint["best_loss"]
    #     start_epoch = checkpoint["epoch"] + 1
    #     if "logger" in checkpoint:
    #         logger.load_checkpoint(checkpoint["logger"])
    #     if "grad_scaler" in checkpoint:
    #         grad_scaler.load_state_dict(checkpoint["grad_scaler"])
    #     torch.distributed.barrier()
    else:
        best_loss = 1e10
        start_epoch = 0

    # %%
    train_dataset = NpyDataset(data_root=args.tr_npy_path, data_aug=args.data_aug)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = [None for _ in range(len(train_loader))]
        logger.log('lrs', optimizer.param_groups[0]['lr'])
        logger.log('epoch_start_timestamps', time())
        pbar = tqdm(train_loader)
        for step, (teacher_input, student_input, image_name) in enumerate(pbar):
            teacher_input = teacher_input.to(gpu)
            student_input = student_input.to(gpu)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_output = teacher_model(teacher_input)
            target = teacher_output.detach().clone()
            with autocast():
                student_output = student_model(student_input)
                loss = F.mse_loss(student_output, target)
            epoch_loss[step] = loss.item()
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            pbar.set_description(f"[RANK {rank}: GPU {gpu}] Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log('epoch_end_timestamps', time())
        epoch_loss_world = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(epoch_loss_world, epoch_loss)
        epoch_loss_reduced = np.vstack(epoch_loss_world).mean()
        logger.log('train_losses', epoch_loss_reduced)
        if is_main_host:
            module_revert_sync_BN = revert_sync_batchnorm(deepcopy(student_model.module))
            weights = module_revert_sync_BN.state_dict()
            checkpoint = {
                "epoch": epoch,
                "model": weights,
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "logger": logger.get_checkpoint(),
                "grad_scaler": grad_scaler.state_dict()
            }
            torch.save(
                checkpoint,
                join(model_save_path, "medsam_lite_encoder_latest.pth")
            )
            if epoch_loss_reduced < best_loss:
                print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
                best_loss = epoch_loss_reduced
                torch.save(
                    checkpoint,
                    join(model_save_path, "medsam_lite_encoder_best.pth")
                )
            logger.plot_progress_png(model_save_path)
        epoch_loss_reduced = 0
        torch.distributed.barrier()

# %%
if __name__ == "__main__":
    args = get_args()
    main(args)
