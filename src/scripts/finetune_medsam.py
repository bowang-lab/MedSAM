#!/usr/bin/env python3
# src/scripts/finetune_medsam.py

from __future__ import annotations

import os
import json
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Iterable
from contextlib import nullcontext

import numpy as np
from PIL import Image as PILImage, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from src.imgpipe.image_factory import ImageFactory
from src.imgpipe.image import Image as IMG
from src.imgpipe.normalized_box import NormalizedBox
from src.imgpipe.enums import Structure, LabelType
import src.training_utils as tu
from src.model.predictor import YoloPredictor, YoloPredictorConfig, _ultralytics_device_str

try:
    from segment_anything import sam_model_registry
except Exception as _e:
    sam_model_registry = None
    _SAM_ERR = _e

try:
    import cv2
except Exception as _e:
    cv2 = None
    _CV2_ERR = _e


def _ensure_cv2_available():
    if cv2 is None:
        raise RuntimeError(
            f"OpenCV (cv2) not available: {_CV2_ERR!r}. "
            "Install: pip install opencv-python"
        )


# Global defaults
DEFAULT_MODEL = "vit_b"
DEFAULT_IMGSZ = 1024
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 8
DEFAULT_WORKERS = 8
DEFAULT_LR = 1e-4
DEFAULT_WD = 1e-4
SEED = 42

DEFAULT_YOLO_DEVICE = "cuda:0"
DEFAULT_YOLO_IMGSZ = 640
DEFAULT_YOLO_CONF = 0.25
DEFAULT_YOLO_IOU = 0.50

DEFAULT_USE_DET_PROB = 0.5
DEFAULT_PAD_JITTER = 0.30
DEFAULT_BOX_TR = 0.05
DEFAULT_BOX_SC = 0.10

from src.utils import set_global_seed


def _ensure_sam_available():
    if sam_model_registry is None:
        raise RuntimeError(
            f"segment-anything not available: {_SAM_ERR!r}. "
            "Install: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )


def _to_device(x: Any, device: torch.device) -> Any:
    return tu.to_device(x, device)


@dataclass
class DistConfig:
    backend: str = "nccl"
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    def __post_init__(self):
        # Auto-detect SLURM vs torchrun
        if "SLURM_NTASKS" in os.environ:
            self.world_size = int(os.environ["SLURM_NTASKS"])
            self.rank = int(os.environ["SLURM_PROCID"])
            self.local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
        elif "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if self.distributed and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


class DistributedContext:
    def __init__(self, cfg: DistConfig):
        self.cfg = cfg
        self._initialized = False

    def setup(self):
        if self.cfg.distributed:
            if torch.cuda.is_available():
                torch.cuda.set_device(self.cfg.local_rank)

            torch.distributed.init_process_group(
                backend=self.cfg.backend,
                init_method=self.cfg.init_method,
                world_size=self.cfg.world_size,
                rank=self.cfg.rank,
            )
            self._initialized = True

    def cleanup(self):
        if self._initialized:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            self._initialized = False

    @property
    def is_main(self) -> bool:
        return self.cfg.rank == 0

    def barrier(self):
        if self.cfg.distributed:
            torch.distributed.barrier()

    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.cfg.distributed:
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return tensor


def build_image_index(data_root: Path, exclude: Optional[List[str]] = None) -> Dict[str, IMG]:
    fac = ImageFactory(root=data_root, auto_scan=True)
    fac.filter_empty_masks()
    if exclude:
        fac.filter_datasets(exclude=exclude)
    images = fac.make_images()
    idx: Dict[str, IMG] = {Path(im.image_path).stem: im for im in images}
    if not idx:
        raise RuntimeError("No images discovered by ImageFactory.")
    return idx


def images_from_yolo_splits(
        yolo_ds: Path, data_root: Path, exclude: Optional[List[str]] = None
) -> Tuple[List[IMG], List[IMG], List[IMG]]:
    mapping = tu.parse_data_yaml(yolo_ds / "data.yaml")
    train_imgs = tu.resolve_split_images(yolo_ds, mapping["train"])
    val_imgs = tu.resolve_split_images(yolo_ds, mapping["val"])
    test_imgs = tu.resolve_split_images(yolo_ds, mapping["test"])

    idx = build_image_index(data_root, exclude)
    miss = 0

    def to_images(ps: List[Path]) -> List[IMG]:
        nonlocal miss
        out: List[IMG] = []
        for p in ps:
            im = idx.get(Path(p).stem)
            if im is None:
                miss += 1
                continue
            im.set_split(None)
            out.append(im)
        return out

    tr, va, te = to_images(train_imgs), to_images(val_imgs), to_images(test_imgs)
    if miss:
        print(f"[WARN] {miss} YOLO-split images not found in ImageFactory index; skipped.")
    print(f"[INFO] YOLO split sizes → train={len(tr)} val={len(va)} test={len(te)}")
    return tr, va, te


def attach_detector_boxes_with_cache(
        images: Iterable[IMG],
        cache_path: Optional[Path],
        dist: DistributedContext,
        provider: Optional[YoloPredictor] = None,
) -> None:
    images = list(images)
    mapping: Dict[str, Dict[str, Optional[Tuple[float, float, float, float]]]] = {}

    if cache_path and cache_path.exists():
        for line in cache_path.read_text().splitlines():
            rec = json.loads(line)
            mapping[rec["stem"]] = {"disc": rec.get("disc"), "cup": rec.get("cup")}
    else:
        if provider is None:
            if dist.is_main:
                print("[INFO] No provider and no cache; skipping detector attachment.")
        else:
            if dist.is_main:
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    jf = cache_path.open("w")
                else:
                    jf = open(os.devnull, "w")

                chunk_size = provider.cfg.batch_size if provider else 32

                with jf:
                    for i in tqdm(range(0, len(images), chunk_size), desc="Generating Detector Prompts"):
                        chunk = images[i: i + chunk_size]
                        paths = [Path(im.image_path) for im in chunk]
                        batch_preds = provider.top1_per_class_batch(paths)

                        for im, preds in zip(chunk, batch_preds):
                            rec = {"stem": Path(im.image_path).stem}
                            disc_res = preds.get(0, (None, None))
                            cup_res = preds.get(1, (None, None))
                            rec["disc"] = disc_res[0].as_tuple() if disc_res[0] else None
                            rec["cup"] = cup_res[0].as_tuple() if cup_res[0] else None
                            jf.write(json.dumps(rec) + "\n")
                            mapping[rec["stem"]] = {"disc": rec["disc"], "cup": rec["cup"]}

        dist.barrier()
        if not mapping and cache_path and cache_path.exists():
            for line in cache_path.read_text().splitlines():
                rec = json.loads(line)
                mapping[rec["stem"]] = {"disc": rec.get("disc"), "cup": rec.get("cup")}

    for im in images:
        stem = Path(im.image_path).stem
        m = mapping.get(stem)
        if not m:
            continue
        if m.get("disc") is not None:
            im.set_box(Structure.DISC, LabelType.PRED, NormalizedBox(*m["disc"]))
        if m.get("cup") is not None:
            im.set_box(Structure.CUP, LabelType.PRED, NormalizedBox(*m["cup"]))


def cartesian_to_polar(
        img: np.ndarray, out_h: int, out_w: int,
        center: Optional[Tuple[float, float]] = None, radius: Optional[float] = None,
) -> np.ndarray:
    _ensure_cv2_available()
    h, w = img.shape[:2]
    if center is None:
        center = (w / 2.0, h / 2.0)
    if radius is None:
        cx, cy = center
        corners = [(0, 0), (w, 0), (0, h), (w, h)]
        radius = max(np.sqrt((cx - x) ** 2 + (cy - y) ** 2) for x, y in corners)
        radius = min(radius, max(w, h))
    flags = cv2.WARP_POLAR_LINEAR
    polar = cv2.warpPolar(img, (out_w, out_h), center, radius, flags)
    polar = cv2.rotate(polar, cv2.ROTATE_90_CLOCKWISE)
    return polar


def polar_to_cartesian(
        img_polar: np.ndarray, out_h: int, out_w: int,
        center: Optional[Tuple[float, float]] = None, radius: Optional[float] = None,
) -> np.ndarray:
    _ensure_cv2_available()
    if center is None:
        center = (out_w / 2.0, out_h / 2.0)
    if radius is None:
        cx, cy = center
        corners = [(0, 0), (out_w, 0), (0, out_h), (out_w, out_h)]
        radius = max(np.sqrt((cx - x) ** 2 + (cy - y) ** 2) for x, y in corners)
        radius = min(radius, max(out_w, out_h))
    p = cv2.rotate(img_polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    flags = cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP
    cart = cv2.warpPolar(p, (out_w, out_h), center, radius, flags)
    return cart


def compute_disc_center_and_radius(
        disc_box_xyxy: np.ndarray, img_w: int, img_h: int, radius_padding: float = 1.5,
) -> Tuple[Tuple[float, float], float]:
    x1, y1, x2, y2 = disc_box_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    disc_w = x2 - x1
    disc_h = y2 - y1
    min_radius = max(disc_w, disc_h) / 2.0 * 1.2
    max_possible = min(cx, cy, img_w - cx, img_h - cy) * 0.95
    desired_radius = max(disc_w, disc_h) / 2.0 * radius_padding
    radius = max(min_radius, min(desired_radius, max_possible))
    return (cx, cy), radius


def make_joint_images(images: List[IMG]) -> List[IMG]:
    out: List[IMG] = []
    for im in images:
        if (
                im.gt_disc_mask is not None
                and getattr(im.gt_disc_mask, "path", None)
                and im.gt_cup_mask is not None
                and getattr(im.gt_cup_mask, "path", None)
        ):
            out.append(im)
    return out


def create_fundu_dataset(
        images: List[IMG], img_size: int, train: bool, prompt_mode: str,
        polar_radius_padding: float, use_det_prob: float = 0.0,
        pad_jitter: float = 0.0, box_tr: float = 0.0, box_sc: float = 0.0,
) -> FunduSAMDataset:
    return FunduSAMDataset(
        images=images, img_size=img_size, train=train,
        use_det_prob=use_det_prob, pad_jitter=pad_jitter,
        box_tr=box_tr, box_sc=box_sc, prompt_mode=prompt_mode,
        polar_radius_padding=polar_radius_padding,
    )


def fundu_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [sample[key] for sample in batch]
        if key == "meta":
            meta_keys = values[0].keys()
            result[key] = {mk: [v[mk] for v in values] for mk in meta_keys}
        elif isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values, dim=0)
        else:
            result[key] = values
    return result


class FunduSAMDataset(Dataset):
    def __init__(
            self, images: List[IMG], img_size: int = DEFAULT_IMGSZ, train: bool = True,
            use_det_prob: float = DEFAULT_USE_DET_PROB, pad_jitter: float = DEFAULT_PAD_JITTER,
            box_tr: float = DEFAULT_BOX_TR, box_sc: float = DEFAULT_BOX_SC,
            prompt_mode: str = "mix", polar_radius_padding: float = 1.5,
    ):
        _ensure_cv2_available()
        self.images = images
        self.img_size = img_size
        self.train = train
        self.use_det_prob = float(np.clip(use_det_prob, 0.0, 1.0))
        self.pad_jitter = pad_jitter
        self.box_tr = box_tr
        self.box_sc = box_sc
        self.prompt_mode = prompt_mode
        self.polar_radius_padding = polar_radius_padding
        self.letterbox = tu.LetterboxToSquare(img_size)

    def __len__(self) -> int:
        return len(self.images)

    def _choose_disc_box(self, im: IMG) -> Optional[NormalizedBox]:
        gt = im.gt_disc_box
        det = im.inter_pred_disc_box
        if self.train and self.prompt_mode == "mix":
            return det if (random.random() < self.use_det_prob and det is not None) else gt
        if self.prompt_mode == "det":
            return det if det is not None else gt
        return gt

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        im: IMG = self.images[idx]
        im.ensure_boxes_from_masks()

        img = PILImage.open(im.image_path).convert("RGB")
        mref_disc = im.gt_disc_mask
        mref_cup = im.gt_cup_mask
        if mref_disc is None or getattr(mref_disc, "path", None) is None:
            raise RuntimeError(f"Missing GT disc mask: {im.image_path}")
        if mref_cup is None or getattr(mref_cup, "path", None) is None:
            raise RuntimeError(f"Missing GT cup mask: {im.image_path}")

        m_disc = PILImage.open(mref_disc.path).convert("L")
        m_cup = PILImage.open(mref_cup.path).convert("L")

        if self.train:
            img = tu.color_jitter_safe(img, brightness=0.10, contrast=0.10)

        base_nbox = self._choose_disc_box(im)
        if base_nbox is None:
            tb = tu.mask_to_tight_box(np.array(m_disc, dtype=np.uint8))
            box_xyxy = tb if tb is not None else np.array([0, 0, img.size[0] - 1, img.size[1] - 1], dtype=np.float32)
        else:
            box_xyxy = tu.nbox_to_xyxy(base_nbox, im.width, im.height)

        pad_frac = (random.uniform(0.0, self.pad_jitter) if self.train else 0.0)
        box_p = tu.pad_box(box_xyxy, pad_frac, im.width, im.height)
        if self.train and (self.box_tr > 0.0 or self.box_sc > 0.0):
            box_p = tu.jitter_box_xyxy(box_p, im.width, im.height, tr=self.box_tr, sc=self.box_sc)

        img_cart, disc_cart, box_t = self.letterbox(img, m_disc, box_p)
        _, cup_cart, _ = self.letterbox(img, m_cup, box_p)

        polar_center, polar_radius = compute_disc_center_and_radius(
            box_t, self.img_size, self.img_size, self.polar_radius_padding,
        )

        img_np = np.array(img_cart)
        disc_np = np.array(disc_cart, dtype=np.uint8)
        cup_np = np.array(cup_cart, dtype=np.uint8)

        img_pol = cartesian_to_polar(img_np, out_h=self.img_size, out_w=self.img_size, center=polar_center,
                                     radius=polar_radius)
        disc_pol = cartesian_to_polar(disc_np, out_h=self.img_size, out_w=self.img_size, center=polar_center,
                                      radius=polar_radius)
        cup_pol = cartesian_to_polar(cup_np, out_h=self.img_size, out_w=self.img_size, center=polar_center,
                                     radius=polar_radius)

        img_pol_pil = PILImage.fromarray(img_pol)
        x = tu.preprocess_for_sam(img_pol_pil)

        disc_bin = (disc_pol > 0).astype(np.float32)
        cup_bin = (cup_pol > 0).astype(np.float32)
        y = torch.from_numpy(np.stack([disc_bin, cup_bin], axis=0))

        b = torch.from_numpy(box_t.astype(np.float32))
        meta = {
            "image_path": str(im.image_path),
            "stem": Path(im.image_path).stem,
            "polar_center": polar_center,
            "polar_radius": polar_radius,
        }
        return {"image": x, "mask": y, "box": b, "meta": meta}


class FunduJointLoss(nn.Module):
    def __init__(self, w_disc: float = 1.0, w_cup: float = 2.0, w_contain: float = 1.0, bce_weight: float = 0.5,
                 smooth: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w_disc = w_disc
        self.w_cup = w_cup
        self.w_contain = w_contain
        self.bce_weight = bce_weight
        self.smooth = smooth

    def _dice_loss(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num = 2.0 * (prob * target).sum(dim=(1, 2)) + self.smooth
        den = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + self.smooth
        dice = num / den
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_disc = logits[:, 0]
        log_cup = logits[:, 1]
        gt_disc = target[:, 0]
        gt_cup = target[:, 1]

        bce_disc = self.bce(log_disc, gt_disc)
        bce_cup = self.bce(log_cup, gt_cup)

        prob_disc = torch.sigmoid(log_disc)
        prob_cup = torch.sigmoid(log_cup)
        dice_disc = self._dice_loss(prob_disc, gt_disc)
        dice_cup = self._dice_loss(prob_cup, gt_cup)

        L_disc = self.bce_weight * bce_disc + (1.0 - self.bce_weight) * dice_disc
        L_cup = self.bce_weight * bce_cup + (1.0 - self.bce_weight) * dice_cup
        L_contain = torch.mean(prob_cup * (1.0 - prob_disc))

        return self.w_disc * L_disc + self.w_cup * L_cup + self.w_contain * L_contain


class Adapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 32):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(bottleneck, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class FunduBlock(nn.Module):
    def __init__(self, block: nn.Module, dim: int, bottleneck: int = 32):
        super().__init__()
        self.block = block
        self.adapter_after_attn = Adapter(dim, bottleneck)
        self.adapter_after_ffn = Adapter(dim, bottleneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.block
        x = x + b.attn(b.norm1(x))
        x = self.adapter_after_attn(x)
        x = x + b.mlp(b.norm2(x))
        x = self.adapter_after_ffn(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        reduced = max(channels // ratio, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=False)
        max_ = torch.amax(x, dim=(2, 3), keepdim=False)
        attn = self.mlp(avg) + self.mlp(max_)
        attn = self.sigmoid(attn).unsqueeze(-1).unsqueeze(-1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, max_], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, ratio)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FunduSegHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int = 2, width: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, width, kernel_size=1, bias=False)
        self.block1 = ConvBlock(width, width)
        self.block2 = ConvBlock(width, width)
        self.block3 = ConvBlock(width, width)
        self.out_conv = nn.Conv2d(width, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        logits = self.out_conv(x)
        return logits


class FunduSAMFinetuner(nn.Module):
    def __init__(
            self, sam_type: str, checkpoint: Path, freeze_encoders: bool = True,
            adapter_bottleneck: int = 64, num_classes: int = 2, adapt_all_blocks: bool = True,
            last_n_adapted: int = 12, seg_width: int = 128, cbam_ratio: int = 16, cbam_kernel_size: int = 7,
    ):
        super().__init__()
        _ensure_sam_available()
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"MedSAM checkpoint not found: {checkpoint}")

        sam = sam_model_registry[sam_type](checkpoint=str(checkpoint))
        enc = sam.image_encoder

        if hasattr(enc, "blocks") and len(enc.blocks) > 0:
            blk0 = enc.blocks[0]
            mlp = getattr(blk0, "mlp", None)
            dim = None
            if mlp and hasattr(mlp, "fc1"):
                dim = mlp.fc1.in_features
            elif mlp and hasattr(mlp, "linear1"):
                dim = mlp.linear1.in_features
            elif mlp:
                for m in mlp.modules():
                    if isinstance(m, nn.Linear):
                        dim = m.in_features
                        break
            if dim is None: raise RuntimeError("Could not infer token dimension.")
        else:
            raise RuntimeError("Unexpected SAM structure.")

        n_blocks = len(enc.blocks)
        wrapped_blocks = []
        for i, b in enumerate(enc.blocks):
            if adapt_all_blocks or i >= n_blocks - max(1, last_n_adapted):
                wrapped_blocks.append(FunduBlock(b, dim=dim, bottleneck=adapter_bottleneck))
            else:
                wrapped_blocks.append(b)
        enc.blocks = nn.ModuleList(wrapped_blocks)
        self.image_encoder = enc

        out_chans = getattr(enc, "out_chans", 256)
        self.cbam = CBAM(out_chans, ratio=cbam_ratio, kernel_size=cbam_kernel_size)
        self.seg_head = FunduSegHead(in_ch=out_chans, num_classes=num_classes, width=seg_width)

        if freeze_encoders:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for blk in self.image_encoder.blocks:
                if isinstance(blk, FunduBlock):
                    for m in [blk.adapter_after_attn, blk.adapter_after_ffn]:
                        for p in m.parameters(): p.requires_grad = True

        for p in self.cbam.parameters(): p.requires_grad = True
        for p in self.seg_head.parameters(): p.requires_grad = True

        self.use_grad_checkpointing = False

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = batch["image"]

        # Manual Gradient Checkpointing Implementation
        if self.use_grad_checkpointing:
            # 1. Patch Embedding
            x = self.image_encoder.patch_embed(x)
            if self.image_encoder.pos_embed is not None:
                x = x + self.image_encoder.pos_embed

            # 2. Blocks with Checkpointing
            for blk in self.image_encoder.blocks:
                # use_reentrant=False is the modern safe way to use checkpointing
                x = checkpoint(blk, x, use_reentrant=False)

            # 3. Neck
            feat = self.image_encoder.neck(x.permute(0, 3, 1, 2))
        else:
            feat = self.image_encoder(x)

        feat = self.cbam(feat)
        logits_low = self.seg_head(feat)
        logits = F.interpolate(logits_low, size=(batch["image"].shape[2], batch["image"].shape[3]), mode="bilinear",
                               align_corners=False)
        return logits, None


MedSAMFinetuner = FunduSAMFinetuner


@dataclass
class TrainConfig:
    data_root: Path
    yolo_ds: Path
    out_dir: Path
    run_dir: Path
    run_name: str
    yolo_weights: Path
    exclude_datasets: Optional[List[str]] = None
    yolo_device: str = DEFAULT_YOLO_DEVICE
    yolo_imgsz: int = DEFAULT_YOLO_IMGSZ
    yolo_conf: float = DEFAULT_YOLO_CONF
    yolo_iou: float = DEFAULT_YOLO_IOU
    det_cache: Optional[Path] = None
    model: str = DEFAULT_MODEL
    ckpt: Path = Path("")
    unfreeze_encoders: bool = False
    save_full: bool = False
    epochs: int = DEFAULT_EPOCHS
    batch: int = DEFAULT_BATCH
    imgsz: int = DEFAULT_IMGSZ
    lr: float = DEFAULT_LR
    wd: float = DEFAULT_WD
    workers: int = DEFAULT_WORKERS
    amp: bool = True
    grad_acc_steps: int = 1
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    seed: int = SEED
    use_det_prob: float = DEFAULT_USE_DET_PROB
    pad_jitter: float = DEFAULT_PAD_JITTER
    box_jitter_tr: float = DEFAULT_BOX_TR
    box_jitter_sc: float = DEFAULT_BOX_SC
    polar_radius_padding: float = 1.5
    do_train: bool = True
    do_test: bool = True
    test_prompt: str = "det"
    test_weights: Optional[Path] = None
    resume: Optional[Path] = None
    compile_model: bool = False
    eval_every: int = 1
    grad_checkpointing: bool = False


class Trainer:
    def __init__(self, cfg: TrainConfig, dist: DistributedContext):
        self.cfg = cfg
        self.dist = dist
        self.device = dist.cfg.device()
        self.run_path = Path(cfg.run_dir) / cfg.run_name
        self.weights_dir = self.run_path / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.weights_dir / "best.pth"
        self.last_path = self.weights_dir / "last.pth"
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

        train_imgs, val_imgs, test_imgs = images_from_yolo_splits(
            cfg.yolo_ds, cfg.data_root, exclude=cfg.exclude_datasets
        )

        provider = None
        if self.dist.is_main:
            yolo_device_str = _ultralytics_device_str(str(self.device))
            provider = YoloPredictor(
                YoloPredictorConfig(
                    weights=cfg.yolo_weights,
                    device=yolo_device_str,
                    imgsz=cfg.yolo_imgsz,
                    conf=cfg.yolo_conf,
                    iou=cfg.yolo_iou,
                    batch_size=32
                )
            )

        det_cache_path = cfg.det_cache if cfg.det_cache else (self.run_path / "detector_cache.jsonl")
        attach_detector_boxes_with_cache(
            train_imgs + val_imgs + test_imgs,
            det_cache_path,
            self.dist,
            provider=provider,
        )

        del provider
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        joint_train = make_joint_images(train_imgs)
        joint_val = make_joint_images(val_imgs)
        joint_test = make_joint_images(test_imgs)

        self.ds_train = create_fundu_dataset(
            images=joint_train, img_size=cfg.imgsz, train=True,
            prompt_mode="mix", polar_radius_padding=cfg.polar_radius_padding,
            use_det_prob=cfg.use_det_prob, pad_jitter=cfg.pad_jitter,
            box_tr=cfg.box_jitter_tr, box_sc=cfg.box_jitter_sc,
        )
        self.ds_val = create_fundu_dataset(
            images=joint_val, img_size=cfg.imgsz, train=False,
            prompt_mode="gt", polar_radius_padding=cfg.polar_radius_padding,
        )
        self.ds_test = create_fundu_dataset(
            images=joint_test, img_size=cfg.imgsz, train=False,
            prompt_mode=("det" if cfg.test_prompt == "det" else "gt"),
            polar_radius_padding=cfg.polar_radius_padding,
        )

        self.sampler_train = (
            DistributedSampler(self.ds_train, num_replicas=dist.cfg.world_size, rank=dist.cfg.rank, shuffle=True)
            if dist.cfg.distributed else None
        )
        self.sampler_val = (
            DistributedSampler(self.ds_val, num_replicas=dist.cfg.world_size, rank=dist.cfg.rank, shuffle=False)
            if dist.cfg.distributed else None
        )

        self.dl_train = DataLoader(
            self.ds_train, batch_size=cfg.batch, shuffle=(self.sampler_train is None),
            num_workers=cfg.workers, pin_memory=True, drop_last=False,
            sampler=self.sampler_train, collate_fn=fundu_collate_fn,
            persistent_workers=(cfg.workers > 0), prefetch_factor=(2 if cfg.workers > 0 else None)
        )
        self.dl_val = DataLoader(
            self.ds_val, batch_size=max(1, cfg.batch // 2), shuffle=False,
            num_workers=cfg.workers, pin_memory=True, sampler=self.sampler_val,
            collate_fn=fundu_collate_fn,
            persistent_workers=(cfg.workers > 0), prefetch_factor=(2 if cfg.workers > 0 else None)
        )
        self.dl_test = DataLoader(
            self.ds_test, batch_size=max(1, cfg.batch // 2), shuffle=False,
            num_workers=cfg.workers, pin_memory=True, collate_fn=fundu_collate_fn,
        )

        self.model = FunduSAMFinetuner(
            sam_type=cfg.model, checkpoint=cfg.ckpt, freeze_encoders=not cfg.unfreeze_encoders,
        ).to(self.device)

        # Gradient Checkpointing Activation
        if cfg.grad_checkpointing:
            if isinstance(self.model, FunduSAMFinetuner):
                self.model.use_grad_checkpointing = True
                if self.dist.is_main:
                    print("[INFO] Gradient checkpointing enabled (manual wrapping).")

        if cfg.compile_model and hasattr(torch, "compile"):
            if self.dist.is_main:
                print("[INFO] Compiling model with torch.compile...")
            self.model.image_encoder = torch.compile(self.model.image_encoder)

        if dist.cfg.distributed:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[dist.cfg.local_rank], output_device=dist.cfg.local_rank,
                gradient_as_bucket_view=True, find_unused_parameters=cfg.find_unused_parameters,
                bucket_cap_mb=cfg.bucket_cap_mb,
            )

        self.criterion = FunduJointLoss(w_disc=1.0, w_cup=2.0, w_contain=1.0)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.wd,
        )

        use_cuda = self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.cfg.amp and use_cuda))

        if cfg.resume and Path(cfg.resume).exists():
            self._load_resume(cfg.resume)
            if self.dist.is_main:
                print(f"[INFO] Resumed from {cfg.resume}")

        self.best_val = -1.0
        self.history: Dict[str, List[float]] = {"train_loss": [], "train_dice": [], "val_dice": []}

    def _unwrap(self) -> FunduSAMFinetuner:
        return self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model

    def _save_ckpt(self, path: Path) -> None:
        mod = self._unwrap()
        state = {"model": mod.state_dict()}
        torch.save(state, str(path))

    def _load_resume(self, path: Path) -> None:
        state = torch.load(str(path), map_location="cpu")
        mod = self._unwrap()
        if "model" in state:
            mod.load_state_dict(state["model"], strict=True)
        elif "sam" in state:
            own_state = mod.state_dict()
            sam_state = state["sam"]
            sam_state = {k: v for k, v in sam_state.items() if k in own_state}
            own_state.update(sam_state)
            mod.load_state_dict(own_state, strict=False)
        else:
            mod.load_state_dict(state, strict=False)

    def _run_epoch(self, train: bool) -> Dict[str, float]:
        model = self.model
        crit = self.criterion
        opt = self.optimizer
        dl = self.dl_train if train else self.dl_val
        if self.sampler_train and train:
            self.sampler_train.set_epoch(self._epoch)

        model.train(mode=train)
        total_loss = 0.0
        total_dice = 0.0
        n = 0

        grad_acc = max(1, self.cfg.grad_acc_steps)
        use_cuda = self.device.type == "cuda"
        amp_enabled = self.cfg.amp and use_cuda

        iterator = dl
        if self.dist.is_main:
            desc = f"Epoch {self._epoch} {'Train' if train else 'Val'}"
            iterator = tqdm(dl, desc=desc, leave=False)

        for step, batch in enumerate(iterator):
            batch = {k: v for k, v in batch.items() if k != "meta"}
            batch = tu.to_device(batch, self.device)

            if use_cuda:
                autocast_ctx = torch.cuda.amp.autocast(enabled=amp_enabled)
            else:
                autocast_ctx = nullcontext()

            with torch.set_grad_enabled(train):
                with autocast_ctx:
                    logits, _ = model(batch)
                    loss = crit(logits, batch["mask"])

                if train:
                    if amp_enabled:
                        self.scaler.scale(loss / grad_acc).backward()
                    else:
                        (loss / grad_acc).backward()

                    if (step + 1) % grad_acc == 0:
                        if amp_enabled:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            opt.step()
                        opt.zero_grad(set_to_none=True)

            if self.dist.is_main and train:
                iterator.set_postfix(loss=float(loss.detach()))

            with torch.no_grad():
                prob = torch.sigmoid(logits)
                for i in range(prob.shape[0]):
                    d_disc = tu.dice_coef_prob(prob[i, 0:1], batch["mask"][i, 0:1])
                    d_cup = tu.dice_coef_prob(prob[i, 1:2], batch["mask"][i, 1:2])
                    total_dice += 0.5 * (d_disc + d_cup)
                    n += 1
                total_loss += float(loss.detach()) * batch["mask"].shape[0]

        loss_tensor = torch.tensor([total_loss, float(n)], device=self.device)
        dice_tensor = torch.tensor([total_dice, float(n)], device=self.device)
        loss_tensor = self.dist.all_reduce_sum(loss_tensor)
        dice_tensor = self.dist.all_reduce_sum(dice_tensor)

        loss_mean = (loss_tensor[0].item() / max(1.0, loss_tensor[1].item()))
        dice_mean = (dice_tensor[0].item() / max(1.0, dice_tensor[1].item()))
        return {"loss": loss_mean, "dice": dice_mean}

    def fit(self):
        if not self.cfg.do_train:
            return
        if self.dist.is_main:
            print("[INFO] Starting training…")
        for epoch in range(1, self.cfg.epochs + 1):
            self._epoch = epoch
            tr = self._run_epoch(train=True)

            va_dice = 0.0
            if epoch % self.cfg.eval_every == 0 or epoch == self.cfg.epochs:
                va = self._run_epoch(train=False)
                va_dice = va["dice"]
                if self.dist.is_main:
                    print(
                        f"[E{epoch:03d}] loss={tr['loss']:.4f} | dice(tr)={tr['dice']:.4f} | dice(val)={va['dice']:.4f}")
                    self._save_ckpt(self.last_path)
                    if va["dice"] > self.best_val:
                        self.best_val = va["dice"]
                        self._save_ckpt(self.best_path)
            else:
                if self.dist.is_main:
                    print(f"[E{epoch:03d}] loss={tr['loss']:.4f} | dice(tr)={tr['dice']:.4f} | (val skipped)")

            self.history["train_loss"].append(tr["loss"])
            self.history["train_dice"].append(tr["dice"])
            if va_dice > 0: self.history["val_dice"].append(va_dice)

            self.dist.barrier()

        if self.dist.is_main:
            metrics_path = Path(self.cfg.out_dir) / f"{self.cfg.run_name}_metrics.json"
            out = {"history": self.history, "best_val_dice": float(self.best_val)}
            metrics_path.write_text(json.dumps(out, indent=2))
            print(f"[OK] Training complete. Best val Dice={self.best_val:.4f} -> {self.best_path}")

    def test(self) -> Dict[str, Any]:
        if not self.cfg.do_test:
            return {}
        if not self.dist.is_main:
            self.dist.barrier()
            return {}

        ckpt = (
            self.cfg.test_weights
            if self.cfg.test_weights
            else (self.best_path if self.best_path.exists() else self.last_path)
        )
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint found for testing: {ckpt}")
        self._load_resume(ckpt)

        self._unwrap().eval()
        out_dir = Path(self.cfg.out_dir)
        disc_dir = out_dir / "pred_masks" / "disc"
        cup_dir = out_dir / "pred_masks" / "cup"
        disc_dir.mkdir(parents=True, exist_ok=True)
        cup_dir.mkdir(parents=True, exist_ok=True)
        jl_path = out_dir / "test_predictions.jsonl"

        summary = {"disc_sum": 0.0, "cup_sum": 0.0, "disc_n": 0, "cup_n": 0}
        with jl_path.open("w") as jf, torch.no_grad():
            for batch in self.dl_test:
                metas = batch["meta"]
                batch_tensors = {k: v for k, v in batch.items() if k != "meta"}
                batch_tensors = tu.to_device(batch_tensors, self.device)
                logits, _ = self._unwrap()(batch_tensors)
                prob = torch.sigmoid(logits)

                for i in range(prob.shape[0]):
                    stem = metas["stem"][i]
                    image_path = metas["image_path"][i]
                    polar_center = metas["polar_center"][i]
                    polar_radius = metas["polar_radius"][i]

                    d_disc = tu.dice_coef_prob(prob[i, 0:1], batch_tensors["mask"][i, 0:1])
                    d_cup = tu.dice_coef_prob(prob[i, 1:2], batch_tensors["mask"][i, 1:2])

                    disc_pol = (prob[i, 0].cpu().numpy() >= 0.5).astype(np.uint8) * 255
                    cup_pol = (prob[i, 1].cpu().numpy() >= 0.5).astype(np.uint8) * 255

                    disc_cart = polar_to_cartesian(disc_pol, out_h=self.cfg.imgsz, out_w=self.cfg.imgsz,
                                                   center=polar_center, radius=polar_radius)
                    cup_cart = polar_to_cartesian(cup_pol, out_h=self.cfg.imgsz, out_w=self.cfg.imgsz,
                                                  center=polar_center, radius=polar_radius)

                    disc_cart = disc_cart.astype(np.uint8)
                    cup_cart = cup_cart.astype(np.uint8)

                    save_disc = disc_dir / f"{stem}.png"
                    save_cup = cup_dir / f"{stem}.png"
                    PILImage.fromarray(disc_cart).save(str(save_disc))
                    PILImage.fromarray(cup_cart).save(str(save_cup))

                    rec = {
                        "image": image_path, "stem": stem, "dice_disc": float(d_disc), "dice_cup": float(d_cup),
                        "prompt": self.cfg.test_prompt,
                        "polar_center": list(polar_center) if isinstance(polar_center, tuple) else polar_center,
                        "polar_radius": float(polar_radius), "mask_disc_path": str(save_disc),
                        "mask_cup_path": str(save_cup),
                    }
                    jf.write(json.dumps(rec) + "\n")

                    summary["disc_sum"] += float(d_disc)
                    summary["cup_sum"] += float(d_cup)
                    summary["disc_n"] += 1
                    summary["cup_n"] += 1

        disc_mean = summary["disc_sum"] / max(1, summary["disc_n"])
        cup_mean = summary["cup_sum"] / max(1, summary["cup_n"])
        overall = (summary["disc_sum"] + summary["cup_sum"]) / max(1, summary["disc_n"] + summary["cup_n"])
        test_summary = {
            "disc_mean_dice": disc_mean, "cup_mean_dice": cup_mean,
            "overall_mean_dice": overall, "counts": {"disc": summary["disc_n"], "cup": summary["cup_n"]},
            "checkpoint": str(ckpt),
        }
        metrics_path = Path(self.cfg.out_dir) / f"{self.cfg.run_name}_metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
        else:
            m = {}
        m["test"] = test_summary
        metrics_path.write_text(json.dumps(m, indent=2))
        print(f"[OK] Test summary: {test_summary}")
        self.dist.barrier()
        return test_summary


def run_finetune(cfg: TrainConfig) -> Dict[str, Any]:
    dist_cfg = DistConfig()
    dist = DistributedContext(dist_cfg)
    set_global_seed(cfg.seed + dist_cfg.rank)
    dist.setup()
    try:
        trainer = Trainer(cfg, dist)
        trainer.fit()
        result = trainer.test()
        return {"best_val_dice": trainer.best_val, "test": result}
    finally:
        dist.cleanup()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Fundu-style MedSAM (multi-GPU ready) with adapters+CBAM, polar coords, joint OD/OC loss."
    )
    p.add_argument("--data-root", type=Path, required=True, help="Root datasets scanned by ImageFactory.")
    p.add_argument("--yolo-ds", type=Path, required=True, help="YOLO dataset directory containing data.yaml.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for metrics/preds.")
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory for checkpoints.")
    p.add_argument("--run-name", type=str, default="FunduSAMFinetune")
    p.add_argument("--exclude-ds", nargs="*", default=None, help="Datasets to exclude by substring.")
    p.add_argument("--yolo-weights", type=Path, required=True, help="Path to trained YOLO weights.")
    p.add_argument("--yolo-device", type=str, default=DEFAULT_YOLO_DEVICE)
    p.add_argument("--yolo-imgsz", type=int, default=DEFAULT_YOLO_IMGSZ)
    p.add_argument("--yolo-conf", type=float, default=DEFAULT_YOLO_CONF)
    p.add_argument("--yolo-iou", type=float, default=DEFAULT_YOLO_IOU)
    p.add_argument("--det-cache", type=Path, default=None)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--ckpt", type=Path, required=True, help="MedSAM checkpoint.")
    p.add_argument("--unfreeze-encoders", action="store_true")
    p.add_argument("--save-full", action="store_true")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--wd", type=float, default=DEFAULT_WD)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-acc-steps", type=int, default=1)
    p.add_argument("--bucket-cap-mb", type=int, default=25)
    p.add_argument("--find-unused-parameters", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--use-det-prob", type=float, default=DEFAULT_USE_DET_PROB)
    p.add_argument("--pad-jitter", type=float, default=DEFAULT_PAD_JITTER)
    p.add_argument("--box-jitter-tr", type=float, default=DEFAULT_BOX_TR)
    p.add_argument("--box-jitter-sc", type=float, default=DEFAULT_BOX_SC)
    p.add_argument("--polar-radius-padding", type=float, default=1.5)
    p.add_argument("--train", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--test-prompt", choices=["det", "gt"], default="det")
    p.add_argument("--test-weights", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    # New Performance Flags
    p.add_argument("--compile-model", action="store_true",
                   help="Compile model with torch.compile (requires PyTorch 2.0+)")
    p.add_argument("--eval-every", type=int, default=1, help="Run validation every N epochs")
    p.add_argument("--grad-checkpointing", action="store_true",
                   help="Enable gradient checkpointing for lower VRAM usage")

    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(
        data_root=args.data_root, yolo_ds=args.yolo_ds, out_dir=args.out_dir, run_dir=args.run_dir,
        run_name=args.run_name,
        yolo_weights=args.yolo_weights, exclude_datasets=args.exclude_ds, yolo_device=args.yolo_device,
        yolo_imgsz=args.yolo_imgsz,
        yolo_conf=args.yolo_conf, yolo_iou=args.yolo_iou, det_cache=args.det_cache, model=args.model, ckpt=args.ckpt,
        unfreeze_encoders=args.unfreeze_encoders, save_full=args.save_full, epochs=args.epochs, batch=args.batch,
        imgsz=args.imgsz,
        lr=args.lr, wd=args.wd, workers=args.workers, amp=args.amp or True, grad_acc_steps=args.grad_acc_steps,
        bucket_cap_mb=args.bucket_cap_mb, find_unused_parameters=args.find_unused_parameters, seed=args.seed,
        use_det_prob=args.use_det_prob, pad_jitter=args.pad_jitter, box_jitter_tr=args.box_jitter_tr,
        box_jitter_sc=args.box_jitter_sc,
        polar_radius_padding=args.polar_radius_padding, do_train=args.train, do_test=args.test,
        test_prompt=args.test_prompt,
        test_weights=args.test_weights, resume=args.resume,
        compile_model=args.compile_model, eval_every=args.eval_every, grad_checkpointing=args.grad_checkpointing
    )
    res = run_finetune(cfg)
    if DistConfig().rank == 0:
        print(json.dumps({"result": res, "config": asdict(cfg)}, indent=2))


if __name__ == "__main__":
    main()