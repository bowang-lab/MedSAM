# File: src/model/predictor.py
#!/usr/bin/env python3
"""
High-level predictor operating on src.imgpipe.image.Image objects.

MULTI-GPU SUPPORT NOTE
---------------------
This module supports multi-GPU in the "one process per GPU" model, which is the
most reliable approach for inference.

- Pass device="cuda:<id>" to YoloPredictorConfig and MedSamPredictorConfig from
  each worker process.
- Do NOT attempt to run YOLO/MedSAM across multiple GPUs inside a single process
  in this module; shard work at the script level.

This file *does* include speed knobs (AMP/TF32/cudnn benchmark) and faster resize.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image as PILImage
from ultralytics import YOLO

from src.imgpipe.image import Image
from src.imgpipe.normalized_box import NormalizedBox
from src.imgpipe.binary_mask_ref import BinaryMaskRef
from src.imgpipe.enums import Structure, LabelType


# =========================
# Utilities
# =========================

def _xyxy_from_norm(nbox: NormalizedBox, W: int, H: int) -> Tuple[float, float, float, float]:
    return nbox.to_pixel_xyxy(W, H)

# Add near the top (imports already include torch)
def _cuda_index_from_device_str(device_str: str) -> Optional[int]:
    ds = (device_str or "").strip().lower()
    if ds.isdigit():
        return int(ds)
    if ds.startswith("cuda:"):
        try:
            return int(ds.split(":", 1)[1])
        except Exception:
            return None
    if ds == "cuda":
        # Means "use current cuda device"
        return torch.cuda.current_device() if torch.cuda.is_available() else None
    return None


def _ultralytics_device_str(device_str: str) -> str:
    """
    Ultralytics is most reliable with numeric device strings ("0","1",...).
    """
    if not torch.cuda.is_available():
        return device_str
    idx = _cuda_index_from_device_str(device_str)
    if idx is None:
        return device_str
    # Clamp to visible devices (CUDA_VISIBLE_DEVICES may remap)
    n = torch.cuda.device_count()
    if n > 0:
        idx = max(0, min(idx, n - 1))
    return str(idx)

def _pad_xyxy_box(
    xyxy: Tuple[float, float, float, float],
    W: int,
    H: int,
    pad_frac: float,
) -> Tuple[float, float, float, float]:
    """
    Pad a pixel-space box by `pad_frac` of its width/height on each side.
    Clamps to image bounds.
    """
    x1, y1, x2, y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    f = float(max(0.0, min(1.0, pad_frac)))
    dx = f * w
    dy = f * h
    x1p = max(0.0, x1 - dx)
    y1p = max(0.0, y1 - dy)
    x2p = min(float(W), x2 + dx)
    y2p = min(float(H), y2 + dy)
    if x2p <= x1p or y2p <= y1p:
        return (x1, y1, x2, y2)
    return (x1p, y1p, x2p, y2p)


def _select_torch_device(device_str: str) -> torch.device:
    """
    Defensive device selection:
    - respects explicit cuda:<id> if available
    - supports mps
    - falls back to cpu
    """
    ds = (device_str or "").strip().lower()

    if ds.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                return torch.device(device_str)
            except Exception:
                return torch.device("cuda:0")
        return torch.device("cpu")

    if ds == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if ds == "cpu":
        return torch.device("cpu")

    # default-ish
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =========================
# YOLO predictor
# =========================

@dataclass
class YoloPredictorConfig:
    weights: Path
    device: str = "cuda:0"
    imgsz: int = 640
    conf: float = 0.001
    iou: float = 0.70
    batch_size: int = 16  # max images per YOLO call


class YoloPredictor:
    """
    Ultralytics YOLO wrapper that returns at most 1 detection per class (0=disc, 1=cup),
    filtered by `conf` and with NMS IoU=`iou` during inference.

    Multi-GPU: set cfg.device to "cuda:<id>" in each worker process.
    """

    def __init__(self, cfg: YoloPredictorConfig) -> None:
        self.cfg = cfg
        if not cfg.weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {cfg.weights}")

        # Pin this process to the intended CUDA device (critical under torchrun)
        if torch.cuda.is_available():
            idx = _cuda_index_from_device_str(cfg.device)
            if idx is not None:
                n = torch.cuda.device_count()
                if n > 0:
                    idx = max(0, min(idx, n - 1))
                torch.cuda.set_device(idx)

        self.model = YOLO(str(cfg.weights))

    @torch.inference_mode()
    def _top1_for_result(
        self,
        res,
    ) -> Dict[int, Tuple[Optional[NormalizedBox], Optional[float]]]:
        if res.boxes is None or len(res.boxes) == 0:
            return {0: (None, None), 1: (None, None)}

        b = res.boxes
        xywh = b.xywh.cpu().numpy().astype(np.float32)
        cls = b.cls.cpu().numpy().astype(np.int64)
        conf = b.conf.cpu().numpy().astype(np.float32)

        H, W = res.orig_shape
        xywhn = xywh.copy()
        xywhn[:, 0] /= W
        xywhn[:, 1] /= H
        xywhn[:, 2] /= W
        xywhn[:, 3] /= H

        out: Dict[int, Tuple[Optional[NormalizedBox], Optional[float]]] = {
            0: (None, None),
            1: (None, None),
        }
        for cls_id in (0, 1):
            idx = np.where(cls == cls_id)[0]
            if idx.size == 0:
                continue
            j = idx[np.argmax(conf[idx])]
            xc, yc, w, h = map(float, xywhn[j, :4])
            out[cls_id] = (NormalizedBox(xc, yc, w, h), float(conf[j]))
        return out

    @torch.inference_mode()
    def top1_per_class_batch(self, img_paths: List[Path]) -> List[Dict[int, Tuple[Optional[NormalizedBox], Optional[float]]]]:
        if not img_paths:
            return []

        batch_size = max(1, int(self.cfg.batch_size))
        all_out: List[Dict[int, Tuple[Optional[NormalizedBox], Optional[float]]]] = []

        ultra_device = _ultralytics_device_str(self.cfg.device)

        for start in range(0, len(img_paths), batch_size):
            chunk = img_paths[start:start + batch_size]
            res_list = self.model.predict(
                source=[str(p) for p in chunk],
                device=ultra_device,      # <---- changed
                imgsz=self.cfg.imgsz,
                conf=self.cfg.conf,
                iou=self.cfg.iou,
                verbose=False,
            )
            for res in res_list:
                all_out.append(self._top1_for_result(res))

        return all_out

    @torch.inference_mode()
    def top1_per_class(
        self,
        img_path: Path,
    ) -> Dict[int, Tuple[Optional[NormalizedBox], Optional[float]]]:
        res_list = self.top1_per_class_batch([img_path])
        return res_list[0] if res_list else {0: (None, None), 1: (None, None)}


# =========================
# MedSAM predictor
# =========================

@dataclass
class MedSamPredictorConfig:
    checkpoint: Path
    device: str = "cuda:0"
    model_type: str = "vit_b"
    mask_threshold: float = 0.5

    # Speed / CUDA knobs (safe defaults)
    use_amp: bool = True
    allow_tf32: bool = True
    cudnn_benchmark: bool = True

    # Resize backend: cv2 is much faster than skimage
    resize_backend: str = "cv2"  # "cv2" or "pil"


class MedSamPredictor:
    """
    Lightweight wrapper around MedSAM (ViT-B) for multi-box prompting.
    Reuses a single image embedding and iterates per box.

    Multi-GPU: set cfg.device to "cuda:<id>" in each worker process.
    """

    def __init__(self, cfg: MedSamPredictorConfig) -> None:
        from segment_anything import sam_model_registry  # type: ignore

        self.cfg = cfg
        if not cfg.checkpoint.exists():
            raise FileNotFoundError(f"MedSAM checkpoint not found: {cfg.checkpoint}")

        self.device = _select_torch_device(cfg.device)

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
            torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
            torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)

        self.model = sam_model_registry[cfg.model_type](checkpoint=str(cfg.checkpoint))
        self.model = self.model.to(self.device).eval()

    def _autocast_ctx(self):
        from contextlib import nullcontext
        if self.device.type == "cuda" and self.cfg.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _resize_to_1024(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        """
        Fast resize to 1024x1024.
        - prefers OpenCV if available & requested
        - falls back to PIL
        """
        if self.cfg.resize_backend == "cv2":
            try:
                import cv2  # type: ignore
                return cv2.resize(img_rgb_u8, (1024, 1024), interpolation=cv2.INTER_CUBIC)
            except Exception:
                pass

        im = PILImage.fromarray(img_rgb_u8)
        im = im.resize((1024, 1024), resample=PILImage.BICUBIC)
        return np.asarray(im)

    @torch.inference_mode()
    def embed_image(self, img_rgb: np.ndarray) -> Tuple[torch.Tensor, int, int]:
        """
        Compute MedSAM image embedding for a single RGB image.

        Faster than skimage.resize; uses /255 normalization (stable + fast).
        """
        H, W = img_rgb.shape[:2]

        if img_rgb.ndim == 2:
            img_rgb = np.repeat(img_rgb[..., None], 3, axis=-1)
        if img_rgb.shape[2] != 3:
            img_rgb = img_rgb[:, :, :3]

        # ensure uint8 before resizing for speed
        if img_rgb.dtype != np.uint8:
            img_rgb_u8 = np.clip(img_rgb, 0, 255).astype(np.uint8)
        else:
            img_rgb_u8 = img_rgb

        img_1024 = self._resize_to_1024(img_rgb_u8).astype(np.float32) / 255.0  # HWC float32 [0,1]

        img_t = (
            torch.from_numpy(img_1024)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, non_blocking=True)
        )

        with self._autocast_ctx():
            emb = self.model.image_encoder(img_t)  # (1,256,64,64)
        return emb, H, W

    @torch.inference_mode()
    def infer_masks_for_boxes(
        self,
        image_embedding: torch.Tensor,
        boxes_xyxy_pixel: np.ndarray,
        H: int,
        W: int,
    ) -> List[np.ndarray]:
        import torch.nn.functional as F

        if boxes_xyxy_pixel is None:
            return []
        boxes_xyxy_pixel = np.asarray(boxes_xyxy_pixel, dtype=np.float32).reshape(-1, 4)
        if boxes_xyxy_pixel.size == 0:
            return []

        masks: List[np.ndarray] = []
        assert image_embedding.dim() == 4 and image_embedding.shape[0] == 1, (
            f"Expected image_embedding with batch=1, got shape {tuple(image_embedding.shape)}"
        )

        # precompute scale to 1024 once
        scale = np.array([1024.0 / W, 1024.0 / H, 1024.0 / W, 1024.0 / H], dtype=np.float32)

        for k in range(boxes_xyxy_pixel.shape[0]):
            box_1024 = boxes_xyxy_pixel[k: k + 1] * scale  # (1,4)
            box_t = torch.as_tensor(box_1024, dtype=torch.float32, device=image_embedding.device)
            box_t = box_t[:, None, :]  # -> (1,1,4)

            with self._autocast_ctx():
                sparse, dense = self.model.prompt_encoder(points=None, boxes=box_t, masks=None)
                low_res_logits, _ = self.model.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse,
                    dense_prompt_embeddings=dense,
                    multimask_output=False,
                )
                prob = torch.sigmoid(low_res_logits)
                up = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)

            mask = (up.squeeze().float().cpu().numpy() > self.cfg.mask_threshold).astype(np.uint8) * 255
            masks.append(mask)

        return masks


# =========================
# High-level Predictor
# =========================

@dataclass
class PredictorConfig:
    """
    Configuration for the high-level Predictor.
    Controls geometric aspects of inference (box padding).
    """
    box_pad_frac: float = 0.05


class Predictor:

    def __init__(
        self,
        yolo_cfg: YoloPredictorConfig,
        sam_cfg: MedSamPredictorConfig,
        pred_cfg: PredictorConfig,
    ) -> None:
        self.yolo = YoloPredictor(yolo_cfg)
        self.sam = MedSamPredictor(sam_cfg)
        self.cfg = pred_cfg
        self.box_pad_frac = float(max(0.0, min(1.0, pred_cfg.box_pad_frac)))

    @torch.inference_mode()
    def _predict_one(
        self,
        img: Image,
        det: Optional[Dict[int, Tuple[Optional[NormalizedBox], Optional[float]]]] = None,
    ) -> None:
        if det is None:
            det = self.yolo.top1_per_class(img.image_path)

        disc_box, disc_conf = det.get(0, (None, None))
        cup_box, cup_conf = det.get(1, (None, None))

        img.set_box(Structure.DISC, LabelType.PRED, disc_box)
        img.set_box(Structure.CUP,  LabelType.PRED, cup_box)

        if hasattr(img, "yolo_disc_conf"):
            img.yolo_disc_conf = disc_conf
        if hasattr(img, "yolo_cup_conf"):
            img.yolo_cup_conf = cup_conf

        # Load RGB safely (ensures file handle is closed)
        with PILImage.open(img.image_path) as im:
            rgb = np.array(im.convert("RGB"))

        emb, H, W = self.sam.embed_image(rgb)

        boxes_xyxy: List[Tuple[float, float, float, float]] = []
        order_map: List[int] = []  # 0=disc, 1=cup

        if disc_box is not None:
            xyxy = _xyxy_from_norm(disc_box, W, H)
            boxes_xyxy.append(_pad_xyxy_box(xyxy, W, H, self.box_pad_frac))
            order_map.append(0)

        if cup_box is not None:
            xyxy = _xyxy_from_norm(cup_box, W, H)
            boxes_xyxy.append(_pad_xyxy_box(xyxy, W, H, self.box_pad_frac))
            order_map.append(1)

        masks_by_class: Dict[int, Optional[np.ndarray]] = {0: None, 1: None}
        if boxes_xyxy:
            boxes_arr = np.asarray(boxes_xyxy, dtype=np.float32)
            masks = self.sam.infer_masks_for_boxes(emb, boxes_arr, H, W)
            for k, cls_id in enumerate(order_map):
                masks_by_class[cls_id] = masks[k] if k < len(masks) else None

        pdm = masks_by_class[0]
        pcm = masks_by_class[1]

        if pdm is not None:
            img.set_mask(Structure.DISC, LabelType.PRED, BinaryMaskRef(array=(pdm > 0)))

        if pcm is not None:
            img.set_mask(Structure.CUP, LabelType.PRED, BinaryMaskRef(array=(pcm > 0)))

        img.ensure_boxes_from_masks()

    @torch.inference_mode()
    def predict(self, images: List[Image]) -> List[Image]:
        if not images:
            return images

        dets = self.yolo.top1_per_class_batch([img.image_path for img in images])
        for img, det in zip(images, dets):
            self._predict_one(img, det=det)

        return images