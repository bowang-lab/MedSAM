# File: src/odcup/predictor.py

#!/usr/bin/env python3
"""
High-level predictor operating on src.imgpipe.image.Image objects.

This module provides:
- YoloPredictorConfig, MedSamPredictorConfig, PredictorConfig
- YoloPredictor, MedSamPredictor
- Predictor: runs YOLO + MedSAM on a list of Image objects, mutating them
  in place and returning the list of Image objects.
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


class YoloPredictor:
    """
    Ultralytics YOLO wrapper that returns at most 1 detection per class (0=disc, 1=cup),
    filtered by `conf` and with NMS IoU=`iou` during inference.
    """

    def __init__(self, cfg: YoloPredictorConfig) -> None:
        self.cfg = cfg
        if not cfg.weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {cfg.weights}")
        self.model = YOLO(str(cfg.weights))

    @torch.no_grad()
    def top1_per_class(
        self,
        img_path: Path,
    ) -> Dict[int, Tuple[Optional[NormalizedBox], Optional[float]]]:
        """
        Returns a dict:
            {
                0: (disc_box_norm_or_None, disc_conf_or_None),
                1: (cup_box_norm_or_None,  cup_conf_or_None),
            }
        Boxes are YOLO-normalized (xc,yc,w,h) in [0,1].
        """
        res = self.model.predict(
            source=str(img_path),
            device=self.cfg.device,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            verbose=False,
        )[0]

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


# =========================
# MedSAM predictor
# =========================

@dataclass
class MedSamPredictorConfig:
    checkpoint: Path
    device: str = "cuda:0"
    model_type: str = "vit_b"
    mask_threshold: float = 0.5


class MedSamPredictor:
    """
    Lightweight wrapper around MedSAM (ViT-B) for multi-box prompting.
    Reuses a single image embedding and iterates per box.
    """

    def __init__(self, cfg: MedSamPredictorConfig) -> None:
        from segment_anything import sam_model_registry  # type: ignore

        self.cfg = cfg
        if not cfg.checkpoint.exists():
            raise FileNotFoundError(f"MedSAM checkpoint not found: {cfg.checkpoint}")

        # Device selection
        if cfg.device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif "cuda" in cfg.device and torch.cuda.is_available():
            device = torch.device(cfg.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        self.device = device

        self.model = sam_model_registry[cfg.model_type](checkpoint=str(cfg.checkpoint))
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def embed_image(self, img_rgb: np.ndarray) -> Tuple[torch.Tensor, int, int]:
        """
        Compute MedSAM image embedding for a single RGB image.

        Returns:
            (embedding, H, W)
        """
        from skimage import transform  # type: ignore

        H, W = img_rgb.shape[:2]
        if img_rgb.ndim == 2:
            img_rgb = np.repeat(img_rgb[..., None], 3, axis=-1)

        img_1024 = transform.resize(
            img_rgb, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)

        # Simple [0,1] normalization
        img_1024 = (img_1024 - img_1024.min()) / max(
            1e-8, (img_1024.max() - img_1024.min())
        )
        img_1024_t = (
            torch.tensor(img_1024)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        emb = self.model.image_encoder(img_1024_t)  # (1,256,64,64)
        return emb, H, W

    @torch.no_grad()
    def infer_masks_for_boxes(
        self,
        image_embedding: torch.Tensor,
        boxes_xyxy_pixel: np.ndarray,
        H: int,
        W: int,
    ) -> List[np.ndarray]:
        """
        Prompt MedSAM with a set of pixel-space boxes (xyxy) on an already
        embedded image. Returns a list of uint8 masks (0 or 255).
        """
        import torch.nn.functional as F  # local import

        if boxes_xyxy_pixel is None:
            return []
        boxes_xyxy_pixel = np.asarray(boxes_xyxy_pixel, dtype=np.float32).reshape(-1, 4)
        if boxes_xyxy_pixel.size == 0:
            return []

        masks: List[np.ndarray] = []
        assert (
            image_embedding.dim() == 4 and image_embedding.shape[0] == 1
        ), f"Expected image_embedding with batch=1, got shape {tuple(image_embedding.shape)}"

        for k in range(boxes_xyxy_pixel.shape[0]):
            box_xyxy = boxes_xyxy_pixel[k: k + 1]  # (1,4)
            box_1024 = (
                box_xyxy / np.array([W, H, W, H], dtype=np.float32) * 1024.0
            )
            box_t = torch.as_tensor(
                box_1024, dtype=torch.float32, device=image_embedding.device
            )  # (1,4)
            box_t = box_t[:, None, :]  # -> (1,1,4)

            sparse, dense = self.model.prompt_encoder(
                points=None, boxes=box_t, masks=None
            )
            low_res_logits, _ = self.model.mask_decoder(
                image_embeddings=image_embedding,  # (1,256,64,64)
                image_pe=self.model.prompt_encoder.get_dense_pe(),  # (1,256,64,64)
                sparse_prompt_embeddings=sparse,  # (1,?,256)
                dense_prompt_embeddings=dense,  # (1,256,64,64)
                multimask_output=False,
            )
            prob = torch.sigmoid(low_res_logits)
            up = F.interpolate(
                prob, size=(H, W), mode="bilinear", align_corners=False
            )  # (1,1,H,W)
            mask = (up.squeeze().detach().cpu().numpy() >
                    self.cfg.mask_threshold).astype(np.uint8) * 255
            masks.append(mask)

        return masks


# =========================
# High-level Predictor
# =========================

@dataclass
class PredictorConfig:
    """
    Configuration for the high-level Predictor.

    Note: This config is now minimal in terms of behaviour:
    it only controls geometric aspects of inference (box padding).
    """
    box_pad_frac: float = 0.05  # padding applied to YOLO boxes before MedSAM


class Predictor:
    """
    High-level predictor operating directly on `Image` objects.

    Responsibilities:
    - For each `Image`:
        * run YOLO and keep the best detection per class;
        * record YOLO normalized boxes and confidences in the Image;
        * run MedSAM segmentation using padded YOLO boxes as prompts;
        * record predicted masks and final boxes (from masks) in the Image.

    It performs **no** saving, visualization, or run-level summarization.
    """

    def __init__(
        self,
        yolo_cfg: YoloPredictorConfig,
        sam_cfg: MedSamPredictorConfig,
        pred_cfg: PredictorConfig,
    ) -> None:
        self.yolo = YoloPredictor(yolo_cfg)
        self.sam = MedSamPredictor(sam_cfg)
        self.cfg = pred_cfg

        self.box_pad_frac = float(
            max(0.0, min(1.0, pred_cfg.box_pad_frac))
        )

    # ------------- per-image core --------------

    def _predict_one(self, img: Image) -> None:
        """
        Mutates a single Image in place with YOLO boxes, confidences and
        MedSAM masks/boxes. No saving or visualization.
        """
        # --- YOLO: best box per class ---
        det = self.yolo.top1_per_class(img.image_path)
        disc_box, disc_conf = det.get(0, (None, None))
        cup_box, cup_conf = det.get(1, (None, None))

        # Store detector (intermediate) boxes on Image (UNPADDED, normalized)
        img.set_box(Structure.DISC, LabelType.PRED, disc_box)  # -> inter_pred_disc_box
        img.set_box(Structure.CUP, LabelType.PRED, cup_box)    # -> inter_pred_cup_box

        # Store YOLO confidences on Image, if fields exist
        if hasattr(img, "yolo_disc_conf"):
            img.yolo_disc_conf = disc_conf
        if hasattr(img, "yolo_cup_conf"):
            img.yolo_cup_conf = cup_conf

        # --- MedSAM: prepare embedding once ---
        rgb = np.array(PILImage.open(img.image_path).convert("RGB"))
        emb, H, W = self.sam.embed_image(rgb)

        # Prompt boxes in pixel space with padding
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

        # Attach predicted masks to Image (BinaryMaskRef with boolean array)
        pdm = masks_by_class[0]
        pcm = masks_by_class[1]

        if pdm is not None:
            img.set_mask(
                Structure.DISC,
                LabelType.PRED,
                BinaryMaskRef(array=(pdm > 0)),
            )

        if pcm is not None:
            img.set_mask(
                Structure.CUP,
                LabelType.PRED,
                BinaryMaskRef(array=(pcm > 0)),
            )

        # Let Image compute normalized boxes from masks where needed
        img.ensure_boxes_from_masks()

    # ------------- public API --------------

    def predict(self, images: List[Image]) -> List[Image]:
        """
        Run YOLO + MedSAM on a list of Image objects.

        Effects:
        - Each Image is mutated in place to include:
            * inter_pred_disc_box / inter_pred_cup_box (YOLO boxes)
            * pred_disc_mask / pred_cup_mask (MedSAM masks, if any)
            * pred_disc_box / pred_cup_box (from masks where needed)

        Returns:
            The same list of Image objects (for convenience/chainability).
        """
        for img in images:
            self._predict_one(img)
        return images