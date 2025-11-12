#!/usr/bin/env python3
"""
End-to-end optic disc/cup prediction → MedSAM segmentation pipeline.

What's new in this revision
---------------------------
- Factory-only pipeline (always builds Image objects via ImageFactory to ensure GT masks).
- Optional YOLO subset filter: if --yolo-ds is provided, we restrict the factory-built list
  to stems listed under <yolo-ds>/images/<split>.
- Summary now reports, for every category:
    • Means + Standard Deviation (std) + Standard Error (se)
    • For GT↔Pred metric *errors* (cdr_v, cdr_h, rim_over_disc, I_over_S, I_over_N, I_over_T):
        - MAE, RMSE, signed bias (mean error), std, se, and count n
- --summary-only recomputes all summary stats (incl. MAE/RMSE/std/se) from predictions.jsonl
  without rerunning detection/segmentation (reconstructs metrics from saved masks and/or GT boxes).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import torch
from PIL import Image as PILImage
from ultralytics import YOLO

# --- Local project imports ---
from src.imgpipe.image_factory import ImageFactory
from src.imgpipe.image import Image
from src.imgpipe.normalized_box import NormalizedBox
from src.imgpipe.binary_mask_ref import BinaryMaskRef
from src.imgpipe.enums import Structure, LabelType

# =========================
# Utilities
# =========================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _xyxy_from_norm(nbox: NormalizedBox, W: int, H: int) -> Tuple[float, float, float, float]:
    return nbox.to_pixel_xyxy(W, H)

def _pad_xyxy_box(
    xyxy: Tuple[float, float, float, float],
    W: int,
    H: int,
    pad_frac: float
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

def _to_uint8_mask(m: np.ndarray) -> np.ndarray:
    m = (m > 0).astype(np.uint8) * 255
    return m

def _save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image as _PILImage
    _PILImage.fromarray(arr).save(str(path))

def _mask_from_norm_box(box_norm: Optional[Tuple[float, float, float, float]], H: int, W: int) -> Optional[np.ndarray]:
    """
    Rasterize a YOLO-normalized (xc,yc,w,h) box as a uint8 mask (0/255) at image size HxW.
    """
    if not box_norm:
        return None
    nb = NormalizedBox(*map(float, box_norm))
    x1, y1, x2, y2 = nb.to_pixel_xyxy(W, H)
    x1, y1 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
    x2, y2 = int(min(W, np.ceil(x2))), int(min(H, np.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    m = np.zeros((H, W), dtype=np.uint8)
    m[y1:y2, x1:x2] = 255
    return m

def _read_mask_path(p: Optional[str]) -> Optional[np.ndarray]:
    if not p:
        return None
    try:
        arr = np.array(PILImage.open(p).convert("L"))
        return (arr > 0).astype(np.uint8) * 255
    except Exception:
        return None

def _box_from_binary_mask(mask: Optional[np.ndarray], W: int, H: int) -> Optional[NormalizedBox]:
    """
    Compute a NormalizedBox from a binary mask (uint8 0/255) at image size (H,W).
    Returns None if mask is empty.
    """
    if mask is None:
        return None
    m = (mask > 0)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    x1, x2 = float(xs.min()), float(xs.max() + 1)
    y1, y2 = float(ys.min()), float(ys.max() + 1)
    return NormalizedBox.from_xyxy(x1, y1, x2, y2, W, H)

def _collect_yolo_split_stems(yolo_ds: Path, splits: List[str]) -> Set[str]:
    """
    Collect filename stems from <yolo_ds>/images/<split>/**/* for provided splits.
    This is used to filter the factory-built Image list efficiently.
    """
    stems: Set[str] = set()
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for split in (s.strip() for s in splits if s.strip()):
        root = yolo_ds / "images" / split
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in valid_exts:
                stems.add(p.stem)
    return stems

def _is_finite(x: Optional[float]) -> bool:
    return (x is not None) and np.isfinite(x)

# ===== Metric helpers (mask-based; mirror Image methods, used for summary_jsonl) =====

def _cdr_from_masks(disc_mask: Optional[np.ndarray], cup_mask: Optional[np.ndarray], axis: str = "vertical") -> Optional[float]:
    if disc_mask is None or cup_mask is None:
        return None
    d = (disc_mask > 0)
    c = (cup_mask > 0)
    if not d.any():
        return None
    ys_d, xs_d = np.nonzero(d)
    ys_c, xs_c = np.nonzero(c)
    if ys_d.size == 0:
        return None
    if axis == "vertical":
        d_extent = float(ys_d.max() - ys_d.min() + 1)
        c_extent = float(ys_c.max() - ys_c.min() + 1) if ys_c.size else 0.0
    else:
        d_extent = float(xs_d.max() - xs_d.min() + 1)
        c_extent = float(xs_c.max() - xs_c.min() + 1) if xs_c.size else 0.0
    return (c_extent / d_extent) if d_extent > 0 else None

def _rim_metrics_from_masks(
    disc_mask: Optional[np.ndarray],
    cup_mask: Optional[np.ndarray],
    *,
    laterality: Optional[str],
) -> Optional[Dict[str, Optional[float]]]:
    if disc_mask is None or cup_mask is None:
        return None
    disc = (disc_mask > 0)
    cup = (cup_mask > 0)
    if not disc.any():
        return None

    H, W = disc.shape
    rim = disc & (~cup)

    disc_area = float(disc.sum())
    rim_area = float(rim.sum())
    r_over_d = rim_area / disc_area if disc_area > 0 else np.nan

    ys, xs = np.nonzero(disc)
    yc = float(ys.mean()) if ys.size else (H / 2.0)
    xc = float(xs.mean()) if xs.size else (W / 2.0)

    superior = (np.arange(H)[:, None] < yc)
    inferior = ~superior

    rim_sup = float((rim & superior).sum())
    rim_inf = float((rim & inferior).sum())

    def _safe(a: float, b: float) -> float:
        return float(a / b) if b > 0 else np.nan

    out: Dict[str, Optional[float]] = {
        "rim_over_disc": r_over_d,
        "I_over_S": _safe(rim_inf, rim_sup),
        "I_over_N": None,
        "I_over_T": None,
    }

    if laterality:
        lat = str(laterality).upper()
        left = (np.arange(W)[None, :] < xc)
        right = ~left
        if lat == "OD":
            nasal, temporal = left, right
        else:
            nasal, temporal = right, left
        rim_nas = float((rim & nasal).sum())
        rim_tem = float((rim & temporal).sum())
        out["I_over_N"] = _safe(rim_inf, rim_nas)
        out["I_over_T"] = _safe(rim_inf, rim_tem)

    return out

# =========================
# MedSAM wrapper
# =========================

class MedSAM:
    """
    Lightweight wrapper around MedSAM (ViT-B) for multi-box prompting.
    Reuses the single-image embedding and iterates per box to satisfy input shape.
    """
    def __init__(self, checkpoint: Path, device: str = "cuda:0", model_type: str = "vit_b") -> None:
        from segment_anything import sam_model_registry  # type: ignore

        if device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif "cuda" in device and torch.cuda.is_available():
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.model = sam_model_registry[model_type](checkpoint=str(checkpoint))
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def embed_image(self, img_rgb: np.ndarray) -> Tuple[torch.Tensor, int, int]:
        from skimage import transform  # type: ignore

        H, W = img_rgb.shape[:2]
        if img_rgb.ndim == 2:
            img_rgb = np.repeat(img_rgb[..., None], 3, axis=-1)
        img_1024 = transform.resize(
            img_rgb, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / max(1e-8, (img_1024.max() - img_1024.min()))
        img_1024_t = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        emb = self.model.image_encoder(img_1024_t)  # (1,256,64,64)
        return emb, H, W

    @torch.no_grad()
    def infer_masks_for_boxes(
            self,
            image_embedding: torch.Tensor,
            boxes_xyxy_pixel: np.ndarray,
            H: int,
            W: int,
            threshold: float = 0.5,
    ) -> List[np.ndarray]:
        import torch.nn.functional as F  # local import

        if boxes_xyxy_pixel is None:
            return []
        boxes_xyxy_pixel = np.asarray(boxes_xyxy_pixel, dtype=np.float32).reshape(-1, 4)
        if boxes_xyxy_pixel.size == 0:
            return []

        masks: List[np.ndarray] = []
        assert image_embedding.dim() == 4 and image_embedding.shape[0] == 1, \
            f"Expected image_embedding with batch=1, got shape {tuple(image_embedding.shape)}"

        for k in range(boxes_xyxy_pixel.shape[0]):
            box_xyxy = boxes_xyxy_pixel[k: k + 1]  # (1,4)
            box_1024 = box_xyxy / np.array([W, H, W, H], dtype=np.float32) * 1024.0
            box_t = torch.as_tensor(box_1024, dtype=torch.float32, device=image_embedding.device)  # (1,4)
            box_t = box_t[:, None, :]  # -> (1,1,4)

            sparse, dense = self.model.prompt_encoder(points=None, boxes=box_t, masks=None)
            low_res_logits, _ = self.model.mask_decoder(
                image_embeddings=image_embedding,  # (1,256,64,64)
                image_pe=self.model.prompt_encoder.get_dense_pe(),  # (1,256,64,64)
                sparse_prompt_embeddings=sparse,  # (1,?,256)
                dense_prompt_embeddings=dense,  # (1,256,64,64)
                multimask_output=False,
            )
            prob = torch.sigmoid(low_res_logits)
            up = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)  # (1,1,H,W)
            mask = (up.squeeze().detach().cpu().numpy() > threshold).astype(np.uint8) * 255
            masks.append(mask)

        return masks

# =========================
# Detector wrapper (YOLO)
# =========================

@dataclass
class DetectorConfig:
    weights: Path
    device: str = "cpu"
    imgsz: int = 640
    conf: float = 0.001
    iou: float = 0.70

class ODCupDetector:
    """
    Ultralytics YOLO wrapper that returns at most 1 detection per class (0,1),
    filtered by `conf` and with NMS IoU=`iou` during inference.
    """
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.model = YOLO(str(cfg.weights))

    @torch.no_grad()
    def top1_per_class(
        self, img_path: Path
    ) -> Dict[int, Tuple[Optional[NormalizedBox], Optional[float]]]:
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

        out: Dict[int, Tuple[Optional[NormalizedBox], Optional[float]]] = {0: (None, None), 1: (None, None)}
        for cls_id in (0, 1):
            idx = np.where(cls == cls_id)[0]
            if idx.size == 0:
                continue
            j = idx[np.argmax(conf[idx])]
            xc, yc, w, h = map(float, xywhn[j, :4])
            out[cls_id] = (NormalizedBox(xc, yc, w, h), float(conf[j]))
        return out

# =========================
# Prediction pipeline
# =========================

@dataclass
class PipelineConfig:
    out_dir: Path
    save_overlays: bool = False
    overlay_dirname: str = "viz"
    mask_dirname: str = "masks"
    records_jsonl: str = "predictions.jsonl"
    summary_json: str = "summary.json"
    per_image_csv: str = "predictions_per_image.csv"

class ODCupPredictor:
    """
    End-to-end detector→MedSAM segmenter operating on Image objects.
    """
    def __init__(
        self,
        det: Optional[ODCupDetector],
        sam: Optional[MedSAM],
        pcfg: PipelineConfig,
        *,
        box_pad_frac: float = 0.05,
    ) -> None:
        self.det = det
        self.sam = sam
        self.pcfg = pcfg
        _ensure_dir(self.pcfg.out_dir)

        self.viz_dir = _ensure_dir(self.pcfg.out_dir / self.pcfg.overlay_dirname)
        self.msk_dir = _ensure_dir(self.pcfg.out_dir / self.pcfg.mask_dirname)
        self.jsonl_path = self.pcfg.out_dir / self.pcfg.records_jsonl
        self.summary_path = self.pcfg.out_dir / self.pcfg.summary_json
        self.per_image_csv_path = self.pcfg.out_dir / self.pcfg.per_image_csv

        self.box_pad_frac = float(max(0.0, min(1.0, box_pad_frac)))

    def _predict_one(self, img: Image) -> Dict[str, Any]:
        assert self.det is not None and self.sam is not None, "Detector and MedSAM must be initialized for prediction."

        # --- Detector: top-1 per class
        det = self.det.top1_per_class(img.image_path)
        disc_box, disc_conf = det.get(0, (None, None))
        cup_box,  cup_conf  = det.get(1, (None, None))

        # Keep detector (intermediate) boxes on the Image (UNPADDED for record/overlay)
        img.set_box(Structure.DISC, LabelType.PRED, disc_box)  # -> inter_pred_disc_box
        img.set_box(Structure.CUP,  LabelType.PRED, cup_box)   # -> inter_pred_cup_box

        # --- MedSAM: prepare embedding once
        rgb = np.array(PILImage.open(img.image_path).convert("RGB"))
        emb, H, W = self.sam.embed_image(rgb)

        # Build joint prompt list (both boxes if available), with padding applied
        boxes_xyxy: List[Tuple[float, float, float, float]] = []
        order_map: List[int] = []  # map back to class ids

        if disc_box is not None:
            xyxy = _xyxy_from_norm(disc_box, W, H)
            boxes_xyxy.append(_pad_xyxy_box(xyxy, W, H, self.box_pad_frac))
            order_map.append(0)

        if cup_box is not None:
            xyxy = _xyxy_from_norm(cup_box, W, H)
            boxes_xyxy.append(_pad_xyxy_box(xyxy, W, H, self.box_pad_frac))
            order_map.append(1)

        masks_by_class: Dict[int, Optional[np.ndarray]] = {0: None, 1: None}
        if len(boxes_xyxy) > 0:
            masks = self.sam.infer_masks_for_boxes(emb, np.asarray(boxes_xyxy, dtype=np.float32), H, W)
            for k, cls_id in enumerate(order_map):
                masks_by_class[cls_id] = masks[k] if k < len(masks) else None

        # Attach predicted masks
        if masks_by_class[0] is not None:
            img.set_mask(Structure.DISC, LabelType.PRED, BinaryMaskRef(array=(masks_by_class[0] > 0)))
        if masks_by_class[1] is not None:
            img.set_mask(Structure.CUP,  LabelType.PRED, BinaryMaskRef(array=(masks_by_class[1] > 0)))

        # Compute final predicted boxes *from masks*
        pdm = masks_by_class[0]
        pcm = masks_by_class[1]
        pred_disc_nb = _box_from_binary_mask(pdm, W, H)
        pred_cup_nb  = _box_from_binary_mask(pcm, W, H)
        img.pred_disc_box = pred_disc_nb
        img.pred_cup_box  = pred_cup_nb

        # Save raw masks
        disc_m_path = self.msk_dir / f"{img.uid}_disc.png"
        cup_m_path  = self.msk_dir / f"{img.uid}_cup.png"
        if pdm is not None:
            _save_png(disc_m_path, _to_uint8_mask(pdm))
        if pcm is not None:
            _save_png(cup_m_path, _to_uint8_mask(pcm))

        # Compute and cache Dice
        dice = img.update_mask_dice(fallback_to_boxes=True)

        # Optional overlay
        if self.pcfg.save_overlays:
            ov_path = self.viz_dir / f"{img.uid}_overlay.png"
            try:
                img.visualize(show=False, save_path=ov_path, dpi=140, mask_alpha=0.7)
            except Exception:
                pass

        # GT mask paths if available
        disc_gt_path = getattr(img.gt_disc_mask, "path", None)
        cup_gt_path  = getattr(img.gt_cup_mask,  "path", None)

        laterality = getattr(img, "laterality", None)
        laterality_str = getattr(laterality, "name", None) if laterality is not None else None

        rec: Dict[str, Any] = {
            "uid": img.uid,
            "dataset": img.dataset,
            "subject_id": img.subject_id,
            "image_path": str(img.image_path),
            "laterality": laterality_str,  # optional
            "detector": {
                "disc": {"conf": disc_conf, "box_norm": (disc_box.as_tuple() if disc_box else None)},
                "cup":  {"conf": cup_conf,  "box_norm": (cup_box.as_tuple()  if cup_box  else None)},
                "conf_th": self.det.cfg.conf,
                "iou_th": self.det.cfg.iou,
            },
            "sam_prompt": {"pad_frac": self.box_pad_frac},
            "pred_masks": {
                "disc_path": str(disc_m_path) if pdm is not None else None,
                "cup_path":  str(cup_m_path)  if pcm is not None else None,
            },
            "pred_boxes_from_masks": {
                "disc": (pred_disc_nb.as_tuple() if pred_disc_nb else None),
                "cup":  (pred_cup_nb.as_tuple()  if pred_cup_nb  else None),
            },
            "gt_boxes": {
                "disc": (img.gt_disc_box.as_tuple() if img.gt_disc_box else None),
                "cup":  (img.gt_cup_box.as_tuple()  if img.gt_cup_box  else None),
            },
            "gt_masks": {
                "disc_path": str(disc_gt_path) if disc_gt_path else None,
                "cup_path":  str(cup_gt_path)  if cup_gt_path  else None,
            },
            "dice": {"disc": dice.get("disc"), "cup": dice.get("cup")},
            "split": img.split,
        }
        return rec

    def predict(self, images: List[Image], limit: Optional[int] = None) -> None:
        assert self.det is not None and self.sam is not None, "Detector and MedSAM must be initialized for prediction."
        if limit is not None:
            images = images[: int(limit)]

        rows_for_csv: List[Tuple[str, Optional[float], Optional[float]]] = []

        with open(self.jsonl_path, "w") as f:
            for img in images:
                rec = self._predict_one(img)
                f.write(json.dumps(rec) + "\n")
                image_name = Path(rec["image_path"]).name
                dd = rec.get("dice", {})
                rows_for_csv.append((image_name, dd.get("disc"), dd.get("cup")))

        with open(self.per_image_csv_path, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["image_name", "dice_disc", "dice_cup"])
            for name, d_disc, d_cup in rows_for_csv:
                writer.writerow([name, "" if d_disc is None else f"{d_disc:.6f}",
                                       "" if d_cup  is None else f"{d_cup:.6f}"])

    # ----------------- Summary statistics (in-memory) -----------------

    @staticmethod
    def _dice_from_masks(pred: Optional[np.ndarray], gt: Optional[np.ndarray]) -> float:
        if pred is None or gt is None:
            return 0.0
        p = (pred > 0).astype(np.uint8)
        g = (gt > 0).astype(np.uint8)
        inter = float((p & g).sum())
        denom = float(p.sum() + g.sum())
        return (2.0 * inter / denom) if denom > 0 else 0.0

    @staticmethod
    def _box_dice(pb: Optional[NormalizedBox], gb: Optional[NormalizedBox]) -> float:
        if pb is None or gb is None:
            return 0.0
        return float(pb.dice(gb))

    @staticmethod
    def _mean_std_se(xs: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
        n = len(xs)
        if n == 0:
            return None, None, None, 0
        mean = float(np.mean(xs))
        std = float(np.std(xs, ddof=1)) if n > 1 else 0.0
        se = (std / float(np.sqrt(n))) if n > 0 else None
        return mean, std, se, n

    @staticmethod
    def _rate_stats(successes: int, n: int) -> Dict[str, Optional[float]]:
        if n <= 0:
            return {"p": None, "std": None, "se": None, "n": 0}
        p = successes / n
        std = float(np.sqrt(p * (1 - p)))  # Bernoulli std per-trial
        se = float(np.sqrt(p * (1 - p) / n))
        return {"p": float(p), "std": std, "se": se, "n": n}

    def summarize(self, images: List[Image]) -> Dict[str, Any]:
        """
        Compute summary statistics using in-memory Image objects.
        Falls back to rasterized GT boxes if GT masks are absent.
        Also computes RMSE/std/se for GT↔Pred metric errors.
        """
        n = len(images)

        # Detection availability (intermediate detector boxes)
        det_disc = sum(1 for im in images if im.inter_pred_disc_box is not None)
        det_cup  = sum(1 for im in images if im.inter_pred_cup_box  is not None)

        # Segmentation availability (MedSAM masks)
        seg_disc = sum(1 for im in images if im.pred_disc_mask is not None)
        seg_cup  = sum(1 for im in images if im.pred_cup_mask  is not None)

        strict_mask_disc, strict_mask_cup = [], []
        paired_mask_disc, paired_mask_cup = [], []
        strict_box_disc, strict_box_cup = [], []
        paired_box_disc, paired_box_cup = [], []

        # Metric error accumulators (signed errors)
        err_values: Dict[str, List[float]] = {
            "cdr_v": [], "cdr_h": [],
            "rim_over_disc": [], "I_over_S": [], "I_over_N": [], "I_over_T": []
        }

        for im in images:
            W, H = im.width, im.height

            # Load masks (aligned)
            gdm = im._mask_to_image_size(im.gt_disc_mask) if im.gt_disc_mask is not None else None
            gcm = im._mask_to_image_size(im.gt_cup_mask)  if im.gt_cup_mask  is not None else None
            pdm = im._mask_to_image_size(im.pred_disc_mask) if im.pred_disc_mask is not None else None
            pcm = im._mask_to_image_size(im.pred_cup_mask)  if im.pred_cup_mask  is not None else None

            # Fallback: rasterize GT boxes if GT masks missing
            if gdm is None and im.gt_disc_box is not None:
                gdm = _mask_from_norm_box(im.gt_disc_box.as_tuple(), H, W)
            if gcm is None and im.gt_cup_box is not None:
                gcm = _mask_from_norm_box(im.gt_cup_box.as_tuple(), H, W)

            # Mask Dice
            if gdm is not None:
                d = self._dice_from_masks(pdm, gdm)
                strict_mask_disc.append(d)
                if pdm is not None:
                    paired_mask_disc.append(d)
            if gcm is not None:
                d = self._dice_from_masks(pcm, gcm)
                strict_mask_cup.append(d)
                if pcm is not None:
                    paired_mask_cup.append(d)

            # Boxes
            strict_box_disc.append(self._box_dice(im.pred_disc_box, im.gt_disc_box))
            strict_box_cup.append(self._box_dice(im.pred_cup_box,  im.gt_cup_box))
            if (im.pred_disc_box is not None) and (im.gt_disc_box is not None):
                paired_box_disc.append(self._box_dice(im.pred_disc_box, im.gt_disc_box))
            if (im.pred_cup_box is not None) and (im.gt_cup_box is not None):
                paired_box_cup.append(self._box_dice(im.pred_cup_box,  im.gt_cup_box))

            # Metric errors (signed)
            gt_v = im.cdr(use_pred=False, axis="vertical")
            pr_v = im.cdr(use_pred=True,  axis="vertical")
            if _is_finite(gt_v) and _is_finite(pr_v):
                err_values["cdr_v"].append(float(pr_v - gt_v))

            gt_h = im.cdr(use_pred=False, axis="horizontal")
            pr_h = im.cdr(use_pred=True,  axis="horizontal")
            if _is_finite(gt_h) and _is_finite(pr_h):
                err_values["cdr_h"].append(float(pr_h - gt_h))

            gt_r = im.rim_metrics(use_pred=False)
            pr_r = im.rim_metrics(use_pred=True)
            if gt_r and pr_r:
                for k in ("rim_over_disc", "I_over_S", "I_over_N", "I_over_T"):
                    gt = gt_r.get(k, None)
                    pr = pr_r.get(k, None)
                    if _is_finite(gt) and _is_finite(pr):
                        err_values[k].append(float(pr - gt))

        # Aggregations
        def _agg_dict(xs: List[float]) -> Dict[str, Optional[float]]:
            mean, std, se, n_ = self._mean_std_se(xs)
            return {"mean": mean, "std": std, "se": se, "n": n_}

        # Dice aggregates
        md_stats = {
            "strict_disc": _agg_dict(strict_mask_disc),
            "strict_cup":  _agg_dict(strict_mask_cup),
            "paired_disc": _agg_dict(paired_mask_disc),
            "paired_cup":  _agg_dict(paired_mask_cup),
        }
        bd_stats = {
            "strict_disc": _agg_dict(strict_box_disc),
            "strict_cup":  _agg_dict(strict_box_cup),
            "paired_disc": _agg_dict(paired_box_disc),
            "paired_cup":  _agg_dict(paired_box_cup),
        }

        # Keep legacy "mean" fields for backward compatibility
        def _mean_or_null(x: List[float]) -> Optional[float]:
            return float(np.mean(x)) if x else None

        # Error metrics: MAE / RMSE / bias / std / se
        metric_error: Dict[str, Dict[str, Optional[float]]] = {}
        for k, vals in err_values.items():
            n_k = len(vals)
            if n_k == 0:
                metric_error[k] = {"mae": None, "rmse": None, "bias": None, "std": None, "se": None, "n": 0}
            else:
                v = np.asarray(vals, dtype=float)
                mae = float(np.mean(np.abs(v)))
                rmse = float(np.sqrt(np.mean(v * v)))
                bias = float(np.mean(v))
                std = float(np.std(v, ddof=1)) if n_k > 1 else 0.0
                se = float(std / np.sqrt(n_k)) if n_k > 0 else None
                metric_error[k] = {"mae": mae, "rmse": rmse, "bias": bias, "std": std, "se": se, "n": n_k}

        # Rate stats (binomial)
        det_rate_disc = det_disc / max(1, n)
        det_rate_cup  = det_cup  / max(1, n)
        seg_rate_disc = seg_disc / max(1, n)
        seg_rate_cup  = seg_cup  / max(1, n)

        summary: Dict[str, Any] = {
            "counts": {
                "images": n,
                "detected_disc": det_disc,
                "detected_cup": det_cup,
                "segmented_disc": seg_disc,
                "segmented_cup": seg_cup,
            },
            "rates": {
                "det_rate_disc": det_rate_disc,
                "det_rate_cup":  det_rate_cup,
                "seg_rate_disc": seg_rate_disc,
                "seg_rate_cup":  seg_rate_cup,
            },
            "rates_stats": {
                "det_rate_disc": self._rate_stats(det_disc, n),
                "det_rate_cup":  self._rate_stats(det_cup, n),
                "seg_rate_disc": self._rate_stats(seg_disc, n),
                "seg_rate_cup":  self._rate_stats(seg_cup, n),
            },
            # Legacy means (back-compat)
            "mask_dice": {
                "strict_mean_disc": _mean_or_null(strict_mask_disc),
                "strict_mean_cup":  _mean_or_null(strict_mask_cup),
                "paired_mean_disc": _mean_or_null(paired_mask_disc),
                "paired_mean_cup":  _mean_or_null(paired_mask_cup),
                "strict_count_disc": len(strict_mask_disc),
                "strict_count_cup":  len(strict_mask_cup),
            },
            "box_dice": {
                "strict_mean_disc": _mean_or_null(strict_box_disc),
                "strict_mean_cup":  _mean_or_null(strict_box_cup),
                "paired_mean_disc": _mean_or_null(paired_box_disc),
                "paired_mean_cup":  _mean_or_null(paired_box_cup),
            },
            # New detailed stats
            "mask_dice_stats": md_stats,
            "box_dice_stats":  bd_stats,
            "metric_error":    metric_error,
        }

        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    # ----------------- Offline summary from JSONL -----------------

    def summarize_jsonl(self, jsonl_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Offline summary: read predictions.jsonl, load mask files, and compute
        mask/box distributions and GT↔Pred metric errors with MAE/RMSE/std/se.
        Uses GT mask paths if present; otherwise falls back to rasterized GT boxes.
        """
        jp = Path(jsonl_path) if jsonl_path else self.jsonl_path
        if not jp.exists():
            raise FileNotFoundError(jp)

        strict_mask_disc, strict_mask_cup = [], []
        paired_mask_disc, paired_mask_cup = [], []
        strict_box_disc,  strict_box_cup  = [], []
        paired_box_disc,  paired_box_cup  = [], []
        det_disc = det_cup = seg_disc = seg_cup = 0
        n_images = 0

        err_values: Dict[str, List[float]] = {
            "cdr_v": [], "cdr_h": [],
            "rim_over_disc": [], "I_over_S": [], "I_over_N": [], "I_over_T": []
        }

        with open(jp, "r") as f:
            for line in f:
                rec = json.loads(line)
                n_images += 1
                img_path = rec["image_path"]
                W, H = PILImage.open(img_path).size

                # Detector availability
                if rec["detector"]["disc"]["box_norm"] is not None: det_disc += 1
                if rec["detector"]["cup"]["box_norm"]  is not None: det_cup  += 1

                # Predicted masks
                pdm = _read_mask_path(rec.get("pred_masks", {}).get("disc_path"))
                pcm = _read_mask_path(rec.get("pred_masks", {}).get("cup_path"))
                if pdm is not None: seg_disc += 1
                if pcm is not None: seg_cup  += 1

                # GT references
                gdm = _read_mask_path(rec.get("gt_masks", {}).get("disc_path"))
                gcm = _read_mask_path(rec.get("gt_masks", {}).get("cup_path"))
                if gdm is None:
                    gdm = _mask_from_norm_box(rec.get("gt_boxes", {}).get("disc"), H, W)
                if gcm is None:
                    gcm = _mask_from_norm_box(rec.get("gt_boxes", {}).get("cup"),  H, W)

                # Mask Dice
                if gdm is not None:
                    d = self._dice_from_masks(pdm, gdm)
                    strict_mask_disc.append(d)
                    if pdm is not None:
                        paired_mask_disc.append(d)
                if gcm is not None:
                    d = self._dice_from_masks(pcm, gcm)
                    strict_mask_cup.append(d)
                    if pcm is not None:
                        paired_mask_cup.append(d)

                # Box Dice (pred boxes from masks vs GT boxes)
                pb_disc = tuple(rec.get("pred_boxes_from_masks", {}).get("disc") or ()) or None
                pb_cup  = tuple(rec.get("pred_boxes_from_masks", {}).get("cup")  or ()) or None
                gb_disc = tuple(rec.get("gt_boxes", {}).get("disc") or ()) or None
                gb_cup  = tuple(rec.get("gt_boxes", {}).get("cup")  or ()) or None

                pb_disc_nb = NormalizedBox(*pb_disc) if pb_disc else None
                pb_cup_nb  = NormalizedBox(*pb_cup)  if pb_cup  else None
                gb_disc_nb = NormalizedBox(*gb_disc) if gb_disc else None
                gb_cup_nb  = NormalizedBox(*gb_cup)  if gb_cup  else None

                strict_box_disc.append(self._box_dice(pb_disc_nb, gb_disc_nb))
                strict_box_cup.append(self._box_dice(pb_cup_nb,  gb_cup_nb))
                if pb_disc_nb and gb_disc_nb:
                    paired_box_disc.append(self._box_dice(pb_disc_nb, gb_disc_nb))
                if pb_cup_nb and gb_cup_nb:
                    paired_box_cup.append(self._box_dice(pb_cup_nb,  gb_cup_nb))

                # Metric errors (signed), using masks; laterality if present
                laterality = rec.get("laterality")

                cdr_v_gt = _cdr_from_masks(gdm, gcm, axis="vertical")
                cdr_v_pr = _cdr_from_masks(pdm, pcm, axis="vertical")
                cdr_h_gt = _cdr_from_masks(gdm, gcm, axis="horizontal")
                cdr_h_pr = _cdr_from_masks(pdm, pcm, axis="horizontal")

                for key, gt, pr in (("cdr_v", cdr_v_gt, cdr_v_pr), ("cdr_h", cdr_h_gt, cdr_h_pr)):
                    if _is_finite(gt) and _is_finite(pr):
                        err_values[key].append(float(pr - gt))

                rm_gt = _rim_metrics_from_masks(gdm, gcm, laterality=laterality)
                rm_pr = _rim_metrics_from_masks(pdm, pcm, laterality=laterality)
                if rm_gt and rm_pr:
                    for k in ("rim_over_disc", "I_over_S", "I_over_N", "I_over_T"):
                        gt = rm_gt.get(k)
                        pr = rm_pr.get(k)
                        if _is_finite(gt) and _is_finite(pr):
                            err_values[k].append(float(pr - gt))

        # Aggregations
        def _mean_or_null(xs: List[float]) -> Optional[float]:
            return float(np.mean(xs)) if xs else None

        def _agg(xs: List[float]) -> Dict[str, Optional[float]]:
            n = len(xs)
            if n == 0:
                return {"mean": None, "std": None, "se": None, "n": 0}
            mean = float(np.mean(xs))
            std = float(np.std(xs, ddof=1)) if n > 1 else 0.0
            se = float(std / np.sqrt(n)) if n > 0 else None
            return {"mean": mean, "std": std, "se": se, "n": n}

        md_stats = {
            "strict_disc": _agg(strict_mask_disc),
            "strict_cup":  _agg(strict_mask_cup),
            "paired_disc": _agg(paired_mask_disc),
            "paired_cup":  _agg(paired_mask_cup),
        }
        bd_stats = {
            "strict_disc": _agg(strict_box_disc),
            "strict_cup":  _agg(strict_box_cup),
            "paired_disc": _agg(paired_box_disc),
            "paired_cup":  _agg(paired_box_cup),
        }

        metric_error: Dict[str, Dict[str, Optional[float]]] = {}
        for k, vals in err_values.items():
            n_k = len(vals)
            if n_k == 0:
                metric_error[k] = {"mae": None, "rmse": None, "bias": None, "std": None, "se": None, "n": 0}
            else:
                v = np.asarray(vals, dtype=float)
                mae = float(np.mean(np.abs(v)))
                rmse = float(np.sqrt(np.mean(v * v)))
                bias = float(np.mean(v))
                std = float(np.std(v, ddof=1)) if n_k > 1 else 0.0
                se = float(std / np.sqrt(n_k)) if n_k > 0 else None
                metric_error[k] = {"mae": mae, "rmse": rmse, "bias": bias, "std": std, "se": se, "n": n_k}

        det_rate_disc = det_disc / max(1, n_images)
        det_rate_cup  = det_cup  / max(1, n_images)
        seg_rate_disc = seg_disc / max(1, n_images)
        seg_rate_cup  = seg_cup  / max(1, n_images)

        summary = {
            "counts": {
                "images": n_images,
                "detected_disc": det_disc,
                "detected_cup": det_cup,
                "segmented_disc": seg_disc,
                "segmented_cup": seg_cup,
            },
            "rates": {
                "det_rate_disc": det_rate_disc,
                "det_rate_cup":  det_rate_cup,
                "seg_rate_disc": seg_rate_disc,
                "seg_rate_cup":  seg_rate_cup,
            },
            "rates_stats": {
                "det_rate_disc": self._rate_stats(det_disc, n_images),
                "det_rate_cup":  self._rate_stats(det_cup, n_images),
                "seg_rate_disc": self._rate_stats(seg_disc, n_images),
                "seg_rate_cup":  self._rate_stats(seg_cup, n_images),
            },
            # Legacy means (back-compat)
            "mask_dice": {
                "strict_mean_disc": _mean_or_null(strict_mask_disc),
                "strict_mean_cup":  _mean_or_null(strict_mask_cup),
                "paired_mean_disc": _mean_or_null(paired_mask_disc),
                "paired_mean_cup":  _mean_or_null(paired_mask_cup),
                "strict_count_disc": len(strict_mask_disc),
                "strict_count_cup":  len(strict_mask_cup),
            },
            "box_dice": {
                "strict_mean_disc": _mean_or_null(strict_box_disc),
                "strict_mean_cup":  _mean_or_null(strict_box_cup),
                "paired_mean_disc": _mean_or_null(paired_box_disc),
                "paired_mean_cup":  _mean_or_null(paired_box_cup),
            },
            # New detailed stats
            "mask_dice_stats": md_stats,
            "box_dice_stats":  bd_stats,
            "metric_error":    metric_error,
        }

        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary

# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end OD/Cup detection→MedSAM segmentation (factory-based, optional YOLO subset).")

    # Offline summary option
    p.add_argument("--summary-only", action="store_true",
                   help="Only compute summary from predictions.jsonl (no detection/segmentation).")
    p.add_argument("--jsonl", type=Path, default=None,
                   help="Path to predictions.jsonl (defaults to <out-dir>/predictions.jsonl).")

    # Common I/O
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory (masks/viz/json).")
    p.add_argument("--limit", type=int, default=None, help="Optional cap on number of images after any filtering.")
    p.add_argument("--save-overlays", action="store_true", help="Save overlay visualizations.")

    # ImageFactory (always used for building images/GT)
    p.add_argument("--data-root", type=Path, help="Root for ImageFactory (expects fundus/oc_mask/od_mask subfolders).")
    p.add_argument("--include-ds", nargs="*", default=None, help="Datasets to include (substring match).")
    p.add_argument("--exclude-ds", nargs="*", default=None, help="Datasets to exclude (substring match).")
    p.add_argument("--require-complete", action="store_true", help="Only include images with both GT masks.")

    # Optional YOLO subset filter
    p.add_argument("--yolo-ds", type=Path, help="YOLO dataset directory containing images/* and labels/*.")
    p.add_argument("--splits", type=str, default="test", help="Comma-separated splits to define subset (e.g., 'val,test').")

    # Detector
    p.add_argument("--yolo-weights", type=Path, required=False, help="YOLO weights (e.g., best.pt).")
    p.add_argument("--device", type=str, default="cuda:0", help="Device for YOLO and MedSAM (e.g., 'cuda:0'|'mps'|'cpu').")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference size.")
    p.add_argument("--conf", type=float, default=0.001, help="YOLO confidence threshold.")
    p.add_argument("--iou", type=float, default=0.70, help="YOLO NMS IoU threshold.")

    # MedSAM
    p.add_argument("--medsam-checkpoint", type=Path, required=False, help="Path to MedSAM ViT-B checkpoint.")

    # Padding of detector boxes before SAM prompting
    p.add_argument("--box-pad-frac", type=float, default=0.05,
                   help="Per-side fractional padding applied to detector inter_pred boxes before MedSAM prompting. "
                        "E.g., 0.05 = 5%% of box width/height on each side.")

    return p.parse_args()

# =========================
# Builders
# =========================

def _build_images_from_factory(args: argparse.Namespace) -> List[Image]:
    if not args.data_root:
        raise ValueError("--data-root is required (factory is always used).")
    fac = ImageFactory(root=args.data_root, auto_scan=True)
    if args.include_ds or args.exclude_ds:
        fac.filter_datasets(include=args.include_ds, exclude=args.exclude_ds)
    if args.require_complete:
        fac.filter_empty_masks()
    images = fac.make_images(require_complete=bool(args.require_complete), compute_boxes=True)
    return images

def _filter_images_by_yolo_subset(images: List[Image], yolo_ds: Optional[Path], splits_str: str) -> List[Image]:
    """
    If yolo_ds is provided, restrict to images whose filename stem appears under
    <yolo_ds>/images/<split> for any requested split. This is O(N) with set membership.
    """
    if not yolo_ds:
        return images
    splits = [s.strip() for s in (splits_str or "").split(",") if s.strip()]
    if not splits:
        return images
    stems = _collect_yolo_split_stems(yolo_ds, splits)
    if not stems:
        return []
    out = [im for im in images if Path(im.image_path).stem in stems]
    return out

# =========================
# Main
# =========================

def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Offline summary path
    if args.summary_only:
        pipe = ODCupPredictor(det=None, sam=None, pcfg=PipelineConfig(out_dir=out_dir, save_overlays=False))
        summary = pipe.summarize_jsonl(args.jsonl)
        print(json.dumps({
            "out_dir": str(out_dir),
            "summary_path": str(pipe.summary_path),
            "jsonl_path": str(args.jsonl or pipe.jsonl_path),
            "counts": summary.get("counts", {}),
            "rates": summary.get("rates", {}),
            "rates_stats": summary.get("rates_stats", {}),
            "mask_dice_stats": summary.get("mask_dice_stats", {}),
            "box_dice_stats": summary.get("box_dice_stats", {}),
            "metric_error": summary.get("metric_error", {}),
        }, indent=2))
        return

    # Validate required inference args
    if not args.yolo_weights or not args.yolo_weights.exists():
        raise FileNotFoundError("--yolo-weights is required and must exist for inference.")
    if not args.medsam_checkpoint or not args.medsam_checkpoint.exists():
        raise FileNotFoundError("--medsam-checkpoint is required and must exist for inference.")

    # Build from factory (always), then optionally filter by YOLO subset
    images_all = _build_images_from_factory(args)
    images = _filter_images_by_yolo_subset(images_all, args.yolo_ds, args.splits)
    if args.limit:
        images = images[: args.limit]

    # Components
    det_cfg = DetectorConfig(
        weights=args.yolo_weights,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )
    detector = ODCupDetector(det_cfg)
    sam = MedSAM(checkpoint=args.medsam_checkpoint, device=args.device)
    pipe = ODCupPredictor(
        detector,
        sam,
        PipelineConfig(out_dir=out_dir, save_overlays=args.save_overlays),
        box_pad_frac=args.box_pad_frac,
    )

    # Run predictions and summarize (in-memory)
    pipe.predict(images, limit=None)
    summary = pipe.summarize(images)

    # Echo a concise summary
    print(json.dumps({
        "out_dir": str(out_dir),
        "factory_images_total": len(images_all),
        "filtered_by_yolo_subset": bool(args.yolo_ds),
        "yolo_subset_splits": (args.splits if args.yolo_ds else None),
        "n_images": len(images),
        "detector": {"conf": args.conf, "iou": args.iou, "imgsz": args.imgsz},
        "sam_prompt": {"box_pad_frac": args.box_pad_frac},
        "summary_path": str(pipe.summary_path),
        "jsonl_path": str(pipe.jsonl_path),
        "per_image_csv": str(pipe.per_image_csv_path),
        "rates": summary.get("rates", {}),
        "rates_stats": summary.get("rates_stats", {}),
        "mask_dice_stats": summary.get("mask_dice_stats", {}),
        "box_dice_stats": summary.get("box_dice_stats", {}),
        "metric_error": summary.get("metric_error", {}),
    }, indent=2))

if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()