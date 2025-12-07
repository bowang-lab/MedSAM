#!/usr/bin/env python3
# src/imgpipe/image.py
# Image record storing normalized boxes; metrics/vis preserved.
# Uses Parquet as canonical persistence format with a stable Arrow schema.

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.patches import Rectangle
from PIL import Image as PILImage

from .binary_mask_ref import BinaryMaskRef
from .enums import Eye, Ethnicity, LabelType, Structure
from .normalized_box import NormalizedBox
from src.utils import gen_uid, read_image_size


# =============================================================================
# Media ref: optional image bytes stored alongside paths
# =============================================================================


@dataclass
class ImageDataRef:
    """
    Reference to an image payload:
      - path: where the image lives on disk (retained)
      - packed: optional raw file bytes (stored in Parquet)
      - encoding: how 'packed' bytes should be interpreted
          - "raw_file" means bytes are exactly the file contents (best for fidelity)
      - ext: original file extension (".jpg", ".png", ...)
    """

    path: Optional[Path] = None
    packed: Optional[bytes] = None
    encoding: str = "raw_file"
    ext: Optional[str] = None

    def has_bytes(self) -> bool:
        return self.packed is not None and len(self.packed) > 0

    def load_bytes(self) -> Optional[bytes]:
        if self.packed is not None:
            return self.packed
        if self.path is None:
            return None
        try:
            return self.path.read_bytes()
        except Exception:
            return None

    def load_pil(self) -> PILImage.Image:
        """
        Load to PIL.Image. Prefers packed bytes when present, else uses path.
        Always returns an RGB image.
        """
        b = self.load_bytes()
        if b is not None:
            with PILImage.open(io.BytesIO(b)) as im:
                return im.convert("RGB").copy()
        if self.path is None:
            raise FileNotFoundError("ImageDataRef has neither packed bytes nor a valid path.")
        with PILImage.open(self.path) as im:
            return im.convert("RGB")

    def to_dict(self, *, include_image_bytes: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "path": str(self.path) if self.path is not None else None,
            "encoding": str(self.encoding) if self.encoding is not None else None,
            "ext": str(self.ext) if self.ext is not None else None,
        }
        out["data"] = self.load_bytes() if include_image_bytes else None
        return out

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> Optional["ImageDataRef"]:
        if not d:
            return None
        path = d.get("path")
        enc = d.get("encoding") or "raw_file"
        ext = d.get("ext")
        data = d.get("data")

        packed: Optional[bytes] = None
        if data is not None:
            if isinstance(data, (bytes, bytearray)):
                packed = bytes(data)
            elif isinstance(data, memoryview):
                packed = data.tobytes()

        return ImageDataRef(
            path=Path(path) if path else None,
            packed=packed,
            encoding=str(enc),
            ext=str(ext) if ext else None,
        )


# =============================================================================
# Image record
# =============================================================================


@dataclass
class Image:
    """
    All metadata and annotations for a single fundus image.

    - Boxes are stored as normalized YOLO (xc,yc,w,h) in [0,1].
    - Masks are stored as BinaryMaskRef (path and/or packed bytes).
    - Paths are always retained.
    - Parquet is the canonical persistence format (save_parquet/load_parquet/iter_parquet).

    IMPORTANT: `extras` is persisted as JSON string (nullable) under the Parquet column name "extras".
    This avoids Parquet limitations with empty structs while keeping forward flexibility.
    """

    # Identity / dataset
    uid: str
    dataset: str
    patient_id: str

    # Image payload
    image_path: Path
    width: int
    height: int
    split: Optional[str] = None  # "train" | "val" | "test" | None

    # Optional embedded image payload (bytes) in addition to path
    image_ref: Optional[ImageDataRef] = None

    # GT masks
    gt_disc_mask: Optional[BinaryMaskRef] = None
    gt_cup_mask: Optional[BinaryMaskRef] = None

    # MedSAM mask predictions (optional)
    pred_disc_mask: Optional[BinaryMaskRef] = None
    pred_cup_mask: Optional[BinaryMaskRef] = None

    # All boxes are normalized (xc,yc,w,h) in [0,1]
    gt_disc_box: Optional[NormalizedBox] = None
    gt_cup_box: Optional[NormalizedBox] = None

    inter_pred_disc_box: Optional[NormalizedBox] = None
    inter_pred_cup_box: Optional[NormalizedBox] = None

    pred_disc_box: Optional[NormalizedBox] = None
    pred_cup_box: Optional[NormalizedBox] = None

    # Detector / segmenter confidences
    yolo_disc_conf: Optional[float] = None
    yolo_cup_conf: Optional[float] = None
    sam_disc_conf: Optional[float] = None
    sam_cup_conf: Optional[float] = None

    # Ratios / cached metrics
    gt_cd_ratio: Optional[float] = None
    pred_cd_ratio: Optional[float] = None

    mask_dice_disc: Optional[float] = None
    mask_dice_cup: Optional[float] = None

    # Optional bookkeeping
    yolo_label_path: Optional[Path] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    # Patient Info (persisted)
    laterality: Optional[Eye] = None
    age: Optional[int] = None
    ethnicity: Optional[Ethnicity] = None
    glaucoma: Optional[bool] = None
    gt_cdr: Optional[float] = None
    pred_cdr: Optional[float] = None
    gt_rdr: Optional[float] = None
    pred_rdr: Optional[float] = None

    # Lazy-loaded image cache (not serialized)
    _pil_image: Optional[PILImage.Image] = field(default=None, init=False, repr=False, compare=False)

    # ----------------- lazy image loading -----------------

    def ensure_image_ref(self) -> ImageDataRef:
        if self.image_ref is None:
            ext = self.image_path.suffix.lower() if self.image_path is not None else None
            self.image_ref = ImageDataRef(path=self.image_path, ext=ext)
        else:
            if self.image_ref.path is None:
                self.image_ref.path = self.image_path
            if self.image_ref.ext is None:
                self.image_ref.ext = self.image_path.suffix.lower()
        return self.image_ref

    @property
    def image(self) -> PILImage.Image:
        """
        Lazy-load and cache the PIL image.
        Prefers embedded bytes (if present) over filesystem path.
        """
        if self._pil_image is None:
            ref = self.ensure_image_ref()
            self._pil_image = ref.load_pil()
        return self._pil_image

    def unload_image(self) -> None:
        """Release the cached PIL image from memory."""
        if self._pil_image is not None:
            try:
                self._pil_image.close()
            finally:
                self._pil_image = None

    # ----------------- construction -----------------

    @staticmethod
    def _coerce_maskref(mask: Union[BinaryMaskRef, Path, np.ndarray]) -> BinaryMaskRef:
        if isinstance(mask, BinaryMaskRef):
            return mask
        if isinstance(mask, Path):
            return BinaryMaskRef(path=mask)
        return BinaryMaskRef(array=mask)

    @staticmethod
    def from_path(
        image_path: Path,
        dataset: str,
        subject_id: str,
        uid: Optional[str] = None,
        split: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> "Image":
        p = Path(image_path)
        if width is None or height is None:
            w, h = read_image_size(p)
        else:
            w, h = int(width), int(height)

        obj = Image(
            uid=uid or gen_uid(),
            dataset=dataset,
            patient_id=subject_id,
            image_path=p,
            width=w,
            height=h,
            split=split,
        )
        obj.ensure_image_ref()
        return obj

    # ----------------- normalized box helpers -----------------

    def _box_attr_name(self, component: Structure, kind: LabelType) -> str:
        return {
            (Structure.DISC, LabelType.GT): "gt_disc_box",
            (Structure.DISC, LabelType.PRED): "inter_pred_disc_box",
            (Structure.CUP, LabelType.GT): "gt_cup_box",
            (Structure.CUP, LabelType.PRED): "inter_pred_cup_box",
        }[(component, kind)]

    def set_box(self, component: Structure, kind: LabelType, nbox: Optional[NormalizedBox]) -> None:
        setattr(self, self._box_attr_name(component, kind), nbox)

    def set_box_norm(
        self,
        component: Structure,
        kind: LabelType,
        nbox: Optional[Union[NormalizedBox, Tuple[float, float, float, float], Dict[str, float]]],
    ) -> None:
        """
        Set a normalized YOLO box (xc,yc,w,h in [0,1]) for the given component/kind.
        Accepts a NormalizedBox, a 4-tuple/list, or a dict with keys {'xc','yc','w','h'}.
        Pass None to clear.
        """
        if nbox is None:
            setattr(self, self._box_attr_name(component, kind), None)
            return

        if isinstance(nbox, NormalizedBox):
            nb = nbox
        elif isinstance(nbox, (tuple, list)) and len(nbox) == 4:
            xc, yc, w, h = (float(nbox[0]), float(nbox[1]), float(nbox[2]), float(nbox[3]))
            xc, yc = float(np.clip(xc, 0.0, 1.0)), float(np.clip(yc, 0.0, 1.0))
            w, h = float(np.clip(w, 0.0, 1.0)), float(np.clip(h, 0.0, 1.0))
            nb = NormalizedBox(xc, yc, w, h)
        elif isinstance(nbox, dict) and all(k in nbox for k in ("xc", "yc", "w", "h")):
            xc, yc, w, h = (float(nbox["xc"]), float(nbox["yc"]), float(nbox["w"]), float(nbox["h"]))
            xc, yc = float(np.clip(xc, 0.0, 1.0)), float(np.clip(yc, 0.0, 1.0))
            w, h = float(np.clip(w, 0.0, 1.0)), float(np.clip(h, 0.0, 1.0))
            nb = NormalizedBox(xc, yc, w, h)
        else:
            raise TypeError(
                "nbox must be a NormalizedBox, (xc,yc,w,h) tuple/list, dict with keys {'xc','yc','w','h'}, or None."
            )

        setattr(self, self._box_attr_name(component, kind), nb)

    def get_box_norm(self, component: Structure, kind: LabelType) -> Optional[NormalizedBox]:
        return getattr(self, self._box_attr_name(component, kind))

    def set_split(self, split: Optional[str]) -> None:
        if split not in (None, "train", "val", "test"):
            raise ValueError("split must be one of None, 'train', 'val', 'test'")
        self.split = split

    # ----------------- mask setters -----------------

    def set_mask(self, component: Structure, kind: LabelType, mask: Union[BinaryMaskRef, Path, np.ndarray]) -> None:
        ref = self._coerce_maskref(mask)
        attr_map = {
            (Structure.DISC, LabelType.GT): "gt_disc_mask",
            (Structure.DISC, LabelType.PRED): "pred_disc_mask",
            (Structure.CUP, LabelType.GT): "gt_cup_mask",
            (Structure.CUP, LabelType.PRED): "pred_cup_mask",
        }
        try:
            setattr(self, attr_map[(component, kind)], ref)
        except KeyError as e:
            raise ValueError(f"Unsupported (component, kind)=({component}, {kind})") from e

    def all_paths(self, *, drop_none: bool = False) -> Dict[str, Optional[Path]]:
        """
        Return a mapping of all path-bearing fields for this image instance.
        Includes image path, YOLO label path, and any mask file paths if present.
        """

        def _mask_path(m: Optional[BinaryMaskRef]) -> Optional[Path]:
            return getattr(m, "path", None) if m is not None else None

        paths: Dict[str, Optional[Path]] = {
            "image_path": self.image_path,
            "yolo_label_path": self.yolo_label_path,
            "gt_disc_mask_path": _mask_path(self.gt_disc_mask),
            "gt_cup_mask_path": _mask_path(self.gt_cup_mask),
            "pred_disc_mask_path": _mask_path(self.pred_disc_mask),
            "pred_cup_mask_path": _mask_path(self.pred_cup_mask),
        }
        return {k: v for k, v in paths.items() if v is not None} if drop_none else paths

    # ----------------- mask → normalized box -----------------

    def _mask_to_image_size(self, mref: Optional[BinaryMaskRef]) -> Optional[np.ndarray]:
        """
        Load mask as bool array aligned to this image's (H, W).
        If sizes differ, safely crop/pad to fit.
        """
        if mref is None:
            return None
        arr = mref.load().astype(bool)
        H, W = self.height, self.width
        if arr.shape == (H, W):
            return arr
        out = np.zeros((H, W), dtype=bool)
        h = min(H, arr.shape[0])
        w = min(W, arr.shape[1])
        out[:h, :w] = arr[:h, :w]
        return out

    def _mask_bbox_norm_aligned(self, mref: Optional[BinaryMaskRef]) -> Optional[NormalizedBox]:
        """Compute a normalized bbox from a mask AFTER aligning it to image (H,W)."""
        m = self._mask_to_image_size(mref)
        if m is None or not m.any():
            return None
        ys, xs = np.nonzero(m)
        x1, x2 = float(xs.min()), float(xs.max() + 1)
        y1, y2 = float(ys.min()), float(ys.max() + 1)
        return NormalizedBox.from_xyxy(x1, y1, x2, y2, self.width, self.height)

    def _rasterize_box(self, nbox: Optional[NormalizedBox]) -> Optional[np.ndarray]:
        """Rasterize a normalized box into a bool mask at (H,W)."""
        if nbox is None:
            return None
        x1, y1, x2, y2 = nbox.to_pixel_xyxy(self.width, self.height)
        x1, y1 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
        x2, y2 = int(min(self.width, np.ceil(x2))), int(min(self.height, np.ceil(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        m = np.zeros((self.height, self.width), dtype=bool)
        m[y1:y2, x1:x2] = True
        return m

    def ensure_boxes_from_masks(self) -> None:
        """Populate missing normalized boxes from masks (aligned to image size)."""
        if self.gt_disc_box is None:
            self.gt_disc_box = self._mask_bbox_norm_aligned(self.gt_disc_mask)
        if self.gt_cup_box is None:
            self.gt_cup_box = self._mask_bbox_norm_aligned(self.gt_cup_mask)

        if self.pred_disc_box is None:
            self.pred_disc_box = self._mask_bbox_norm_aligned(self.pred_disc_mask)
        if self.pred_cup_box is None:
            self.pred_cup_box = self._mask_bbox_norm_aligned(self.pred_cup_mask)

    # ----------------- YOLO export -----------------

    def yolo_lines_2class(self, use_gt: bool = True) -> Iterator[str]:
        """
        Yield YOLO-normalized lines: '<cls> <xc> <yc> <w> <h>' (0=disc, 1=cup).
        """
        self.ensure_boxes_from_masks()
        boxes = (
            ((0, self.gt_disc_box), (1, self.gt_cup_box))
            if use_gt
            else ((0, self.inter_pred_disc_box), (1, self.inter_pred_cup_box))
        )
        for cls_id, nbox in boxes:
            if nbox is None:
                continue
            xc, yc, w, h = nbox.as_tuple()
            yield f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

    # ----------------- metrics -----------------

    @staticmethod
    def _dice_from_bool(pred: Optional[np.ndarray], gt: Optional[np.ndarray]) -> Optional[float]:
        if pred is None or gt is None:
            return None
        p = pred.astype(bool)
        g = gt.astype(bool)
        if not p.any() and not g.any():
            return 1.0
        inter = float((p & g).sum())
        denom = float(p.sum() + g.sum())
        return (2.0 * inter / denom) if denom > 0 else 0.0

    def cdr(self, *, use_pred: bool = False, axis: str = "vertical") -> Optional[float]:
        """
        Cup-to-Disc Ratio (mask-based preferred; box-based fallback).
        axis: "vertical" (default) or "horizontal".
        """
        if axis not in ("vertical", "horizontal"):
            raise ValueError("axis must be 'vertical' or 'horizontal'")

        dm = self._mask_to_image_size(self.pred_disc_mask if use_pred else self.gt_disc_mask)
        cm = self._mask_to_image_size(self.pred_cup_mask if use_pred else self.gt_cup_mask)

        def _extent(mask: np.ndarray, kind: str) -> Optional[float]:
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                return None
            return float(ys.max() - ys.min() + 1) if kind == "vertical" else float(xs.max() - xs.min() + 1)

        if dm is not None and cm is not None:
            d = _extent(dm, axis)
            c = _extent(cm, axis)
            if d and d > 0:
                return (c or 0.0) / d

        dbox = self.pred_disc_box if use_pred else self.gt_disc_box
        cbox = self.pred_cup_box if use_pred else self.gt_cup_box
        if dbox and cbox:
            if axis == "vertical":
                d = max(0.0, dbox.height)
                c = max(0.0, cbox.height)
            else:
                d = max(0.0, dbox.width)
                c = max(0.0, cbox.width)
            return (c / d) if d > 0 else None
        return None

    def compute_mask_dice(self, component: Structure, *, fallback_to_boxes: bool = True) -> Optional[float]:
        if component not in (Structure.DISC, Structure.CUP):
            raise ValueError("component must be Structure.DISC or Structure.CUP")

        pred_m = self._mask_to_image_size(self.pred_disc_mask if component is Structure.DISC else self.pred_cup_mask)
        gt_m = self._mask_to_image_size(self.gt_disc_mask if component is Structure.DISC else self.gt_cup_mask)

        if pred_m is None and fallback_to_boxes:
            nb = self.pred_disc_box if component is Structure.DISC else self.pred_cup_box
            pred_m = self._rasterize_box(nb)
        if gt_m is None and fallback_to_boxes:
            nb = self.gt_disc_box if component is Structure.DISC else self.gt_cup_box
            gt_m = self._rasterize_box(nb)

        d = self._dice_from_bool(pred_m, gt_m)
        if component is Structure.DISC:
            self.mask_dice_disc = d
        else:
            self.mask_dice_cup = d
        return d

    def update_mask_dice(self, *, fallback_to_boxes: bool = True) -> Dict[str, Optional[float]]:
        return {
            "disc": self.compute_mask_dice(Structure.DISC, fallback_to_boxes=fallback_to_boxes),
            "cup": self.compute_mask_dice(Structure.CUP, fallback_to_boxes=fallback_to_boxes),
        }

    def rim_metrics(self, *, use_pred: bool = False) -> Optional[Dict[str, Optional[float]]]:
        disc = self._mask_to_image_size(self.pred_disc_mask if use_pred else self.gt_disc_mask)
        cup = self._mask_to_image_size(self.pred_cup_mask if use_pred else self.gt_cup_mask)
        if disc is None or cup is None or not disc.any():
            return None

        H, W = self.height, self.width
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

        metrics: Dict[str, Optional[float]] = {
            "rim_over_disc": r_over_d,
            "I_over_S": _safe(rim_inf, rim_sup),
            "I_over_N": None,
            "I_over_T": None,
            "rim_area": rim_area,
            "disc_area": disc_area,
            "inferior_area": rim_inf,
            "superior_area": rim_sup,
            "nasal_area": None,
            "temporal_area": None,
        }

        if self.laterality is not None:
            left = (np.arange(W)[None, :] < xc)
            right = ~left
            if str(self.laterality.name).upper() == "OD":
                nasal, temporal = left, right
            else:
                nasal, temporal = right, left

            rim_nas = float((rim & nasal).sum())
            rim_tem = float((rim & temporal).sum())

            metrics["nasal_area"] = rim_nas
            metrics["temporal_area"] = rim_tem
            metrics["I_over_N"] = _safe(rim_inf, rim_nas)
            metrics["I_over_T"] = _safe(rim_inf, rim_tem)

        return metrics

    def metrics_summary(self) -> Dict[str, Dict[str, Optional[float]]]:
        out: Dict[str, Dict[str, Optional[float]]] = {"gt": {}, "pred": {}}
        out["gt"]["cdr_v"] = self.cdr(use_pred=False, axis="vertical")
        out["gt"]["cdr_h"] = self.cdr(use_pred=False, axis="horizontal")
        out["pred"]["cdr_v"] = self.cdr(use_pred=True, axis="vertical")
        out["pred"]["cdr_h"] = self.cdr(use_pred=True, axis="horizontal")

        gt_r = self.rim_metrics(use_pred=False)
        pr_r = self.rim_metrics(use_pred=True)
        if gt_r:
            out["gt"].update(gt_r)
        if pr_r:
            out["pred"].update(pr_r)
        return out

    def visualize(
            self,
            *,
            show: bool = True,
            save_path: Optional[Path] = None,
            dpi: int = 150,
            figsize: Tuple[int, int] = (14, 6),
            mask_alpha: float = 0.35,
            show_metrics: bool = True,
            show_conf: bool = True,
            show_boxes: bool = True,
            show_original_image: bool = False
    ) -> None:
        img = self.image
        W, H = img.size

        def _mask_arr(mref: Optional[BinaryMaskRef]) -> Optional[np.ndarray]:
            return self._mask_to_image_size(mref)

        def _draw_mask(ax, mask: np.ndarray, color: str, alpha: float) -> None:
            overlay = np.zeros((H, W, 4), dtype=float)
            rgba = plt.matplotlib.colors.to_rgba(color, alpha=alpha)
            overlay[mask] = rgba
            ax.imshow(overlay, origin="upper")

        def _draw_nbox(ax, nbox: Optional[NormalizedBox], color: str, ls: str = "-", lw: float = 2.0) -> None:
            if nbox is None:
                return
            x1, y1, x2, y2 = nbox.to_pixel_xyxy(W, H)
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            ax.add_patch(Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=lw, linestyle=ls))

        gt_disc_m = _mask_arr(self.gt_disc_mask)
        gt_cup_m = _mask_arr(self.gt_cup_mask)
        pred_disc_m = _mask_arr(self.pred_disc_mask)
        pred_cup_m = _mask_arr(self.pred_cup_mask)

        has_gt = (gt_disc_m is not None) or (gt_cup_m is not None)
        has_pred = (
                (pred_disc_m is not None)
                or (pred_cup_m is not None)
                or (self.inter_pred_disc_box is not None)
                or (self.inter_pred_cup_box is not None)
        )

        panels: List[Tuple[str, Dict[str, Any]]] = []

        # 1. Optional Original Image Panel
        if show_original_image:
            panels.append(
                (
                    "Original",
                    {
                        "disc_m": None, "cup_m": None,
                        "disc_box": None, "cup_box": None,
                        "use_pred": False, "is_original": True
                    }
                )
            )

        # 2. Ground Truth Panel
        if has_gt:
            panels.append(
                (
                    "Ground Truth",
                    {
                        "disc_m": gt_disc_m, "cup_m": gt_cup_m,
                        "disc_box": self.gt_disc_box,
                        "cup_box": self.gt_cup_box,
                        "use_pred": False, "is_original": False
                    },
                )
            )

        # 3. Predictions Panel
        if has_pred:
            panels.append(
                (
                    "Predictions",
                    {
                        "disc_m": pred_disc_m,
                        "cup_m": pred_cup_m,
                        "disc_box": self.inter_pred_disc_box,
                        "cup_box": self.inter_pred_cup_box,
                        "use_pred": True, "is_original": False
                    },
                )
            )

        if not panels:
            panels.append(("Image",
                           {"disc_m": None, "cup_m": None, "disc_box": None, "cup_box": None, "use_pred": False,
                            "is_original": True}))

        ncols = len(panels)
        panel_width = figsize[0] / 2
        plot_width = panel_width * ncols
        plot_height = figsize[1]

        # UPDATE: gridspec_kw used to reduce whitespace (wspace=0.02)
        fig, axes = plt.subplots(
            nrows=1,
            ncols=ncols,
            figsize=(plot_width, plot_height),
            dpi=dpi,
            squeeze=False,
            gridspec_kw={"wspace": 0.02, "hspace": 0}
        )

        fig.suptitle(f"{self.dataset} / {self.patient_id} — {self.image_path.name}", fontsize=12)

        disc_color_mask = "tab:red"
        cup_color_mask = "tab:blue"
        disc_color_box = "red"
        cup_color_box = "blue"

        self.update_mask_dice(fallback_to_boxes=True)

        for j, (title, dct) in enumerate(panels):
            ax = axes[0, j]
            ax.imshow(img, origin="upper")
            ax.set_axis_off()
            ax.set_title(title, fontsize=11)

            if dct.get("is_original"):
                continue

            if dct["disc_m"] is not None:
                _draw_mask(ax, dct["disc_m"], disc_color_mask, mask_alpha)
            if dct["cup_m"] is not None:
                _draw_mask(ax, dct["cup_m"], cup_color_mask, mask_alpha)

            if show_boxes:
                _draw_nbox(ax, dct["disc_box"], disc_color_box, ls="-", lw=2.0)
                _draw_nbox(ax, dct["cup_box"], cup_color_box, ls="--", lw=2.0)

            if show_metrics or show_conf:
                use_pred = bool(dct["use_pred"])
                lines: List[str] = []

                if show_metrics:
                    cdr_v = self.cdr(use_pred=use_pred, axis="vertical")
                    rims = self.rim_metrics(use_pred=use_pred)

                    if use_pred:
                        if self.mask_dice_disc is not None:
                            lines.append(f"Dice(D): {self.mask_dice_disc:.2f}")
                        if self.mask_dice_cup is not None:
                            lines.append(f"Dice(C): {self.mask_dice_cup:.2f}")

                    if cdr_v is not None:
                        lines.append(f"CDR(V): {cdr_v:.2f}")
                    if rims is not None:
                        if np.isfinite(rims.get("rim_over_disc", np.nan)):
                            lines.append(f"R/D: {rims['rim_over_disc']:.2f}")

                if show_conf and dct["use_pred"]:
                    if lines: lines.append("")
                    if self.yolo_disc_conf is not None:
                        lines.append(f"Conf(D): {self.yolo_disc_conf:.2f}")
                    if self.yolo_cup_conf is not None:
                        lines.append(f"Conf(C): {self.yolo_cup_conf:.2f}")

                if lines:
                    ax.text(
                        0.02, 0.02, "\n".join(lines),
                        transform=ax.transAxes, fontsize=9, va="top", ha="left",
                        color="white",
                        bbox=dict(facecolor="black", alpha=0.35, boxstyle="round,pad=0.3", edgecolor="none"),
                    )

            handles: List[Any] = []
            if (dct["disc_m"] is not None) or (dct["disc_box"] is not None and show_boxes):
                handles.append(plt.Line2D([0], [0], color=disc_color_box, lw=2, linestyle="-", label="Disc"))
            if (dct["cup_m"] is not None) or (dct["cup_box"] is not None and show_boxes):
                handles.append(plt.Line2D([0], [0], color=cup_color_box, lw=2, linestyle="--", label="Cup"))
            if handles:
                ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True)

        # Tighter layout logic
        fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.02)

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), bbox_inches="tight", dpi=dpi, pad_inches=0.05)

        if show:
            plt.show()
        else:
            plt.close(fig)

    # ----------------- serialization helpers -----------------

    @staticmethod
    def _path_to_str(p: Optional[Path]) -> Optional[str]:
        return str(p) if p is not None else None

    @staticmethod
    def _path_from_str(s: Optional[str]) -> Optional[Path]:
        return Path(s) if s else None

    @staticmethod
    def _enum_to_str(e: Optional[Enum]) -> Optional[str]:
        if e is None:
            return None
        return getattr(e, "name", str(e))

    @staticmethod
    def _enum_from_str(enum_cls, s: Optional[str]):
        if s is None:
            return None
        try:
            return enum_cls[s]
        except Exception:
            try:
                return enum_cls(s)
            except Exception:
                return None

    @staticmethod
    def _boxn_to_dict(nbox: Optional[NormalizedBox]) -> Optional[Dict[str, float]]:
        if nbox is None:
            return None
        xc, yc, w, h = nbox.as_tuple()
        return {"xc": float(xc), "yc": float(yc), "w": float(w), "h": float(h)}

    @staticmethod
    def _boxn_from_dict(dct: Optional[Dict[str, Any]]) -> Optional[NormalizedBox]:
        if not dct:
            return None
        return NormalizedBox(float(dct["xc"]), float(dct["yc"]), float(dct["w"]), float(dct["h"]))

    @staticmethod
    def _mask_to_dict(m: Optional[BinaryMaskRef], *, include_mask_bytes: bool = False) -> Optional[Dict[str, Any]]:
        return m.to_dict(include_mask_bytes=include_mask_bytes) if m is not None else None

    @staticmethod
    def _mask_from_dict(dct: Optional[Dict[str, Any]]) -> Optional[BinaryMaskRef]:
        if not dct:
            return None

        path = dct.get("path")
        data = dct.get("data")
        h = dct.get("h")
        w = dct.get("w")
        enc = dct.get("encoding")

        packed: Optional[bytes] = None
        if data is not None:
            if isinstance(data, (bytes, bytearray)):
                packed = bytes(data)
            elif isinstance(data, memoryview):
                packed = data.tobytes()

        if packed is not None and h is not None and w is not None:
            return BinaryMaskRef(
                path=Path(path) if path else None,
                packed=packed,
                shape=(int(h), int(w)),
                encoding=str(enc) if enc else "packbits_little",
            )

        return BinaryMaskRef(path=Path(path) if path else None)

    @staticmethod
    def _extras_to_storage(extras: Dict[str, Any]) -> Optional[str]:
        """
        Persist extras as JSON string (nullable). This avoids Parquet errors from empty structs
        and keeps future flexibility for arbitrary nested content.
        """
        if not extras:
            return None
        try:
            return json.dumps(extras, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)
        except Exception:
            # Fallback: last resort stringification (still stored as a string)
            try:
                return json.dumps({"_unserializable_extras": str(extras)}, ensure_ascii=False)
            except Exception:
                return str(extras)

    @staticmethod
    def _extras_from_storage(v: Any) -> Dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, (bytes, bytearray, memoryview)):
            try:
                v = bytes(v).decode("utf-8", errors="replace")
            except Exception:
                return {}
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return {}
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, dict) else {"_extras": obj}
            except Exception:
                return {"_extras_raw": s}
        return {"_extras": v}

    # ----------------- dict API (for Parquet) -----------------

    def to_dict(
        self,
        *,
        drop_none: bool = False,
        include_mask_bytes: bool = False,
        include_image_bytes: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert to a plain Python dict suitable for Arrow/Parquet.

        For Parquet, prefer drop_none=False so all columns appear with nulls.
        Schema is fixed by Image._canonical_arrow_schema().
        """
        self.ensure_boxes_from_masks()
        img_ref = self.ensure_image_ref()

        d: Dict[str, Any] = {
            "_schema": 7,
            # identity
            "uid": self.uid,
            "dataset": self.dataset,
            "subject_id": self.patient_id,
            # image payload (path retained + optional embedded bytes)
            "image_path": str(self.image_path),
            "image": img_ref.to_dict(include_image_bytes=include_image_bytes),
            "width": int(self.width),
            "height": int(self.height),
            "split": self.split,
            # masks
            "gt_disc_mask": self._mask_to_dict(self.gt_disc_mask, include_mask_bytes=include_mask_bytes),
            "gt_cup_mask": self._mask_to_dict(self.gt_cup_mask, include_mask_bytes=include_mask_bytes),
            "pred_disc_mask": self._mask_to_dict(self.pred_disc_mask, include_mask_bytes=include_mask_bytes),
            "pred_cup_mask": self._mask_to_dict(self.pred_cup_mask, include_mask_bytes=include_mask_bytes),
            # boxes (normalized)
            "gt_disc_box": self._boxn_to_dict(self.gt_disc_box),
            "gt_cup_box": self._boxn_to_dict(self.gt_cup_box),
            "inter_pred_disc_box": self._boxn_to_dict(self.inter_pred_disc_box),
            "inter_pred_cup_box": self._boxn_to_dict(self.inter_pred_cup_box),
            "pred_disc_box": self._boxn_to_dict(self.pred_disc_box),
            "pred_cup_box": self._boxn_to_dict(self.pred_cup_box),
            # lightweight cached metrics
            "gt_cd_ratio": self.gt_cd_ratio,
            "pred_cd_ratio": self.pred_cd_ratio,
            # cached dice
            "mask_dice_disc": self.mask_dice_disc,
            "mask_dice_cup": self.mask_dice_cup,
            # detector / segmenter confidences
            "yolo_disc_conf": self.yolo_disc_conf,
            "yolo_cup_conf": self.yolo_cup_conf,
            "sam_disc_conf": self.sam_disc_conf,
            "sam_cup_conf": self.sam_cup_conf,
            # bookkeeping
            "yolo_label_path": self._path_to_str(self.yolo_label_path),
            # IMPORTANT: persist as JSON string (nullable)
            "extras": Image._extras_to_storage(self.extras),
            # patient info
            "eye": self._enum_to_str(self.laterality),
            "age": self.age,
            "ethnicity": self._enum_to_str(self.ethnicity),
            "glaucoma": self.glaucoma,
            "gt_cdr": self.gt_cdr,
            "pred_cdr": self.pred_cdr,
            "gt_rdr": self.gt_rdr,
            "pred_rdr": self.pred_rdr,
        }

        if drop_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Image":
        P = Image._path_from_str

        image_path = P(d.get("image_path")) or Path(".")
        obj = Image(
            uid=d.get("uid", ""),
            dataset=d.get("dataset", ""),
            patient_id=d.get("subject_id", d.get("patient_id", "")) or "",
            image_path=image_path,
            width=int(d.get("width", 0) or 0),
            height=int(d.get("height", 0) or 0),
            split=d.get("split"),
            yolo_label_path=P(d.get("yolo_label_path")),
            extras=Image._extras_from_storage(d.get("extras")),
        )

        # Image bytes (optional)
        img_ref = ImageDataRef.from_dict(d.get("image"))
        if img_ref is None:
            img_ref = ImageDataRef(path=image_path, ext=image_path.suffix.lower())
        if img_ref.path is None:
            img_ref.path = image_path
        if img_ref.ext is None:
            img_ref.ext = image_path.suffix.lower()
        obj.image_ref = img_ref

        # masks
        obj.gt_disc_mask = Image._mask_from_dict(d.get("gt_disc_mask"))
        obj.gt_cup_mask = Image._mask_from_dict(d.get("gt_cup_mask"))
        obj.pred_disc_mask = Image._mask_from_dict(d.get("pred_disc_mask"))
        obj.pred_cup_mask = Image._mask_from_dict(d.get("pred_cup_mask"))

        # boxes
        obj.gt_disc_box = Image._boxn_from_dict(d.get("gt_disc_box"))
        obj.gt_cup_box = Image._boxn_from_dict(d.get("gt_cup_box"))
        obj.inter_pred_disc_box = Image._boxn_from_dict(d.get("inter_pred_disc_box"))
        obj.inter_pred_cup_box = Image._boxn_from_dict(d.get("inter_pred_cup_box"))
        obj.pred_disc_box = Image._boxn_from_dict(d.get("pred_disc_box"))
        obj.pred_cup_box = Image._boxn_from_dict(d.get("pred_cup_box"))

        # cached metrics
        obj.gt_cd_ratio = float(d["gt_cd_ratio"]) if d.get("gt_cd_ratio") is not None else None
        obj.pred_cd_ratio = float(d["pred_cd_ratio"]) if d.get("pred_cd_ratio") is not None else None

        # cached dice
        obj.mask_dice_disc = float(d["mask_dice_disc"]) if d.get("mask_dice_disc") is not None else None
        obj.mask_dice_cup = float(d["mask_dice_cup"]) if d.get("mask_dice_cup") is not None else None

        # confidences
        obj.yolo_disc_conf = float(d["yolo_disc_conf"]) if d.get("yolo_disc_conf") is not None else None
        obj.yolo_cup_conf = float(d["yolo_cup_conf"]) if d.get("yolo_cup_conf") is not None else None
        obj.sam_disc_conf = float(d["sam_disc_conf"]) if d.get("sam_disc_conf") is not None else None
        obj.sam_cup_conf = float(d["sam_cup_conf"]) if d.get("sam_cup_conf") is not None else None

        # patient info
        obj.laterality = Image._enum_from_str(Eye, d.get("eye"))

        age_val = d.get("age")
        if isinstance(age_val, float) and math.isnan(age_val):
            obj.age = None
        elif age_val is None:
            obj.age = None
        else:
            try:
                obj.age = int(age_val)
            except (TypeError, ValueError):
                obj.age = None

        obj.ethnicity = Image._enum_from_str(Ethnicity, d.get("ethnicity"))

        glau_val = d.get("glaucoma")
        if isinstance(glau_val, float) and math.isnan(glau_val):
            obj.glaucoma = None
        elif glau_val is None:
            obj.glaucoma = None
        else:
            obj.glaucoma = bool(glau_val)

        obj.gt_cdr = float(d["gt_cdr"]) if d.get("gt_cdr") is not None else None
        obj.pred_cdr = float(d["pred_cdr"]) if d.get("pred_cdr") is not None else None
        obj.gt_rdr = float(d["gt_rdr"]) if d.get("gt_rdr") is not None else None
        obj.pred_rdr = float(d["pred_rdr"]) if d.get("pred_rdr") is not None else None

        return obj

    # ----------------- Arrow schema normalisation helpers -----------------

    @staticmethod
    def _make_field_nullable(field: pa.Field) -> pa.Field:
        t = field.type

        if pa.types.is_struct(t):
            new_fields = [Image._make_field_nullable(sf) for sf in t]
            return pa.field(field.name, pa.struct(new_fields), nullable=True, metadata=field.metadata)

        if pa.types.is_list(t):
            vf = t.value_field
            new_vf = Image._make_field_nullable(vf)
            return pa.field(field.name, pa.list_(new_vf), nullable=True, metadata=field.metadata)

        if pa.types.is_large_list(t):
            vf = t.value_field
            new_vf = Image._make_field_nullable(vf)
            return pa.field(field.name, pa.large_list(new_vf), nullable=True, metadata=field.metadata)

        # key must remain non-nullable in Arrow map types
        if pa.types.is_map(t):
            kf = t.key_field
            vf = t.item_field
            new_vf = Image._make_field_nullable(vf)
            return pa.field(field.name, pa.map_(kf.type, new_vf.type), nullable=True, metadata=field.metadata)

        return pa.field(field.name, t, nullable=True, metadata=field.metadata)

    @staticmethod
    def _schema_all_nullable(schema: pa.Schema) -> pa.Schema:
        return pa.schema([Image._make_field_nullable(f) for f in schema], metadata=schema.metadata)

    @staticmethod
    def _normalize_value_to_type(value: Any, dtype: pa.DataType) -> Any:
        if value is None:
            return None

        if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
            if isinstance(value, Path):
                return str(value)
            return value if isinstance(value, str) else str(value)

        if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
            if isinstance(value, memoryview):
                return value.tobytes()
            if isinstance(value, bytearray):
                return bytes(value)
            return value

        if pa.types.is_struct(dtype):
            if isinstance(value, dict):
                out: Dict[str, Any] = {}
                for f in dtype:
                    out[f.name] = Image._normalize_value_to_type(value.get(f.name), f.type)
                return out
            return value

        if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
            if not isinstance(value, list):
                return value
            elem_t = dtype.value_type
            return [Image._normalize_value_to_type(v, elem_t) for v in value]

        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return value.item()
            except Exception:
                pass

        return value

    @staticmethod
    def _normalize_row_to_schema(d: Dict[str, Any], schema: pa.Schema) -> Dict[str, Any]:
        """
        Produce a dict whose keys and value types are aligned with the target schema.
        Missing keys become None. Values are normalised recursively for nested types.
        """
        out: Dict[str, Any] = {}
        for f in schema:
            out[f.name] = Image._normalize_value_to_type(d.get(f.name), f.type)
        return out

    # ----------------- Canonical Arrow schema (stable, Parquet-safe) -----------------

    @staticmethod
    def _canonical_arrow_schema() -> pa.Schema:
        """
        Stable, Parquet-safe schema for Image records.

        Key decisions:
          - `extras` stored as LARGE_STRING (nullable) JSON; avoids empty-struct Parquet limitation.
          - Nested structs for image/masks/boxes are explicitly defined.
          - All fields are later made nullable for robustness across partial records.
        """
        s = pa.large_string()
        i32 = pa.int32()
        f64 = pa.float64()
        b = pa.binary()
        bl = pa.bool_()

        box_struct = pa.struct(
            [
                pa.field("xc", f64, nullable=True),
                pa.field("yc", f64, nullable=True),
                pa.field("w", f64, nullable=True),
                pa.field("h", f64, nullable=True),
            ]
        )

        mask_struct = pa.struct(
            [
                pa.field("path", s, nullable=True),
                pa.field("data", b, nullable=True),
                pa.field("h", i32, nullable=True),
                pa.field("w", i32, nullable=True),
                pa.field("encoding", s, nullable=True),
            ]
        )

        image_struct = pa.struct(
            [
                pa.field("path", s, nullable=True),
                pa.field("encoding", s, nullable=True),
                pa.field("ext", s, nullable=True),
                pa.field("data", b, nullable=True),
            ]
        )

        return pa.schema(
            [
                pa.field("_schema", i32, nullable=True),

                pa.field("uid", s, nullable=True),
                pa.field("dataset", s, nullable=True),
                pa.field("subject_id", s, nullable=True),

                pa.field("image_path", s, nullable=True),
                pa.field("image", image_struct, nullable=True),
                pa.field("width", i32, nullable=True),
                pa.field("height", i32, nullable=True),
                pa.field("split", s, nullable=True),

                pa.field("gt_disc_mask", mask_struct, nullable=True),
                pa.field("gt_cup_mask", mask_struct, nullable=True),
                pa.field("pred_disc_mask", mask_struct, nullable=True),
                pa.field("pred_cup_mask", mask_struct, nullable=True),

                pa.field("gt_disc_box", box_struct, nullable=True),
                pa.field("gt_cup_box", box_struct, nullable=True),
                pa.field("inter_pred_disc_box", box_struct, nullable=True),
                pa.field("inter_pred_cup_box", box_struct, nullable=True),
                pa.field("pred_disc_box", box_struct, nullable=True),
                pa.field("pred_cup_box", box_struct, nullable=True),

                pa.field("gt_cd_ratio", f64, nullable=True),
                pa.field("pred_cd_ratio", f64, nullable=True),

                pa.field("mask_dice_disc", f64, nullable=True),
                pa.field("mask_dice_cup", f64, nullable=True),

                pa.field("yolo_disc_conf", f64, nullable=True),
                pa.field("yolo_cup_conf", f64, nullable=True),
                pa.field("sam_disc_conf", f64, nullable=True),
                pa.field("sam_cup_conf", f64, nullable=True),

                pa.field("yolo_label_path", s, nullable=True),
                pa.field("extras", s, nullable=True),

                pa.field("eye", s, nullable=True),
                pa.field("age", i32, nullable=True),
                pa.field("ethnicity", s, nullable=True),
                pa.field("glaucoma", bl, nullable=True),

                pa.field("gt_cdr", f64, nullable=True),
                pa.field("pred_cdr", f64, nullable=True),
                pa.field("gt_rdr", f64, nullable=True),
                pa.field("pred_rdr", f64, nullable=True),
            ]
        )

    # ----------------- Parquet helpers -----------------

    @staticmethod
    def save_parquet(
        images: Iterable["Image"],
        path: Union[Path, str],
        *,
        drop_none: bool = False,  # kept for API compatibility; should remain False for Parquet
        include_image_bytes: bool = False,
        include_mask_bytes: bool = True,
        compression: str = "zstd",
        write_batch: int = 1024,
    ) -> None:
        """
        Save Image objects to a Parquet file using a robust, streaming writer.

        Guarantees:
          - Stable schema (does NOT depend on batch inference).
          - Nested structs written safely.
          - `extras` is always a nullable string column (JSON), never an empty struct.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Writer uses a stable schema; make it fully nullable to accept partial rows.
        writer_schema = Image._schema_all_nullable(Image._canonical_arrow_schema())

        writer: Optional[pq.ParquetWriter] = None
        buffer_rows: List[Dict[str, Any]] = []

        def flush_rows(rows: List[Dict[str, Any]]) -> None:
            nonlocal writer
            if not rows:
                return

            if writer is None:
                writer = pq.ParquetWriter(where=str(out_path), schema=writer_schema, compression=compression)

            assert writer is not None

            rows_norm = [Image._normalize_row_to_schema(r, writer_schema) for r in rows]
            table = pa.Table.from_pylist(rows_norm)
            writer.write_table(table.cast(writer_schema, safe=False))

        try:
            for img in images:
                row = img.to_dict(
                    drop_none=bool(drop_none),  # generally keep False
                    include_image_bytes=include_image_bytes,
                    include_mask_bytes=include_mask_bytes,
                )
                buffer_rows.append(row)

                if len(buffer_rows) >= int(write_batch):
                    flush_rows(buffer_rows)
                    buffer_rows = []

            flush_rows(buffer_rows)

        finally:
            if writer is not None:
                writer.close()

    @staticmethod
    def load_parquet(path: Union[Path, str]) -> List["Image"]:
        p = Path(path)
        if not p.exists():
            return []
        table = pq.read_table(p)
        return [Image.from_dict(rec) for rec in table.to_pylist()]


    @staticmethod
    def iter_parquet(path: Union[Path, str], batch_size: int = 1024) -> Iterator["Image"]:
        """
        Stream Image objects from a Parquet file OR a directory of Parquet files.
        Robustly handles nested structs by using iter_batches().
        """
        p = Path(path).resolve()

        # 1. Resolve input to a list of files
        files: List[Path] = []
        if p.is_file():
            if p.suffix.lower() != ".parquet":
                raise ValueError(f"Expected .parquet file, got: {p}")
            files = [p]
        elif p.is_dir():
            files = sorted(p.rglob("*.parquet"))
            if not files:
                # It is valid to have an empty directory, just yield nothing
                return
        else:
            raise FileNotFoundError(f"Input path not found: {p}")

        # 2. Stream from all files
        for f in files:
            pf = pq.ParquetFile(str(f))
            for batch in pf.iter_batches(batch_size=int(batch_size)):
                for rec in batch.to_pylist():
                    yield Image.from_dict(rec)

    # ----------------- repr -----------------

    def __repr__(self) -> str:
        return (
            f"ImageSample(uid={self.uid!r}, ds={self.dataset!r}, subj={self.patient_id!r}, "
            f"size=({self.width}x{self.height}), split={self.split!r})"
        )