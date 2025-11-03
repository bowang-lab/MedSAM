# src/imgpipe/normalized_box.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np

from src.utils import xc_yc_wh_to_xyxy, xyxy_to_xc_yc_wh


@dataclass(frozen=True, slots=True)
class NormalizedBox:
    """
    Resolution-agnostic YOLO-format box (xc, yc, w, h) in [0,1].
    """
    xc: float
    yc: float
    w: float
    h: float

    def __post_init__(self) -> None:
        vals = (self.xc, self.yc, self.w, self.h)
        if not all(map(math.isfinite, vals)):
            raise ValueError("NormalizedBox values must be finite.")
        if not (0.0 <= self.xc <= 1.0 and 0.0 <= self.yc <= 1.0):
            raise ValueError("xc,yc must be within [0,1].")
        if not (0.0 <= self.w <= 1.0 and 0.0 <= self.h <= 1.0):
            raise ValueError("w,h must be within [0,1].")

    # -------- conversions --------

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return self.xc, self.yc, self.w, self.h

    def to_numpy(self) -> np.ndarray:
        return np.array([self.xc, self.yc, self.w, self.h], dtype=float)

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> "NormalizedBox":
        if img_w <= 0 or img_h <= 0:
            raise ValueError("img_w and img_h must be positive.")
        xc, yc, w, h = xyxy_to_xc_yc_wh(x1, y1, x2, y2)
        return cls(xc / img_w, yc / img_h, w / img_w, h / img_h)

    def to_pixel_xyxy(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """
        Returns (x1, y1, x2, y2) in pixel space for drawing/eval (not persisted).
        """
        if img_w <= 0 or img_h <= 0:
            raise ValueError("img_w and img_h must be positive.")
        Xc, Yc = self.xc * img_w, self.yc * img_h
        W, H = self.w * img_w, self.h * img_h
        return xc_yc_wh_to_xyxy(Xc, Yc, W, H)

    # -------- geometry in normalized space --------

    @property
    def width(self) -> float:
        return self.w

    @property
    def height(self) -> float:
        return self.h

    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def intersection(self, other: "NormalizedBox") -> float:
        ax1, ay1, ax2, ay2 = self.to_pixel_xyxy(1, 1)
        bx1, by1, bx2, by2 = other.to_pixel_xyxy(1, 1)
        xi1, yi1 = max(ax1, bx1), max(ay1, by1)
        xi2, yi2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, xi2 - xi1), max(0.0, yi2 - yi1)
        return iw * ih  # normalized area since img_w=img_h=1

    def iou(self, other: "NormalizedBox") -> float:
        inter = self.intersection(other)
        denom = self.area() + other.area() - inter + 1e-9
        return inter / denom

    def dice(self, other: "NormalizedBox") -> float:
        inter = self.intersection(other)
        denom = self.area() + other.area() + 1e-9
        return (2.0 * inter) / denom

    def box_loss(self, other: "NormalizedBox") -> float:
        """
        Ultralytics-style CIoU box loss between this box (pred) and `other` (gt),
        both in normalized YOLO (xc, yc, w, h) coordinates.

        Returns:
            float: 1 - CIoU, where CIoU = IoU - (ρ^2/c^2 + α·v)
        """
        # IoU term (normalized space)
        iou = self.iou(other)

        # Convert to xyxy on a unit canvas (equivalent to normalized space)
        ax1, ay1, ax2, ay2 = self.to_pixel_xyxy(1, 1)
        bx1, by1, bx2, by2 = other.to_pixel_xyxy(1, 1)

        # Center distance squared ρ^2
        rho2 = (self.xc - other.xc) ** 2 + (self.yc - other.yc) ** 2

        # Enclosing box diagonal squared c^2
        cw = max(ax2, bx2) - min(ax1, bx1)
        ch = max(ay2, by2) - min(ay1, by1)
        c2 = cw * cw + ch * ch + 1e-9  # avoid div-by-zero

        # Aspect-ratio consistency term v and weight α
        w1 = max(self.w, 1e-9)
        h1 = max(self.h, 1e-9)
        w2 = max(other.w, 1e-9)
        h2 = max(other.h, 1e-9)
        v = (4.0 / (math.pi ** 2)) * (math.atan(w2 / h2) - math.atan(w1 / h1)) ** 2
        alpha = v / (1.0 - iou + v + 1e-9)

        # Complete CIoU and loss
        ciou = iou - (rho2 / c2 + alpha * v)
        return 1.0 - ciou

    # -------- transforms (normalized units) --------

    def translate(self, dx: float, dy: float) -> "NormalizedBox":
        return NormalizedBox(self.xc + dx, self.yc + dy, self.w, self.h)

    def scale(self, sx: float, sy: float, about_center: bool = False) -> "NormalizedBox":
        if about_center:
            return NormalizedBox(self.xc, self.yc, self.w * sx, self.h * sy)
        # scale about origin (0,0)
        x1, y1, x2, y2 = self.to_pixel_xyxy(1, 1)
        x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
        return NormalizedBox.from_xyxy(x1, y1, x2, y2, 1, 1)