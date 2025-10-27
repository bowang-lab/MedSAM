from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.utils import (
    ensure_bool_mask,
    xc_yc_wh_to_xyxy,
    xyxy_to_xc_yc_wh,
)


@dataclass
class BoundingBox:
    """Pixel-space axis-aligned box."""
    x1: float
    y1: float
    x2: float
    y2: float

    @staticmethod
    def from_mask(mask: np.ndarray) -> Optional["BoundingBox"]:
        m = ensure_bool_mask(mask)
        if not m.any():
            return None
        ys, xs = np.nonzero(m)
        y1, y2 = float(ys.min()), float(ys.max() + 1)
        x1, x2 = float(xs.min()), float(xs.max() + 1)
        return BoundingBox(x1, y1, x2, y2)

    @staticmethod
    def from_yolo_norm(
        xc: float, yc: float, w: float, h: float, img_w: int, img_h: int
    ) -> "BoundingBox":
        Xc = xc * img_w
        Yc = yc * img_h
        W = w * img_w
        H = h * img_h
        x1, y1, x2, y2 = xc_yc_wh_to_xyxy(Xc, Yc, W, H)
        return BoundingBox(x1, y1, x2, y2).clipped(img_w, img_h)

    def to_yolo_norm(self, img_w: int, img_h: int):
        xc, yc, w, h = xyxy_to_xc_yc_wh(self.x1, self.y1, self.x2, self.y2)
        return (xc / img_w, yc / img_h, w / img_w, h / img_h)

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: "BoundingBox") -> float:
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)
        iw = max(0.0, xi2 - xi1)
        ih = max(0.0, yi2 - yi1)
        inter = iw * ih
        denom = self.area() + other.area() - inter + 1e-9
        return inter / denom

    def clipped(self, img_w: int, img_h: int) -> "BoundingBox":
        x1 = min(max(self.x1, 0.0), img_w)
        y1 = min(max(self.y1, 0.0), img_h)
        x2 = min(max(self.x2, 0.0), img_w)
        y2 = min(max(self.y2, 0.0), img_h)
        return BoundingBox(x1, y1, x2, y2)