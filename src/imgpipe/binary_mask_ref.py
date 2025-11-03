# src/imgpipe/binary_mask_ref.py
# Mask holder; exposes optional convenience to compute a normalized bbox

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from src.imgpipe.normalized_box import NormalizedBox
from src.utils import ensure_bool_mask
from PIL import Image  # type: ignore
import imageio.v3 as iio  # type: ignore
import cv2  # type: ignore


@dataclass
class BinaryMaskRef:
    """
    Holds a reference to a binary mask — either a file path or an in-memory array.
    Loads lazily and caches the array when read from disk.
    """
    path: Optional[Path] = None
    array: Optional[np.ndarray] = field(default=None, repr=False)

    def load(self) -> np.ndarray:
        """Return mask as a boolean array."""
        if self.array is not None:
            return ensure_bool_mask(self.array)
        if self.path is None:
            raise ValueError("Mask has neither array nor path.")
        arr = self._read_mask(self.path)
        self.array = ensure_bool_mask(arr)
        return self.array

    @staticmethod
    def _read_mask(p: Path) -> np.ndarray:
        # Try PIL
        try:
            with Image.open(p) as im:
                return np.array(im.convert("L")) > 0
        except Exception:
            pass
        # Try OpenCV
        try:
            arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                raise ValueError("cv2.imread returned None")
            return arr > 0
        except Exception:
            pass
        # Try imageio
        try:
            arr = iio.imread(str(p))
            if arr.ndim == 3:
                arr = arr[..., 0]
            return arr > 0
        except Exception as e:
            raise RuntimeError(f"Unable to read mask at {p}: {e}") from e

    def to_dict(self) -> Dict[str, Any]:
        return {"path": str(self.path) if self.path else None, "has_array": self.array is not None}

    # -------- Optional convenience (not required by Image; Image aligns sizes itself) --------

    def bbox_norm(self, img_w: int, img_h: int) -> Optional[NormalizedBox]:
        """
        YOLO-normalized (xc, yc, w, h) derived from this mask.
        NOTE: This assumes the mask is aligned to the image dimensions. If not,
        prefer Image._mask_bbox_norm_aligned(), which pads/crops to image size first.
        """
        m = self.load()
        if not m.any():
            return None
        ys, xs = np.nonzero(m)
        # +1 on max edges (exclusive upper edge convention)
        x1, x2 = float(xs.min()), float(xs.max() + 1)
        y1, y2 = float(ys.min()), float(ys.max() + 1)
        return NormalizedBox.from_xyxy(x1, y1, x2, y2, img_w, img_h)