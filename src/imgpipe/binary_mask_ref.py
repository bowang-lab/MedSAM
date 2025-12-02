# src/imgpipe/binary_mask_ref.py

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.imgpipe.normalized_box import NormalizedBox
from src.utils import ensure_bool_mask
from PIL import Image  # type: ignore
import imageio.v3 as iio  # type: ignore
import cv2  # type: ignore


def _pack_bool_mask(mask: np.ndarray) -> Tuple[bytes, int, int]:
    m = ensure_bool_mask(mask)
    h, w = m.shape
    flat = np.ascontiguousarray(m.reshape(-1).astype(np.uint8))
    packed = np.packbits(flat, bitorder="little")
    return packed.tobytes(), int(h), int(w)


def _unpack_bool_mask(data: bytes, h: int, w: int) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(buf, bitorder="little")
    bits = bits[: h * w]
    return bits.reshape(h, w).astype(bool)


@dataclass
class BinaryMaskRef:
    """
    Holds a reference to a binary mask — either a file path, a packed-bytes blob, or an in-memory array.
    Loads lazily and caches the array when decoded/read.
    """
    path: Optional[Path] = None
    array: Optional[np.ndarray] = field(default=None, repr=False)
    packed: Optional[bytes] = field(default=None, repr=False)
    shape: Optional[Tuple[int, int]] = field(default=None, repr=False)  # (H, W)
    encoding: Optional[str] = field(default=None, repr=False)  # e.g. "packbits_little"

    def load(self) -> np.ndarray:
        """Return mask as a boolean array."""
        if self.array is not None:
            return ensure_bool_mask(self.array)

        # Decode packed bytes if present
        if self.packed is not None and self.shape is not None:
            h, w = self.shape
            arr = _unpack_bool_mask(self.packed, h, w)
            self.array = ensure_bool_mask(arr)
            return self.array

        # Fall back to path
        if self.path is None:
            raise ValueError("Mask has neither array nor packed bytes nor path.")
        arr = self._read_mask(self.path)
        self.array = ensure_bool_mask(arr)
        return self.array

    def pack_inplace(self) -> None:
        """
        Convert any in-memory array into packed bytes and drop the array to reduce RAM.
        """
        if self.array is None:
            return
        data, h, w = _pack_bool_mask(self.array)
        self.packed = data
        self.shape = (h, w)
        self.encoding = "packbits_little"
        self.array = None

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

    def to_dict(self, *, include_mask_bytes: bool = False) -> Dict[str, Any]:
        """
        JSON-safe by default. When include_mask_bytes=True, returns a Parquet-friendly dict with bytes.
        """
        d: Dict[str, Any] = {
            "path": str(self.path) if self.path else None,
            "has_array": self.array is not None,
        }

        if include_mask_bytes:
            # Ensure packed representation exists if we can derive it
            if self.packed is None and self.array is not None:
                self.pack_inplace()

            d.update(
                {
                    "encoding": self.encoding,
                    "h": self.shape[0] if self.shape else None,
                    "w": self.shape[1] if self.shape else None,
                    "data": self.packed,  # bytes (Parquet can store this; JSON cannot)
                }
            )

        return d

    def bbox_norm(self, img_w: int, img_h: int) -> Optional[NormalizedBox]:
        m = self.load()
        if not m.any():
            return None
        ys, xs = np.nonzero(m)
        x1, x2 = float(xs.min()), float(xs.max() + 1)
        y1, y2 = float(ys.min()), float(ys.max() + 1)
        return NormalizedBox.from_xyxy(x1, y1, x2, y2, img_w, img_h)