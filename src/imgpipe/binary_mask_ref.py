# MedSAM/src/imgpipe/binary_mask_ref.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

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
        if self.array is not None:
            return ensure_bool_mask(self.array)
        if self.path is None:
            raise ValueError("Mask has neither array nor path.")
        arr = self._read_mask(self.path)
        self.array = ensure_bool_mask(arr)
        return self.array

    @staticmethod
    def _read_mask(p: Path) -> np.ndarray:
        try:
            with Image.open(p) as im:
                return np.array(im.convert("L")) > 0
        except Exception:
            pass
        try:
            arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                raise ValueError("cv2.imread returned None")
            return arr > 0
        except Exception:
            pass
        try:
            arr = iio.imread(str(p))
            if arr.ndim == 3:
                arr = arr[..., 0]
            return arr > 0
        except Exception as e:
            raise RuntimeError(f"Unable to read mask at {p}: {e}") from e

    def to_dict(self) -> Dict[str, Any]:
        return {"path": str(self.path) if self.path else None, "has_array": self.array is not None}