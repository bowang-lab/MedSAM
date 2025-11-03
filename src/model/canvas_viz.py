# canvas_viz.py
from __future__ import annotations
import inspect
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES: Tuple[str, ...] = ("disc", "cup") # class name lookup by id


# ---- Your dataclasses/enums ----
from src.imgpipe.normalized_box import NormalizedBox
from src.imgpipe.enums import Structure  # only used for readable comments / naming
# =============================
# Canvas visualization (no coord changes)
# =============================

def _tensor_canvas_to_pil(img_t: torch.Tensor) -> Image.Image:
    """
    img_t: (3,H,W) float in [0,1] or uint8 in [0,255]
    Returns a PIL RGB image that is exactly the *letterboxed canvas* used by the trainer.
    """
    if img_t.ndim != 3:
        raise ValueError("Expected CHW image tensor")
    if img_t.dtype.is_floating_point:
        arr = (img_t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    else:
        arr = img_t.permute(1, 2, 0).cpu().numpy()
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")

def _denorm_xywhn_to_xyxy_on_canvas(xc: float, yc: float, w: float, h: float, W: int, H: int) -> Tuple[float,float,float,float]:
    """Convert GT normalized (xc,yc,w,h) to pixel xyxy on the same canvas."""
    x1 = (xc - w/2.0) * W
    y1 = (yc - h/2.0) * H
    x2 = (xc + w/2.0) * W
    y2 = (yc + h/2.0) * H
    return x1, y1, x2, y2

def _draw_box(draw: ImageDraw.ImageDraw, xyxy: Tuple[float,float,float,float], color: str, width: int = 2) -> None:
    x1, y1, x2, y2 = xyxy
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

def _put_label(draw: ImageDraw.ImageDraw, xyxy: Tuple[float,float,float,float], text: str, color: str) -> None:
    x1, y1, _, _ = xyxy
    x1_i, y1_i = int(x1), int(y1)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    try:
        x0, y0, x2, y2 = draw.textbbox((0, 0), text, font=font)
        tw, th = (x2 - x0), (y2 - y0)
    except Exception:
        tw = 6 * len(text) + 2
        th = 11
    pad = 2
    bg_x1 = x1_i
    bg_y1 = max(0, y1_i - th - pad)
    bg_x2 = x1_i + tw + 2 * pad
    bg_y2 = y1_i
    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=color)
    draw.text((bg_x1 + pad, max(0, y1_i - th - pad + 1)), text, fill="white", font=font)

def save_val_canvas_debug(
    out_dir: str,
    img_tensor_chw: torch.Tensor,                   # (3,H,W) from batch["img"][i]
    fname: Optional[str],                           # original filename for naming
    preds_px: List[Tuple[float,float,float,float,int,float]],   # predictions: xyxy on canvas
    gts_norm: List[Tuple[float,float,float,float,int]],         # GT: normalized on canvas
    class_names: Tuple[str, ...] = CLASS_NAMES
) -> str:
    """
    Saves a PNG with RED predicted boxes and GREEN GT boxes over the trainer canvas.
    Returns the saved file path.
    """
    os.makedirs(out_dir, exist_ok=True)

    canvas = _tensor_canvas_to_pil(img_tensor_chw)
    W, H = canvas.size
    draw = ImageDraw.Draw(canvas)

    # Draw predictions (already xyxy on canvas)
    for (x1, y1, x2, y2, cid, conf) in preds_px:
        _draw_box(draw, (x1, y1, x2, y2), color="red", width=2)
        cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        _put_label(draw, (x1, y1, x2, y2), f"P:{cname} {conf:.2f}", color="red")

    # Draw ground truths (denormalize to canvas)
    for (xc, yc, w, h, cid) in gts_norm:
        x1, y1, x2, y2 = _denorm_xywhn_to_xyxy_on_canvas(xc, yc, w, h, W, H)
        _draw_box(draw, (x1, y1, x2, y2), color="green", width=2)
        cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        _put_label(draw, (x1, y1, x2, y2), f"G:{cname}", color="green")

    # Optional title strip
    title = f"{os.path.basename(fname) if fname else 'val'}  |  canvas {W}x{H}"
    draw.rectangle([0, 0, min(W, 520), 16], fill="black")
    draw.text((2, 2), title, fill="white")

    stem = os.path.splitext(os.path.basename(fname or "val"))[0]
    out_path = os.path.join(out_dir, f"{stem}_canvas_debug.png")
    canvas.save(out_path)
    return out_path
