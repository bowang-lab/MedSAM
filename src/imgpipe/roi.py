from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image as PILImage

from src.utils import ensure_dir
from .yolo_io import YoloDatasetIO


def _read_first_line_for_class(label_file: Path, class_id: int) -> Optional[Tuple[float, float, float, float]]:
    if not label_file.exists():
        return None
    for ln in label_file.read_text().splitlines():
        parts = ln.strip().split()
        if not parts:
            continue
        try:
            cid = int(float(parts[0]))
        except Exception:
            continue
        if cid == class_id:
            x, y, w, h = map(float, parts[1:5])
            return x, y, w, h
    return None


def _xywhn_to_xyxy(px_w: int, px_h: int, x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int]:
    cx = x * px_w
    cy = y * px_h
    bw = w * px_w
    bh = h * px_h
    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    return x1, y1, x2, y2


def _pad_box(x1: int, y1: int, x2: int, y2: int, W: int, H: int, pad_pct: float) -> Tuple[int, int, int, int]:
    pad = int(round(max(W, H) * pad_pct))
    nx1 = max(0, x1 - pad)
    ny1 = max(0, y1 - pad)
    nx2 = min(W, x2 + pad)
    ny2 = min(H, y2 + pad)
    return nx1, ny1, nx2, ny2


def _clip(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def _xyxy_to_xywhn_in_crop(W_roi: int, H_roi: int, x1: int, y1: int, x2: int, y2: int) -> Tuple[float, float, float, float]:
    cx = (x1 + x2) / 2.0 / W_roi
    cy = (y1 + y2) / 2.0 / H_roi
    w = (x2 - x1) / W_roi
    h = (y2 - y1) / H_roi
    return cx, cy, w, h


class ROIDatasetBuilder:
    """
    Build a cup-ROI dataset from an existing YOLO split:
    - ROI is a crop around the disc (class 0) with padding.
    - Labels contain ONLY the cup (class 1) mapped to the ROI coordinate system (class 0 in the ROI dataset).
    """

    def __init__(self, pad_pct: float, keep_negatives: bool = False) -> None:
        self.pad_pct = pad_pct
        self.keep_negatives = keep_negatives

    def build(self, src_root: Path, out_root: Path) -> None:
        for split in ("train", "val", "test"):
            src_img_dir = src_root / "images" / split
            src_lbl_dir = src_root / "labels" / split
            if not src_img_dir.exists() or not src_lbl_dir.exists():
                continue

            dst_img_dir = out_root / "images" / split
            dst_lbl_dir = out_root / "labels" / split
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

            for ip in sorted(src_img_dir.iterdir()):
                if not ip.is_file():
                    continue
                lp = src_lbl_dir / f"{ip.stem}.txt"
                if not lp.exists():
                    continue

                with PILImage.open(ip).convert("RGB") as im:
                    W, H = im.size

                    disc_xywhn = _read_first_line_for_class(lp, 0)  # disc
                    if disc_xywhn is None:
                        # cannot form ROI without disc
                        continue
                    dx, dy, dw, dh = disc_xywhn
                    dx1, dy1, dx2, dy2 = _xywhn_to_xyxy(W, H, dx, dy, dw, dh)
                    rx1, ry1, rx2, ry2 = _pad_box(dx1, dy1, dx2, dy2, W, H, self.pad_pct)

                    roi = im.crop((rx1, ry1, rx2, ry2))
                    W_roi, H_roi = roi.size

                    cup_xywhn = _read_first_line_for_class(lp, 1)  # cup
                    cup_lines_out: list[str] = []
                    if cup_xywhn is not None:
                        cx, cy, cw, ch = cup_xywhn
                        cx1, cy1, cx2, cy2 = _xywhn_to_xyxy(W, H, cx, cy, cw, ch)
                        # map to ROI coords
                        nx1 = _clip(cx1 - rx1, 0, W_roi - 1)
                        ny1 = _clip(cy1 - ry1, 0, H_roi - 1)
                        nx2 = _clip(cx2 - rx2 + W_roi, 0, W_roi - 1)
                        ny2 = _clip(cy2 - ry2 + H_roi, 0, H_roi - 1)
                        # discard degenerate
                        if nx2 > nx1 and ny2 > ny1:
                            x, y, w, h = _xyxy_to_xywhn_in_crop(W_roi, H_roi, nx1, ny1, nx2, ny2)
                            # single-class 'cup' => class 0 in ROI dataset
                            cup_lines_out.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

                    if (not cup_lines_out) and (not self.keep_negatives):
                        # skip negative ROIs unless asked to keep
                        continue

                    # save
                    dst_ip = dst_img_dir / ip.name
                    dst_lp = dst_lbl_dir / f"{ip.stem}.txt"
                    roi.save(dst_ip)
                    with open(dst_lp, "w") as f:
                        for ln in cup_lines_out:
                            f.write(ln + "\n")

        # single name
        YoloDatasetIO.write_data_yaml(out_root, names=["cup"])