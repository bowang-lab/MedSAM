# src/imgpipe/image.py
# Image record storing ONLY normalized boxes; metrics/vis preserved.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple
from enum import Enum

import numpy as np
from pathlib import Path
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .binary_mask_ref import BinaryMaskRef
from .normalized_box import NormalizedBox
from .enums import LabelType, Structure, Eye, Ethnicity
from src.utils import gen_uid, read_image_size


@dataclass
class Image:
    """
    All metadata and annotations for a single fundus image.
    Masks are optional; boxes are stored as normalized YOLO (xc,yc,w,h).
    """
    # Identity / dataset
    uid: str
    dataset: str
    subject_id: str

    # Image payload
    image_path: Path
    width: int
    height: int
    split: Optional[str] = None  # "train" | "val" | "test" | None

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

    # Simple cached metrics
    gt_cd_ratio: Optional[float] = None
    pred_cd_ratio: Optional[float] = None

    # --- NEW: cached mask Dice (prediction vs GT), per class ---
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

    # ----------------- construction -----------------

    @staticmethod
    def _coerce_maskref(mask: BinaryMaskRef | Path | np.ndarray) -> BinaryMaskRef:
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
        return Image(
            uid=uid or gen_uid(),
            dataset=dataset,
            subject_id=subject_id,
            image_path=p,
            width=w,
            height=h,
            split=split,
        )

    def set_box_norm(
            self,
            component: Structure,
            kind: LabelType,
            nbox: Optional[NormalizedBox | Tuple[float, float, float, float] | Dict[str, float]],
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
            # Clip to [0,1] and enforce non-negativity for robustness against tiny drift.
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
                "nbox must be a NormalizedBox, (xc,yc,w,h) tuple/list, dict with keys {'xc','yc','w','h'}, or None.")

        setattr(self, self._box_attr_name(component, kind), nb)

    # ----------------- setters -----------------

    def set_mask(self, component: Structure, kind: LabelType, mask: BinaryMaskRef | Path | np.ndarray) -> None:
        ref = self._coerce_maskref(mask)
        attr_map = {
            (Structure.DISC, LabelType.GT): "gt_disc_mask",
            (Structure.DISC, LabelType.PRED): "pred_disc_mask",
            (Structure.CUP,  LabelType.GT): "gt_cup_mask",
            (Structure.CUP,  LabelType.PRED): "pred_cup_mask",
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

    def _box_attr_name(self, component: Structure, kind: LabelType) -> str:
        return {
            (Structure.DISC, LabelType.GT):   "gt_disc_box",
            (Structure.DISC, LabelType.PRED): "inter_pred_disc_box",
            (Structure.CUP,  LabelType.GT):   "gt_cup_box",
            (Structure.CUP,  LabelType.PRED): "inter_pred_cup_box",
        }[(component, kind)]

    def set_box(self, component: Structure, kind: LabelType, nbox: Optional[NormalizedBox]) -> None:
        setattr(self, self._box_attr_name(component, kind), nbox)

    def set_split(self, split: Optional[str]) -> None:
        if split not in (None, "train", "val", "test"):
            raise ValueError("split must be one of None, 'train', 'val', 'test'")
        self.split = split

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
        """
        Compute a normalized bbox from a mask AFTER aligning it to image (H,W).
        """
        m = self._mask_to_image_size(mref)
        if m is None or not m.any():
            return None
        ys, xs = np.nonzero(m)
        # +1 on max edges (exclusive upper edge convention)
        x1, x2 = float(xs.min()), float(xs.max() + 1)
        y1, y2 = float(ys.min()), float(ys.max() + 1)
        return NormalizedBox.from_xyxy(x1, y1, x2, y2, self.width, self.height)

    # Helper: rasterize a normalized box into a bool mask at (H,W)
    def _rasterize_box(self, nbox: Optional[NormalizedBox]) -> Optional[np.ndarray]:
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

    # in src/imgpipe/image.py
    def ensure_boxes_from_masks(self) -> None:
        """Populate missing normalized boxes from masks (aligned to image size)."""
        if self.gt_disc_box is None:
            self.gt_disc_box = self._mask_bbox_norm_aligned(self.gt_disc_mask)
        if self.gt_cup_box is None:
            self.gt_cup_box = self._mask_bbox_norm_aligned(self.gt_cup_mask)

        # FIX: SAM output masks must drive the *final* predicted boxes
        if self.pred_disc_box is None:
            self.pred_disc_box = self._mask_bbox_norm_aligned(self.pred_disc_mask)
        if self.pred_cup_box is None:
            self.pred_cup_box = self._mask_bbox_norm_aligned(self.pred_cup_mask)

    # ----------------- normalized accessors -----------------

    def get_box_norm(self, component: Structure, kind: LabelType) -> Optional[NormalizedBox]:
        return getattr(self, self._box_attr_name(component, kind))

    # ----------------- YOLO export -----------------

    def yolo_lines_2class(self, use_gt: bool = True) -> Iterable[str]:
        """
        Yield YOLO-normalized lines: '<cls> <xc> <yc> <w> <h>' (0=disc, 1=cup).
        """
        self.ensure_boxes_from_masks()
        boxes = (
            (0, self.gt_disc_box), (1, self.gt_cup_box)
        ) if use_gt else (
            (0, self.inter_pred_disc_box), (1, self.inter_pred_cup_box)
        )
        for cls_id, nbox in boxes:
            if nbox is None:
                continue
            xc, yc, w, h = nbox.as_tuple()
            yield f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

    # ----------------- metrics -----------------

    def cdr(self, *, use_pred: bool = False, axis: str = "vertical") -> Optional[float]:
        """
        Cup-to-Disc Ratio (mask-based preferred; box-based fallback).
        axis: "vertical" (default) or "horizontal".
        """
        assert axis in ("vertical", "horizontal")
        # Prefer masks
        dm = self._mask_to_image_size(self.pred_disc_mask if use_pred else self.gt_disc_mask)
        cm = self._mask_to_image_size(self.pred_cup_mask  if use_pred else self.gt_cup_mask)

        def _extent(mask: np.ndarray, kind: str) -> Optional[float]:
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                return None
            if kind == "vertical":
                return float(ys.max() - ys.min() + 1)
            return float(xs.max() - xs.min() + 1)

        if dm is not None and cm is not None:
            d = _extent(dm, axis)
            c = _extent(cm, axis)
            if d and d > 0:
                return (c or 0.0) / d

        # Fallback to normalized boxes (ratio is scale-invariant)
        dbox = (self.pred_disc_box if use_pred else self.gt_disc_box)
        cbox = (self.pred_cup_box if use_pred else self.gt_cup_box)
        if dbox and cbox:
            if axis == "vertical":
                d = max(0.0, dbox.height)
                c = max(0.0, cbox.height)
            else:
                d = max(0.0, dbox.width)
                c = max(0.0, cbox.width)
            return (c / d) if d > 0 else None
        return None

    # --- NEW: class-wise Dice computation & caching ---

    @staticmethod
    def _dice_from_bool(pred: Optional[np.ndarray], gt: Optional[np.ndarray]) -> Optional[float]:
        if pred is None or gt is None:
            return None
        p = pred.astype(bool)
        g = gt.astype(bool)
        if not p.any() and not g.any():
            return 1.0  # both empty → perfect agreement
        inter = float((p & g).sum())
        denom = float(p.sum() + g.sum())
        return (2.0 * inter / denom) if denom > 0 else 0.0

    def compute_mask_dice(
        self,
        component: Structure,
        *,
        fallback_to_boxes: bool = True
    ) -> Optional[float]:
        """
        Compute (and return) Dice between predicted and GT masks for a component.
        If masks are missing and fallback_to_boxes is True, rasterize available boxes.
        Does not change existing fields except updating the cached dice attribute.
        """
        assert component in (Structure.DISC, Structure.CUP)
        # Load masks aligned to image size
        pred_m = self._mask_to_image_size(self.pred_disc_mask if component is Structure.DISC else self.pred_cup_mask)
        gt_m   = self._mask_to_image_size(self.gt_disc_mask   if component is Structure.DISC else self.gt_cup_mask)

        # Optional fallbacks from boxes
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
        """
        Compute Dice for both classes and cache them.
        """
        return {
            "disc": self.compute_mask_dice(Structure.DISC, fallback_to_boxes=fallback_to_boxes),
            "cup":  self.compute_mask_dice(Structure.CUP,  fallback_to_boxes=fallback_to_boxes),
        }

    def rim_metrics(self, *, use_pred: bool = False) -> Optional[Dict[str, Optional[float]]]:
        """
        Compute neuroretinal rim metrics from masks:
          - Rim-to-Disc area ratio (R/D)
          - I/S : Inferior-to-Superior rim area ratio
          - I/N : Inferior-to-Nasal rim area ratio (requires known laterality)
          - I/T : Inferior-to-Temporal rim area ratio (requires known laterality)
        """
        disc = self._mask_to_image_size(self.pred_disc_mask if use_pred else self.gt_disc_mask)
        cup  = self._mask_to_image_size(self.pred_cup_mask  if use_pred else self.gt_cup_mask)
        if disc is None or cup is None:
            return None

        H, W = self.height, self.width
        if not disc.any():
            return None

        rim = disc & (~cup)

        # Areas
        disc_area = float(disc.sum())
        rim_area  = float(rim.sum())
        r_over_d  = rim_area / disc_area if disc_area > 0 else np.nan

        # Disc center from disc mask
        ys, xs = np.nonzero(disc)
        yc = float(ys.mean()) if ys.size else (H / 2.0)
        xc = float(xs.mean()) if xs.size else (W / 2.0)

        # Superior/Inferior split
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
            # Optional raw areas
            "rim_area": rim_area,
            "disc_area": disc_area,
            "inferior_area": rim_inf,
            "superior_area": rim_sup,
            "nasal_area": None,
            "temporal_area": None,
        }

        # Only compute Nasal/Temporal if laterality is known
        if self.laterality is not None:
            left  = (np.arange(W)[None, :] <  xc)
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
        out["gt"]["cdr_v"]   = self.cdr(use_pred=False, axis="vertical")
        out["gt"]["cdr_h"]   = self.cdr(use_pred=False, axis="horizontal")
        out["pred"]["cdr_v"] = self.cdr(use_pred=True,  axis="vertical")
        out["pred"]["cdr_h"] = self.cdr(use_pred=True,  axis="horizontal")

        gt_r = self.rim_metrics(use_pred=False)
        pr_r = self.rim_metrics(use_pred=True)
        if gt_r:
            out["gt"].update(gt_r)
        if pr_r:
            out["pred"].update(pr_r)
        return out

    # ----------------- visualization -----------------
    def visualize(
            self,
            *,
            show: bool = True,
            save_path: Optional[Path] = None,
            dpi: int = 150,
            figsize: Tuple[int, int] = (14, 6),
            mask_alpha: float = 0.35,
            show_metrics: bool = True,
    ) -> None:
        """
        Render 1–2 panels:
          • Panel 1 (if available): Ground-truth disc/cup masks + GT metrics.
          • Panel 2 (if available): Predicted disc/cup masks + detector inter_pred boxes + predicted metrics.
        """
        # Load base image
        img = PILImage.open(self.image_path).convert("RGB")
        W, H = img.size

        # Helpers
        def _mask_arr(mref: Optional[BinaryMaskRef]) -> Optional[np.ndarray]:
            return self._mask_to_image_size(mref)

        def _draw_mask(ax, mask: np.ndarray, color: str, alpha: float):
            overlay = np.zeros((H, W, 4), dtype=float)
            rgba = plt.matplotlib.colors.to_rgba(color, alpha=alpha)
            overlay[mask] = rgba
            ax.imshow(overlay, origin="upper")

        def _draw_nbox(ax, nbox: Optional[NormalizedBox], color: str, ls: str = "-", lw: float = 2.0):
            if nbox is None:
                return
            x1, y1, x2, y2 = nbox.to_pixel_xyxy(W, H)
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            ax.add_patch(Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linewidth=lw, linestyle=ls))

        # Determine availability
        gt_disc_m = _mask_arr(self.gt_disc_mask)
        gt_cup_m = _mask_arr(self.gt_cup_mask)
        pred_disc_m = _mask_arr(self.pred_disc_mask)
        pred_cup_m = _mask_arr(self.pred_cup_mask)

        has_gt = (gt_disc_m is not None) or (gt_cup_m is not None)
        has_pred = (pred_disc_m is not None) or (pred_cup_m is not None) or \
                   (self.inter_pred_disc_box is not None) or (self.inter_pred_cup_box is not None)

        # Nothing to show → single image only
        if not has_gt and not has_pred:
            fig, ax = plt.subplots(1, 1, figsize=(figsize[0] / 2, figsize[1]), dpi=dpi)
            ax.imshow(img, origin="upper")
            ax.set_axis_off()
            if save_path is not None:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(save_path), bbox_inches="tight", dpi=dpi)
            if show:
                plt.show()
            else:
                plt.close(fig)
            return

        # Assemble panels: strictly at most two
        panels = []
        if has_gt:
            panels.append(("Ground Truth", {"disc_m": gt_disc_m, "cup_m": gt_cup_m,
                                            "disc_box": None, "cup_box": None, "use_pred": False}))
        if has_pred:
            panels.append(("Predictions", {"disc_m": pred_disc_m, "cup_m": pred_cup_m,
                                           "disc_box": self.inter_pred_disc_box,
                                           "cup_box": self.inter_pred_cup_box,
                                           "use_pred": True}))

        ncols = len(panels)
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize if ncols == 2 else (figsize[0] / 2, figsize[1]),
                                 dpi=dpi, squeeze=False)

        # Figure title
        suptitle = f"{self.dataset} / {self.subject_id} — {self.image_path.name}"
        fig.suptitle(suptitle, fontsize=12)

        # Colors
        disc_color_mask = "tab:red"
        cup_color_mask = "tab:blue"
        disc_color_box = "red"
        cup_color_box = "blue"

        # Update cached Dice (non-intrusive; uses masks and optional box fallbacks)
        self.update_mask_dice(fallback_to_boxes=True)

        for j, (title, dct) in enumerate(panels):
            ax = axes[0, j]
            ax.imshow(img, origin="upper")
            ax.set_axis_off()
            ax.set_title(title, fontsize=11)

            # Masks
            if dct["disc_m"] is not None:
                _draw_mask(ax, dct["disc_m"], disc_color_mask, mask_alpha)
            if dct["cup_m"] is not None:
                _draw_mask(ax, dct["cup_m"], cup_color_mask, mask_alpha)

            # Boxes (only inter_pred on predictions panel as requested)
            _draw_nbox(ax, dct["disc_box"], disc_color_box, ls="-", lw=2.0)
            _draw_nbox(ax, dct["cup_box"], cup_color_box, ls="--", lw=2.0)

            # Metrics block
            if show_metrics:
                use_pred = bool(dct["use_pred"])
                cdr_v = self.cdr(use_pred=use_pred, axis="vertical")
                cdr_h = self.cdr(use_pred=use_pred, axis="horizontal")
                rims = self.rim_metrics(use_pred=use_pred)

                lines = []
                # Show Dice on prediction panel (GT vs Pred)
                if use_pred:
                    if self.mask_dice_disc is not None:
                        lines.append(f"Dice (Disc): {self.mask_dice_disc:.3f}")
                    if self.mask_dice_cup is not None:
                        lines.append(f"Dice (Cup):  {self.mask_dice_cup:.3f}")

                if cdr_v is not None:
                    lines.append(f"CDR (V): {cdr_v:.3f}")
                if cdr_h is not None:
                    lines.append(f"CDR (H): {cdr_h:.3f}")
                if rims is not None:
                    if np.isfinite(rims.get("rim_over_disc", np.nan)):
                        lines.append(f"R/D: {rims['rim_over_disc']:.3f}")
                    if np.isfinite(rims.get("I_over_S", np.nan)):
                        lines.append(f"I/S: {rims['I_over_S']:.3f}")
                    ion = rims.get("I_over_N")
                    iot = rims.get("I_over_T")
                    if ion is not None and np.isfinite(ion):
                        lines.append(f"I/N: {ion:.3f}")
                    if iot is not None and np.isfinite(iot):
                        lines.append(f"I/T: {iot:.3f}")

                if lines:
                    ax.text(
                        0.02, 0.02, "\n".join(lines),
                        transform=ax.transAxes,
                        fontsize=9, va="top", ha="left", color="white",
                        bbox=dict(facecolor="black", alpha=0.35, boxstyle="round,pad=0.3", edgecolor="none"),
                    )

            # Legend (only if something was drawn)
            handles = []
            if (dct["disc_m"] is not None) or (dct["disc_box"] is not None):
                handles.append(plt.Line2D([0], [0], color=disc_color_box, lw=2, linestyle="-", label="Disc"))
            if (dct["cup_m"] is not None) or (dct["cup_box"] is not None):
                handles.append(plt.Line2D([0], [0], color=cup_color_box, lw=2, linestyle="--", label="Cup"))
            if handles:
                ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=True)

        fig.tight_layout(rect=(0, 0, 1, 0.96))

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), bbox_inches="tight", dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close(fig)

    # ----------------- serialization -----------------

    @staticmethod
    def _path_to_str(p: Optional[Path]) -> Optional[str]:
        return str(p) if p is not None else None

    @staticmethod
    def _path_from_str(s: Optional[str]) -> Optional[Path]:
        return Path(s) if s else None

    @staticmethod
    def _mask_to_dict(m: Optional[BinaryMaskRef]) -> Optional[Dict[str, Any]]:
        return m.to_dict() if m is not None else None

    @staticmethod
    def _mask_from_dict(dct: Optional[Dict[str, Any]]) -> Optional[BinaryMaskRef]:
        if not dct:
            return None
        path = dct.get("path")
        return BinaryMaskRef(path=Path(path) if path else None)

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

    def to_dict(self, *, drop_none: bool = False) -> Dict[str, Any]:
        self.ensure_boxes_from_masks()  # ensure normalized GT exist if masks available
        d: Dict[str, Any] = {
            "_schema": 5,  # bumped since format changed (added cached mask dice)

            # identity
            "uid": self.uid,
            "dataset": self.dataset,
            "subject_id": self.subject_id,

            # image payload
            "image_path": str(self.image_path),
            "width": int(self.width),
            "height": int(self.height),
            "split": self.split,

            # masks
            "gt_disc_mask": self._mask_to_dict(self.gt_disc_mask),
            "gt_cup_mask": self._mask_to_dict(self.gt_cup_mask),
            "pred_disc_mask": self._mask_to_dict(self.pred_disc_mask),
            "pred_cup_mask": self._mask_to_dict(self.pred_cup_mask),

            # boxes (normalized only)
            "gt_disc_box": self._boxn_to_dict(self.gt_disc_box),
            "gt_cup_box":  self._boxn_to_dict(self.gt_cup_box),
            "inter_pred_disc_box": self._boxn_to_dict(self.inter_pred_disc_box),
            "inter_pred_cup_box":  self._boxn_to_dict(self.inter_pred_cup_box),
            "pred_disc_box": self._boxn_to_dict(self.pred_disc_box),
            "pred_cup_box":  self._boxn_to_dict(self.pred_cup_box),

            # lightweight cached metrics
            "gt_cd_ratio": self.gt_cd_ratio,
            "pred_cd_ratio": self.pred_cd_ratio,

            # NEW: cached dice
            "mask_dice_disc": self.mask_dice_disc,
            "mask_dice_cup": self.mask_dice_cup,

            # bookkeeping
            "yolo_label_path": self._path_to_str(self.yolo_label_path),
            "extras": self.extras or {},

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
        obj = Image(
            uid=d["uid"],
            dataset=d["dataset"],
            subject_id=d["subject_id"],
            image_path=P(d.get("image_path")) or Path("."),
            width=int(d["width"]),
            height=int(d["height"]),
            split=d.get("split"),
            yolo_label_path=P(d.get("yolo_label_path")),
            extras=d.get("extras") or {},
        )

        # masks
        obj.gt_disc_mask  = Image._mask_from_dict(d.get("gt_disc_mask"))
        obj.gt_cup_mask   = Image._mask_from_dict(d.get("gt_cup_mask"))
        obj.pred_disc_mask = Image._mask_from_dict(d.get("pred_disc_mask"))
        obj.pred_cup_mask  = Image._mask_from_dict(d.get("pred_cup_mask"))

        # boxes (normalized only)
        obj.gt_disc_box         = Image._boxn_from_dict(d.get("gt_disc_box"))
        obj.gt_cup_box          = Image._boxn_from_dict(d.get("gt_cup_box"))
        obj.inter_pred_disc_box = Image._boxn_from_dict(d.get("inter_pred_disc_box"))
        obj.inter_pred_cup_box  = Image._boxn_from_dict(d.get("inter_pred_cup_box"))
        obj.pred_disc_box       = Image._boxn_from_dict(d.get("pred_disc_box"))
        obj.pred_cup_box        = Image._boxn_from_dict(d.get("pred_cup_box"))

        # cached metrics
        obj.gt_cd_ratio   = (float(d["gt_cd_ratio"])   if d.get("gt_cd_ratio")   is not None else None)
        obj.pred_cd_ratio = (float(d["pred_cd_ratio"]) if d.get("pred_cd_ratio") is not None else None)

        # NEW: cached dice
        obj.mask_dice_disc = (float(d["mask_dice_disc"]) if d.get("mask_dice_disc") is not None else None)
        obj.mask_dice_cup  = (float(d["mask_dice_cup"])  if d.get("mask_dice_cup")  is not None else None)

        # patient info
        obj.laterality = Image._enum_from_str(Eye, d.get("eye"))
        obj.age        = (int(d["age"]) if d.get("age") is not None else None)
        obj.ethnicity  = Image._enum_from_str(Ethnicity, d.get("ethnicity"))
        obj.glaucoma   = (bool(d["glaucoma"]) if d.get("glaucoma") is not None else None)
        obj.gt_cdr     = (float(d["gt_cdr"]) if d.get("gt_cdr") is not None else None)
        obj.pred_cdr   = (float(d["pred_cdr"]) if d.get("pred_cdr") is not None else None)
        obj.gt_rdr     = (float(d["gt_rdr"]) if d.get("gt_rdr") is not None else None)
        obj.pred_rdr   = (float(d["pred_rdr"]) if d.get("pred_rdr") is not None else None)

        return obj

    def to_json(self, *, drop_none: bool = False) -> str:
        return json.dumps(
            self.to_dict(drop_none=drop_none),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @staticmethod
    def from_json(s: str) -> "Image":
        return Image.from_dict(json.loads(s))

    def __repr__(self) -> str:
        return (f"ImageSample(uid={self.uid!r}, ds={self.dataset!r}, subj={self.subject_id!r}, "
                f"size=({self.width}x{self.height}), split={self.split!r})")