# MedSAM/src/imgpipe/image.py

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .binary_mask_ref import BinaryMaskRef
from .bounding_box import BoundingBox
from .enums import LabelType, Structure
from src.utils import gen_uid, read_image_size


@dataclass
class Image:
    """
    All metadata and annotations for a single fundus image.
    You can attach masks/boxes later via set_mask/set_box.
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

    # GT Mask
    gt_disc_mask: Optional[BinaryMaskRef] = None
    gt_cup_mask: Optional[BinaryMaskRef] = None

    # MedSam Mask Prediction
    pred_disc_mask: Optional[BinaryMaskRef] = None
    pred_cup_mask: Optional[BinaryMaskRef] = None

    # BB from GT Mask
    gt_disc_box: Optional[BoundingBox] = None
    gt_cup_box: Optional[BoundingBox] = None

    # Predicted BB from Object Recognition Model
    inter_pred_disc_box: Optional[BoundingBox] = None
    inter_pred_cup_box: Optional[BoundingBox] = None

    # Predicted BB from MedSam Mask
    pred_disc_box: Optional[BoundingBox] = None
    pred_cup_box: Optional[BoundingBox] = None

    gt_cd_ratio: Optional[float] = None
    pred_cd_ratio: Optional[float] = None


    # Optional bookkeeping
    yolo_label_path: Optional[Path] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    # ---------- factories ----------

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

    # ---------- tiny setters ----------

    def set_mask(self, which: Structure, kind: LabelType, mask: BinaryMaskRef | Path | Any) -> None:
        ref = mask if isinstance(mask, BinaryMaskRef) else BinaryMaskRef(path=mask if isinstance(mask, Path) else None,
                                                                         array=mask if not isinstance(mask, Path) else None)
        if which == Structure.DISC and kind == LabelType.GT:
            self.gt_disc_mask = ref
        elif which == Structure.DISC and kind == LabelType.PRED:
            self.pred_disc_mask = ref
        elif which == Structure.CUP and kind == LabelType.GT:
            self.gt_cup_mask = ref
        elif which == Structure.CUP and kind == LabelType.PRED:
            self.pred_cup_mask = ref
        else:
            raise ValueError(f"Unsupported (which, kind)=({which},{kind})")

    def set_box(self, which: Structure, kind: LabelType, box: Optional[BoundingBox]) -> None:
        if which == Structure.DISC and kind == LabelType.GT:
            self.gt_disc_box = box
        elif which == Structure.DISC and kind == LabelType.PRED:
            self.inter_pred_disc_box = box
        elif which == Structure.CUP and kind == LabelType.GT:
            self.gt_cup_box = box
        elif which == Structure.CUP and kind == LabelType.PRED:
            self.inter_pred_cup_box = box
        else:
            raise ValueError(f"Unsupported (which, kind)=({which},{kind})")

    def set_split(self, split: Optional[str]) -> None:
        if split not in (None, "train", "val", "test"):
            raise ValueError("split must be one of None, 'train', 'val', 'test'")
        self.split = split

    # ---------- derivations ----------

    def ensure_boxes_from_masks(self) -> None:
        if self.gt_disc_box is None and self.gt_disc_mask is not None:
            self.gt_disc_box = self.gt_disc_mask.bbox()
        if self.gt_cup_box is None and self.gt_cup_mask is not None:
            self.gt_cup_box = self.gt_cup_mask.bbox()
        if self.inter_pred_disc_box is None and self.pred_disc_mask is not None:
            self.inter_pred_disc_box = self.pred_disc_mask.bbox()
        if self.inter_pred_cup_box is None and self.pred_cup_mask is not None:
            self.inter_pred_cup_box = self.pred_cup_mask.bbox()

    def yolo_lines_2class(self, use_gt: bool = True) -> Iterable[str]:
        """Yield normalized YOLO lines: '<cls> <xc> <yc> <w> <h>' (0=disc, 1=cup)."""
        self.ensure_boxes_from_masks()
        boxes = (
            (0, self.gt_disc_box), (1, self.gt_cup_box)
        ) if use_gt else (
            (0, self.inter_pred_disc_box), (1, self.inter_pred_cup_box)
        )
        W, H = self.width, self.height
        for cls, box in boxes:
            if box is None:
                continue
            xc, yc, w, h = box.to_yolo_norm(W, H)
            yield f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

    # ---------- small metrics ----------

    def disc_iou(self) -> Optional[float]:
        if self.gt_disc_box and self.inter_pred_disc_box:
            return self.gt_disc_box.iou(self.inter_pred_disc_box)
        return None

    def cup_iou(self) -> Optional[float]:
        if self.gt_cup_box and self.inter_pred_cup_box:
            return self.gt_cup_box.iou(self.inter_pred_cup_box)
        return None

    # ---------- serialization ----------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["image_path"] = str(self.image_path)
        if self.yolo_label_path:
            d["yolo_label_path"] = str(self.yolo_label_path)
        # replace mask refs with light dicts
        for k in ("gt_disc_mask", "gt_cup_mask", "pred_disc_mask", "pred_cup_mask"):
            ref = getattr(self, k)
            d[k] = ref.to_dict() if ref else None
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Image":
        def _p(x): return Path(x) if x else None

        obj = Image(
            uid=d["uid"],
            dataset=d["dataset"],
            subject_id=d["subject_id"],
            image_path=_p(d["image_path"]),
            width=int(d["width"]),
            height=int(d["height"]),
            split=d.get("split"),
            yolo_label_path=_p(d.get("yolo_label_path")),
            extras=d.get("extras") or {},
        )

        # masks
        def _mask(md):
            if not md:
                return None
            return BinaryMaskRef(path=_p(md.get("path")))

        obj.gt_disc_mask = _mask(d.get("gt_disc_mask"))
        obj.gt_cup_mask = _mask(d.get("gt_cup_mask"))
        obj.pred_disc_mask = _mask(d.get("pred_disc_mask"))
        obj.pred_cup_mask = _mask(d.get("pred_cup_mask"))

        # boxes
        def _box(bd):
            if not bd:
                return None
            return BoundingBox(float(bd["x1"]), float(bd["y1"]), float(bd["x2"]), float(bd["y2"]))

        obj.gt_disc_box = _box(d.get("gt_disc_box"))
        obj.gt_cup_box = _box(d.get("gt_cup_box"))
        obj.inter_pred_disc_box = _box(d.get("pred_disc_box"))
        obj.inter_pred_cup_box = _box(d.get("pred_cup_box"))
        return obj

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "Image":
        return Image.from_dict(json.loads(s))

    # ---------- niceties ----------

    def __repr__(self) -> str:
        return (f"ImageSample(uid={self.uid!r}, ds={self.dataset!r}, subj={self.subject_id!r}, "
                f"size=({self.width}x{self.height}), split={self.split!r})")