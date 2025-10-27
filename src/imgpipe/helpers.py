from collections import defaultdict
from pathlib import Path
from typing import List

from src.imgpipe.config import PipelineConfig
from src.imgpipe.image import Image


def normalize_steps(steps_arg: str | None) -> List[str]:
    if not steps_arg:
        return ["collect", "split", "save", "augment", "disc_only", "roi"]
    return [s.strip() for s in steps_arg.split(",") if s.strip()]

def has_multi_datasets(cfg: PipelineConfig) -> bool:
    raw = getattr(cfg, "raw", None)
    return isinstance(raw, dict) and isinstance(raw.get("datasets"), list) and len(raw["datasets"]) > 0

def group_by_subject(images: List[Image]) -> dict[str, List[Image]]:
    by_subj: dict[str, List[Image]] = defaultdict(list)
    for im in images:
        sid = im.subject_id or im.uid  # fallback to uid if subject_id missing
        by_subj[sid].append(im)
    return by_subj

def subset_roots(base_project_dir: Path, n: int, seed: int) -> dict[str, Path]:
    root = base_project_dir / "data" / "_subset" / f"N{n}_seed{seed}"
    return {
        "root": root,
        "yolo_split":      root / "yolo_split",
        "yolo_aug":        root / "yolo_split_aug",
        "yolo_roi":        root / "yolo_split_cupROI",
        "yolo_disc_only":  root / "yolo_split_disc_only",
        "viz_out":         root / "viz_labels",
    }