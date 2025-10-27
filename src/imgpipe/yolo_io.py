from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from .image import Image
from src.utils import ensure_dir, safe_link_or_copy


class YoloDatasetIO:
    """Helpers to write and transform YOLO datasets."""

    @staticmethod
    def write_one_image_and_label(im: Image, out_root: Path, prefer_copy: bool = False) -> None:
        assert im.split in ("train", "val", "test"), "Image.split must be set"
        out_img = out_root / "images" / im.split / im.image_path.name
        out_lbl = out_root / "labels" / im.split / f"{im.image_path.stem}.txt"
        safe_link_or_copy(im.image_path, out_img, prefer_copy=prefer_copy)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_lbl, "w") as f:
            for ln in im.yolo_lines_2class(use_gt=True):
                f.write(ln + "\n")

    @staticmethod
    def write_split(images: List[Image], out_root: Path, write_yaml: bool = True, prefer_copy: bool = False) -> None:
        for split in ("train", "val", "test"):
            (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        for im in images:
            if im.split not in ("train", "val", "test"):
                continue
            YoloDatasetIO.write_one_image_and_label(im, out_root, prefer_copy=prefer_copy)

        if write_yaml:
            YoloDatasetIO.write_data_yaml(out_root, names=["disc", "cup"])

    @staticmethod
    def write_data_yaml(root: Path, names: List[str], test_if_exists: bool = True) -> None:
        data_yaml = {
            "path": str(root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": names,
        }
        if not test_if_exists or ((root / "images" / "test").exists()):
            data_yaml["test"] = "images/test"

        import yaml
        with open(root / "data.yaml", "w") as f:
            yaml.safe_dump(data_yaml, f, sort_keys=False)

    # --------- transforms on an existing on-disk YOLO dataset ---------

    @staticmethod
    def filter_to_single_class(base_root: Path, out_root: Path, keep_class_id: int, drop_empty: bool = False, prefer_copy: bool = False) -> None:
        """
        Build a derived YOLO dataset that keeps only the given class id in labels,
        copying/symlinking images and rewriting labels.
        """
        for split in ("train", "val", "test"):
            src_img = base_root / "images" / split
            src_lbl = base_root / "labels" / split
            if not src_img.exists() or not src_lbl.exists():
                continue
            dst_img = out_root / "images" / split
            dst_lbl = out_root / "labels" / split
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)

            for p in sorted(src_img.iterdir()):
                if not p.is_file():
                    continue
                # copy/symlink image
                q = dst_img / p.name
                safe_link_or_copy(p, q, prefer_copy=prefer_copy)

            for lp in sorted(src_lbl.glob("*.txt")):
                lines_in = [ln.strip() for ln in lp.read_text().splitlines() if ln.strip()]
                kept = []
                for ln in lines_in:
                    parts = ln.split()
                    try:
                        cls = int(float(parts[0]))
                    except Exception:
                        continue
                    if cls == keep_class_id:
                        kept.append(" ".join(["0"] + parts[1:]))  # remap to single-class '0'
                if not kept and drop_empty:
                    continue
                (dst_lbl / lp.name).write_text("\n".join(kept) + ("\n" if kept else ""))

        # write one-name data.yaml
        YoloDatasetIO.write_data_yaml(out_root, names=["disc" if keep_class_id == 0 else "cup"])