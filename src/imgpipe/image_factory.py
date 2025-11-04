from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Dict, List, Optional, Iterable, Tuple, Set

from src.imgpipe.enums import Structure, LabelType
from src.imgpipe.image import Image

_ALLOWED_EXTS: Set[str] = {".png"}


def _list_files_with_ext(root: Path, exts: Set[str], recursive: bool) -> List[Path]:
    if not root or not root.exists():
        return []
    if recursive:
        return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _stem_map(paths: Iterable[Path]) -> Dict[str, Path]:
    """Map filename stem -> first matching Path (stable by sorted order)."""
    m: Dict[str, Path] = {}
    for p in paths:
        m.setdefault(p.stem, p)
    return m


@dataclass(slots=True)
class ImageFactory:
    """
    Scans a root directory containing multiple datasets. Each dataset directory is
    expected to contain the subdirectories: 'fundus', 'oc_mask', 'od_mask'.
    """
    root: Path
    recursive: bool = True
    allowed_exts: Set[str] = field(default_factory=lambda: set(_ALLOWED_EXTS))
    auto_scan: bool = True

    # populated by scan()
    index: Dict[str, Dict[str, List[Path]]] = field(init=False, default_factory=dict)
    stem_index: Dict[str, Dict[str, Dict[str, Path]]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize root to Path (in case a str sneaks in)
        if not isinstance(self.root, Path):
            self.root = Path(self.root)
        if self.auto_scan:
            self.scan()

    def scan(self) -> None:
        """Walk the root and index all dataset files by expected subdirectory."""
        self.index.clear()
        self.stem_index.clear()

        if not self.root.exists() or not self.root.is_dir():
            raise ValueError(f"Root does not exist or is not a directory: {self.root}")

        for ds_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            dataset = ds_dir.name

            # Expected subdirectories (create empty lists if absent)
            fundus_dir = ds_dir / "fundus"
            oc_mask_dir = ds_dir / "oc_mask"
            od_mask_dir = ds_dir / "od_mask"

            fundus_paths = _list_files_with_ext(fundus_dir, self.allowed_exts, self.recursive)
            oc_mask_paths = _list_files_with_ext(oc_mask_dir, self.allowed_exts, self.recursive)
            od_mask_paths = _list_files_with_ext(od_mask_dir, self.allowed_exts, self.recursive)

            self.index[dataset] = {
                "fundus": fundus_paths,
                "oc_mask": oc_mask_paths,
                "od_mask": od_mask_paths,
            }

            self.stem_index[dataset] = {
                "fundus": _stem_map(fundus_paths),
                "oc_mask": _stem_map(oc_mask_paths),
                "od_mask": _stem_map(od_mask_paths),
            }

    # ---------------- convenience accessors ----------------

    def datasets(self) -> List[str]:
        """List dataset names discovered under root."""
        return sorted(self.index.keys())

    def paths(self, dataset: str, kind: str) -> List[Path]:
        """
        Get paths for a given dataset and kind ('fundus' | 'oc_mask' | 'od_mask').
        Returns an empty list if missing.
        """
        return list(self.index.get(dataset, {}).get(kind, []))

    def stem_paths(self, dataset: str, kind: str) -> Dict[str, Path]:
        """Get stem->path map for a dataset and kind."""
        return dict(self.stem_index.get(dataset, {}).get(kind, {}))

    def summary(self) -> Dict[str, Dict[str, int]]:
        """
        Simple count summary: per dataset, number of files in each kind.
        Example: {'DS1': {'fundus': 120, 'oc_mask': 118, 'od_mask': 118}, ...}
        """
        out: Dict[str, Dict[str, int]] = {}
        for ds, kinds in self.index.items():
            out[ds] = {k: len(v) for k, v in kinds.items()}
        return out

    def paired_triplets(self, dataset: str) -> List[Tuple[Optional[Path], Optional[Path], Optional[Path]]]:
        """
        Align by stem within a dataset and return triplets
        (fundus_path, oc_mask_path, od_mask_path) for every stem seen in any kind.
        Missing items are returned as None. (Path alignment only; no I/O.)
        """
        si = self.stem_index.get(dataset, {})
        f = si.get("fundus", {})
        oc = si.get("oc_mask", {})
        od = si.get("od_mask", {})

        all_stems = sorted(set().union(f.keys(), oc.keys(), od.keys()))
        return [(f.get(s), oc.get(s), od.get(s)) for s in all_stems]

    def filter_datasets(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> None:
        """
        Filter the indexed datasets in-place, keeping only those that match the given
        inclusion or exclusion lists.
        """
        if not self.index:
            raise RuntimeError("No datasets have been scanned yet. Call scan() first.")

        def _match(name: str) -> bool:
            n = name.lower()
            if include and not any(s.lower() in n for s in include):
                return False
            if exclude and any(s.lower() in n for s in exclude):
                return False
            return True

        keep = [ds for ds in self.index.keys() if _match(ds)]

        # Filter both structures in place
        self.index = {k: v for k, v in self.index.items() if k in keep}
        self.stem_index = {k: v for k, v in self.stem_index.items() if k in keep}

    def filter_empty_masks(self) -> None:
        """
        Filter datasets in-place to keep only entries (stems) that have
        all three paths present: fundus image, optic cup mask (oc_mask),
        and optic disc mask (od_mask).

        Notes
        -----
        - This modifies `stem_index` and rebuilds `index` accordingly.
        - Datasets that become empty after filtering are retained (but empty).
        - No file I/O is performed; filtering is path-based.
        """
        if not self.stem_index:
            raise RuntimeError("No datasets indexed. Call scan() first.")

        new_stem_index: Dict[str, Dict[str, Dict[str, Path]]] = {}
        new_index: Dict[str, Dict[str, List[Path]]] = {}

        for ds, kinds in self.stem_index.items():
            f = kinds.get("fundus", {})
            oc = kinds.get("oc_mask", {})
            od = kinds.get("od_mask", {})

            # Keep only stems that have all three
            valid_stems = sorted(set(f.keys()) & set(oc.keys()) & set(od.keys()))

            new_stem_index[ds] = {
                "fundus": {s: f[s] for s in valid_stems},
                "oc_mask": {s: oc[s] for s in valid_stems},
                "od_mask": {s: od[s] for s in valid_stems},
            }

            new_index[ds] = {
                "fundus": [f[s] for s in valid_stems],
                "oc_mask": [oc[s] for s in valid_stems],
                "od_mask": [od[s] for s in valid_stems],
            }

        self.stem_index = new_stem_index
        self.index = new_index

    def filter_random_subset(self, subset_n: int) -> None:
        """
        In-place filter that keeps a reproducible random subset of images across ALL datasets.

        Behavior
        --------
        - Treats each *fundus* stem as one "image".
        - Shuffles the union of (dataset, stem) pairs with a fixed seed (0) for reproducibility.
        - Retains only the first `subset_n` items (or all, if fewer exist).
        - Updates both `stem_index` and `index` accordingly.
        - Datasets are retained even if they become empty (consistent with other filters).

        Parameters
        ----------
        subset_n : int
            Desired number of images to keep across all datasets (cap at total available).

        Raises
        ------
        RuntimeError
            If no datasets have been scanned yet (i.e., `scan()` not called).
        ValueError
            If `subset_n` is not positive.
        """
        if not self.stem_index:
            raise RuntimeError("No datasets indexed. Call scan() first.")
        if subset_n <= 0:
            raise ValueError("subset_n must be a positive integer.")

        # Collect all (dataset, stem) pairs where a fundus image exists.
        all_items: List[Tuple[str, str]] = []
        for ds, kinds in self.stem_index.items():
            fundus_stems = kinds.get("fundus", {})
            for s in fundus_stems.keys():
                all_items.append((ds, s))

        if not all_items:
            # Nothing to filter; leave indices as-is.
            return

        # Reproducible shuffle using a fixed seed.
        rng = Random(0)
        rng.shuffle(all_items)

        # Take the first k items.
        k = min(subset_n, len(all_items))
        chosen = set(all_items[:k])  # set of (dataset, stem)

        # Build new (filtered) stem_index and index.
        new_stem_index: Dict[str, Dict[str, Dict[str, Path]]] = {}
        new_index: Dict[str, Dict[str, List[Path]]] = {}

        for ds, kinds in self.stem_index.items():
            f_map = kinds.get("fundus", {})
            oc_map = kinds.get("oc_mask", {})
            od_map = kinds.get("od_mask", {})

            # Keep only stems selected for this dataset.
            selected_stems = sorted([s for s in f_map.keys() if (ds, s) in chosen])

            new_stem_index[ds] = {
                "fundus": {s: f_map[s] for s in selected_stems},
                "oc_mask": {s: oc_map[s] for s in selected_stems if s in oc_map},
                "od_mask": {s: od_map[s] for s in selected_stems if s in od_map},
            }

            new_index[ds] = {
                "fundus": [f_map[s] for s in selected_stems],
                "oc_mask": [oc_map[s] for s in selected_stems if s in oc_map],
                "od_mask": [od_map[s] for s in selected_stems if s in od_map],
            }

        self.stem_index = new_stem_index
        self.index = new_index

    def make_images(
            self,
            *,
            require_complete: bool = True,
            compute_boxes: bool = True,
    ) -> List[Image]:
        """
        Create Image objects from the indexed paths and return them as a list.

        Parameters
        ----------
        require_complete : bool, default True
            If True, only include stems that have all three paths present:
            fundus, oc_mask (optic cup), and od_mask (optic disc).
            If False, include any stem that has a fundus image; masks are attached
            when available.
        compute_boxes : bool, default True
            If True, immediately derive bounding boxes from any attached masks
            via Image.ensure_boxes_from_masks().

        Returns
        -------
        List[Image]
            A list of constructed Image objects.
        """
        if not self.stem_index:
            raise RuntimeError("No datasets indexed. Call scan() first.")

        items: List[Image] = []

        for ds, kinds in self.stem_index.items():
            f = kinds.get("fundus", {})
            oc = kinds.get("oc_mask", {})
            od = kinds.get("od_mask", {})

            if require_complete:
                stems = sorted(set(f.keys()) & set(oc.keys()) & set(od.keys()))
            else:
                # Always require fundus; attach masks if present
                stems = sorted(set(f.keys()))

            for s in stems:
                fundus_path = f.get(s)
                if fundus_path is None:
                    # Skip if no fundus image (shouldn't happen under either branch, but safe)
                    continue

                img = Image.from_path(
                    image_path=fundus_path,
                    dataset=ds,
                    subject_id=s,
                    uid=f"{ds}:{s}",
                )

                # Attach masks via paths; Image.set_mask will coerce Path -> BinaryMaskRef
                oc_path = oc.get(s)
                if oc_path is not None:
                    img.set_mask(Structure.CUP, LabelType.GT, oc_path)

                od_path = od.get(s)
                if od_path is not None:
                    img.set_mask(Structure.DISC, LabelType.GT, od_path)

                if compute_boxes:
                    img.ensure_boxes_from_masks()

                items.append(img)

        return items

    def save_images(
            self,
            images: List[Image],
            out_path: Path,
            *,
            relative_to_root: bool = False,
            drop_none: bool = False,
    ) -> Path:
        """
        Serialize a list of Image objects to a single JSONL file (one JSON per line).
        A small manifest is written on the first line.

        Parameters
        ----------
        images : List[Image]
            Images to persist.
        out_path : Path
            Destination .jsonl file (e.g., Path("images.jsonl")).
        relative_to_root : bool, default False
            If True, file paths inside each record are made relative to self.root.
        drop_none : bool, default False
            If True, omit None-valued fields from each Image record.

        Returns
        -------
        Path
            The written file path.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _relativize_path(s: Optional[str]) -> Optional[str]:
            if not s:
                return s
            p = Path(s)
            try:
                return str(p.relative_to(self.root))
            except Exception:
                return s  # leave as-is if not under root

        def _relativize_record(d: Dict[str, Any]) -> Dict[str, Any]:
            if not relative_to_root:
                return d
            d = dict(d)  # shallow copy
            # top-level file paths
            if "image_path" in d and d["image_path"]:
                d["image_path"] = _relativize_path(d["image_path"])
            if "yolo_label_path" in d and d["yolo_label_path"]:
                d["yolo_label_path"] = _relativize_path(d["yolo_label_path"])
            # mask paths (nested)
            for key in ("gt_disc_mask", "gt_cup_mask", "pred_disc_mask", "pred_cup_mask"):
                md = d.get(key)
                if isinstance(md, dict) and md.get("path"):
                    md = dict(md)
                    md["path"] = _relativize_path(md["path"])
                    d[key] = md
            return d

        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            manifest = {
                "_manifest": {
                    "schema": "ImageJSONL.v1",
                    "base": str(self.root),
                    "relative": bool(relative_to_root),
                }
            }
            f.write(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n")
            for img in images:
                rec = _relativize_record(img.to_dict(drop_none=drop_none))
                f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
        tmp.replace(out_path)
        return out_path

    @staticmethod
    def load_images(
            in_path: Path,
            *,
            validate_files: bool = False,
    ) -> List[Image]:
        """
        Load Image objects from a JSONL file produced by save_images().

        Parameters
        ----------
        in_path : Path
            Source JSONL file.
        validate_files : bool, default False
            If True, filter out records whose image_path does not exist on disk.

        Returns
        -------
        List[Image]
            Reconstructed Image objects.
        """
        in_path = Path(in_path)
        if not in_path.exists():
            raise FileNotFoundError(in_path)

        base: Optional[Path] = None
        relative_flag: bool = False
        out: List[Image] = []

        def _abspath(s: Optional[str]) -> Optional[str]:
            if not s:
                return s
            p = Path(s)
            if not relative_flag or base is None or p.is_absolute():
                return str(p)
            return str((base / p).resolve())

        def _absolutize_record(d: Dict[str, Any]) -> Dict[str, Any]:
            d = dict(d)
            if "image_path" in d and d["image_path"]:
                d["image_path"] = _abspath(d["image_path"])
            if "yolo_label_path" in d and d["yolo_label_path"]:
                d["yolo_label_path"] = _abspath(d["yolo_label_path"])
            for key in ("gt_disc_mask", "gt_cup_mask", "pred_disc_mask", "pred_cup_mask"):
                md = d.get(key)
                if isinstance(md, dict) and md.get("path"):
                    md = dict(md)
                    md["path"] = _abspath(md["path"])
                    d[key] = md
            return d

        with in_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # first line may be a manifest
                if i == 0 and isinstance(rec, dict) and "_manifest" in rec:
                    man = rec["_manifest"] or {}
                    base = Path(man.get("base")) if man.get("base") else None
                    relative_flag = bool(man.get("relative", False))
                    continue
                # normal image record
                rec = _absolutize_record(rec)
                img = Image.from_dict(rec)
                if validate_files and not Path(img.image_path).exists():
                    continue
                out.append(img)

        return out

    def __repr__(self) -> str:
        parts = [f"ImageFactory(root={str(self.root)!r}, recursive={self.recursive})"]
        if self.index:
            parts.append(f"datasets={self.summary()}")
        return " | ".join(parts)