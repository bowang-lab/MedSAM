#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image as PILImage

from src.imgpipe.image import Image

# =========================
# Global configuration
# =========================
PARQUET_PATH = Path("/Users/carlosperez/PycharmProjects/MedSAM/src/image_data/parquets/hpc/og_images/combined_dataset.parquet")
N_PREVIEW = 3
BATCH_SIZE = 256

# Where to save preview masks (and small metadata)
OUT_DIR = Path("/Users/carlosperez/PycharmProjects/MedSAM/OUT/mask_previews")

# Mask validation behavior
MASK_CHECK_LIMIT = 0  # 0 = check all rows; else stop after this many rows
MASK_PIXEL_FRACTION_MAX = 0.95  # flag masks that cover >95% of image as suspicious

# Only validate masks that are embedded within the parquet (avoid filesystem/path loads)
CHECK_EMBEDDED_MASKS_ONLY = True


def _ds_name(img: Image) -> str:
    v = getattr(img, "dataset", None)
    s = str(v).strip() if v is not None else ""
    return s if s else "UNKNOWN"


def _split_name(img: Image) -> str:
    v = getattr(img, "split", None)
    s = str(v).strip() if v is not None else ""
    return s if s else "NONE"


def _has_any_gt(img: Image) -> bool:
    """Counts GT if either GT boxes or GT masks exist."""
    return bool(
        getattr(img, "gt_disc_box", None) is not None
        or getattr(img, "gt_cup_box", None) is not None
        or getattr(img, "gt_disc_mask", None) is not None
        or getattr(img, "gt_cup_mask", None) is not None
    )


def _has_any_pred(img: Image) -> bool:
    """Counts predictions if either predicted boxes or predicted masks exist."""
    return bool(
        getattr(img, "pred_disc_box", None) is not None
        or getattr(img, "pred_cup_box", None) is not None
        or getattr(img, "inter_pred_disc_box", None) is not None
        or getattr(img, "inter_pred_cup_box", None) is not None
        or getattr(img, "pred_disc_mask", None) is not None
        or getattr(img, "pred_cup_mask", None) is not None
    )


def _is_empty_box(box: Any) -> bool:
    """
    Best-effort test for "empty" boxes across common representations.

    Accepts:
      - None
      - dict with numeric fields (x1,y1,x2,y2) or (cx,cy,w,h) or similar
      - list/tuple/np.ndarray length 4
      - objects with x1/y1/x2/y2 or cx/cy/w/h attributes
    """
    if box is None:
        return True

    # Arrow/Parquet sometimes yields empty structs as {}
    if isinstance(box, dict) and len(box) == 0:
        return True

    def _get(d: dict, keys: tuple[str, ...]) -> Optional[float]:
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return float(d[k])
                except Exception:
                    return None
        return None

    # dict form
    if isinstance(box, dict):
        # xyxy
        x1 = _get(box, ("x1", "xmin", "left"))
        y1 = _get(box, ("y1", "ymin", "top"))
        x2 = _get(box, ("x2", "xmax", "right"))
        y2 = _get(box, ("y2", "ymax", "bottom"))
        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            return (x2 <= x1) or (y2 <= y1)

        # cxcywh
        w = _get(box, ("w", "width"))
        h = _get(box, ("h", "height"))
        if w is not None and h is not None:
            return (w <= 0) or (h <= 0)

        # If dict but can't interpret (treat as non-empty to avoid false positives)
        return False

    # list/tuple/array
    if isinstance(box, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(box, dtype=float).reshape(-1)
            if arr.size != 4:
                return False
            a, b, c, d = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
            # could be xyxy or cxcywh; in either case, "c,d" typically sizes or max coords
            # we do a permissive emptiness check:
            if c <= 0 or d <= 0:
                # cxcywh empty OR nonsense
                # (xyxy with negative coords is possible, but <=0 for x2/y2 is uncommon)
                return True
            # if interpreted xyxy, check x2>x1 and y2>y1
            return (c <= a) or (d <= b)
        except Exception:
            return False

    # attribute form
    for attrs in (("x1", "y1", "x2", "y2"), ("xmin", "ymin", "xmax", "ymax")):
        if all(hasattr(box, a) for a in attrs):
            try:
                x1, y1, x2, y2 = (float(getattr(box, a)) for a in attrs)
                return (x2 <= x1) or (y2 <= y1)
            except Exception:
                return False

    if hasattr(box, "w") and hasattr(box, "h"):
        try:
            w, h = float(getattr(box, "w")), float(getattr(box, "h"))
            return (w <= 0) or (h <= 0)
        except Exception:
            return False

    return False


def _print_counter(title: str, ctr: Counter) -> None:
    total = sum(int(v) for v in ctr.values())
    print(f"\n{title} (total={total})")
    for k in sorted(ctr.keys()):
        print(f"  {k:<28} {int(ctr[k])}")


# =========================
# Mask embedded-vs-path accounting
# =========================
def _mask_has_embedded_bytes(mask_ref) -> bool:
    if mask_ref is None:
        return False

    if hasattr(mask_ref, "has_bytes") and callable(getattr(mask_ref, "has_bytes")):
        try:
            return bool(mask_ref.has_bytes())
        except Exception:
            return False

    return getattr(mask_ref, "packed", None) is not None


def _mask_has_path(mask_ref) -> bool:
    if mask_ref is None:
        return False
    p = getattr(mask_ref, "path", None)
    return p is not None and str(p).strip() != ""


# =========================
# Mask loading/validation + saving previews
# =========================
def _safe_load_bool_mask(img: Image, mask_ref) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if mask_ref is None:
        return None, None

    if CHECK_EMBEDDED_MASKS_ONLY and (not _mask_has_embedded_bytes(mask_ref)):
        return None, "skipped_no_embedded_bytes"

    try:
        arr = mask_ref.load()
    except Exception as e:
        return None, f"load_failed: {type(e).__name__}: {e}"

    try:
        arr = np.asarray(arr)
        if arr.ndim != 2:
            return None, f"bad_ndim={arr.ndim}"

        H, W = int(getattr(img, "height", 0) or 0), int(getattr(img, "width", 0) or 0)
        if H <= 0 or W <= 0:
            return None, f"bad_image_size(H={H},W={W})"

        m = arr.astype(bool)

        if m.shape != (H, W):
            out = np.zeros((H, W), dtype=bool)
            h = min(H, m.shape[0])
            w = min(W, m.shape[1])
            out[:h, :w] = m[:h, :w]
            m = out

        return m, None
    except Exception as e:
        return None, f"coerce_failed: {type(e).__name__}: {e}"


def _validate_binary_mask(m: np.ndarray) -> Optional[str]:
    if m.dtype != bool:
        return f"not_bool(dtype={m.dtype})"
    if m.ndim != 2:
        return f"bad_ndim={m.ndim}"
    if m.size == 0:
        return "empty_mask"
    return None


def _check_masks_validity(img: Image) -> Dict[str, str]:
    problems: Dict[str, str] = {}

    for key in ("gt_disc_mask", "gt_cup_mask", "pred_disc_mask", "pred_cup_mask"):
        mref = getattr(img, key, None)
        m, err = _safe_load_bool_mask(img, mref)

        if err == "skipped_no_embedded_bytes":
            continue

        if err is not None:
            problems[key] = err
            continue
        if m is None:
            continue

        bad = _validate_binary_mask(m)
        if bad is not None:
            problems[key] = bad
            continue

        frac = float(m.mean())
        if frac <= 0.0:
            problems[key] = "all_zero_mask"
        elif frac >= float(MASK_PIXEL_FRACTION_MAX):
            problems[key] = f"suspicious_large_mask(frac={frac:.3f})"

    return problems


def _save_preview_masks(img: Image, idx: int, out_dir: Path) -> None:
    uid = str(getattr(img, "uid", f"row{idx}"))
    ds = _ds_name(img)
    sp = _split_name(img)

    safe_stub = f"{idx:04d}__{ds}__{sp}__{uid}".replace("/", "_")

    row_dir = out_dir / safe_stub
    row_dir.mkdir(parents=True, exist_ok=True)

    meta_txt = [
        f"uid: {getattr(img, 'uid', None)}",
        f"dataset: {getattr(img, 'dataset', None)}",
        f"patient_id: {getattr(img, 'patient_id', None)}",
        f"split: {getattr(img, 'split', None)}",
        f"image_path: {getattr(img, 'image_path', None)}",
        f"width,height: {getattr(img, 'width', None)},{getattr(img, 'height', None)}",
        "",
        f"gt_disc_box: {getattr(img, 'gt_disc_box', None)}",
        f"gt_cup_box: {getattr(img, 'gt_cup_box', None)}",
        f"pred_disc_box: {getattr(img, 'pred_disc_box', None)}",
        f"pred_cup_box: {getattr(img, 'pred_cup_box', None)}",
        f"inter_pred_disc_box: {getattr(img, 'inter_pred_disc_box', None)}",
        f"inter_pred_cup_box: {getattr(img, 'inter_pred_cup_box', None)}",
    ]
    (row_dir / "meta.txt").write_text("\n".join(meta_txt) + "\n", encoding="utf-8")

    for key in ("gt_disc_mask", "gt_cup_mask", "pred_disc_mask", "pred_cup_mask"):
        mref = getattr(img, key, None)
        m, err = _safe_load_bool_mask(img, mref)

        if err == "skipped_no_embedded_bytes" or m is None:
            continue

        png = (m.astype(np.uint8) * 255)
        PILImage.fromarray(png).save(row_dir / f"{key}.png")

    lines = []
    for key in ("gt_disc_mask", "gt_cup_mask", "pred_disc_mask", "pred_cup_mask"):
        mref = getattr(img, key, None)
        if mref is None:
            lines.append(f"{key}: NONE")
            continue
        emb = _mask_has_embedded_bytes(mref)
        pth = _mask_has_path(mref)
        lines.append(f"{key}: embedded={emb} path={pth} path_value={getattr(mref, 'path', None)}")
    (row_dir / "mask_sources.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_PATH}")
    if PARQUET_PATH.is_dir():
        raise IsADirectoryError(f"Expected a parquet FILE, got a directory: {PARQUET_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Streaming preview + counts from: {PARQUET_PATH}\n")
    print(f"[INFO] Preview masks (first {N_PREVIEW} rows) will be saved under: {OUT_DIR}\n")
    if CHECK_EMBEDDED_MASKS_ONLY:
        print("[INFO] Mask validation will ONLY check masks embedded in the parquet (no path loads).\n")

    per_dataset = Counter()
    per_split = Counter()
    n_total = 0

    n_with_gt = 0
    n_with_pred = 0
    n_with_both = 0
    n_with_neither = 0

    # mask validation stats
    n_mask_checked = 0
    n_mask_rows_with_any_problem = 0
    mask_problem_types = Counter()
    mask_problem_fields = Counter()

    # embedded vs path counts (by field)
    mask_present = Counter()
    mask_embedded = Counter()
    mask_path = Counter()
    mask_embedded_only = Counter()
    mask_path_only = Counter()
    mask_both = Counter()

    # NEW: mask presence/emptiness stats
    mask_null = Counter()
    mask_empty_struct_like = Counter()

    # NEW: box presence/emptiness stats
    box_present = Counter()
    box_null = Counter()
    box_empty = Counter()  # includes empty struct-like or zero-area boxes

    mask_fields = ("gt_disc_mask", "gt_cup_mask", "pred_disc_mask", "pred_cup_mask")
    box_fields = (
        "gt_disc_box",
        "gt_cup_box",
        "pred_disc_box",
        "pred_cup_box",
        "inter_pred_disc_box",
        "inter_pred_cup_box",
    )

    n_previewed = 0
    for img in Image.iter_parquet(PARQUET_PATH, batch_size=BATCH_SIZE):
        n_total += 1
        per_dataset[_ds_name(img)] += 1
        per_split[_split_name(img)] += 1

        has_gt = _has_any_gt(img)
        has_pred = _has_any_pred(img)

        if has_gt:
            n_with_gt += 1
        if has_pred:
            n_with_pred += 1
        if has_gt and has_pred:
            n_with_both += 1
        if (not has_gt) and (not has_pred):
            n_with_neither += 1

        # -------------------------
        # Mask accounting
        # -------------------------
        for key in mask_fields:
            mref = getattr(img, key, None)
            if mref is None:
                mask_null[key] += 1
                continue

            mask_present[key] += 1
            has_emb = _mask_has_embedded_bytes(mref)
            has_pth = _mask_has_path(mref)

            if has_emb:
                mask_embedded[key] += 1
            if has_pth:
                mask_path[key] += 1

            if has_emb and has_pth:
                mask_both[key] += 1
            elif has_emb:
                mask_embedded_only[key] += 1
            elif has_pth:
                mask_path_only[key] += 1

            # empty/ref-broken detection (common culprit for your crash)
            has_any_storage = bool(getattr(mref, "packed", None) is not None) or bool(getattr(mref, "array", None) is not None) or bool(getattr(mref, "arr", None) is not None) or has_pth
            if not has_any_storage:
                mask_empty_struct_like[key] += 1

        # -------------------------
        # Box accounting
        # -------------------------
        for bkey in box_fields:
            box = getattr(img, bkey, None)
            if box is None:
                box_null[bkey] += 1
                continue

            box_present[bkey] += 1
            if _is_empty_box(box):
                box_empty[bkey] += 1

        # -------------------------
        # Mask validation (optional limit)
        # -------------------------
        if MASK_CHECK_LIMIT <= 0 or n_mask_checked < int(MASK_CHECK_LIMIT):
            probs = _check_masks_validity(img)
            n_mask_checked += 1
            if probs:
                n_mask_rows_with_any_problem += 1
                for field, msg in probs.items():
                    mask_problem_fields[field] += 1
                    mask_problem_types[msg] += 1

                if n_mask_rows_with_any_problem <= 10:
                    print(
                        f"[MASK-ISSUE] uid={getattr(img, 'uid', None)} "
                        f"dataset={_ds_name(img)} split={_split_name(img)}"
                    )
                    print(f"  image_path: {getattr(img, 'image_path', None)}")
                    for field, msg in probs.items():
                        print(f"  - {field}: {msg}")
                    print()

        # -------------------------
        # Preview printing + save masks for first N_PREVIEW rows
        # -------------------------
        if n_previewed < int(N_PREVIEW):
            print(f"[{n_previewed}] {img}")
            print(f"    dataset:        {getattr(img, 'dataset', None)}")
            print(f"    patient_id:     {getattr(img, 'patient_id', None)}")
            print(f"    split:          {getattr(img, 'split', None)}")
            print(f"    image_path:     {getattr(img, 'image_path', None)}")

            # Print boxes (GT + pred + inter_pred)
            print("    boxes:")
            for bkey in box_fields:
                bval = getattr(img, bkey, None)
                print(f"      {bkey:<18} {bval}  empty={_is_empty_box(bval)}")

            # Print masks (refs)
            print("    masks:")
            for mkey in mask_fields:
                mref = getattr(img, mkey, None)
                if mref is None:
                    print(f"      {mkey:<14} NONE")
                else:
                    emb = _mask_has_embedded_bytes(mref)
                    pth = _mask_has_path(mref)
                    has_any_storage = bool(getattr(mref, 'packed', None) is not None) or bool(getattr(mref, 'array', None) is not None) or bool(getattr(mref, 'arr', None) is not None) or pth
                    print(f"      {mkey:<14} embedded={emb} path={pth} storage_ok={has_any_storage} ref={mref}")

            print(f"    has_gt:         {has_gt}")
            print(f"    has_pred:       {has_pred}")
            print()

            _save_preview_masks(img, n_previewed, OUT_DIR)
            n_previewed += 1

    print(f"Previewed {n_previewed} rows.")
    print(f"Total rows scanned: {n_total}")

    print("\nAnnotation/prediction coverage:")
    print(f"  With any GT:       {n_with_gt}")
    print(f"  With any pred:     {n_with_pred}")
    print(f"  With both:         {n_with_both}")
    print(f"  With neither:      {n_with_neither}")

    _print_counter("Counts by dataset", per_dataset)
    _print_counter("Counts by split", per_split)

    # Report embedded-vs-path counts
    print("\nMask storage (counts are per-row-per-field; NULL counted separately):")
    _print_counter("Mask present (non-None)", mask_present)
    _print_counter("Mask is NULL (None)", mask_null)
    _print_counter("Mask empty struct-like (no bytes/array/path)", mask_empty_struct_like)
    _print_counter("Mask has embedded bytes", mask_embedded)
    _print_counter("Mask has path set", mask_path)
    _print_counter("Mask embedded-only", mask_embedded_only)
    _print_counter("Mask path-only", mask_path_only)
    _print_counter("Mask both (embedded + path)", mask_both)

    print("\nBox presence/emptiness (counts are per-row-per-field; NULL counted separately):")
    _print_counter("Box present (non-None)", box_present)
    _print_counter("Box is NULL (None)", box_null)
    _print_counter("Box empty/invalid (zero-area/empty-struct)", box_empty)

    print("\nMask validation (embedded-only):" if CHECK_EMBEDDED_MASKS_ONLY else "\nMask validation:")
    print(f"  Rows checked:                {n_mask_checked}")
    print(f"  Rows with any mask problem:  {n_mask_rows_with_any_problem}")
    if n_mask_rows_with_any_problem > 0:
        _print_counter("Mask problem fields", mask_problem_fields)
        _print_counter("Mask problem types", mask_problem_types)

    print(f"\n[INFO] Preview masks saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()