#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}


def iter_image_files(directory: Path) -> Iterable[Path]:
    """Yield non-hidden image files in a (non-recursive) directory (fast scandir, no sort)."""
    with os.scandir(directory) as it:
        for entry in it:
            if not entry.is_file():
                continue
            name = entry.name
            if name.startswith("."):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            yield Path(entry.path)


def extract_image_id(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise RuntimeError(
            f"Filename does not match '<num>_<num>_<image_id>_<description>': {path.name}"
        )
    return parts[2]


def scan_existing_outputs(out_dir: Path) -> Set[str]:
    """Return a set of existing output filenames (case-sensitive)."""
    existing: Set[str] = set()
    if not out_dir.exists():
        return existing
    with os.scandir(out_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            if entry.name.startswith("."):
                continue
            existing.add(entry.name)
    return existing


@dataclass(frozen=True)
class Job:
    src: str
    dst: str
    is_png: bool


def _process_one(job: Job, *, compress_level: int, force_rgb: bool, preserve_metadata: bool) -> Tuple[str, Optional[str]]:
    """
    Worker process: copy or convert. Returns (status, errstr).
    status in {"copied", "converted"}.
    """
    src = Path(job.src)
    dst = Path(job.dst)

    try:
        if job.is_png:
            if preserve_metadata:
                shutil.copy2(src, dst)
            else:
                shutil.copyfile(src, dst)
            return "copied", None

        with Image.open(src) as img:
            if force_rgb and img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            img.save(dst, format="PNG", compress_level=compress_level)
        return "converted", None
    except Exception as e:
        return "error", f"{src.name}: {e}"


def convert_directory(
    in_dir: Path,
    out_dir: Path,
    *,
    workers: int,
    compress_level: int,
    force_rgb: bool,
    preserve_metadata: bool,
    verify: bool,
) -> None:
    if not in_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # One output directory scan (cheap compared to per-file exists checks)
    existing = scan_existing_outputs(out_dir)

    # Build jobs sequentially (also detects naming collisions up-front)
    jobs: List[Job] = []
    count_total = 0
    count_skipped_existing = 0
    count_bad_name = 0
    count_collisions = 0

    reserved = set(existing)  # reserve names so two inputs don’t race to same dst
    for src in iter_image_files(in_dir):
        count_total += 1
        try:
            image_id = extract_image_id(src)
        except RuntimeError as e:
            logging.error("Naming pattern error for %s: %s", src.name, e)
            count_bad_name += 1
            continue

        out_name = f"EYEPACS_{image_id}.png"
        if out_name in reserved:
            # Existing output OR collision with another input in this run
            if out_name in existing:
                count_skipped_existing += 1
            else:
                logging.error("Naming collision: multiple inputs map to %s (e.g., %s)", out_name, src.name)
                count_collisions += 1
            continue

        reserved.add(out_name)
        jobs.append(Job(src=str(src), dst=str(out_dir / out_name), is_png=(src.suffix.lower() == ".png")))

    logging.info("Found %d input images", count_total)
    logging.info("Will process %d (skipped existing=%d, bad_name=%d, collisions=%d)",
                 len(jobs), count_skipped_existing, count_bad_name, count_collisions)

    if not jobs:
        if verify and (count_bad_name > 0 or count_collisions > 0):
            raise RuntimeError("No jobs to run due to errors/collisions.")
        return

    # Parallel work: best speedup on CPU nodes
    workers = max(1, int(workers))
    count_converted = 0
    count_copied = 0
    count_errors = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                _process_one,
                job,
                compress_level=compress_level,
                force_rgb=force_rgb,
                preserve_metadata=preserve_metadata,
            )
            for job in jobs
        ]

        for fut in as_completed(futures):
            status, err = fut.result()
            if status == "converted":
                count_converted += 1
            elif status == "copied":
                count_copied += 1
            else:
                count_errors += 1
                logging.error("Error: %s", err)

    logging.info("Finished.")
    logging.info("Total images in input       : %d", count_total)
    logging.info("Converted to PNG            : %d", count_converted)
    logging.info("Copied PNGs                 : %d", count_copied)
    logging.info("Skipped (already existed)   : %d", count_skipped_existing)
    logging.info("Bad filenames               : %d", count_bad_name)
    logging.info("Naming collisions           : %d", count_collisions)
    logging.info("Worker errors               : %d", count_errors)

    if verify:
        # Optional: expensive full scan (only do when you need it)
        out_count = sum(1 for _ in iter_image_files(out_dir))
        logging.info("Total images in output      : %d", out_count)

        # Minimal sanity check: everything that should have an output is either present or errored/collided/bad-name.
        # (Collisions/bad names can reduce output count below input count.)
        expected_min = count_total - (count_bad_name + count_collisions + count_errors)
        if out_count < expected_min:
            raise RuntimeError(
                f"Verification failed: output has {out_count}, expected at least {expected_min}. "
                "Check logs for errors/collisions."
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Fast EYEPACS -> PNG converter (parallel, resume-safe).")
    p.add_argument("input_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    p.add_argument("--compress-level", type=int, default=1, choices=range(0, 10))
    p.add_argument("--force-rgb", action="store_true", help="Convert non-RGB modes to RGB before saving PNG.")
    p.add_argument("--preserve-metadata", action="store_true", help="Use copy2 for PNG->PNG (slower).")
    p.add_argument("--verify", action="store_true", help="Scan output dir at end (slow) and sanity check counts.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    convert_directory(
        args.input_dir,
        args.output_dir,
        workers=args.workers,
        compress_level=args.compress_level,
        force_rgb=args.force_rgb,
        preserve_metadata=args.preserve_metadata,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()