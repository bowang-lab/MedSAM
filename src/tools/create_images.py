#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.imgpipe.image import Image
from src.imgpipe.image_factory import ImageFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load Image objects from a CSV using ImageFactory.make_images_from_csv "
            "and save to a Parquet file, embedding image bytes by default."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/Volumes/ResearchUSB/All_Datasets_Organized"),
        help="Root directory for datasets (used to resolve relative fundus/mask paths).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/Users/carlosperez/PycharmProjects/MedSAM/src/image_data/Cleaned_Metadata.csv"),
        help="Path to the CSV file.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Optional: sample at most N rows from the CSV for quick tests.",
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=Path("/Volumes/ResearchUSB/saved_images_with_img_data.parquet"),
        help="Output Parquet file.",
    )
    parser.add_argument(
        "--no-embed-images",
        action="store_true",
        help="If set, do NOT embed image bytes in Parquet (paths only).",
    )
    parser.add_argument(
        "--no-embed-masks",
        action="store_true",
        help="If set, do NOT embed mask bytes in Parquet (paths only).",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        help="Parquet compression codec (e.g., zstd, snappy, gzip, none).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    factory = ImageFactory(root=args.root, auto_scan=False)
    images = factory.make_images_from_csv(
        csv_path=args.csv,
        sample_n=args.sample_n,
    )

    # Ensure each Image has an ImageDataRef that points to image_path, so embedding works.
    # (If image_ref already exists, this is a no-op.)
    for im in images:
        # relies on Image._ensure_image_ref added in your refactor
        im.ensure_image_ref()  # type: ignore[attr-defined]

    Image.save_parquet(
        images,
        path=args.out_parquet,
        drop_none=False,
        include_image_bytes=(not args.no_embed_images),
        include_mask_bytes=(not args.no_embed_masks),
        compression=args.compression,
    )

    print(f"Loaded {len(images)} images from {args.csv}")
    print(f"Saved Parquet: {args.out_parquet}")
    print(f"Embedded images: {not args.no_embed_images}")
    print(f"Embedded masks:  {not args.no_embed_masks}")
    if images:
        print("First image sample:")
        print(images[0])


if __name__ == "__main__":
    main()