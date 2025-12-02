# %% [markdown]
# # JPG/PNG Copier & Converter (Jupyter Notebook)
# Recursively copy all .png images and convert all .jpg/.jpeg images under ROOT_DIR to .png
# and write them to a separate output directory (OUT_DIR),
# preserving the relative folder structure and pixel dimensions.

# %% [markdown]
# ## Global configuration
# Set your root directory, output directory, and behavior flags here.

# %%
from pathlib import Path

# Path to the root folder containing your images (read from here)
ROOT_DIR = Path(
    "/arc/project/st-ipor-1/carlosp/fundus_data/All_Datasets_Organized/CHAKSU_jpg"
).expanduser().resolve()

# Path to the root folder where images will be written (write to here)
OUT_DIR = Path(
    "/scratch/st-ipor-1/cperez/MedSAM/data/CHAKSU_V2"
).expanduser().resolve()

# If True, delete original JPG/JPEG files after successful conversion
# NOTE: deletion happens in ROOT_DIR only, never in OUT_DIR.
# Since this script *copies* images, it is safest to keep this False.
DELETE_JPG = False

# If True, overwrite PNGs in OUT_DIR if they already exist
OVERWRITE = False

print(f"ROOT_DIR:   {ROOT_DIR}")
print(f"OUT_DIR:    {OUT_DIR}")
print(f"DELETE_JPG: {DELETE_JPG}")
print(f"OVERWRITE:  {OVERWRITE}")

# %% [markdown]
# ## Imports

# %%
import os
import shutil
from PIL import Image

# %% [markdown]
# ## Core copy/convert function

# %%
def copy_and_convert_images_separate_dir(
    root: Path,
    out_root: Path,
    delete_jpg: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Recursively walk `root` and:
      * copy all PNG images to `out_root`, preserving relative paths and file names
      * convert all JPG/JPEG images to PNG in `out_root` (same stem, .png extension),
        preserving relative paths and pixel dimensions.

    Parameters
    ----------
    root : Path
        Root directory to search for images.
    out_root : Path
        Root directory to write outputs to.
    delete_jpg : bool
        If True, delete original JPG/JPEG images after successful conversion.
    overwrite : bool
        If True, overwrite existing files in out_root. Otherwise skip.
    """
    jpg_extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    png_extensions = {".png", ".PNG"}

    if not root.exists() or not root.is_dir():
        raise ValueError(f"{root} is not a valid directory")

    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Scanning under: {root}")
    print(f"Writing outputs to: {out_root}")

    copied_png = 0
    converted_jpg = 0
    skipped_existing = 0
    errors = 0
    warned_mismatch = 0

    for dirpath, _, filenames in os.walk(root):
        dirpath = Path(dirpath)

        for fname in filenames:
            src = dirpath / fname
            ext = src.suffix

            # Only handle PNG and JPG/JPEG
            if ext not in jpg_extensions and ext not in png_extensions:
                continue

            # Compute relative path from ROOT_DIR, and mirror it under OUT_DIR
            rel_dir = src.parent.relative_to(root)
            dst_dir = out_root / rel_dir
            dst_dir.mkdir(parents=True, exist_ok=True)

            # Destination path logic
            if ext in png_extensions:
                # Copy PNG as-is (do not change the name)
                dst = dst_dir / src.name
                if dst.exists() and not overwrite:
                    print(f"[skip] PNG already exists: {dst}")
                    skipped_existing += 1
                    continue

                try:
                    shutil.copy2(src, dst)
                    print(f"[copy] {src} -> {dst}")
                    copied_png += 1
                except Exception as e:
                    print(f"[ERROR] Failed to copy {src}: {e}")
                    errors += 1

            elif ext in jpg_extensions:
                # Convert JPG/JPEG to PNG; keep same stem, change only extension
                dst = (dst_dir / src.stem).with_suffix(".png")
                if dst.exists() and not overwrite:
                    print(f"[skip] Converted PNG already exists: {dst}")
                    skipped_existing += 1
                    continue

                try:
                    with Image.open(src) as im:
                        width, height = im.size  # original dimensions
                        im.save(dst, format="PNG")

                    # Re-open to verify size
                    with Image.open(dst) as im_png:
                        w2, h2 = im_png.size

                    if (width, height) != (w2, h2):
                        print(
                            f"[WARN] Dimension mismatch for {src} -> {dst}: "
                            f"{width}x{height} vs {w2}x{h2}"
                        )
                        warned_mismatch += 1
                    else:
                        print(f"[ok] {src} -> {dst} ({width}x{height})")
                        converted_jpg += 1

                        if delete_jpg:
                            src.unlink()
                            print(f"      removed original JPG: {src}")

                except Exception as e:
                    print(f"[ERROR] Failed on {src}: {e}")
                    errors += 1

    print("\nSummary")
    print("-------")
    print(f"Copied PNGs:          {copied_png}")
    print(f"Converted JPG→PNG:    {converted_jpg}")
    print(f"Skipped (exists):     {skipped_existing}")
    print(f"Warnings (mismatch):  {warned_mismatch}")
    print(f"Errors:               {errors}")

# %% [markdown]
# ## Run the copy/convert

# %%
copy_and_convert_images_separate_dir(
    ROOT_DIR,
    OUT_DIR,
    delete_jpg=DELETE_JPG,
    overwrite=OVERWRITE,
)