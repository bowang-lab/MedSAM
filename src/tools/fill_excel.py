#!/usr/bin/env python
from pathlib import Path
import pandas as pd

# =========================
# Global configuration
# =========================
DATA_ROOT = Path("/Volumes/ResearchUSB/All_Datasets_Organized")  # Root folder containing all datasets
DATASET_NAME = "GRAPE"
CSV_PATH = Path("/Users/carlosperez/PycharmProjects/MedSAM/src/image_data/Cleaned_Metadata.csv")

# =========================
# Derived paths
# =========================
dataset_root = DATA_ROOT / DATASET_NAME
fundus_dir = dataset_root / "fundus"
oc_mask_dir = dataset_root / "oc_mask"
od_mask_dir = dataset_root / "od_mask"

print(f"CSV:          {CSV_PATH}")
print(f"DATA_ROOT:    {DATA_ROOT}")
print(f"Dataset root: {dataset_root}")
print(f"fundus dir:   {fundus_dir}")
print(f"oc_mask dir:  {oc_mask_dir}")
print(f"od_mask dir:  {od_mask_dir}")

# =========================
# Load CSV
# =========================
df = pd.read_csv(CSV_PATH)

if "names" not in df.columns:
    raise ValueError("CSV file must contain a 'names' column.")

# Ensure target path columns exist
for col in ["fundus", "oc_mask", "od_mask"]:
    if col not in df.columns:
        df[col] = ""

# =========================
# Build mapping from name -> row index
# Support both raw value in 'names' and its stem (without extension).
# =========================
name_to_idx = {}

for idx, val in df["names"].items():
    if not isinstance(val, str):
        continue
    raw = val.strip()
    stem = Path(raw).stem

    # Prefer first occurrence; do not overwrite existing mapping
    if raw not in name_to_idx:
        name_to_idx[raw] = idx
    if stem not in name_to_idx:
        name_to_idx[stem] = idx

print(f"Built mapping for {len(name_to_idx)} names from CSV.")

# =========================
# Helper to compute relative path (POSIX style) w.r.t DATA_ROOT
# This will include DATASET_NAME (e.g., "CRFO/fundus/img.png").
# =========================
def rel_to_data_root(p: Path) -> str:
    return p.relative_to(DATA_ROOT).as_posix()

# =========================
# Collect files and basic counts
# =========================
if not fundus_dir.is_dir():
    raise FileNotFoundError(f"fundus directory not found: {fundus_dir}")

fundus_files = sorted([p for p in fundus_dir.iterdir() if p.is_file()])
fundus_files_count = len(fundus_files)

oc_files_count = 0
if oc_mask_dir.is_dir():
    oc_files_count = sum(1 for p in oc_mask_dir.iterdir() if p.is_file())

od_files_count = 0
if od_mask_dir.is_dir():
    od_files_count = sum(1 for p in od_mask_dir.iterdir() if p.is_file())

print("\n=== On-disk file counts ===")
print(f"Fundus files found:                  {fundus_files_count}")
print(f"oc_mask files found:                 {oc_files_count}")
print(f"od_mask files found:                 {od_files_count}")

# =========================
# Iterate over fundus files and fill paths
# =========================
missing_in_csv = []
missing_oc_mask = []
missing_od_mask = []

fundus_matches = 0
oc_matches = 0
od_matches = 0

for fpath in fundus_files:
    fname = fpath.name
    stem = fpath.stem

    # Try to match by stem first, then full filename
    idx = name_to_idx.get(stem)
    if idx is None:
        idx = name_to_idx.get(fname)

    if idx is None:
        # print(f"[WARNING] No matching 'names' entry in CSV for fundus file: {fname}")
        print(fname)
        missing_in_csv.append(fname)
        continue

    # Fill fundus relative path (relative to DATA_ROOT)
    df.at[idx, "fundus"] = rel_to_data_root(fpath)
    fundus_matches += 1

    # Check oc_mask
    oc_path = oc_mask_dir / fname
    if oc_path.is_file():
        df.at[idx, "oc_mask"] = rel_to_data_root(oc_path)
        oc_matches += 1
    else:
        print(fname)
        missing_oc_mask.append(fname)

    # Check od_mask
    od_path = od_mask_dir / fname
    if od_path.is_file():
        df.at[idx, "od_mask"] = rel_to_data_root(od_path)
        od_matches += 1
    else:
        missing_od_mask.append(fname)

# =========================
# Save updated CSV
# =========================
print(f"\nOverwriting original CSV: {CSV_PATH}")
df.to_csv(CSV_PATH, index=False)

# =========================
# Summary
# =========================
fundus_filled = (
    df["fundus"]
      .fillna("")              # convert NaN -> ""
      .astype(str)             # ensure string ops work
      .str.strip()             # remove whitespace
      .ne("")                  # not equal to empty string
).sum()

oc_filled = (
    df["oc_mask"].fillna("").astype(str).str.strip().ne("")
).sum()

od_filled = (
    df["od_mask"].fillna("").astype(str).str.strip().ne("")
).sum()


print("\n=== Summary ===")
print(f"Total rows in CSV:                   {len(df)}")
print(f"Fundus rows with path filled:        {int(fundus_filled)}")
print(f"Rows with oc_mask path filled:       {int(oc_filled)}")
print(f"Rows with od_mask path filled:       {int(od_filled)}")
print(f"Fundus files with no CSV match:      {len(missing_in_csv)}")
print(f"Fundus files missing oc_mask file:   {len(missing_oc_mask)}")
print(f"Fundus files missing od_mask file:   {len(missing_od_mask)}")

print("\n=== Match statistics ===")
print(f"Fundus files found on disk:          {fundus_files_count}")
print(f"Fundus files matched to CSV rows:    {fundus_matches}")
print(f"oc_mask files found on disk:         {oc_files_count}")
print(f"oc_mask files matched (path filled): {oc_matches}")
print(f"od_mask files found on disk:         {od_files_count}")
print(f"od_mask files matched (path filled): {od_matches}")