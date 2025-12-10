# %% [markdown]
# # Update `names` for GRAPE rows (robust to missing `original_name`)
# For rows where Dataset == "GRAPE":
#   - Prefer `original_name` if it exists and is not NaN
#   - Otherwise fall back to the current `names` value
#   - Strip any existing "GRAPE_" prefix to avoid double-prefixing
#   - Set names = "GRAPE_<base>"

# %%
from pathlib import Path
import pandas as pd

# ---- set these ----
CSV_IN  = Path("/Users/carlosperez/PycharmProjects/MedSAM/src/image_data/Cleaned_Metadata.csv").expanduser().resolve()
CSV_OUT = Path("/src/image_data/Cleaned_Metadata.csv").expanduser().resolve()
OVERWRITE_INPUT = False  # True to overwrite input

print(f"CSV_IN:  {CSV_IN}")
print(f"CSV_OUT: {CSV_OUT}")
print(f"OVERWRITE_INPUT: {OVERWRITE_INPUT}")

# %%
required_cols = [
    "Dataset", "uid", "names", "types", "type_expanded", "fundus",
    "od_mask", "oc_mask", "original_name", "patient_id", "sex", "age",
    "eye", "ethnicity", "vcdr", "notchI_present", "notchS_present",
    "notchN_present", "notchT_present", "expert1_grade", "expert2_grade",
    "expert3_grade", "expert4_grade", "expert5_grade", "cdr_avg",
    "cdr_expert1", "cdr_expert2", "cdr_expert3", "cdr_expert4"
]

df = pd.read_csv(CSV_IN)

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# %%
mask = df["Dataset"] == "GRAPE"

# base name: original_name if valid else current names
orig = df.loc[mask, "original_name"]
curr = df.loc[mask, "names"]

base = orig.where(orig.notna() & (orig.astype(str).str.strip() != ""), curr)

# make sure it's string, and strip any existing GRAPE_ prefix
base = (
    base.astype(str)
        .str.strip()
        .str.replace(r"^GRAPE_", "", regex=True)
)

# only update where base is not empty/"nan"
valid_base = base.notna() & (base != "") & (base.str.lower() != "nan")
df.loc[mask & valid_base, "names"] = "GRAPE_" + base[valid_base]

print(f"GRAPE rows: {mask.sum()}")
print(f"Updated GRAPE rows with valid base: {valid_base.sum()}")

# %%
out_path = CSV_IN if OVERWRITE_INPUT else CSV_OUT
df.to_csv(out_path, index=False)
print(f"Wrote updated CSV to: {out_path}")