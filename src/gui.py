#!/usr/bin/env python3
# src/scripts/gui.py
"""
Parquet Explorer GUI (Streamlit)

Features
- Load an Image-Parquet (rows produced by Image.to_dict()).
- Show counts: total, has predictions, has GT.
- Dataset + split filtering.
- Confidence filtering (disc / cup / both).
- Aggregate metrics on filtered set.
- Visual Analysis: Histograms and Confidence vs Performance plots.
- Per-image table with sortable columns + per-image viewer with overlays.

Run
  pip install streamlit pandas pyarrow pillow matplotlib numpy
  streamlit run src/gui.py
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from src.imgpipe.image import Image


# -------------------------
# Small utilities
# -------------------------

def _is_finite(x: Any) -> bool:
    try:
        return (x is not None) and bool(np.isfinite(float(x)))
    except Exception:
        return False


def _safe_float(x: Any) -> Optional[float]:
    if _is_finite(x):
        return float(x)
    return None


def _mae(a: pd.Series, b: pd.Series) -> Optional[float]:
    m = a.notna() & b.notna()
    if not bool(m.any()):
        return None
    return float((a[m].astype(float) - b[m].astype(float)).abs().mean())


def _extract_mask_path(mask_obj: Any) -> Optional[str]:
    """
    Parquet rows store masks in a nested struct/dict (e.g., {"path": "..."}).
    Pandas may represent it as dict-like, or as a pyarrow.StructScalar-like object.
    """
    if mask_obj is None or (isinstance(mask_obj, float) and math.isnan(mask_obj)):
        return None

    # dict case (common)
    if isinstance(mask_obj, dict):
        p = mask_obj.get("path")
        return str(p) if p else None

    # pyarrow struct scalar in pandas can behave like dict-ish
    try:
        p = mask_obj["path"]
        return str(p) if p else None
    except Exception:
        pass

    # fallback: try attribute
    try:
        p = getattr(mask_obj, "path")
        return str(p) if p else None
    except Exception:
        return None


def _remap_path_str(s: Optional[str], old_prefix: str, new_prefix: str) -> Optional[str]:
    if not s:
        return s
    if old_prefix and s.startswith(old_prefix):
        return new_prefix + s[len(old_prefix):]
    return s


def _remap_row_paths(row: Dict[str, Any], old_prefix: str, new_prefix: str) -> Dict[str, Any]:
    """
    Remap image/mask paths in an Image.to_dict() record.
    Intended for moving Parquet across machines.
    """
    if not old_prefix:
        return row

    r = dict(row)
    r["image_path"] = _remap_path_str(r.get("image_path"), old_prefix, new_prefix)

    for k in ("gt_disc_mask", "gt_cup_mask", "pred_disc_mask", "pred_cup_mask"):
        m = r.get(k)
        if isinstance(m, dict) and m.get("path"):
            m2 = dict(m)
            m2["path"] = _remap_path_str(str(m2["path"]), old_prefix, new_prefix)
            r[k] = m2

    return r


@st.cache_data(show_spinner=False)
def load_parquet_as_df(parquet_path: str) -> pd.DataFrame:
    """
    Load Parquet into pandas.
    Note: If your Parquet is huge, consider writing a column-pruned Parquet for GUI use.
    """
    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet not found: {p}")

    table = pq.read_table(str(p))
    df = table.to_pandas()
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add convenience boolean columns for filtering/summary.
    """
    out = df.copy()

    # basic identity fields
    for c in ("uid", "dataset", "split", "subject_id", "image_path"):
        if c not in out.columns:
            out[c] = None

    # mask-path presence
    out["gt_disc_mask_path"] = out.get("gt_disc_mask", pd.Series([None] * len(out))).apply(_extract_mask_path)
    out["gt_cup_mask_path"] = out.get("gt_cup_mask", pd.Series([None] * len(out))).apply(_extract_mask_path)
    out["pred_disc_mask_path"] = out.get("pred_disc_mask", pd.Series([None] * len(out))).apply(_extract_mask_path)
    out["pred_cup_mask_path"] = out.get("pred_cup_mask", pd.Series([None] * len(out))).apply(_extract_mask_path)

    out["has_gt"] = out["gt_disc_mask_path"].notna() | out["gt_cup_mask_path"].notna()
    out["has_pred"] = out["pred_disc_mask_path"].notna() | out["pred_cup_mask_path"].notna()

    # also count as "pred present" if YOLO boxes/conf exist even when masks were not saved
    if "inter_pred_disc_box" in out.columns or "inter_pred_cup_box" in out.columns:
        out["has_yolo_boxes"] = out.get("inter_pred_disc_box", pd.Series([None] * len(out))).notna() | out.get(
            "inter_pred_cup_box", pd.Series([None] * len(out))
        ).notna()
    else:
        out["has_yolo_boxes"] = False

    out["has_any_pred"] = out["has_pred"] | out["has_yolo_boxes"]

    # numeric normalization
    for c in (
    "yolo_disc_conf", "yolo_cup_conf", "mask_dice_disc", "mask_dice_cup", "gt_cdr", "pred_cdr", "gt_rdr", "pred_rdr"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # CDR/RDR abs errors if columns exist
    if "gt_cdr" in out.columns and "pred_cdr" in out.columns:
        out["cdr_abs_err"] = (out["pred_cdr"] - out["gt_cdr"]).abs()
    else:
        out["cdr_abs_err"] = np.nan

    if "gt_rdr" in out.columns and "pred_rdr" in out.columns:
        out["rdr_abs_err"] = (out["pred_rdr"] - out["gt_rdr"]).abs()
    else:
        out["rdr_abs_err"] = np.nan

    return out


def compute_fast_aggregates(df_f: pd.DataFrame) -> Dict[str, Any]:
    """
    Fast aggregate metrics that do NOT load masks/images from disk.
    Uses stored columns if available.
    """
    out: Dict[str, Any] = {}

    out["n_images"] = int(len(df_f))
    out["n_has_pred"] = int(df_f["has_any_pred"].sum()) if "has_any_pred" in df_f.columns else None
    out["n_has_gt"] = int(df_f["has_gt"].sum()) if "has_gt" in df_f.columns else None

    # Dice means
    out["mean_dice_disc"] = float(df_f["mask_dice_disc"].dropna().mean()) if "mask_dice_disc" in df_f.columns else None
    out["mean_dice_cup"] = float(df_f["mask_dice_cup"].dropna().mean()) if "mask_dice_cup" in df_f.columns else None

    # MAE from stored fields (if present)
    if "gt_cdr" in df_f.columns and "pred_cdr" in df_f.columns:
        out["cdr_mae"] = _mae(df_f["pred_cdr"], df_f["gt_cdr"])
        out["cdr_n"] = int((df_f["pred_cdr"].notna() & df_f["gt_cdr"].notna()).sum())
    else:
        out["cdr_mae"] = None
        out["cdr_n"] = 0

    if "gt_rdr" in df_f.columns and "pred_rdr" in df_f.columns:
        out["rdr_mae"] = _mae(df_f["pred_rdr"], df_f["gt_rdr"])
        out["rdr_n"] = int((df_f["pred_rdr"].notna() & df_f["gt_rdr"].notna()).sum())
    else:
        out["rdr_mae"] = None
        out["rdr_n"] = 0

    # Conf stats
    for k in ("yolo_disc_conf", "yolo_cup_conf"):
        if k in df_f.columns:
            out[f"{k}_mean"] = float(df_f[k].dropna().mean()) if df_f[k].notna().any() else None

    return out


def slow_metrics_summary(df_f: pd.DataFrame, *, old_prefix: str, new_prefix: str, limit: int) -> Dict[str, Any]:
    """
    Slow metrics: loads masks and computes Image.metrics_summary() per image
    to produce MAE for cdr_v/cdr_h/rim_over_disc/I_over_S.
    """
    keys = ["cdr_v", "cdr_h", "rim_over_disc", "I_over_S"]
    errs = {k: [] for k in keys}
    n_done = 0
    n_skipped = 0

    # iterate with limit
    for _, row in df_f.head(limit).iterrows():
        rec = row.to_dict()
        rec = _remap_row_paths(rec, old_prefix, new_prefix)
        try:
            img = Image.from_dict(rec)
        except Exception:
            n_skipped += 1
            continue

        try:
            m = img.metrics_summary()
            gt = m.get("gt", {}) or {}
            pr = m.get("pred", {}) or {}
            for k in keys:
                a = gt.get(k)
                b = pr.get(k)
                if _is_finite(a) and _is_finite(b):
                    errs[k].append(abs(float(b) - float(a)))
            n_done += 1
        except Exception:
            n_skipped += 1
        finally:
            # avoid file handles lingering
            try:
                img.unload_image()
            except Exception:
                pass

    out: Dict[str, Any] = {"n_evaluated": n_done, "n_skipped": n_skipped, "limit": limit}
    for k in keys:
        out[f"{k}_mae"] = float(np.mean(errs[k])) if errs[k] else None
        out[f"{k}_n"] = len(errs[k])

    return out


def render_image_viewer(rec: Dict[str, Any], *, old_prefix: str, new_prefix: str) -> None:
    """
    Render a detailed view for a single Image record.
    """
    rec = _remap_row_paths(rec, old_prefix, new_prefix)

    st.subheader("Selected image")

    # Basic fields
    st.write(
        {
            "uid": rec.get("uid"),
            "dataset": rec.get("dataset"),
            "split": rec.get("split"),
            "subject_id": rec.get("subject_id"),
        }
    )

    img_obj: Optional[Image] = None
    try:
        img_obj = Image.from_dict(rec)
    except Exception as e:
        st.error(f"Failed to construct Image.from_dict for this row: {e!r}")

    # Metric Columns
    cols = st.columns(4)
    cols[0].metric("Dice (Disc)",
                   f"{_safe_float(rec.get('mask_dice_disc')):.3f}" if _is_finite(rec.get("mask_dice_disc")) else "NA")
    cols[1].metric("Dice (Cup)",
                   f"{_safe_float(rec.get('mask_dice_cup')):.3f}" if _is_finite(rec.get("mask_dice_cup")) else "NA")
    cols[2].metric("YOLO conf (Disc)",
                   f"{_safe_float(rec.get('yolo_disc_conf')):.3f}" if _is_finite(rec.get("yolo_disc_conf")) else "NA")
    cols[3].metric("YOLO conf (Cup)",
                   f"{_safe_float(rec.get('yolo_cup_conf')):.3f}" if _is_finite(rec.get("yolo_cup_conf")) else "NA")

    if img_obj is None:
        return

    # --- Visualization Controls ---
    st.markdown("### Visualization Settings")
    c_ctrl1, c_ctrl2, c_ctrl3 = st.columns(3)
    show_overlay = c_ctrl1.checkbox("Show Overlay", value=True)
    show_boxes = c_ctrl2.checkbox("Show Boxes", value=True)
    show_orig = c_ctrl3.checkbox("Show Original Image Panel", value=False)

    # --- Render Overlay ---
    if show_overlay:
        tmp_png = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tmp_png = f.name

            # Pass new toggles to visualize
            img_obj.visualize(
                show=False,
                save_path=Path(tmp_png),
                dpi=150,
                mask_alpha=0.6,
                show_metrics=False,
                show_conf=False,
                show_boxes=show_boxes,  # <--- Linked to checkbox
                show_original_image=show_orig  # <--- Linked to checkbox
            )
            st.image(tmp_png, caption="Visualization", use_container_width=True)
        except Exception as e:
            st.warning(f"Overlay failed: {e!r}")
        finally:
            try:
                img_obj.unload_image()
            except Exception:
                pass
            if tmp_png:
                try:
                    Path(tmp_png).unlink(missing_ok=True)
                except Exception:
                    pass

def render_histograms(df: pd.DataFrame) -> None:
    """
    Render distribution plots and confidence analysis charts.
    """
    st.subheader("Visual Analysis")

    tab1, tab2 = st.tabs(["Distributions (Histograms)", "Confidence vs Performance"])

    # --- TAB 1: Histograms ---
    with tab1:
        # Available numeric metrics
        all_cols = [
            "mask_dice_disc", "mask_dice_cup",
            "yolo_disc_conf", "yolo_cup_conf",
            "cdr_abs_err", "rdr_abs_err",
            "gt_cdr", "pred_cdr"
        ]
        available = [c for c in all_cols if c in df.columns]

        selected_metrics = st.multiselect(
            "Select metrics",
            options=available,
            default=["mask_dice_disc", "mask_dice_cup"] if "mask_dice_disc" in available else available[:2]
        )

        bins = st.slider("Bins", min_value=5, max_value=100, value=30, key="hist_bins")

        if selected_metrics:
            # Arrange plots in a grid
            cols = st.columns(min(len(selected_metrics), 2))
            for i, metric in enumerate(selected_metrics):
                with cols[i % 2]:
                    data = df[metric].dropna()
                    if len(data) == 0:
                        st.warning(f"No valid data for {metric}")
                        continue

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.hist(data, bins=bins, edgecolor="black", alpha=0.7)
                    ax.set_title(f"{metric} (N={len(data)})")
                    ax.set_ylabel("Count")

                    # Mean Line
                    mean_val = data.mean()
                    ax.axvline(mean_val, color='red', linestyle='--', label=f"Mean: {mean_val:.3f}")
                    ax.legend()

                    st.pyplot(fig)
                    plt.close(fig)

    # --- TAB 2: Confidence Analysis ---
    with tab2:
        st.write("Analyze how performance metrics change across confidence levels.")

        c1, c2, c3 = st.columns(3)
        x_metric = c1.selectbox("X Axis (Confidence)", ["yolo_disc_conf", "yolo_cup_conf"], index=0)
        y_metric = c2.selectbox("Y Axis (Metric)", ["mask_dice_disc", "mask_dice_cup", "cdr_abs_err"], index=0)
        n_bins = c3.slider("Number of Bins", 5, 50, 10, key="conf_bins")

        if x_metric in df.columns and y_metric in df.columns:
            df_plot = df[[x_metric, y_metric]].dropna()

            if len(df_plot) > 0:
                # Bin the data
                bins_edges = np.linspace(0, 1.0, n_bins + 1)
                df_plot['bin'] = pd.cut(df_plot[x_metric], bins=bins_edges, include_lowest=True)

                # Calculate stats per bin
                grouped = df_plot.groupby('bin', observed=False)[y_metric].agg(['mean', 'count']).reset_index()

                # Prepare plotting data
                x_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
                counts = grouped['count'].fillna(0).values
                means = grouped['mean'].fillna(0).values  # Fill NaN means with 0 for plotting

                # Create Dual-Axis Plot
                fig, ax1 = plt.subplots(figsize=(8, 5))

                # Bar plot for Counts (Left Axis)
                ax1.bar(x_centers, counts, width=1.0 / n_bins * 0.8, alpha=0.4, color='gray', label='Count')
                ax1.set_xlabel(f"{x_metric} (Bins)")
                ax1.set_ylabel("Sample Count", color='gray')
                ax1.tick_params(axis='y', labelcolor='gray')
                ax1.set_xlim(0, 1.0)

                # Line plot for Metric Mean (Right Axis)
                ax2 = ax1.twinx()
                ax2.plot(x_centers, means, color='blue', marker='o', linewidth=2, label=f"Avg {y_metric}")
                ax2.set_ylabel(f"Average {y_metric}", color='blue')
                ax2.tick_params(axis='y', labelcolor='blue')

                # Legend and Layout
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("No overlapping valid data for these metrics.")
        else:
            st.info("Selected metrics not available in this parquet.")


# -------------------------
# Streamlit app
# -------------------------

st.set_page_config(page_title="Parquet Explorer", layout="wide")
st.title("Parquet Explorer (Image predictions)")

with st.sidebar:
    st.header("Input")
    parquet_path = st.text_input("Predictions parquet path", value="predictions.parquet")

    st.subheader("Path mapping (optional)")
    old_prefix = st.text_input("Old prefix to replace", value="")
    new_prefix = st.text_input("New prefix", value="")

    st.subheader("Filters")
    min_conf = st.slider("Min YOLO confidence", 0.0, 1.0, 0.0, 0.001)
    conf_mode = st.selectbox("Apply confidence to", ["none", "disc", "cup", "both"], index=0)

    show_only_pred = st.checkbox("Only rows with any predictions (masks or YOLO boxes)", value=False)
    show_only_gt = st.checkbox("Only rows with GT masks present", value=False)

    st.subheader("Slow metrics (mask-loading)")
    enable_slow = st.checkbox("Compute slow metrics (loads masks; can be slow)", value=False)
    slow_limit = st.number_input("Slow metrics max rows", min_value=1, max_value=50000, value=2000, step=100)

    st.subheader("Table")
    sort_col = st.text_input("Sort column", value="mask_dice_cup")
    sort_asc = st.checkbox("Ascending sort", value=False)
    max_rows = st.number_input("Display max rows", min_value=50, max_value=200000, value=5000, step=50)

# Load
try:
    raw_df = load_parquet_as_df(parquet_path)
except Exception as e:
    st.error(f"Failed to load parquet: {e!r}")
    st.stop()

df = add_derived_columns(raw_df)

# Dataset + split selectors
datasets = sorted([d for d in df["dataset"].dropna().unique().tolist()])
splits = sorted([s for s in df["split"].dropna().unique().tolist()])

cA, cB = st.columns(2)
with cA:
    selected_datasets = st.multiselect("Datasets", options=datasets, default=datasets)
with cB:
    selected_splits = st.multiselect("Splits", options=splits, default=splits)

# Apply filters
df_f = df.copy()
if selected_datasets:
    df_f = df_f[df_f["dataset"].isin(selected_datasets)]
if selected_splits:
    df_f = df_f[df_f["split"].isin(selected_splits)]

if show_only_pred:
    df_f = df_f[df_f["has_any_pred"] == True]  # noqa: E712
if show_only_gt:
    df_f = df_f[df_f["has_gt"] == True]  # noqa: E712

if conf_mode != "none" and min_conf > 0.0:
    if conf_mode == "disc" and "yolo_disc_conf" in df_f.columns:
        df_f = df_f[df_f["yolo_disc_conf"].fillna(-1.0) >= float(min_conf)]
    elif conf_mode == "cup" and "yolo_cup_conf" in df_f.columns:
        df_f = df_f[df_f["yolo_cup_conf"].fillna(-1.0) >= float(min_conf)]
    elif conf_mode == "both":
        a = df_f["yolo_disc_conf"].fillna(-1.0) >= float(min_conf) if "yolo_disc_conf" in df_f.columns else False
        b = df_f["yolo_cup_conf"].fillna(-1.0) >= float(min_conf) if "yolo_cup_conf" in df_f.columns else False
        df_f = df_f[a & b]

# Header counts
st.subheader("Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total rows in parquet", f"{len(df):,}")
c2.metric("Rows after filters", f"{len(df_f):,}")
c3.metric("With predictions", f"{int(df_f['has_any_pred'].sum()):,}" if "has_any_pred" in df_f.columns else "NA")
c4.metric("With ground truth", f"{int(df_f['has_gt'].sum()):,}" if "has_gt" in df_f.columns else "NA")

# Aggregates
st.subheader("Aggregate metrics (filtered)")
agg = compute_fast_aggregates(df_f)
st.write({k: v for k, v in agg.items()})

# Render the new histograms section
render_histograms(df_f)

if enable_slow:
    with st.spinner("Computing slow metrics (loading masks)..."):
        slow = slow_metrics_summary(df_f, old_prefix=old_prefix, new_prefix=new_prefix, limit=int(slow_limit))
    st.write(slow)

# Table
st.subheader("Per-image table")

display_cols = [
    "uid",
    "dataset",
    "split",
    "subject_id",
    "has_gt",
    "has_any_pred",
    "yolo_disc_conf",
    "yolo_cup_conf",
    "mask_dice_disc",
    "mask_dice_cup",
    "cdr_abs_err",
    "rdr_abs_err",
    "image_path",
    "pred_disc_mask_path",
    "pred_cup_mask_path",
    "gt_disc_mask_path",
    "gt_cup_mask_path",
]
display_cols = [c for c in display_cols if c in df_f.columns]

df_show = df_f[display_cols].copy()

if sort_col and sort_col in df_show.columns:
    try:
        df_show = df_show.sort_values(by=sort_col, ascending=bool(sort_asc), kind="mergesort")
    except Exception:
        pass

df_show = df_show.head(int(max_rows))

st.dataframe(df_show, use_container_width=True, height=420)

# Viewer selection
st.subheader("Image viewer")
uids = df_show["uid"].dropna().astype(str).tolist() if "uid" in df_show.columns else []
selection = st.selectbox("Select uid", options=[""] + uids, index=0)

if selection:
    # find the first matching row in df_f (not just df_show)
    row = df_f[df_f["uid"].astype(str) == str(selection)].head(1)
    if len(row) == 0:
        st.warning("Selected uid not found in filtered data.")
    else:
        rec = row.iloc[0].to_dict()
        render_image_viewer(rec, old_prefix=old_prefix, new_prefix=new_prefix)