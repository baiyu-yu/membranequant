"""CSV, GraphPad, mask, and summary export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from .utils import ensure_dir


# Primary columns (order for results.csv). DualCellQuant metrics first.
CSV_COLUMNS = [
    "Image",
    "Field",
    "Experiment",
    "Drug",
    "Group",
    "Condition",
    "CellID",
    "Area",
    "Area_um2",
    "Perimeter",
    "AND_Area_px",
    "AND_Area_um2",
    "WholeGreen",
    "WholeGreenIntegrated",
    "WholeRed",
    "WholeRedIntegrated",
    "MembraneGreen",
    "MembraneGreenIntegrated",
    "MembraneRed",
    "MembraneRedIntegrated",
    "MembranePixels",
    "CytoGreen",
    "CytoGreenIntegrated",
    "CytoPixels",
    # DualCellQuant primary
    "Ratio_T_over_R",
    "RatioOfMeans_T_R",
    "StdRatio_T_over_R",
    "SumRatio_T_over_R",
    "Enrichment_Membrane_vs_Whole",
    "MembraneFraction",
    "Std_target_on_mask",
    "Std_reference_on_mask",
    "Std_target_whole",
    "Std_reference_whole",
    "RedCoverage",
    "RedCoverageArea",
    "Backend",
    "QC",
    "QC_Reason",
]

# Metrics exported to multi-metric summaries / GraphPad
PRIMARY_METRICS = [
    "Ratio_T_over_R",
    "RatioOfMeans_T_R",
    "Enrichment_Membrane_vs_Whole",
    "MembraneGreen",
    "MembraneRed",
    "MembraneFraction",
    "WholeGreen",
]


def rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.DataFrame(rows)
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    # Keep known columns first, then any extras
    extra = [c for c in df.columns if c not in CSV_COLUMNS]
    return df[CSV_COLUMNS + extra]


def write_results_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def _sem(x: pd.Series) -> float:
    n = x.count()
    if n <= 1:
        return 0.0
    return float(x.std(ddof=1) / np.sqrt(n))


def write_summary_csv(df: pd.DataFrame, path: Path, metric: str = "Ratio_T_over_R") -> pd.DataFrame:
    """Summary of one metric for QC-pass cells, by Experiment + Drug + Group + Condition."""
    ensure_dir(path.parent)
    passed = df[df["QC"] == "pass"].copy() if "QC" in df.columns else df.copy()
    passed = passed.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
    if metric in ("Ratio_T_over_R", "RatioOfMeans_T_R", "Enrichment_Membrane_vs_Whole", "M/C_DiI", "M/C", "MEI"):
        passed = passed[(passed[metric].isna()) | ((passed[metric] > 0) & (passed[metric] <= 50.0))].copy()

    # Prefer Dual T/R ratio; fall back to enrichment
    if metric not in passed.columns or passed[metric].isna().all():
        for fallback in ("Ratio_T_over_R", "RatioOfMeans_T_R", "Enrichment_Membrane_vs_Whole"):
            if fallback in passed.columns and not passed[fallback].isna().all():
                metric = fallback
                break

    out_cols = [
        "Experiment",
        "Drug",
        "Group",
        "Condition",
        "Metric",
        "Mean",
        "SD",
        "SEM",
        "Median",
        "N_Cells",
        "N_Images",
        # Backward-compatible aliases used by older plot code
        "Mean_M/C",
    ]
    if passed.empty:
        summary = pd.DataFrame(columns=out_cols)
        summary.to_csv(path, index=False)
        return summary

    if "Condition" not in passed.columns:
        passed["Condition"] = (
            passed.get("Experiment", pd.Series([""] * len(passed))).astype(str)
            + passed.get("Drug", pd.Series([""] * len(passed))).astype(str)
            + passed.get("Group", pd.Series([""] * len(passed))).astype(str)
        )

    keys = ["Experiment", "Drug", "Group", "Condition"]
    for k in keys:
        if k not in passed.columns:
            passed[k] = ""

    # Drop NaN metric rows for aggregation
    work = passed.dropna(subset=[metric]) if metric in passed.columns else passed
    if work.empty:
        summary = pd.DataFrame(columns=out_cols)
        summary.to_csv(path, index=False)
        return summary

    grouped = work.groupby(keys, dropna=False)
    summary = grouped.agg(
        Mean=(metric, "mean"),
        SD=(metric, "std"),
        Median=(metric, "median"),
        N_Cells=("CellID", "count"),
        N_Images=("Image", "nunique"),
    ).reset_index()
    summary["SEM"] = grouped[metric].apply(_sem).values
    summary["Metric"] = metric
    summary["Mean_M/C"] = summary["Mean"]  # alias for bar plots
    summary = summary[out_cols]
    summary.to_csv(path, index=False)
    return summary


def write_multi_metric_summary(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """One row per Condition × Metric for multi-panel comparison plots."""
    ensure_dir(path.parent)
    passed = df[df["QC"] == "pass"].copy() if "QC" in df.columns else df.copy()
    if passed.empty:
        empty = pd.DataFrame()
        empty.to_csv(path, index=False)
        return empty

    if "Condition" not in passed.columns or passed["Condition"].isna().all():
        passed["Condition"] = passed.get("Group", "Unknown").astype(str)

    rows: list[dict[str, Any]] = []
    for metric in PRIMARY_METRICS:
        if metric not in passed.columns:
            continue
        for cond, sub in passed.groupby("Condition", dropna=False):
            vals = sub[metric].replace([np.inf, -np.inf], np.nan).infer_objects(copy=False).dropna()
            if metric in ("Ratio_T_over_R", "RatioOfMeans_T_R", "Enrichment_Membrane_vs_Whole", "M/C_DiI", "M/C", "MEI"):
                vals = vals[(vals > 0) & (vals <= 50.0)]
            if vals.empty:
                continue
            n = int(vals.count())
            mean = float(vals.mean())
            sd = float(vals.std(ddof=1)) if n > 1 else 0.0
            sem = sd / np.sqrt(n) if n > 1 else 0.0
            rows.append(
                {
                    "Condition": cond,
                    "Metric": metric,
                    "Mean": mean,
                    "SD": sd,
                    "SEM": sem,
                    "Median": float(vals.median()),
                    "N_Cells": n,
                    "Mean_M/C": mean,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def write_graphpad_csv(df: pd.DataFrame, path: Path, metric: str = "Ratio_T_over_R") -> None:
    """Wide GraphPad CSV: one column per Condition for the chosen metric."""
    ensure_dir(path.parent)
    passed = df[df["QC"] == "pass"].copy() if "QC" in df.columns else df.copy()
    if passed.empty:
        pd.DataFrame().to_csv(path, index=False)
        return

    if metric not in passed.columns or passed[metric].isna().all():
        for fallback in ("Ratio_T_over_R", "RatioOfMeans_T_R", "Enrichment_Membrane_vs_Whole"):
            if fallback in passed.columns and not passed[fallback].isna().all():
                metric = fallback
                break

    if "Condition" not in passed.columns or passed["Condition"].isna().all():
        col_key = "Group"
    else:
        col_key = "Condition"

    groups = list(passed[col_key].dropna().unique())
    series = []
    max_len = 0
    for g in groups:
        vals = passed.loc[passed[col_key] == g, metric].dropna().tolist()
        series.append(vals)
        max_len = max(max_len, len(vals))

    data: dict[str, list] = {}
    for g, vals in zip(groups, series):
        padded = vals + [np.nan] * (max_len - len(vals))
        data[str(g)] = padded
    pd.DataFrame(data).to_csv(path, index=False)


def write_all_graphpad(df: pd.DataFrame, out_dir: Path) -> None:
    """Write GraphPad-ready wide CSVs for each primary metric."""
    ensure_dir(out_dir)
    for metric in PRIMARY_METRICS:
        if metric not in df.columns:
            continue
        safe = metric.replace("/", "_").replace("\\", "_")
        write_graphpad_csv(df, out_dir / f"graphpad_{safe}.csv", metric=metric)
    # Keep classic filename pointing at Dual primary metric
    write_graphpad_csv(df, out_dir / "graphpad_MC.csv", metric="Ratio_T_over_R")


def save_label_mask(labels: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    tifffile.imwrite(str(path), labels.astype(np.uint16))


def write_qc_log(rejected_seg: list[dict], rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    lines = ["# MembraneQuant QC log", ""]
    if rejected_seg:
        lines.append("## Segmentation rejects")
        for r in rejected_seg:
            lines.append(f"- cell_id={r.get('cell_id')}: {r.get('reason')}")
        lines.append("")
    lines.append("## Measurement QC")
    for row in rows:
        if row.get("QC") != "pass":
            lines.append(
                f"- Image={row.get('Image')} CellID={row.get('CellID')}: {row.get('QC_Reason')}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
