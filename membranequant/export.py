"""CSV, GraphPad, mask, and summary export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from .utils import ensure_dir


CSV_COLUMNS = [
    "Image",
    "Field",
    "Experiment",
    "Drug",
    "Group",
    "Condition",
    "CellID",
    "Area",
    "Perimeter",
    "WholeGreen",
    "WholeGreenIntegrated",
    "WholeRed",
    "WholeRedIntegrated",
    "MembraneGreen",
    "MembraneGreenIntegrated",
    "MembraneRed",
    "MembraneRedIntegrated",
    "CytoGreen",
    "CytoGreenIntegrated",
    "MembranePixels",
    "CytoPixels",
    "M/C",
    "MembraneFraction",
    "RedCoverage",
    "RedCoverageArea",
    "PearsonMem",
    "QC",
    "QC_Reason",
]


def rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.DataFrame(rows)
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[CSV_COLUMNS]


def write_results_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def write_summary_csv(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Summary of M/C for QC-pass cells, by Experiment + Drug + Group (条件)."""
    ensure_dir(path.parent)
    passed = df[df["QC"] == "pass"].copy() if "QC" in df.columns else df.copy()
    out_cols = [
        "Experiment",
        "Drug",
        "Group",
        "Condition",
        "Mean_M/C",
        "SD",
        "SEM",
        "N_Cells",
        "N_Images",
    ]
    if passed.empty:
        summary = pd.DataFrame(columns=out_cols)
        summary.to_csv(path, index=False)
        return summary

    if "Condition" not in passed.columns:
        passed["Condition"] = (
            passed.get("Experiment", "").astype(str)
            + passed.get("Drug", "").astype(str)
            + passed.get("Group", "").astype(str)
        )

    def sem(x: pd.Series) -> float:
        n = x.count()
        if n <= 1:
            return 0.0
        return float(x.std(ddof=1) / np.sqrt(n))

    keys = ["Experiment", "Drug", "Group", "Condition"]
    for k in keys:
        if k not in passed.columns:
            passed[k] = ""
    grouped = passed.groupby(keys, dropna=False)
    summary = grouped.agg(
        **{
            "Mean_M/C": ("M/C", "mean"),
            "SD": ("M/C", "std"),
            "N_Cells": ("CellID", "count"),
            "N_Images": ("Image", "nunique"),
        }
    ).reset_index()
    summary["SEM"] = grouped["M/C"].apply(sem).values
    summary = summary[out_cols]
    summary.to_csv(path, index=False)
    return summary


def write_graphpad_csv(df: pd.DataFrame, path: Path) -> None:
    """Wide GraphPad CSV: one column per Condition (如 104d1 / wd1)."""
    ensure_dir(path.parent)
    passed = df[df["QC"] == "pass"].copy() if "QC" in df.columns else df.copy()
    if passed.empty:
        pd.DataFrame().to_csv(path, index=False)
        return

    if "Condition" not in passed.columns or passed["Condition"].isna().all():
        col_key = "Group"
    else:
        col_key = "Condition"

    groups = list(passed[col_key].dropna().unique())
    series = []
    max_len = 0
    for g in groups:
        vals = passed.loc[passed[col_key] == g, "M/C"].dropna().tolist()
        series.append(vals)
        max_len = max(max_len, len(vals))

    data: dict[str, list] = {}
    for g, vals in zip(groups, series):
        padded = vals + [np.nan] * (max_len - len(vals))
        data[str(g)] = padded
    pd.DataFrame(data).to_csv(path, index=False)


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
