"""Quality control filters for DualCellQuant-measured cells."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import Config


def apply_qc(rows: list[dict[str, Any]], cfg: Config) -> list[dict[str, Any]]:
    """Annotate cells by QC rules. Failed cells keep QC=fail + reason.

    Rules (Dual backend):
      - Cell area outside [minimum_cell_area, maximum_cell_area]
      - AND-mask pixels < minimum_and_pixels
      - RedCoverage < minimum_red_coverage (if threshold > 0)
      - Invalid primary ratio
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []

        area = int(row.get("Area", 0) or 0)
        if area < cfg.minimum_cell_area:
            reasons.append("cell_area_low")
        if area > cfg.maximum_cell_area:
            reasons.append("cell_area_high")

        and_px = int(row.get("MembranePixels", row.get("AND_Area_px", 0)) or 0)
        min_and = int(getattr(cfg, "minimum_and_pixels", getattr(cfg, "minimum_ring_pixels", 50)))
        if and_px < min_and:
            reasons.append("and_pixels_low")

        red_cov_thr = float(cfg.minimum_red_coverage)
        if red_cov_thr > 0:
            red_cov = float(row.get("RedCoverage", 1.0) or 0.0)
            if red_cov < red_cov_thr:
                reasons.append("red_coverage_low")

        ratio = row.get("Ratio_T_over_R", row.get("RatioOfMeans_T_R"))
        if ratio is None or not isinstance(ratio, (int, float)) or np.isnan(ratio) or np.isinf(ratio) or ratio <= 0 or ratio > 50:
            reasons.append("invalid_ratio")

        row = dict(row)
        if reasons:
            row["QC"] = "fail"
            row["QC_Reason"] = ";".join(reasons)
        else:
            row["QC"] = "pass"
            row["QC_Reason"] = ""
        out.append(row)
    return out


def passed_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("QC") == "pass"]
