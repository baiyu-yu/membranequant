"""Quality control filters for measured cells."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import Config


def apply_qc(rows: list[dict[str, Any]], cfg: Config) -> list[dict[str, Any]]:
    """Annotate and filter cells by QC rules. Failed cells keep QC=fail + reason.

    Rules (from design):
      - Red Coverage < minimum_red_coverage  (DiI gap / broken membrane stain)
      - Membrane pixels < minimum_ring_pixels
      - Green saturation already handled at segmentation; re-check if present
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        reasons: list[str] = []

        mem_px = int(row.get("MembranePixels", 0))
        if mem_px < cfg.minimum_ring_pixels:
            reasons.append("membrane_pixels_low")

        red_cov = float(row.get("RedCoverage", 0.0))
        if red_cov < cfg.minimum_red_coverage:
            reasons.append("red_coverage_low")

        # NaN metrics are suspicious
        mc = row.get("M/C")
        if mc is None or (isinstance(mc, float) and np.isnan(mc)):
            reasons.append("invalid_mc_ratio")

        if reasons:
            row = dict(row)
            row["QC"] = "fail"
            row["QC_Reason"] = ";".join(reasons)
        else:
            row = dict(row)
            row["QC"] = "pass"
            row["QC_Reason"] = ""
        out.append(row)
    return out


def passed_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("QC") == "pass"]
