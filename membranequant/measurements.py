"""Per-cell intensity and morphology measurements."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.measure import regionprops
from scipy.stats import pearsonr

from .config import Config
from .segmentation import CellMasks


def _sum_mean(image: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    pixels = image[mask]
    n = int(pixels.size)
    if n == 0:
        return 0.0, 0.0, 0
    total = float(np.sum(pixels))
    mean = total / n
    return mean, total, n


def measure_cells(
    green: np.ndarray,
    red: np.ndarray,
    masks: CellMasks,
    meta: dict[str, Any],
    cfg: Config,
) -> list[dict[str, Any]]:
    """Measure each kept cell; return list of row dicts (pre-QC status)."""
    rows: list[dict[str, Any]] = []
    labels = masks.labels
    membrane = masks.membrane
    cytoplasm = masks.cytoplasm

    props = {int(p.label): p for p in regionprops(labels)}

    for cid in masks.kept_ids:
        whole = labels == cid
        mem = membrane == cid
        cyto = cytoplasm == cid

        prop = props.get(cid)
        area = int(prop.area) if prop is not None else int(np.count_nonzero(whole))
        perimeter = float(prop.perimeter) if prop is not None else 0.0

        g_mean, g_int, _ = _sum_mean(green, whole)
        r_mean, r_int, _ = _sum_mean(red, whole)

        mg_mean, mg_int, mem_n = _sum_mean(green, mem)
        mr_mean, mr_int, _ = _sum_mean(red, mem)
        cg_mean, cg_int, cyto_n = _sum_mean(green, cyto)

        # Red coverage on membrane ring: fraction of ring pixels with DiI signal
        # Use adaptive-ish floor: mean of ring red * 0.5 or global low threshold
        if mem_n > 0:
            red_ring = red[mem]
            red_thr = max(float(np.percentile(red_ring, 25)) * 0.5, 1e-6)
            red_cov_area = int(np.count_nonzero(red_ring > red_thr))
            red_coverage = red_cov_area / mem_n
        else:
            red_cov_area = 0
            red_coverage = 0.0

        mc_ratio = (mg_mean / cg_mean) if cg_mean > 0 else np.nan
        mem_fraction = (mg_int / g_int) if g_int > 0 else np.nan

        pearson = np.nan
        if cfg.compute_pearson and mem_n >= 10:
            try:
                pearson, _ = pearsonr(green[mem], red[mem])
            except Exception:
                pearson = np.nan

        row: dict[str, Any] = {
            "Image": meta.get("image_id", ""),
            "Field": meta.get("field", ""),
            "Experiment": meta.get("experiment", ""),
            "Drug": meta.get("drug", ""),
            "Group": meta.get("group", ""),  # 组别（文件名中药物后的数字）
            "Condition": meta.get("condition_id", ""),  # 如 104d1 / wd1
            "CellID": cid,
            "Area": area,
            "Perimeter": perimeter,
            "WholeGreen": g_mean,
            "WholeGreenIntegrated": g_int,
            "WholeRed": r_mean,
            "WholeRedIntegrated": r_int,
            "MembraneGreen": mg_mean,
            "MembraneGreenIntegrated": mg_int,
            "MembraneRed": mr_mean,
            "MembraneRedIntegrated": mr_int,
            "CytoGreen": cg_mean,
            "CytoGreenIntegrated": cg_int,
            "MembranePixels": mem_n,
            "CytoPixels": cyto_n,
            "M/C": mc_ratio,
            "MembraneFraction": mem_fraction,
            "RedCoverage": red_coverage,
            "RedCoverageArea": red_cov_area,
            "PearsonMem": pearson,
            "QC": "pass",
            "QC_Reason": "",
        }
        rows.append(row)

    return rows
