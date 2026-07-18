"""Per-cell intensity, membrane enrichment, and colocalization measurements.

Scientific framing
------------------
This pipeline answers two related but distinct questions:

1) **Membrane enrichment** (primary for EGFP membrane localization):
   Is green signal higher at the plasma membrane than in the cytoplasm?
   Metrics: M/C, M/C_DiI, MEI, Edge/Center, MembraneFraction.

2) **Co-occurrence with DiI** (classic colocalization with membrane dye):
   What fraction of green intensity sits on DiI-positive membrane pixels?
   Metrics: Manders M1/M2 (Costes thresholds), Pearson r (whole-cell / membrane).

References (ImageJ Coloc2 / JaCoP style):
- Pearson correlation coefficient (PCC)
- Manders' split coefficients M1, M2
- Costes automatic thresholding for objective Manders thresholds
- Membrane/cytoplasm ratio used widely for PM recruitment assays
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import disk, erosion, opening

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


def _safe_pearson(a: np.ndarray, b: np.ndarray, min_n: int = 10) -> float:
    if a.size < min_n or b.size < min_n:
        return np.nan
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return np.nan
    try:
        r, _ = pearsonr(a.astype(np.float64, copy=False), b.astype(np.float64, copy=False))
        return float(r)
    except Exception:
        return np.nan


def _costes_thresholds(
    ch1: np.ndarray,
    ch2: np.ndarray,
    n_steps: int = 100,
) -> tuple[float, float]:
    """Costes automatic thresholds (simplified Coloc2-style walk).

    Walk intensity pairs down the regression line and pick the last thresholds
    where Pearson correlation of pixels *below* both thresholds is ~0.
    Falls back to Otsu if the walk fails.
    """
    x = ch1.astype(np.float64, copy=False).ravel()
    y = ch2.astype(np.float64, copy=False).ravel()
    if x.size < 30:
        return 0.0, 0.0

    # Linear regression y ~ a*x + b (channel2 on channel1)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-12 or sy < 1e-12:
        try:
            return float(threshold_otsu(x)), float(threshold_otsu(y))
        except Exception:
            return float(np.percentile(x, 50)), float(np.percentile(y, 50))

    a = float(np.cov(x, y, ddof=0)[0, 1] / (sx * sx))
    b = float(np.mean(y) - a * np.mean(x))

    xmax = float(np.max(x))
    ymax = float(np.max(y))
    # Parameterize threshold candidates along the regression segment
    t_vals = np.linspace(0.95, 0.05, n_steps)
    thr1_best, thr2_best = 0.0, 0.0
    found = False

    for t in t_vals:
        t1 = t * xmax
        t2 = a * t1 + b
        if t2 < 0:
            t2 = 0.0
        if t2 > ymax:
            t2 = ymax
        below = (x < t1) & (y < t2)
        if int(np.count_nonzero(below)) < 20:
            continue
        r = _safe_pearson(x[below], y[below], min_n=20)
        if np.isnan(r):
            continue
        # Costes: stop when below-threshold correlation crosses ~0 from positive
        if r <= 0.0:
            thr1_best, thr2_best = float(t1), float(t2)
            found = True
            break
        thr1_best, thr2_best = float(t1), float(t2)
        found = True

    if not found:
        try:
            thr1_best = float(threshold_otsu(x))
            thr2_best = float(threshold_otsu(y))
        except Exception:
            thr1_best = float(np.percentile(x, 50))
            thr2_best = float(np.percentile(y, 50))

    return thr1_best, thr2_best


def _manders_coefficients(
    green: np.ndarray,
    red: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Return M1, M2, thr_green, thr_red, pearson_above.

    M1 = fraction of green intensity in pixels where red > thr_red
    M2 = fraction of red intensity in pixels where green > thr_green
    Thresholds from Costes on pixels inside ``mask`` (whole cell).
    """
    g = green[mask].astype(np.float64, copy=False)
    r = red[mask].astype(np.float64, copy=False)
    if g.size < 30:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    thr_g, thr_r = _costes_thresholds(g, r)
    g_sum = float(np.sum(g))
    r_sum = float(np.sum(r))
    if g_sum <= 0 or r_sum <= 0:
        return np.nan, np.nan, thr_g, thr_r, np.nan

    # Manders: intensity in co-occurring pixels / total intensity
    m1 = float(np.sum(g[r > thr_r]) / g_sum)
    m2 = float(np.sum(r[g > thr_g]) / r_sum)
    above = (g > thr_g) & (r > thr_r)
    p_above = _safe_pearson(g[above], r[above], min_n=10) if np.count_nonzero(above) >= 10 else np.nan
    return m1, m2, thr_g, thr_r, p_above


def _build_dil_membrane_mask(
    whole: np.ndarray,
    red: np.ndarray,
    geometric_mem: np.ndarray,
    ring_width: int,
    global_red_thr: float,
) -> np.ndarray:
    """DiI-guided membrane mask inside a border search band.

    Strategy:
    1. Take outer border band (max(ring_width+2, 5) px) as candidate membrane zone.
    2. Within the band, keep pixels with red above max(local Otsu, global thr*0.3,
       local 60th percentile) — adaptive, not a fixed top-35%.
    3. Morphological open to remove speckles; fall back to geometric ring if empty.
    """
    band = max(int(ring_width) + 2, 5)
    selem = disk(band)
    eroded = erosion(whole, selem)
    if np.count_nonzero(eroded) < 10:
        selem = disk(max(3, ring_width))
        eroded = erosion(whole, selem)

    if np.count_nonzero(eroded) < 10:
        candidate = geometric_mem
    else:
        candidate = whole & ~eroded

    red_cand = red[candidate]
    if red_cand.size == 0:
        return geometric_mem

    # Adaptive local threshold: blend percentile + Otsu + global floor
    try:
        local_otsu = float(threshold_otsu(red_cand)) if red_cand.max() > red_cand.min() else 0.0
    except Exception:
        local_otsu = 0.0
    local_pct = float(np.percentile(red_cand, 60))
    thr = max(local_otsu * 0.85, local_pct, global_red_thr * 0.3, 1e-6)

    mem_dil = candidate & (red >= thr)
    if np.count_nonzero(mem_dil) > 0:
        mem_dil = opening(mem_dil, disk(1))

    if np.count_nonzero(mem_dil) == 0:
        # Looser fallback: top 40% red in the band
        thr2 = float(np.percentile(red_cand, 60))
        mem_dil = candidate & (red >= thr2)

    if np.count_nonzero(mem_dil) == 0:
        mem_dil = geometric_mem

    return mem_dil


def _edge_center_ratio(green: np.ndarray, whole: np.ndarray) -> float:
    """Mean intensity in outer shell / mean intensity in inner core.

    Independent of DiI; pure geometric membrane-enrichment proxy.
    """
    if np.count_nonzero(whole) < 50:
        return np.nan
    # Core: strong erosion; shell: remainder
    core = erosion(whole, disk(6))
    if np.count_nonzero(core) < 20:
        core = erosion(whole, disk(3))
    if np.count_nonzero(core) < 10:
        return np.nan
    shell = whole & ~core
    if np.count_nonzero(shell) < 10:
        return np.nan
    c_mean = float(np.mean(green[core]))
    e_mean = float(np.mean(green[shell]))
    if c_mean <= 1e-12:
        return np.nan
    return e_mean / c_mean


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

    # Global red Otsu as QC baseline (DiI present vs noise)
    try:
        if red.max() > red.min() + 1e-6:
            global_red_thr = float(threshold_otsu(red))
        else:
            global_red_thr = 0.1
    except Exception:
        global_red_thr = 0.1

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

        # ---- DiI-guided membrane (optimized) ----
        mem_dil = _build_dil_membrane_mask(whole, red, mem, cfg.ring_width, global_red_thr)
        cyto_dil = whole & ~mem_dil

        mg_mean_dil, mg_int_dil, mem_n_dil = _sum_mean(green, mem_dil)
        mr_mean_dil, mr_int_dil, _ = _sum_mean(red, mem_dil)
        cg_mean_dil, cg_int_dil, cyto_n_dil = _sum_mean(green, cyto_dil)

        mc_ratio_dil = (mg_mean_dil / cg_mean_dil) if cg_mean_dil > 0 else np.nan
        # Legacy name kept for UI compatibility: green mass on DiI membrane / total green
        mem_frac_dil = (mg_int_dil / g_int) if g_int > 0 else np.nan

        # Membrane Enrichment Index ∈ (-1, 1): 0=equal, >0 membrane-enriched
        if (mg_mean_dil + cg_mean_dil) > 0:
            mei = (mg_mean_dil - cg_mean_dil) / (mg_mean_dil + cg_mean_dil)
        else:
            mei = np.nan

        # Geometric M/C and membrane fraction
        mc_ratio = (mg_mean / cg_mean) if cg_mean > 0 else np.nan
        mem_fraction = (mg_int / g_int) if g_int > 0 else np.nan
        edge_center = _edge_center_ratio(green, whole)

        # Red coverage on geometric ring (QC)
        if mem_n > 0:
            red_ring = red[mem]
            red_thr = max(global_red_thr * 0.5, 1e-6)
            red_cov_area = int(np.count_nonzero(red_ring > red_thr))
            red_coverage = red_cov_area / mem_n
        else:
            red_cov_area = 0
            red_coverage = 0.0

        # Pearson correlations
        pearson_mem = _safe_pearson(green[mem], red[mem]) if mem_n >= 10 else np.nan
        pearson_whole = _safe_pearson(green[whole], red[whole]) if np.count_nonzero(whole) >= 10 else np.nan
        pearson_dil = (
            _safe_pearson(green[mem_dil], red[mem_dil]) if mem_n_dil >= 10 else np.nan
        )

        # True Manders with Costes thresholds (inside whole cell)
        m1, m2, thr_g, thr_r, pearson_above = _manders_coefficients(green, red, whole)

        # Optional: also report geometric-ring-only Pearson if flag set (always stored)
        if not cfg.compute_pearson:
            # Still compute; flag only existed historically. Values always useful.
            pass

        row: dict[str, Any] = {
            "Image": meta.get("image_id", ""),
            "Field": meta.get("field", ""),
            "Experiment": meta.get("experiment", ""),
            "Drug": meta.get("drug", ""),
            "Group": meta.get("group", ""),
            "Condition": meta.get("condition_id", ""),
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
            # Geometric membrane metrics
            "M/C": mc_ratio,
            "MembraneFraction": mem_fraction,
            "RedCoverage": red_coverage,
            "RedCoverageArea": red_cov_area,
            "PearsonMem": pearson_mem,
            # DiI-guided enrichment (recommended primary)
            "M/C_DiI": mc_ratio_dil,
            "MEI": mei,
            "EdgeCenterRatio": edge_center,
            "MembraneFraction_DiI": mem_frac_dil,
            "MembraneGreen_DiI": mg_mean_dil,
            "CytoGreen_DiI": cg_mean_dil,
            "MembraneRed_DiI": mr_mean_dil,
            "MembranePixels_DiI": mem_n_dil,
            "CytoPixels_DiI": cyto_n_dil,
            # Colocalization (Coloc2-style)
            "Manders_M1": m1,  # fraction of green co-occurring with DiI
            "Manders_M2": m2,  # fraction of DiI co-occurring with green
            "Costes_ThrGreen": thr_g,
            "Costes_ThrRed": thr_r,
            "PearsonWhole": pearson_whole,
            "PearsonDiI": pearson_dil,
            "PearsonAboveThr": pearson_above,
            # Legacy alias: old "Manders_M1" was really membrane fraction under DiI mask
            "DiI_OverlapFraction": mem_frac_dil,
            "QC": "pass",
            "QC_Reason": "",
        }
        rows.append(row)

    return rows
