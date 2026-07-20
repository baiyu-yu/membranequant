"""DualCellQuant image-analysis backend.

MembraneQuant keeps only:
  - experiment folder scanning / Red–Green pairing (``io``)
  - post-analysis: QC, CSV/GraphPad export, plots, Web UI

All per-image analysis (background, Cellpose segmentation, EDT radial
membrane ROI, masking, per-cell intensity) is delegated to DualCellQuant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import Config
from .io import FieldPair


def dualcellquant_status() -> dict[str, Any]:
    """Return DualCellQuant / Cellpose availability."""
    info: dict[str, Any] = {
        "available": False,
        "version": None,
        "cellpose_available": False,
        "cuda_available": False,
        "message": "",
    }
    try:
        import dualcellquant

        info["available"] = True
        info["version"] = getattr(dualcellquant, "__version__", None)
    except Exception as exc:
        info["message"] = f"DualCellQuant not installed: {exc}"
        return info

    try:
        import cellpose  # noqa: F401
        import torch

        info["cellpose_available"] = True
        info["cuda_available"] = bool(torch.cuda.is_available())
        ver = info["version"] or "?"
        gpu = "CUDA available" if info["cuda_available"] else "CPU only"
        info["message"] = f"DualCellQuant {ver}; Cellpose OK ({gpu})"
    except Exception as exc:
        info["message"] = f"DualCellQuant present but Cellpose/torch missing: {exc}"
    return info


def load_image_pil(path: Path) -> Image.Image:
    """Load a microscopy TIFF as PIL Image, preserving bit depth when possible."""
    import tifffile

    raw = np.asarray(tifffile.imread(str(path)))
    # Squeeze singleton dims (e.g. 1xHxW)
    while raw.ndim > 2 and 1 in raw.shape:
        raw = np.squeeze(raw)
    if raw.ndim == 3:
        # Prefer HWC with last dim channels
        if raw.shape[0] in (3, 4) and raw.shape[-1] not in (3, 4):
            raw = np.moveaxis(raw, 0, -1)
        if raw.shape[-1] >= 3:
            rgb = raw[..., :3]
            if np.issubdtype(rgb.dtype, np.floating):
                vmax = float(np.nanmax(rgb)) if rgb.size else 1.0
                if vmax <= 1.0:
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                else:
                    rgb = np.clip(rgb, 0, None)
                    rgb = (rgb / rgb.max() * 255).astype(np.uint8) if rgb.max() > 0 else rgb.astype(np.uint8)
            elif rgb.dtype != np.uint8:
                info = np.iinfo(rgb.dtype) if np.issubdtype(rgb.dtype, np.integer) else None
                if info is not None and info.max > 0:
                    rgb = (rgb.astype(np.float32) / info.max * 255).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)
            return Image.fromarray(rgb, mode="RGB")
        # Multi-plane but not RGB — take first plane
        raw = raw[..., 0] if raw.shape[-1] < raw.shape[0] else raw[0]

    if raw.ndim != 2:
        raise ValueError(f"Unsupported image shape {raw.shape} for {path}")

    if np.issubdtype(raw.dtype, np.floating):
        vmax = float(np.nanmax(raw)) if raw.size else 1.0
        if vmax <= 1.0 + 1e-6:
            arr8 = (np.clip(raw, 0, 1) * 255).astype(np.uint8)
        else:
            arr8 = (np.clip(raw / vmax, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(arr8, mode="L")

    if raw.dtype == np.uint8:
        return Image.fromarray(raw, mode="L")
    if raw.dtype == np.uint16:
        return Image.fromarray(raw, mode="I;16")
    if np.issubdtype(raw.dtype, np.integer):
        # Scale other integer depths to uint16 range for Dual's dtype max logic
        info = np.iinfo(raw.dtype)
        scaled = (raw.astype(np.float32) / max(info.max, 1) * 65535).astype(np.uint16)
        return Image.fromarray(scaled, mode="I;16")
    arr8 = np.clip(raw, 0, 255).astype(np.uint8)
    return Image.fromarray(arr8, mode="L")


@dataclass
class DualFieldResult:
    """Outputs of DualCellQuant for one field pair."""

    rows: list[dict[str, Any]]
    labels: np.ndarray
    membrane: np.ndarray  # radial (or AND) label mask
    cytoplasm: np.ndarray
    and_mask: np.ndarray
    green_vis: np.ndarray  # float 0-1 for overlay
    red_vis: np.ndarray
    dual_df_columns: list[str]
    method: str = "dualcellquant"
    rejected: list[dict[str, Any]] | None = None


def _require_dual():
    st = dualcellquant_status()
    if not st["available"]:
        raise ImportError(
            "DualCellQuant is required for image analysis.\n"
            'Install with: pip install "git+https://github.com/fuji3to4/DualCellQuant.git"\n'
            f"Detail: {st['message']}"
        )
    if not st["cellpose_available"]:
        raise ImportError(
            "Cellpose (and PyTorch) are required by DualCellQuant.\n"
            "Install with: pip install cellpose torch\n"
            f"Detail: {st['message']}"
        )
    import dualcellquant.core as core
    import dualcellquant.radial as radial

    return core, radial


def _channel_index(cfg: Config) -> int:
    """Map config channel name to DualCellQuant channel index."""
    # Dual: 0 gray, 1 R, 2 G, 3 B. Single-channel TIFFs always use 0.
    name = (cfg.dual_measure_channel or "gray").strip().lower()
    return {"gray": 0, "grey": 0, "r": 1, "red": 1, "g": 2, "green": 2, "b": 3, "blue": 3}.get(
        name, 0
    )


def _fallback_analyze_field(
    pair: FieldPair,
    cfg: Config,
    meta: dict[str, Any] | None = None,
) -> DualFieldResult:
    import tifffile
    from .preprocess import preprocess_pair
    from .segmentation import build_cell_masks
    from .measurements import measure_cells

    red_raw = tifffile.imread(str(pair.red_path))
    green_raw = tifffile.imread(str(pair.green_path))
    red_p, green_p = preprocess_pair(red_raw, green_raw, cfg)
    masks = build_cell_masks(green_p, red_p, cfg)
    rows = measure_cells(green_p, red_p, masks, meta=meta or {}, cfg=cfg)

    for r in rows:
        ratio = r.get("M/C_DiI", r.get("M/C", np.nan))
        r.setdefault("Ratio_T_over_R", ratio)
        r.setdefault("RatioOfMeans_T_R", ratio)
        r.setdefault("Enrichment_Membrane_vs_Whole", r.get("MEI", np.nan))
        r.setdefault("Backend", "fallback")

    return DualFieldResult(
        rows=rows,
        labels=masks.labels,
        membrane=masks.membrane,
        cytoplasm=masks.cytoplasm,
        and_mask=masks.membrane > 0,
        green_vis=green_p,
        red_vis=red_p,
        dual_df_columns=["Ratio_T_over_R", "RatioOfMeans_T_R", "Enrichment_Membrane_vs_Whole"],
        method="fallback",
        rejected=masks.rejected,
    )


def analyze_field_dual(
    pair: FieldPair,
    cfg: Config,
    meta: dict[str, Any] | None = None,
) -> DualFieldResult:
    """Run full DualCellQuant pipeline on one Red/Green pair.

    Target  = Green (EGFP)
    Reference = Red (DiI)
    """
    try:
        core, radial = _require_dual()
    except ImportError:
        return _fallback_analyze_field(pair, cfg, meta=meta)

    target_img = load_image_pil(pair.green_path)  # EGFP
    reference_img = load_image_pil(pair.red_path)  # DiI

    chan = _channel_index(cfg)
    seg_source = (cfg.dual_seg_source or "target").strip().lower()
    if seg_source not in {"target", "reference"}:
        seg_source = "target"

    # 1) Segmentation (Cellpose via DualCellQuant)
    _overlay, _seg_tiff, _mask_viz, labels = core.run_segmentation(
        target_img,
        reference_img,
        seg_source=seg_source,
        seg_channel=int(cfg.dual_seg_channel),
        diameter=float(cfg.dual_diameter),
        flow_threshold=float(cfg.dual_flow_threshold),
        cellprob_threshold=float(cfg.dual_cellprob_threshold),
        use_gpu=bool(cfg.dual_use_gpu),
        drop_edge_cells=bool(cfg.dual_drop_edge_cells),
        inside_fraction_min=float(cfg.dual_inside_fraction_min),
        edge_margin_pct=float(cfg.dual_edge_margin_pct),
    )
    labels = np.asarray(labels, dtype=np.int32)
    if labels.size == 0 or int(labels.max()) == 0:
        return DualFieldResult(
            rows=[],
            labels=labels if labels.size else np.zeros((1, 1), dtype=np.int32),
            membrane=np.zeros_like(labels, dtype=np.int32),
            cytoplasm=np.zeros_like(labels, dtype=np.int32),
            and_mask=np.zeros_like(labels, dtype=bool),
            green_vis=np.zeros(labels.shape, dtype=np.float32),
            red_vis=np.zeros(labels.shape, dtype=np.float32),
            dual_df_columns=[],
            rejected=[],
        )

    # 2) EDT radial membrane ROI
    _rad_overlay, radial_bool, radial_labels, _rb_tiff, _rl_tiff = radial.radial_mask(
        labels,
        inner_pct=float(cfg.dual_radial_inner_pct),
        outer_pct=float(cfg.dual_radial_outer_pct),
        min_obj_size=int(cfg.dual_radial_min_obj),
    )
    radial_bool = np.asarray(radial_bool, dtype=bool)
    radial_labels = np.asarray(radial_labels, dtype=np.int32)

    # 3) Target / Reference masks (optionally restricted to radial ROI)
    tgt_roi = radial_labels if cfg.dual_use_radial_for_target else None
    ref_roi = radial_labels if cfg.dual_use_radial_for_ref else None

    _t_ov, _t_tiff, tgt_mask = core.apply_mask(
        target_img,
        labels,
        measure_channel=chan,
        sat_limit=float(cfg.dual_sat_limit),
        mask_mode=str(cfg.dual_target_mask_mode),
        pct=float(cfg.dual_target_mask_percentile),
        min_obj_size=int(cfg.dual_min_obj_size),
        roi_labels=tgt_roi,
        mask_name="target_mask",
    )
    _r_ov, _r_tiff, ref_mask = core.apply_mask(
        reference_img,
        labels,
        measure_channel=chan,
        sat_limit=float(cfg.dual_sat_limit),
        mask_mode=str(cfg.dual_ref_mask_mode),
        pct=float(cfg.dual_ref_mask_percentile),
        min_obj_size=int(cfg.dual_min_obj_size),
        roi_labels=ref_roi,
        mask_name="reference_mask",
    )
    tgt_mask = np.asarray(tgt_mask, dtype=bool)
    ref_mask = np.asarray(ref_mask, dtype=bool)

    roi_mask = radial_bool if cfg.dual_and_with_radial else None
    roi_labels = radial_labels if cfg.dual_and_with_radial else None

    # 4) Integrate & quantify (DualCellQuant native metrics)
    (
        _ov_t,
        _ov_r,
        _and_tiff,
        dual_df,
        _csv_path,
        _tgt_on,
        _ref_on,
        _ratio_ov,
    ) = core.integrate_and_quantify(
        target_img,
        reference_img,
        labels,
        tgt_mask,
        ref_mask,
        tgt_chan=chan,
        ref_chan=chan,
        pixel_width_um=float(cfg.dual_pixel_width_um),
        pixel_height_um=float(cfg.dual_pixel_height_um),
        pp_bg_enable=bool(cfg.dual_bg_enable),
        pp_bg_radius=int(cfg.dual_bg_radius),
        pp_norm_enable=bool(cfg.dual_norm_enable),
        pp_norm_method=str(cfg.dual_norm_method),
        bg_mode=str(cfg.dual_bg_mode),
        bg_dark_pct=float(cfg.dual_bg_dark_pct),
        roi_mask=roi_mask,
        roi_labels=roi_labels,
        ratio_ref_epsilon=float(cfg.dual_ratio_ref_epsilon),
    )

    # Visualization arrays (0–1, with Dual preprocess if enabled)
    green_vis = core.extract_single_channel(
        core.preprocess_for_processing(
            target_img,
            use_native_scale=False,
            bg_enable=bool(cfg.dual_bg_enable),
            bg_radius=int(cfg.dual_bg_radius),
            bg_mode=str(cfg.dual_bg_mode),
            bg_dark_pct=float(cfg.dual_bg_dark_pct),
            norm_enable=bool(cfg.dual_norm_enable),
            norm_method=str(cfg.dual_norm_method),
        ),
        chan,
    ).astype(np.float32)
    red_vis = core.extract_single_channel(
        core.preprocess_for_processing(
            reference_img,
            use_native_scale=False,
            bg_enable=bool(cfg.dual_bg_enable),
            bg_radius=int(cfg.dual_bg_radius),
            bg_mode=str(cfg.dual_bg_mode),
            bg_dark_pct=float(cfg.dual_bg_dark_pct),
            norm_enable=bool(cfg.dual_norm_enable),
            norm_method=str(cfg.dual_norm_method),
        ),
        chan,
    ).astype(np.float32)
    # Robust display stretch
    for arr in (green_vis, red_vis):
        lo, hi = np.percentile(arr, (1, 99)) if arr.size else (0.0, 1.0)
        if hi <= lo:
            hi = lo + 1e-6
        arr[:] = np.clip((arr - lo) / (hi - lo), 0, 1)

    # AND selection mask (same logic as Dual integrate)
    and_base = tgt_mask & ref_mask
    if roi_mask is not None and roi_labels is not None:
        and_mask = np.zeros_like(and_base, dtype=bool)
        for lab in np.unique(labels):
            lab = int(lab)
            if lab <= 0:
                continue
            cell = labels == lab
            inside = and_base & cell & roi_mask & (roi_labels == lab)
            outside = roi_mask & (roi_labels == lab) & (~cell)
            and_mask |= inside | outside
    elif roi_mask is not None:
        and_mask = and_base & roi_mask
    else:
        and_mask = and_base

    # Membrane label mask from radial; cytoplasm = cell \ membrane
    membrane = radial_labels.copy()
    cytoplasm = np.zeros_like(labels, dtype=np.int32)
    for lab in np.unique(labels):
        lab = int(lab)
        if lab <= 0:
            continue
        cell = labels == lab
        mem = membrane == lab
        cyto = cell & ~mem
        cytoplasm[cyto] = lab

    meta = meta or {}
    rows = dual_df_to_rows(dual_df, meta=meta, labels=labels, and_mask=and_mask, ref_mask=ref_mask)
    dual_cols = list(dual_df.columns) if dual_df is not None and len(dual_df.columns) else []

    return DualFieldResult(
        rows=rows,
        labels=labels,
        membrane=membrane,
        cytoplasm=cytoplasm,
        and_mask=and_mask,
        green_vis=green_vis,
        red_vis=red_vis,
        dual_df_columns=dual_cols,
        method="dualcellquant",
        rejected=[],
    )


def dual_df_to_rows(
    dual_df,
    meta: dict[str, Any],
    labels: np.ndarray | None = None,
    and_mask: np.ndarray | None = None,
    ref_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Map DualCellQuant quantification DataFrame rows to MembraneQuant result dicts."""
    import pandas as pd

    if dual_df is None or (isinstance(dual_df, pd.DataFrame) and dual_df.empty):
        return []

    rows: list[dict[str, Any]] = []
    for _, r in dual_df.iterrows():
        lab = int(r.get("label", 0))
        area_cell = float(r.get("area_cell_px", np.nan))
        area_and = float(r.get("area_and_px", np.nan))
        mean_t_mem = float(r.get("mean_target_on_mask", np.nan))
        sum_t_mem = float(r.get("sum_target_on_mask", np.nan))
        mean_r_mem = float(r.get("mean_reference_on_mask", np.nan))
        sum_r_mem = float(r.get("sum_reference_on_mask", np.nan))
        mean_t_whole = float(r.get("mean_target_whole", np.nan))
        sum_t_whole = float(r.get("sum_target_whole", np.nan))
        mean_r_whole = float(r.get("mean_reference_whole", np.nan))
        sum_r_whole = float(r.get("sum_reference_whole", np.nan))
        mean_ratio = float(r.get("mean_ratio_T_over_R", np.nan))
        ratio_of_means = float(r.get("ratio_of_means_on_mask", np.nan))
        std_ratio = float(r.get("std_ratio_T_over_R", np.nan))
        sum_ratio = float(r.get("sum_ratio_T_over_R", np.nan))

        # Membrane enrichment vs whole cell (from Dual mask intensities)
        if np.isfinite(mean_t_mem) and np.isfinite(mean_t_whole) and mean_t_whole > 1e-12:
            enrichment = mean_t_mem / mean_t_whole
        else:
            enrichment = np.nan

        # Fraction of whole-cell green signal on AND mask
        if np.isfinite(sum_t_mem) and np.isfinite(sum_t_whole) and sum_t_whole > 1e-12:
            mem_frac = sum_t_mem / sum_t_whole
        else:
            mem_frac = np.nan

        # Red coverage on membrane: fraction of radial/AND pixels with ref mask
        red_coverage = np.nan
        red_cov_area = 0
        mem_px = int(area_and) if np.isfinite(area_and) else 0
        if labels is not None and and_mask is not None and ref_mask is not None:
            cell = labels == lab
            mem_sel = and_mask & cell
            mem_px = int(np.count_nonzero(mem_sel))
            if mem_px > 0:
                red_cov_area = int(np.count_nonzero(mem_sel & ref_mask))
                # and_mask already intersects ref; coverage of radial band by ref:
                # prefer radial membrane pixels if available via labels outside and
                red_coverage = red_cov_area / mem_px if mem_px else np.nan

        row: dict[str, Any] = {
            "Image": meta.get("image_id", ""),
            "Field": meta.get("field", ""),
            "Experiment": meta.get("experiment", ""),
            "Drug": meta.get("drug", ""),
            "Group": meta.get("group", ""),
            "Condition": meta.get("condition_id", ""),
            "CellID": lab,
            "Area": int(area_cell) if np.isfinite(area_cell) else 0,
            "Perimeter": np.nan,  # Dual does not export perimeter
            "Area_um2": float(r.get("area_cell_um2", np.nan)),
            "AND_Area_px": int(area_and) if np.isfinite(area_and) else 0,
            "AND_Area_um2": float(r.get("area_and_um2", np.nan)),
            # Whole-cell (Dual)
            "WholeGreen": mean_t_whole,
            "WholeGreenIntegrated": sum_t_whole,
            "WholeRed": mean_r_whole,
            "WholeRedIntegrated": sum_r_whole,
            # Membrane / AND mask (Dual Target=EGFP, Reference=DiI)
            "MembraneGreen": mean_t_mem,
            "MembraneGreenIntegrated": sum_t_mem,
            "MembraneRed": mean_r_mem,
            "MembraneRedIntegrated": sum_r_mem,
            "MembranePixels": mem_px,
            "CytoGreen": np.nan,
            "CytoGreenIntegrated": np.nan,
            "CytoPixels": np.nan,
            # Dual primary metrics
            "Ratio_T_over_R": mean_ratio,
            "RatioOfMeans_T_R": ratio_of_means,
            "StdRatio_T_over_R": std_ratio,
            "SumRatio_T_over_R": sum_ratio,
            "Enrichment_Membrane_vs_Whole": enrichment,
            "MembraneFraction": mem_frac,
            "Std_target_on_mask": float(r.get("std_target_on_mask", np.nan)),
            "Std_reference_on_mask": float(r.get("std_reference_on_mask", np.nan)),
            "Std_target_whole": float(r.get("std_target_whole", np.nan)),
            "Std_reference_whole": float(r.get("std_reference_whole", np.nan)),
            # QC helpers
            "RedCoverage": red_coverage if np.isfinite(red_coverage) else 1.0,
            "RedCoverageArea": red_cov_area,
            "Backend": "dualcellquant",
            "QC": "pass",
            "QC_Reason": "",
        }
        rows.append(row)
    return rows
