"""Configuration loading and defaults for MembraneQuant + DualCellQuant."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, get_type_hints

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass
class Config:
    """All tunable pipeline parameters.

    Image analysis is performed by DualCellQuant. MembraneQuant only handles
    experiment I/O and post-analysis (QC / CSV / plots).
    """

    # ----- DualCellQuant: preprocess -----
    dual_bg_enable: bool = True
    dual_bg_radius: int = 50
    dual_bg_mode: str = "rolling"  # rolling | dark_subtract | manual
    dual_bg_dark_pct: float = 5.0
    dual_norm_enable: bool = False
    dual_norm_method: str = "z-score"  # z-score | robust z-score | min-max | percentile

    # ----- DualCellQuant: segmentation (Cellpose) -----
    dual_seg_source: str = "target"  # target=EGFP | reference=DiI
    dual_seg_channel: int = 0  # 0=gray, 1=R, 2=G, 3=B
    dual_measure_channel: str = "gray"  # gray|r|g|b for single-channel TIFs use gray
    dual_diameter: float = 0.0  # 0 = auto
    dual_flow_threshold: float = 0.4
    dual_cellprob_threshold: float = 0.0
    dual_use_gpu: bool = True  # use CUDA when available; set false to force CPU
    dual_auto_gpu: bool = True  # if True and CUDA is free, force dual_use_gpu on unless overridden
    dual_drop_edge_cells: bool = True
    dual_inside_fraction_min: float = 0.9
    dual_edge_margin_pct: float = 0.0

    # ----- DualCellQuant: EDT radial membrane -----
    dual_radial_inner_pct: float = 85.0  # 0=center, 100=boundary
    dual_radial_outer_pct: float = 100.0
    dual_radial_min_obj: int = 0

    # ----- DualCellQuant: masks -----
    dual_target_mask_mode: str = "none"  # none|global_otsu|global_percentile|per_cell_otsu|per_cell_percentile
    dual_target_mask_percentile: float = 90.0
    dual_ref_mask_mode: str = "per_cell_otsu"
    dual_ref_mask_percentile: float = 50.0
    dual_sat_limit: float = 255.0
    dual_min_obj_size: int = 10
    dual_use_radial_for_target: bool = True
    dual_use_radial_for_ref: bool = True
    dual_and_with_radial: bool = True
    dual_ratio_ref_epsilon: float = 0.0
    dual_pixel_width_um: float = 1.0
    dual_pixel_height_um: float = 1.0

    # ----- QC (on Dual outputs) -----
    minimum_cell_area: int = 500
    maximum_cell_area: int = 50000
    minimum_and_pixels: int = 50
    minimum_red_coverage: float = 0.0  # 0 = disabled when Dual AND already requires ref

    # ----- Export -----
    save_overlay: bool = True
    save_mask: bool = True
    save_graphpad: bool = True
    output_dir: str = "Results"

    # ----- Legacy aliases (ignored by Dual path; kept so old YAML still loads) -----
    rolling_ball_radius: int = 50
    gaussian_sigma: float = 1.0
    enable_denoise: bool = True
    segmentation_method: str = "dualcellquant"
    segmentation_channel: str = "green"
    threshold: str = "otsu"
    max_eccentricity: float = 0.98
    max_saturation_fraction: float = 0.05
    cellpose_model: str = "cyto2"
    cellpose_diameter: float = 0.0
    cellpose_gpu: bool = True  # legacy alias; synced into dual_use_gpu
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0
    ring_width: int = 3
    minimum_ring_pixels: int = 100
    compute_pearson: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_field_types() -> dict[str, Any]:
    try:
        return get_type_hints(Config)
    except Exception:
        return {f.name: f.type for f in fields(Config)}


def _coerce_value(field_type: Any, value: Any) -> Any:
    if value is None:
        return value
    if isinstance(field_type, str):
        type_map = {"int": int, "float": float, "bool": bool, "str": str}
        field_type = type_map.get(field_type, field_type)

    origin = getattr(field_type, "__origin__", None)
    if origin is not None:
        return value
    if field_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if field_type is int:
        return int(value)
    if field_type is float:
        return float(value)
    if field_type is str:
        return str(value)
    return value


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load config from YAML, then apply optional CLI/runtime overrides.

    GPU policy
    ----------
    - Default / auto: if CUDA is available and ``dual_auto_gpu`` is True, GPU is ON.
    - Explicit override wins: ``dual_use_gpu`` or legacy ``cellpose_gpu`` in overrides.
    - Set ``dual_use_gpu: false`` (and optionally ``dual_auto_gpu: false``) to force CPU.
    """
    cfg = Config()
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    type_map = _resolve_field_types()
    yaml_keys: set[str] = set()
    override_keys: set[str] = set(overrides.keys()) if overrides else set()

    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config file must be a mapping: {config_path}")
        for key, value in raw.items():
            if key not in type_map:
                continue
            yaml_keys.add(key)
            setattr(cfg, key, _coerce_value(type_map[key], value))

    if overrides:
        for key, value in overrides.items():
            if key not in type_map or value is None:
                continue
            setattr(cfg, key, _coerce_value(type_map[key], value))

    # Sync legacy knobs into Dual when user only set old names
    if overrides:
        if "rolling_ball_radius" in overrides and "dual_bg_radius" not in overrides:
            cfg.dual_bg_radius = int(cfg.rolling_ball_radius)
        if "cellpose_gpu" in overrides and "dual_use_gpu" not in overrides:
            cfg.dual_use_gpu = bool(cfg.cellpose_gpu)
        if "cellpose_diameter" in overrides and "dual_diameter" not in overrides:
            cfg.dual_diameter = float(cfg.cellpose_diameter)
        if "cellpose_flow_threshold" in overrides and "dual_flow_threshold" not in overrides:
            cfg.dual_flow_threshold = float(cfg.cellpose_flow_threshold)
        if "cellpose_cellprob_threshold" in overrides and "dual_cellprob_threshold" not in overrides:
            cfg.dual_cellprob_threshold = float(cfg.cellpose_cellprob_threshold)
        if "minimum_ring_pixels" in overrides and "minimum_and_pixels" not in overrides:
            cfg.minimum_and_pixels = int(cfg.minimum_ring_pixels)

    # Auto-enable GPU when CUDA is free and user did not explicitly force CPU.
    explicit_gpu = ("dual_use_gpu" in override_keys) or ("cellpose_gpu" in override_keys)
    if cfg.dual_auto_gpu and not explicit_gpu and _cuda_available():
        # YAML may still say false from older configs; auto-on is intentional.
        if not cfg.dual_use_gpu:
            cfg.dual_use_gpu = True
        cfg.cellpose_gpu = True
    else:
        # Keep legacy alias in sync for UI reporting
        if "dual_use_gpu" in override_keys or "dual_use_gpu" in yaml_keys:
            cfg.cellpose_gpu = bool(cfg.dual_use_gpu)
        elif "cellpose_gpu" in override_keys:
            cfg.dual_use_gpu = bool(cfg.cellpose_gpu)

    _validate_config(cfg)
    return cfg


_MASK_MODES = {
    "none",
    "global_otsu",
    "global_percentile",
    "per_cell_otsu",
    "per_cell_percentile",
}


def _validate_config(cfg: Config) -> None:
    cfg.dual_seg_source = cfg.dual_seg_source.strip().lower()
    if cfg.dual_seg_source not in {"target", "reference"}:
        raise ValueError("dual_seg_source must be 'target' (EGFP) or 'reference' (DiI)")
    cfg.dual_bg_mode = cfg.dual_bg_mode.strip().lower()
    if cfg.dual_bg_mode not in {"rolling", "dark_subtract", "manual"}:
        raise ValueError("dual_bg_mode must be rolling|dark_subtract|manual")
    cfg.dual_target_mask_mode = cfg.dual_target_mask_mode.strip().lower()
    cfg.dual_ref_mask_mode = cfg.dual_ref_mask_mode.strip().lower()
    if cfg.dual_target_mask_mode not in _MASK_MODES:
        raise ValueError(f"dual_target_mask_mode must be one of {sorted(_MASK_MODES)}")
    if cfg.dual_ref_mask_mode not in _MASK_MODES:
        raise ValueError(f"dual_ref_mask_mode must be one of {sorted(_MASK_MODES)}")
    if cfg.dual_radial_inner_pct < 0:
        raise ValueError("dual_radial_inner_pct must be >= 0")
    if cfg.dual_radial_outer_pct < cfg.dual_radial_inner_pct:
        raise ValueError("dual_radial_outer_pct must be >= dual_radial_inner_pct")
    if cfg.minimum_cell_area >= cfg.maximum_cell_area:
        raise ValueError("minimum_cell_area must be < maximum_cell_area")
    if not 0.0 <= cfg.minimum_red_coverage <= 1.0:
        raise ValueError("minimum_red_coverage must be in [0, 1]")
    if cfg.dual_diameter < 0:
        raise ValueError("dual_diameter must be >= 0 (0 = auto)")
    valid_methods = {
        "dualcellquant",
        "otsu",
        "cellpose",
        "watershed_distance",
        "watershed_gradient",
        "hminima_watershed",
        "morphological_opening",
        "combined_markers",
    }
    if cfg.segmentation_method and cfg.segmentation_method.strip().lower() not in valid_methods:
        raise ValueError(f"Unknown segmentation_method: {cfg.segmentation_method}")
    if not cfg.segmentation_method:
        cfg.segmentation_method = "dualcellquant"
