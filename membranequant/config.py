"""Configuration loading and defaults for MembraneQuant."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, get_type_hints

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass
class Config:
    """All tunable pipeline parameters."""

    rolling_ball_radius: int = 50
    gaussian_sigma: float = 1.0
    enable_denoise: bool = True
    # Whole-cell segmentation: otsu (default) | cellpose (optional dependency)
    segmentation_method: str = "otsu"
    segmentation_channel: str = "green"  # green | green_red
    threshold: str = "otsu"
    minimum_cell_area: int = 500
    maximum_cell_area: int = 50000
    max_eccentricity: float = 0.98
    max_saturation_fraction: float = 0.05
    # Cellpose options (only used when segmentation_method == cellpose)
    cellpose_model: str = "cyto2"  # cyto | cyto2 | cyto3 | nuclei | ...
    cellpose_diameter: float = 0.0  # 0 = auto
    cellpose_gpu: bool = False
    cellpose_flow_threshold: float = 0.4
    cellpose_cellprob_threshold: float = 0.0
    # Membrane ring (geometry from cell boundary, not DiI threshold)
    ring_width: int = 3
    minimum_ring_pixels: int = 100
    minimum_red_coverage: float = 0.5
    save_overlay: bool = True
    save_mask: bool = True
    save_graphpad: bool = True
    compute_pearson: bool = False
    output_dir: str = "Results"

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
    # With from __future__ import annotations, types may be strings
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


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load config from YAML, then apply optional CLI/runtime overrides."""
    cfg = Config()
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    type_map = _resolve_field_types()

    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config file must be a mapping: {config_path}")
        for key, value in raw.items():
            if key not in type_map:
                continue
            setattr(cfg, key, _coerce_value(type_map[key], value))

    if overrides:
        for key, value in overrides.items():
            if key not in type_map or value is None:
                continue
            setattr(cfg, key, _coerce_value(type_map[key], value))

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Config) -> None:
    method = cfg.segmentation_method.strip().lower()
    if method not in {"otsu", "cellpose"}:
        raise ValueError("segmentation_method must be 'otsu' or 'cellpose'")
    cfg.segmentation_method = method
    if cfg.segmentation_channel not in {"green", "green_red"}:
        raise ValueError("segmentation_channel must be 'green' or 'green_red'")
    if cfg.ring_width < 1:
        raise ValueError("ring_width must be >= 1")
    if cfg.minimum_cell_area >= cfg.maximum_cell_area:
        raise ValueError("minimum_cell_area must be < maximum_cell_area")
    if not 0.0 <= cfg.minimum_red_coverage <= 1.0:
        raise ValueError("minimum_red_coverage must be in [0, 1]")
    if not 0.0 <= cfg.max_saturation_fraction <= 1.0:
        raise ValueError("max_saturation_fraction must be in [0, 1]")
    if cfg.cellpose_diameter < 0:
        raise ValueError("cellpose_diameter must be >= 0 (0 = auto)")
