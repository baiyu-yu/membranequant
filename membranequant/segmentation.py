"""Whole-cell, membrane ring, and cytoplasm segmentation.

Design principle:
  - Cell boundaries come from EGFP (or EGFP+DiI), NEVER from DiI alone.
  - Membrane ROI is a fixed-width geometric ring from the cell boundary.
  - DiI is used only later for QC (red coverage), not for defining ROIs.

Segmentation backends:
  - otsu     : classical threshold pipeline (default, no extra deps)
  - cellpose : optional deep-learning path (pip install cellpose)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, erosion, remove_small_objects
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

from .config import Config
from .utils import saturation_fraction


@dataclass
class CellMasks:
    """Per-image labeled masks and derived ROIs."""

    labels: np.ndarray  # int32 label image (whole cell)
    membrane: np.ndarray  # int32 labels on membrane ring only
    cytoplasm: np.ndarray  # int32 labels on cytoplasm only
    kept_ids: list[int]
    rejected: list[dict]  # {cell_id, reason}
    method: str = "otsu"


def cellpose_available() -> bool:
    """Return True if the cellpose package can be imported."""
    try:
        import cellpose  # noqa: F401

        return True
    except ImportError:
        return False


def cellpose_status() -> dict[str, Any]:
    """Human-readable Cellpose availability for UI / CLI."""
    if not cellpose_available():
        return {
            "available": False,
            "version": None,
            "message": "Cellpose not installed. Run: pip install cellpose",
        }
    try:
        import cellpose

        ver = getattr(cellpose, "__version__", "unknown")
    except Exception:
        ver = "unknown"
    return {
        "available": True,
        "version": ver,
        "message": f"Cellpose {ver} available",
    }


def _segmentation_image(green: np.ndarray, red: np.ndarray, mode: str) -> np.ndarray:
    if mode == "green_red":
        return np.maximum(green, red)
    return green


def _filter_labeled_cells(
    labels: np.ndarray,
    green: np.ndarray,
    cfg: Config,
    *,
    clear_border_cells: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """Apply area / eccentricity / saturation / border filters; renumber labels."""
    rejected: list[dict] = []
    work = labels.astype(np.int32, copy=True)

    if clear_border_cells:
        labels_raw = work.copy()
        cleared = clear_border(labels_raw)
        kept_raw = set(int(x) for x in np.unique(cleared) if x != 0)
        for rid in np.unique(labels_raw):
            rid = int(rid)
            if rid == 0:
                continue
            if rid not in kept_raw:
                rejected.append({"cell_id": rid, "reason": "touch_border"})
        work = cleared.astype(np.int32, copy=False)

    props = regionprops(work, intensity_image=green)
    remap: dict[int, int] = {}
    next_id = 1
    filtered_rejected: list[dict] = list(rejected)

    for prop in props:
        cid = int(prop.label)
        area = int(prop.area)
        if area < cfg.minimum_cell_area:
            filtered_rejected.append({"cell_id": cid, "reason": "area_too_small"})
            continue
        if area > cfg.maximum_cell_area:
            filtered_rejected.append({"cell_id": cid, "reason": "area_too_large"})
            continue
        ecc = float(prop.eccentricity) if prop.eccentricity is not None else 0.0
        if ecc > cfg.max_eccentricity:
            filtered_rejected.append({"cell_id": cid, "reason": "eccentricity"})
            continue

        mask = work == cid
        sat = saturation_fraction(green[mask])
        if sat > cfg.max_saturation_fraction:
            filtered_rejected.append({"cell_id": cid, "reason": "green_saturation"})
            continue

        remap[cid] = next_id
        next_id += 1

    out = np.zeros_like(work, dtype=np.int32)
    for old, new in remap.items():
        out[work == old] = new

    return out, filtered_rejected


def segment_whole_cells_otsu(
    green: np.ndarray,
    red: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict]]:
    """Otsu-based whole-cell segmentation with morphological cleanup and filters."""
    seg_img = _segmentation_image(green, red, cfg.segmentation_channel)

    if cfg.threshold == "otsu":
        thr = threshold_otsu(seg_img)
    else:
        thr = float(cfg.threshold)

    binary = seg_img > thr
    binary = closing(binary, disk(3))
    binary = remove_small_objects(binary, max_size=max(1, cfg.minimum_cell_area // 4))
    binary = ndi.binary_fill_holes(binary)

    labels_raw = label(binary)
    return _filter_labeled_cells(labels_raw, green, cfg, clear_border_cells=True)


def _to_cellpose_input(image: np.ndarray) -> np.ndarray:
    """Convert float [0,1] image to uint16 range Cellpose handles well."""
    img = np.asarray(image, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)
    return (img * 65535.0).astype(np.uint16)


def segment_whole_cells_cellpose(
    green: np.ndarray,
    red: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict]]:
    """Cellpose whole-cell segmentation on EGFP (or EGFP+DiI max).

    Requires optional dependency: pip install cellpose
    """
    if not cellpose_available():
        raise ImportError(
            "segmentation_method='cellpose' requires cellpose. "
            "Install with: pip install cellpose"
        )

    from cellpose import models

    seg_img = _segmentation_image(green, red, cfg.segmentation_channel)
    cp_img = _to_cellpose_input(seg_img)

    diameter = None if cfg.cellpose_diameter <= 0 else float(cfg.cellpose_diameter)
    model_type = (cfg.cellpose_model or "cyto2").strip()

    # Cellpose 2.x / 3.x compatibility
    masks: np.ndarray
    try:
        model = models.CellposeModel(gpu=bool(cfg.cellpose_gpu), model_type=model_type)
    except TypeError:
        # Newer cellpose may use pretrained_model=
        model = models.CellposeModel(gpu=bool(cfg.cellpose_gpu), pretrained_model=model_type)

    eval_kwargs: dict[str, Any] = {
        "diameter": diameter,
        "flow_threshold": float(cfg.cellpose_flow_threshold),
        "cellprob_threshold": float(cfg.cellpose_cellprob_threshold),
    }

    try:
        # channels=[0,0] => grayscale / single channel
        result = model.eval(cp_img, channels=[0, 0], **eval_kwargs)
    except TypeError:
        # Some versions dropped channels= in favor of channel_axis=None
        result = model.eval(cp_img, channel_axis=None, **eval_kwargs)

    if isinstance(result, (tuple, list)):
        masks = np.asarray(result[0])
    else:
        masks = np.asarray(result)

    masks = masks.astype(np.int32, copy=False)
    return _filter_labeled_cells(masks, green, cfg, clear_border_cells=True)


def segment_whole_cells(
    green: np.ndarray,
    red: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict]]:
    """Dispatch to otsu or cellpose backend."""
    method = (cfg.segmentation_method or "otsu").strip().lower()
    if method == "cellpose":
        return segment_whole_cells_cellpose(green, red, cfg)
    if method == "otsu":
        return segment_whole_cells_otsu(green, red, cfg)
    raise ValueError(f"Unknown segmentation_method: {method}")


# Back-compat alias used by older tests / imports
def segment_whole_cells_legacy(
    green: np.ndarray,
    red: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict]]:
    return segment_whole_cells_otsu(green, red, cfg)


def membrane_ring(labels: np.ndarray, ring_width: int = 3) -> np.ndarray:
    """Build fixed-width membrane ring: whole_cell - eroded_cell."""
    if ring_width < 1:
        raise ValueError("ring_width must be >= 1")

    selem = disk(ring_width)
    membrane = np.zeros_like(labels, dtype=np.int32)
    for cid in np.unique(labels):
        if cid == 0:
            continue
        cell = labels == cid
        eroded = erosion(cell, selem)
        ring = cell & ~eroded
        if not np.any(ring):
            ring = cell
        membrane[ring] = cid
    return membrane


def cytoplasm_from_ring(labels: np.ndarray, membrane: np.ndarray) -> np.ndarray:
    """Cytoplasm = whole cell minus membrane ring."""
    cyto = np.zeros_like(labels, dtype=np.int32)
    for cid in np.unique(labels):
        if cid == 0:
            continue
        mask = (labels == cid) & (membrane != cid)
        cyto[mask] = cid
    return cyto


def build_cell_masks(
    green: np.ndarray,
    red: np.ndarray,
    cfg: Config,
) -> CellMasks:
    """Run whole-cell segmentation and derive membrane / cytoplasm ROIs."""
    method = (cfg.segmentation_method or "otsu").strip().lower()
    labels, rejected = segment_whole_cells(green, red, cfg)
    membrane = membrane_ring(labels, ring_width=cfg.ring_width)
    cytoplasm = cytoplasm_from_ring(labels, membrane)
    kept_ids = [int(i) for i in np.unique(labels) if i != 0]
    return CellMasks(
        labels=labels,
        membrane=membrane,
        cytoplasm=cytoplasm,
        kept_ids=kept_ids,
        rejected=rejected,
        method=method,
    )
