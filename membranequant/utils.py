"""Shared helpers for MembraneQuant."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root-style logger for the pipeline run."""
    logger = logging.getLogger("membranequant")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert multi-channel fluorescence export to a single 2D plane.

    Real data is often saved as RGB even for single-channel acquisitions:
      - DiI (red)   → signal in R plane, G/B ~ 0
      - EGFP (green)→ signal in G plane, R/B ~ 0

    Taking ``[..., 0]`` would zero out green files. We therefore:
      1) For RGB/RGBA: take the channel with the highest mean intensity
         (falls back to max-projection if means are nearly equal).
      2) For other stacks (C, H, W) or (H, W, C>4): take brightest plane.
    """
    arr = np.asarray(image)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        # HWC RGB/RGBA
        if arr.shape[-1] in (3, 4):
            rgb = arr[..., :3].astype(np.float32, copy=False)
            means = rgb.reshape(-1, 3).mean(axis=0)
            # If one channel clearly dominates (typical single-color export)
            if float(means.max()) > 1e-6 and float(means.max()) >= 1.5 * float(means.mean() + 1e-12):
                return rgb[..., int(np.argmax(means))]
            # Otherwise max-projection across colors (keeps both if mixed)
            return rgb.max(axis=-1)

        # CHW-like (few planes first)
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            planes = arr.astype(np.float32, copy=False)
            means = planes.reshape(planes.shape[0], -1).mean(axis=1)
            return planes[int(np.argmax(means))]

        # HWC with many channels, or HW C
        if arr.shape[-1] <= 8:
            planes = np.moveaxis(arr, -1, 0).astype(np.float32, copy=False)
            means = planes.reshape(planes.shape[0], -1).mean(axis=1)
            return planes[int(np.argmax(means))]

        return arr[..., 0]

    # 4D+ : collapse extra dims by max
    while arr.ndim > 2:
        arr = arr.max(axis=0)
    return arr


def normalize_to_unit(image: np.ndarray) -> np.ndarray:
    """Normalize image intensities to float32 range [0, 1] (2D grayscale)."""
    arr = to_grayscale(image)
    arr = np.asarray(arr, dtype=np.float32)
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    if amax <= amin:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)


def saturation_fraction(image: np.ndarray, high: float = 0.99) -> float:
    """Fraction of pixels near the top of the intensity range (normalized image)."""
    if image.size == 0:
        return 0.0
    return float(np.mean(image >= high))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
