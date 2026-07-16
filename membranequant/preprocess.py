"""Background correction and denoising."""

from __future__ import annotations

import numpy as np
from skimage.filters import gaussian
from skimage.restoration import rolling_ball

from .config import Config


def correct_background(image: np.ndarray, radius: int = 50) -> np.ndarray:
    """Subtract rolling-ball background (ImageJ-compatible approach)."""
    img = np.asarray(image, dtype=np.float32)
    # rolling_ball expects non-negative float or integer-like range
    bg = rolling_ball(img, radius=radius)
    corrected = img - bg
    corrected = np.clip(corrected, 0.0, None)
    # Re-normalize to [0, 1] after subtraction for stable downstream thresholds
    maxv = float(corrected.max())
    if maxv > 0:
        corrected = corrected / maxv
    return corrected.astype(np.float32, copy=False)


def denoise(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian blur for mild denoising."""
    if sigma <= 0:
        return np.asarray(image, dtype=np.float32)
    out = gaussian(np.asarray(image, dtype=np.float32), sigma=sigma, preserve_range=True)
    return out.astype(np.float32, copy=False)


def preprocess_pair(
    red: np.ndarray,
    green: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply background correction (+ optional denoise) to both channels."""
    red_p = correct_background(red, radius=cfg.rolling_ball_radius)
    green_p = correct_background(green, radius=cfg.rolling_ball_radius)
    if cfg.enable_denoise:
        red_p = denoise(red_p, sigma=cfg.gaussian_sigma)
        green_p = denoise(green_p, sigma=cfg.gaussian_sigma)
    return red_p, green_p
