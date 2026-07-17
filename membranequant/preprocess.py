"""Background correction and denoising."""

from __future__ import annotations

import numpy as np
from skimage.filters import gaussian
from skimage.restoration import rolling_ball

from .config import Config


def correct_background(image: np.ndarray, radius: int = 50) -> np.ndarray:
    """Subtract rolling-ball background (ImageJ-compatible approach)."""
    import cv2
    img = np.asarray(image, dtype=np.float32)
    
    # ImageJ shrinkFactor logic based on radius
    shrink_factor = 1
    if radius > 100:
        shrink_factor = 8
    elif radius > 30:
        shrink_factor = 4
    elif radius > 10:
        shrink_factor = 2

    if shrink_factor > 1:
        # Downsample using local block minimum (mimics ImageJ shrinkImage)
        h, w = img.shape
        pad_h = (shrink_factor - h % shrink_factor) % shrink_factor
        pad_w = (shrink_factor - w % shrink_factor) % shrink_factor
        if pad_h > 0 or pad_w > 0:
            img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
        else:
            img_padded = img
            
        h_p, w_p = img_padded.shape
        img_small = img_padded.reshape(
            h_p // shrink_factor, shrink_factor,
            w_p // shrink_factor, shrink_factor
        ).min(axis=(1, 3))
        
        radius_small = max(1, radius // shrink_factor)
        bg_small = rolling_ball(img_small, radius=radius_small)
        
        # Upscale the background back to original image dimensions using bilinear interpolation
        bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
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
