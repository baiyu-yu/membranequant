"""Overlay and debug visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import ensure_dir


def _to_rgb_base(green: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Build a simple RGB composite: R=DiI, G=EGFP, B=0."""
    g = np.clip(green, 0, 1)
    r = np.clip(red, 0, 1)
    rgb = np.stack([r, g, np.zeros_like(g)], axis=-1)
    return rgb.astype(np.float32)


def draw_overlay(
    green: np.ndarray,
    red: np.ndarray,
    labels: np.ndarray,
    membrane: np.ndarray,
    cytoplasm: np.ndarray,
    path: Path,
    title: str = "",
) -> None:
    """Save overlay: original composite + ROI colors + cell IDs.

    Colors (design):
      Whole cell boundary — green
      Membrane ring       — red
      Cytoplasm           — blue
    """
    ensure_dir(path.parent)
    base = _to_rgb_base(green, red)
    # Dim base so overlays stand out
    vis = base * 0.55

    # Cytoplasm fill (blue, semi-transparent)
    cyto_mask = cytoplasm > 0
    vis[cyto_mask, 2] = np.clip(vis[cyto_mask, 2] + 0.45, 0, 1)

    # Membrane fill (red)
    mem_mask = membrane > 0
    vis[mem_mask, 0] = np.clip(vis[mem_mask, 0] + 0.55, 0, 1)
    vis[mem_mask, 1] = vis[mem_mask, 1] * 0.5

    # Whole-cell boundary (green)
    bounds = find_boundaries(labels, mode="outer")
    vis[bounds, 1] = 1.0
    vis[bounds, 0] = np.minimum(vis[bounds, 0], 0.3)
    vis[bounds, 2] = np.minimum(vis[bounds, 2], 0.3)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(np.clip(vis, 0, 1))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)

    for prop in regionprops(labels):
        y, x = prop.centroid
        ax.text(
            x,
            y,
            f"Cell {prop.label}",
            color="white",
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, ec="none"),
        )

    fig.tight_layout(pad=0.1)
    # 300 dpi for direct PPT insertion
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)


def save_debug_panel(
    green: np.ndarray,
    red: np.ndarray,
    labels: np.ndarray,
    path: Path,
) -> None:
    """Optional multi-panel debug figure."""
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    axes[0].imshow(red, cmap="magma")
    axes[0].set_title("DiI (Red)")
    axes[1].imshow(green, cmap="gray")
    axes[1].set_title("EGFP (Green)")
    axes[2].imshow(label2rgb(labels, image=green, bg_label=0))
    axes[2].set_title("Labels")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_coloc_scatter(
    green: np.ndarray,
    red: np.ndarray,
    labels: np.ndarray,
    path: Path,
    title: str = "",
    max_points: int = 8000,
) -> None:
    """Per-field green vs red scatter (Coloc2-style fluorogram) for PPT QC."""
    ensure_dir(path.parent)
    mask = labels > 0
    if not np.any(mask):
        return
    g = green[mask].ravel()
    r = red[mask].ravel()
    if g.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(g.size, size=max_points, replace=False)
        g, r = g[idx], r[idx]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)
    ax.scatter(r, g, s=2, alpha=0.25, c="0.2", edgecolors="none")
    ax.set_xlabel("DiI (Red)", fontweight="bold")
    ax.set_ylabel("EGFP (Green)", fontweight="bold")
    ax.set_title(title or "Colocalization scatter", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
