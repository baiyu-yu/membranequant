"""Integration-style tests with synthetic cells."""

from pathlib import Path

import numpy as np
import tifffile

from membranequant.config import Config
from membranequant.export import rows_to_dataframe, write_summary_csv
from membranequant.measurements import measure_cells
from membranequant.preprocess import preprocess_pair
from membranequant.qc import apply_qc
from membranequant.segmentation import build_cell_masks, membrane_ring


def _make_synthetic_pair(size: int = 128, n_cells: int = 2):
    """Create synthetic EGFP whole-cell blobs + DiI rings."""
    yy, xx = np.mgrid[0:size, 0:size]
    green = np.zeros((size, size), dtype=np.float32)
    red = np.zeros((size, size), dtype=np.float32)

    centers = [(40, 40), (90, 85)][:n_cells]
    for cy, cx in centers:
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        cell = dist < 18
        green[cell] = 0.6 + 0.2 * (1 - dist[cell] / 18)
        # membrane brighter in green slightly
        ring = (dist >= 14) & (dist < 18)
        green[ring] = 0.9
        red[ring] = 0.85
        red[cell & ~ring] = 0.05

    green += np.random.RandomState(0).normal(0, 0.01, green.shape).astype(np.float32)
    red += np.random.RandomState(1).normal(0, 0.01, red.shape).astype(np.float32)
    green = np.clip(green, 0, 1)
    red = np.clip(red, 0, 1)
    return red, green


def test_membrane_ring_width():
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[10:40, 10:40] = 1
    mem = membrane_ring(labels, ring_width=3)
    assert mem.max() == 1
    # ring should be thinner than full cell
    assert np.count_nonzero(mem == 1) < np.count_nonzero(labels == 1)
    # cytoplasm residual
    cyto_count = np.count_nonzero((labels == 1) & (mem != 1))
    assert cyto_count > 0


def test_end_to_end_synthetic():
    red, green = _make_synthetic_pair()
    cfg = Config(
        rolling_ball_radius=20,
        gaussian_sigma=0.5,
        minimum_cell_area=100,
        maximum_cell_area=20000,
        ring_width=3,
        minimum_ring_pixels=20,
        minimum_red_coverage=0.3,
        max_eccentricity=0.99,
    )
    red_p, green_p = preprocess_pair(red, green, cfg)
    masks = build_cell_masks(green_p, red_p, cfg)
    assert len(masks.kept_ids) >= 1

    rows = measure_cells(
        green_p,
        red_p,
        masks,
        meta={"image_id": "G/test", "field": 1, "group": "Control", "experiment": "1", "drug": "d0"},
        cfg=cfg,
    )
    rows = apply_qc(rows, cfg)
    assert len(rows) >= 1
    assert "M/C" in rows[0]
    assert rows[0]["MembranePixels"] > 0
    # New colocalization / enrichment metrics
    assert "M/C_DiI" in rows[0]
    assert "MEI" in rows[0]
    assert "Manders_M1" in rows[0]
    assert "PearsonWhole" in rows[0]


def test_pipeline_on_disk(tmp_path: Path):
    from membranequant.main import run_pipeline

    exp = tmp_path / "Experiment"
    gA = exp / "Control"
    gB = exp / "Treatment"
    gA.mkdir(parents=True)
    gB.mkdir(parents=True)

    red, green = _make_synthetic_pair()
    for group_dir, stem in [(gA, "C2_100d0-1"), (gB, "C2_100d1-1")]:
        tifffile.imwrite(group_dir / f"{stem}-1.tif", (red * 65535).astype(np.uint16))
        tifffile.imwrite(group_dir / f"{stem}-2.tif", (green * 65535).astype(np.uint16))

    out = tmp_path / "Results"
    cfg = Config(
        rolling_ball_radius=15,
        gaussian_sigma=0.5,
        minimum_cell_area=80,
        ring_width=3,
        minimum_ring_pixels=15,
        minimum_red_coverage=0.25,
        save_overlay=True,
        save_mask=True,
        save_graphpad=True,
    )
    results = run_pipeline(exp, out, cfg)
    assert results.is_file()
    assert (out / "csv" / "summary.csv").is_file()
    assert (out / "csv" / "graphpad_MC.csv").is_file()
    df = rows_to_dataframe([])  # just ensure import path
    assert (out / "masks").is_dir()
