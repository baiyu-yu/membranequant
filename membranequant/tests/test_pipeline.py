"""Integration-style tests with synthetic cells."""

from pathlib import Path

import numpy as np
import pytest
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
    from membranequant.dual_backend import dualcellquant_status
    from membranequant.main import run_pipeline

    st = dualcellquant_status()
    if not st.get("available") or not st.get("cellpose_available"):
        pytest.skip("DualCellQuant/Cellpose not installed — pipeline hard-requires Dual")

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
        dual_bg_radius=15,
        dual_bg_mode="dark_subtract",
        dual_use_gpu=False,
        dual_auto_gpu=False,
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
    assert (out / "masks").is_dir()


def test_pipeline_hard_fails_without_dual(tmp_path: Path, monkeypatch):
    from membranequant import dual_backend, main

    monkeypatch.setattr(
        dual_backend,
        "dualcellquant_status",
        lambda: {
            "available": False,
            "version": None,
            "cellpose_available": False,
            "cuda_available": False,
            "message": "not installed (test)",
        },
    )
    cfg = Config(dual_use_gpu=False, dual_auto_gpu=False, save_overlay=False, save_mask=False)
    with pytest.raises(ImportError):
        main.run_pipeline(tmp_path, tmp_path / "out", cfg)


def test_generate_all_plots_and_outlier_filter(tmp_path: Path):
    import pandas as pd
    from membranequant.plots import generate_all_plots, build_summary_from_results

    # Generate synthetic results dataframe with extreme ratio outliers
    rows = []
    conditions = ["104d1", "104d2", "104e1", "wd1", "we1"]
    cell_id = 1
    for cond in conditions:
        exp = cond[:3] if cond.startswith("104") else cond[0]
        drug = cond[3] if cond.startswith("104") else cond[1]
        grp = cond[4] if cond.startswith("104") else cond[2]
        for i in range(20):
            val = 1.5 + np.random.randn() * 0.2
            # Insert extreme division-by-zero ratio outliers in wd1
            if cond == "wd1" and i == 0:
                val = 3.5e13
            rows.append({
                "Image": f"{cond}_img1.tif",
                "Experiment": exp,
                "Drug": drug,
                "Group": grp,
                "Condition": cond,
                "CellID": cell_id,
                "Ratio_T_over_R": val,
                "RatioOfMeans_T_R": val,
                "Enrichment_Membrane_vs_Whole": val if val < 100 else 1000,
                "MembraneGreen": 100.0,
                "MembraneRed": 50.0,
                "CytoGreen_DiI": 60.0,
                "Area": 500,
                "QC": "pass",
                "EdgeCenterRatio": 1.5,
            })
            cell_id += 1

    df = pd.DataFrame(rows)
    summary_df = build_summary_from_results(df, metric="RatioOfMeans_T_R")
    
    # Check that wd1 mean is not 3.5e13 / 20 = 1.75e12
    wd1_summary = summary_df[summary_df["Condition"] == "wd1"]
    assert not wd1_summary.empty
    assert wd1_summary["Mean"].iloc[0] < 10.0

    # Generate all plots and verify 08_coloc_dashboard and all other 11 plots are created
    out_dir = tmp_path / "plot_test"
    saved = generate_all_plots(df, summary_df, out_dir, metric="RatioOfMeans_T_R")
    assert len(saved) == 11
    coloc_file = out_dir / "plots" / "08_coloc_dashboard.png"
    assert coloc_file.is_file()

