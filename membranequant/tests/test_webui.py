"""Lightweight tests for Web UI helpers (no browser / no server)."""

from pathlib import Path

import numpy as np
import tifffile

from membranequant.webui import _preview_pairs, run_from_ui


def test_preview_pairs_empty():
    msg = _preview_pairs("")
    assert "无效" in msg or "路径" in msg


def test_preview_and_run(tmp_path: Path):
    exp = tmp_path / "Experiment"
    group = exp / "GroupA"
    group.mkdir(parents=True)

    size = 96
    yy, xx = np.mgrid[0:size, 0:size]
    green = ((yy - 48) ** 2 + (xx - 48) ** 2 < 22**2).astype(np.float32)
    red = np.zeros_like(green)
    ring = ((yy - 48) ** 2 + (xx - 48) ** 2 >= 18**2) & ((yy - 48) ** 2 + (xx - 48) ** 2 < 22**2)
    red[ring] = 1.0
    green = green * 0.75
    green[ring] = 0.95

    tifffile.imwrite(group / "C2_100d0-1-1.tif", (red * 65535).astype(np.uint16))
    tifffile.imwrite(group / "C2_100d0-1-2.tif", (green * 65535).astype(np.uint16))

    preview = _preview_pairs(str(exp))
    assert "1" in preview
    assert "配对" in preview or "视野" in preview

    out = tmp_path / "Results"
    report, results_df, summary_df, overlays = run_from_ui(
        input_dir=str(exp),
        output_dir=str(out),
        seg_method="otsu",
        ring_width=3,
        min_area=80,
        max_area=20000,
        min_red_cov=0.2,
        rolling_ball=15,
        gaussian_sigma=0.5,
        enable_denoise=True,
        seg_channel="green",
        cellpose_model="cyto2",
        cellpose_diameter=0.0,
        cellpose_gpu=False,
        save_overlay=True,
        save_mask=True,
        save_graphpad=True,
        compute_pearson=False,
        config_path="",
        progress=None,
    )
    assert "完成" in report or "分析" in report
    assert results_df is not None
    assert (out / "csv" / "results.csv").is_file()


def test_run_cellpose_without_install_errors_gracefully(tmp_path: Path):
    exp = tmp_path / "Experiment"
    exp.mkdir()
    report, *_ = run_from_ui(
        input_dir=str(exp),
        output_dir=str(tmp_path / "out"),
        seg_method="cellpose",
        ring_width=3,
        min_area=100,
        max_area=50000,
        min_red_cov=0.5,
        rolling_ball=50,
        gaussian_sigma=1.0,
        enable_denoise=True,
        seg_channel="green",
        cellpose_model="cyto2",
        cellpose_diameter=0.0,
        cellpose_gpu=False,
        save_overlay=True,
        save_mask=True,
        save_graphpad=True,
        compute_pearson=False,
        config_path="",
    )
    # 无图对 / 未装 Cellpose / 分析完成 都不应崩溃
    assert "❌" in report or "⚠️" in report or "完成" in report or report.startswith("#")
