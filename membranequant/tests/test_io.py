"""Tests for filename parsing and pair discovery."""

from pathlib import Path

import numpy as np
import tifffile

from membranequant.io import parse_filename, scan_pairs, scan_with_report


def test_parse_104d1_green():
    """C2_104d1-3-2 → 实验104, 药物d, 组别1, 视野3, 绿."""
    p = Path("C2_104d1-3-2.tif")
    parsed = parse_filename(p)
    assert parsed is not None
    assert parsed.experiment == "104"
    assert parsed.drug == "d"
    assert parsed.group == "1"
    assert parsed.field == 3
    assert parsed.channel == "green"
    assert parsed.base_key == "104d1-3"
    assert parsed.condition_id == "104d1"


def test_parse_104d1_red():
    p = Path("C2_104d1-1-1.tif")
    parsed = parse_filename(p)
    assert parsed is not None
    assert parsed.experiment == "104"
    assert parsed.drug == "d"
    assert parsed.group == "1"
    assert parsed.field == 1
    assert parsed.channel == "red"


def test_parse_wd1_user_example():
    """C2_wd1-3-1 → 实验w, 药物d, 组别1, 视野3, 红."""
    p = Path("C2_wd1-3-1.tif")
    parsed = parse_filename(p)
    assert parsed is not None
    assert parsed.experiment == "w"
    assert parsed.drug == "d"
    assert parsed.group == "1"
    assert parsed.field == 3
    assert parsed.channel == "red"
    assert "实验 w" in parsed.describe_cn()
    assert "药物 d" in parsed.describe_cn()
    assert "组别 1" in parsed.describe_cn()
    assert "视野 3" in parsed.describe_cn()


def test_parse_merge():
    p = Path("C2_104d1-3.tif")
    parsed = parse_filename(p)
    assert parsed is not None
    assert parsed.channel == "merge"
    assert parsed.field == 3
    assert parsed.drug == "d"
    assert parsed.group == "1"


def test_parse_e2_group():
    p = Path("C2_104e2-4-2.tif")
    parsed = parse_filename(p)
    assert parsed is not None
    assert parsed.experiment == "104"
    assert parsed.drug == "e"
    assert parsed.group == "2"
    assert parsed.field == 4
    assert parsed.channel == "green"


def test_scan_pairs_flat(tmp_path: Path):
    group = tmp_path / "任意文件夹"
    group.mkdir()
    img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    tifffile.imwrite(group / "C2_104d1-1-1.tif", img)
    tifffile.imwrite(group / "C2_104d1-1-2.tif", img)
    tifffile.imwrite(group / "C2_104d1-1.tif", img)

    pairs = scan_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0].experiment == "104"
    assert pairs[0].drug == "d"
    assert pairs[0].group == "1"
    assert pairs[0].field == 1
    assert pairs[0].merge_path is not None
    assert pairs[0].image_id == "104d1-1"


def test_rgb_green_channel_not_zero():
    """EGFP often saved as RGB with signal only in G plane."""
    from membranequant.utils import normalize_to_unit, to_grayscale

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[..., 1] = 120  # green plane only
    g = to_grayscale(rgb)
    assert g.shape == (32, 32)
    assert float(g.mean()) > 0
    n = normalize_to_unit(rgb)
    assert float(n.max()) == 1.0

    rgb_r = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb_r[..., 0] = 200
    r = to_grayscale(rgb_r)
    assert float(r.mean()) > 0


def test_scan_nested_one_folder_per_image(tmp_path: Path):
    """Layout like: 3B_7.15/C2_104d1-1-1/C2_104d1-1-1.tif"""
    img = (np.random.rand(32, 32) * 255).astype(np.uint8)
    for stem in ("C2_104d1-1", "C2_104d1-1-1", "C2_104d1-1-2", "C2_wd1-3-1", "C2_wd1-3-2"):
        d = tmp_path / stem
        d.mkdir()
        tifffile.imwrite(d / f"{stem}.tif", img)

    report = scan_with_report(tmp_path)
    assert report.all_tifs == 5
    assert len(report.pairs) == 2

    p104 = next(p for p in report.pairs if p.experiment == "104")
    assert p104.drug == "d" and p104.group == "1" and p104.field == 1
    assert p104.merge_path is not None

    pwd = next(p for p in report.pairs if p.experiment == "w")
    assert pwd.drug == "d" and pwd.group == "1" and pwd.field == 3
    assert pwd.image_id == "wd1-3"
