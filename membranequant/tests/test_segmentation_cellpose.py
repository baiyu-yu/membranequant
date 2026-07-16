"""Tests for optional Cellpose backend and config dispatch."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from membranequant.config import Config, load_config
from membranequant.segmentation import (
    build_cell_masks,
    cellpose_available,
    cellpose_status,
    segment_whole_cells,
    segment_whole_cells_cellpose,
)


def test_cellpose_status_dict():
    st = cellpose_status()
    assert "available" in st
    assert "message" in st
    assert st["available"] is cellpose_available()


def test_invalid_method_raises():
    cfg = Config(segmentation_method="magic")
    with pytest.raises(ValueError):
        load_config(overrides={"segmentation_method": "magic"})


def test_cellpose_missing_import_raises():
    red = np.zeros((32, 32), dtype=np.float32)
    green = np.zeros((32, 32), dtype=np.float32)
    green[8:24, 8:24] = 0.8
    cfg = Config(
        segmentation_method="cellpose",
        minimum_cell_area=10,
        maximum_cell_area=5000,
    )
    with patch("membranequant.segmentation.cellpose_available", return_value=False):
        with pytest.raises(ImportError, match="cellpose"):
            segment_whole_cells_cellpose(green, red, cfg)


def test_cellpose_mocked_eval():
    """When cellpose is present, labels from model.eval are filtered like otsu."""
    size = 64
    green = np.zeros((size, size), dtype=np.float32)
    red = np.zeros((size, size), dtype=np.float32)
    green[10:40, 10:40] = 0.7
    red[10:40, 10:40] = 0.2
    # ring-ish red
    red[10:14, 10:40] = 0.9
    red[36:40, 10:40] = 0.9
    red[10:40, 10:14] = 0.9
    red[10:40, 36:40] = 0.9

    fake_masks = np.zeros((size, size), dtype=np.int32)
    fake_masks[10:40, 10:40] = 1

    mock_model = MagicMock()
    mock_model.eval.return_value = (fake_masks, None, None)

    mock_models = MagicMock()
    mock_models.CellposeModel.return_value = mock_model

    cfg = Config(
        segmentation_method="cellpose",
        minimum_cell_area=50,
        maximum_cell_area=20000,
        ring_width=3,
        cellpose_model="cyto2",
        cellpose_diameter=0,
    )

    with patch("membranequant.segmentation.cellpose_available", return_value=True):
        with patch.dict("sys.modules", {"cellpose": MagicMock(), "cellpose.models": mock_models}):
            # Ensure import cellpose.models resolves to mock
            import sys

            sys.modules["cellpose"].models = mock_models
            labels, rejected = segment_whole_cells(green, red, cfg)

    assert labels.max() >= 1
    assert int(np.count_nonzero(labels == 1)) > 0


def test_build_masks_records_method_otsu():
    size = 80
    yy, xx = np.mgrid[0:size, 0:size]
    green = ((yy - 40) ** 2 + (xx - 40) ** 2 < 20**2).astype(np.float32) * 0.8
    red = ((yy - 40) ** 2 + (xx - 40) ** 2 < 20**2).astype(np.float32) * 0.1
    ring = ((yy - 40) ** 2 + (xx - 40) ** 2 >= 16**2) & ((yy - 40) ** 2 + (xx - 40) ** 2 < 20**2)
    red[ring] = 0.9
    cfg = Config(
        segmentation_method="otsu",
        minimum_cell_area=50,
        maximum_cell_area=20000,
        ring_width=2,
        max_eccentricity=0.99,
    )
    masks = build_cell_masks(green, red, cfg)
    assert masks.method == "otsu"
