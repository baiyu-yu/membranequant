"""Cellpose 全细胞分割（优先红通道 / 膜标记，模型 cyto3）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import clear_border


@dataclass
class SegmentResult:
    """一次分割的结果。"""

    labels: np.ndarray  # int32, 0=背景, 1..N=细胞
    image_for_display: np.ndarray  # 用于显示的归一化图
    model: str
    diameter: float | None
    channel_used: str
    n_cells_raw: int
    border_ids: list[int] = field(default_factory=list)
    scalebar_ids: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def cellpose_available() -> bool:
    try:
        import cellpose  # noqa: F401

        return True
    except ImportError:
        return False


def cellpose_status() -> dict[str, Any]:
    if not cellpose_available():
        return {
            "available": False,
            "version": None,
            "cuda": False,
            "message": "未安装 cellpose。请在 mem 环境中: pip install cellpose",
        }
    import cellpose

    ver = getattr(cellpose, "__version__", "unknown")
    cuda = False
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        pass
    return {
        "available": True,
        "version": ver,
        "cuda": cuda,
        "message": f"Cellpose {ver} (CUDA={'是' if cuda else '否'})",
    }


def _to_cellpose_input(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)
    return (img * 65535.0).astype(np.uint16)


def _find_border_labels(labels: np.ndarray) -> list[int]:
    """返回触边细胞 ID 列表。"""
    if labels.size == 0:
        return []
    raw = labels.astype(np.int32, copy=False)
    cleared = clear_border(raw)
    kept = set(int(x) for x in np.unique(cleared) if x != 0)
    border = []
    for lid in np.unique(raw):
        lid = int(lid)
        if lid == 0:
            continue
        if lid not in kept:
            border.append(lid)
    return border


def _find_scalebar_overlap_labels(
    labels: np.ndarray,
    box: tuple[int, int, int, int],
    min_overlap_frac: float = 0.15,
) -> list[int]:
    """与右下角标尺区域重叠较多的细胞。"""
    y0, y1, x0, x1 = box
    h, w = labels.shape
    region = np.zeros((h, w), dtype=bool)
    region[y0:y1, x0:x1] = True

    flagged: list[int] = []
    for prop in regionprops(labels.astype(np.int32)):
        cid = int(prop.label)
        mask = labels == cid
        area = int(mask.sum())
        if area == 0:
            continue
        overlap = int((mask & region).sum())
        if overlap / area >= min_overlap_frac:
            flagged.append(cid)
    return flagged


def run_cellpose(
    image: np.ndarray,
    *,
    model: str = "cyto3",
    diameter: float | None = None,
    gpu: bool = True,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    channel_name: str = "red",
) -> np.ndarray:
    """对单通道图像跑 Cellpose，返回 label 图。

    Parameters
    ----------
    model : ``cyto3``（推荐）或 ``cyto2``（细胞特别小时）
    diameter : ``None`` / ``0`` / ``<=0`` 表示 Auto
    """
    if not cellpose_available():
        raise ImportError(
            "需要安装 cellpose。请激活 conda 环境 mem 后执行: pip install cellpose"
        )

    from cellpose import models

    model_type = (model or "cyto3").strip()
    # 不要用 nuclei
    if model_type.lower() in ("nuclei", "nuc", "nucleus"):
        raise ValueError("请使用 cyto3 / cyto2 全细胞模型，不要用 nuclei。")

    diam = None if diameter is None or float(diameter) <= 0 else float(diameter)
    cp_img = _to_cellpose_input(image)
    use_gpu = bool(gpu)

    try:
        cp_model = models.CellposeModel(gpu=use_gpu, model_type=model_type)
    except TypeError:
        cp_model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_type)
    except Exception:
        # 旧版 Cellpose 类
        try:
            from cellpose.models import Cellpose

            cp_model = Cellpose(gpu=use_gpu, model_type=model_type)
        except Exception as e:
            raise RuntimeError(f"无法加载 Cellpose 模型 '{model_type}': {e}") from e

    eval_kwargs: dict[str, Any] = {
        "diameter": diam,
        "flow_threshold": float(flow_threshold),
        "cellprob_threshold": float(cellprob_threshold),
    }

    try:
        result = cp_model.eval(cp_img, channels=[0, 0], **eval_kwargs)
    except TypeError:
        try:
            result = cp_model.eval(cp_img, channel_axis=None, **eval_kwargs)
        except TypeError:
            result = cp_model.eval([cp_img], channels=[0, 0], **eval_kwargs)

    if isinstance(result, (tuple, list)):
        masks = np.asarray(result[0])
    else:
        masks = np.asarray(result)

    # eval 有时返回 list of masks
    if isinstance(masks, list):
        masks = np.asarray(masks[0])

    return masks.astype(np.int32, copy=False)


def segment_field(
    red: np.ndarray,
    *,
    green: np.ndarray | None = None,
    model: str = "cyto3",
    diameter: float | None = None,
    gpu: bool = True,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    use_channel: str = "red",
    mask_scalebar: bool = True,
    scalebar_width_frac: float = 0.22,
    scalebar_height_frac: float = 0.12,
    auto_flag_border: bool = True,
    auto_flag_scalebar: bool = True,
) -> SegmentResult:
    """对一个视野做全细胞分割。

    use_channel:
      - ``red``   : 红通道（默认，膜标记更清楚时）
      - ``green`` : 绿通道
      - ``max``   : 红绿取 max
    """
    from .io_scan import mask_scale_bar_region

    use_channel = (use_channel or "red").lower()
    if use_channel == "green":
        if green is None:
            raise ValueError("请求使用绿通道，但未加载绿通道图像")
        seg_img = green
        ch_name = "green"
    elif use_channel == "max":
        if green is None:
            seg_img = red
            ch_name = "red(no green for max)"
        else:
            seg_img = np.maximum(red, green)
            ch_name = "max(red,green)"
    else:
        seg_img = red
        ch_name = "red"

    display = seg_img.copy()
    scale_box: tuple[int, int, int, int] | None = None
    work = seg_img
    if mask_scalebar:
        work, scale_box = mask_scale_bar_region(
            seg_img,
            width_frac=scalebar_width_frac,
            height_frac=scalebar_height_frac,
        )

    labels = run_cellpose(
        work,
        model=model,
        diameter=diameter,
        gpu=gpu,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channel_name=ch_name,
    )

    n_raw = int(len([x for x in np.unique(labels) if x != 0]))
    border_ids: list[int] = []
    scalebar_ids: list[int] = []
    if auto_flag_border:
        border_ids = _find_border_labels(labels)
    if auto_flag_scalebar and scale_box is not None:
        scalebar_ids = _find_scalebar_overlap_labels(labels, scale_box)

    return SegmentResult(
        labels=labels,
        image_for_display=display,
        model=model,
        diameter=diameter,
        channel_used=ch_name,
        n_cells_raw=n_raw,
        border_ids=border_ids,
        scalebar_ids=scalebar_ids,
        meta={
            "scale_box": scale_box,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
        },
    )


def renumber_labels(labels: np.ndarray, keep_ids: set[int] | list[int]) -> np.ndarray:
    """只保留 keep_ids，并按 1..N 重新编号。"""
    keep = set(int(x) for x in keep_ids)
    out = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for old in sorted(keep):
        if old <= 0:
            continue
        mask = labels == old
        if not np.any(mask):
            continue
        out[mask] = next_id
        next_id += 1
    return out


def merge_cell_labels(labels: np.ndarray, keep_id: int, remove_id: int) -> np.ndarray:
    """合并两个细胞：将 remove_id 替换为 keep_id，消除它们之间的分隔边框。"""
    out = labels.copy()
    out[out == int(remove_id)] = int(keep_id)
    return out


def find_adjacent_cells_at(
    labels: np.ndarray, y: int, x: int, radius: int = 5
) -> list[int]:
    """给定图像中的 (y, x) 坐标，在该坐标附近的正方形区域内查找涉及的不同 Cell ID 列表。"""
    h, w = labels.shape
    y_min = max(0, int(y) - radius)
    y_max = min(h, int(y) + radius + 1)
    x_min = max(0, int(x) - radius)
    x_max = min(w, int(x) + radius + 1)

    patch = labels[y_min:y_max, x_min:x_max]
    unique_ids = [int(cid) for cid in np.unique(patch) if cid != 0]
    return unique_ids

