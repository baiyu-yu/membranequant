"""Whole-cell, membrane ring, and cytoplasm segmentation.

Design principle:
  - Cell boundaries come from EGFP (or EGFP+DiI), NEVER from DiI alone.
  - Membrane ROI is a fixed-width geometric ring from the cell boundary.
  - DiI is used only later for QC (red coverage), not for defining ROIs.

Segmentation backends:
  - otsu                    : 经典Otsu阈值 (default, no extra deps)
  - watershed_distance      : 距离变换+分水岭 (适合圆形粘连细胞)
  - watershed_gradient      : 梯度+分水岭 (适合边界清晰的细胞)
  - hminima_watershed       : H-minima+分水岭 (适合密集粘连细胞)
  - morphological_opening   : 形态学开运算 (适合轻度粘连)
  - combined_markers        : 距离+梯度双重markers (综合方法)
  - cellpose                : 深度学习 (需安装cellpose)
  
For severely adhered cells, try:
  1. watershed_distance (ImageJ经典方法)
  2. hminima_watershed (抑制过度分割)
  3. combined_markers (综合效果最稳定)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, erosion, remove_small_objects
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi

from .config import Config
from .utils import saturation_fraction


@dataclass
class CellMasks:
    """Per-image labeled masks and derived ROIs."""

    labels: np.ndarray  # int32 label image (whole cell)
    membrane: np.ndarray  # int32 labels on membrane ring only
    cytoplasm: np.ndarray  # int32 labels on cytoplasm only
    kept_ids: list[int]
    rejected: list[dict]  # {cell_id, reason}
    method: str = "otsu"


def cellpose_available() -> bool:
    """Return True if the cellpose package can be imported."""
    try:
        import cellpose  # noqa: F401

        return True
    except ImportError:
        return False


def cellpose_status() -> dict[str, Any]:
    """Human-readable Cellpose availability for UI / CLI."""
    if not cellpose_available():
        return {
            "available": False,
            "version": None,
            "cuda_available": False,
            "message": "Cellpose not installed. Run: pip install cellpose",
        }
    try:
        import cellpose

        ver = getattr(cellpose, "__version__", "unknown")
    except Exception:
        ver = "unknown"

    cuda_available = False
    cuda_error = None
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception as e:
        cuda_error = str(e)

    return {
        "available": True,
        "version": ver,
        "cuda_available": cuda_available,
        "cuda_error": cuda_error,
        "message": f"Cellpose {ver} available (CUDA: {'Available' if cuda_available else 'Not Available'})",
    }


def _segmentation_image(green: np.ndarray, red: np.ndarray, mode: str) -> np.ndarray:
    if mode == "green_red":
        return np.maximum(green, red)
    return green


def _filter_labeled_cells(
    labels: np.ndarray,
    green: np.ndarray,
    cfg: Config,
    *,
    clear_border_cells: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """Apply area / eccentricity / saturation / border filters; renumber labels."""
    rejected: list[dict] = []
    work = labels.astype(np.int32, copy=True)

    if clear_border_cells:
        labels_raw = work.copy()
        cleared = clear_border(labels_raw)
        kept_raw = set(int(x) for x in np.unique(cleared) if x != 0)
        for rid in np.unique(labels_raw):
            rid = int(rid)
            if rid == 0:
                continue
            if rid not in kept_raw:
                rejected.append({"cell_id": rid, "reason": "touch_border"})
        work = cleared.astype(np.int32, copy=False)

    props = regionprops(work, intensity_image=green)
    remap: dict[int, int] = {}
    next_id = 1
    filtered_rejected: list[dict] = list(rejected)

    for prop in props:
        cid = int(prop.label)
        area = int(prop.area)
        if area < cfg.minimum_cell_area:
            filtered_rejected.append({"cell_id": cid, "reason": "area_too_small"})
            continue
        if area > cfg.maximum_cell_area:
            filtered_rejected.append({"cell_id": cid, "reason": "area_too_large"})
            continue
        ecc = float(prop.eccentricity) if prop.eccentricity is not None else 0.0
        if ecc > cfg.max_eccentricity:
            filtered_rejected.append({"cell_id": cid, "reason": "eccentricity"})
            continue

        mask = work == cid
        sat = saturation_fraction(green[mask])
        if sat > cfg.max_saturation_fraction:
            filtered_rejected.append({"cell_id": cid, "reason": "green_saturation"})
            continue

        remap[cid] = next_id
        next_id += 1

    out = np.zeros_like(work, dtype=np.int32)
    for old, new in remap.items():
        out[work == old] = new

    return out, filtered_rejected


def segment_whole_cells_otsu(
    green: np.ndarray,
    red: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict]]:
    """Otsu-based whole-cell segmentation with morphological cleanup and filters."""
    seg_img = _segmentation_image(green, red, cfg.segmentation_channel)

    if cfg.threshold == "otsu":
        thr = threshold_otsu(seg_img)
    else:
        thr = float(cfg.threshold)

    binary = seg_img > thr
    binary = closing(binary, disk(3))
    binary = remove_small_objects(binary, max_size=max(1, cfg.minimum_cell_area // 4))
    binary = ndi.binary_fill_holes(binary)

    labels_raw = label(binary)
    return _filter_labeled_cells(labels_raw, green, cfg, clear_border_cells=True)


def segment_whole_cells_watershed_distance(
    green: np.ndarray,
    red: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict]]:
    """Distance-transform watershed: 距离变换+分水岭（经典方法，适合圆形/椭圆粘连细胞）
    
    Reference: 
    - ImageJ Watershed Plugin
    - OpenCV Tutorial: Watershed segmentation
    - 基于距离变换的分水岭算法常用于粘连物体分割
    
    步骤：
    1. 二值化获取前景
    2. 距离变换得到每个像素到边界的距离
    3. 在距离变换的局部最大值处作为种子点（细胞中心）
    4. 分水岭算法从种子扩展分割
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from scipy import ndimage as ndi
    
    seg_img = _segmentation_image(green, red, cfg.segmentation_channel)
    
    # 二值化
    if cfg.threshold == "otsu":
        thr = threshold_otsu(seg_img)
    else:
        thr = float(cfg.threshold)
    
    binary = seg_img > thr
    binary = closing(binary, disk(3))
    binary = remove_small_objects(binary, max_size=max(1, cfg.minimum_cell_area // 4))
    binary = ndi.binary_fill_holes(binary)
    
    # 距离变换
    distance = ndi.distance_transform_edt(binary)
    
    # 寻找局部最大值作为种子点（细胞中心）
    # min_distance参数控制两个峰值之间的最小距离，可以根据细胞大小调整
    min_distance = max(5, int(np.sqrt(cfg.minimum_cell_area / np.pi) * 0.5))
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary)
    
    # 创建markers
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask)
    
    # 分水岭分割
    labels_raw = watershed(-distance, markers, mask=binary)
    
    return _filter_labeled_cells(labels_raw, green, cfg, clear_border_cells=True)


def segment_whole_cells_watershed_gradient(
    green: np.ndarray,
    red: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict]]:
    """Gradient-based watershed: 基于梯度的分水岭（适合边界清晰的细胞）
    
    Reference:
    - ImageJ Morphological Segmentation Plugin
    - Marker-controlled watershed for cell segmentation
    
    步骤：
    1. 计算图像梯度（边界强度）
    2. 使用形态学操作找到内部markers（确定的细胞中心）
    3. 使用分水岭算法基于梯度图分割
    """
    from skimage.filters import sobel
    from skimage.segmentation import watershed
    from skimage.morphology import opening
    
    seg_img = _segmentation_image(green, red, cfg.segmentation_channel)
    
    # 计算梯度
    edges = sobel(seg_img)
    
    # 二值化获取粗略前景
    if cfg.threshold == "otsu":
        thr = threshold_otsu(seg_img)
    else:
        thr = float(cfg.threshold)
    
    binary = seg_img > thr
    binary = closing(binary, disk(3))
    binary = remove_small_objects(binary, max_size=max(1, cfg.minimum_cell_area // 4))
    
    # 获取确定的内部区域作为markers（通过opening操作缩小）
    markers = opening(binary, disk(5))
    markers = label(markers)
    
    # 基于梯度的分水岭
    labels_raw = watershed(edges, markers, mask=binary)
    
    return _filter_labeled_cells(labels_raw, green, cfg, clear_border_cells=True)


def segment_whole_cells_hminima_watershed(
    green: np.ndarray,
    red: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict]]:
    """H-minima + Watershed: 抑制浅局部极小值的分水岭（文献中常用于密集细胞）
    
    Reference:
    - "Iterative h-minima-based marker-controlled watershed for cell nucleus segmentation"
    - "Marker-controlled watershed with deep edge emphasis and optimized H-minima transform"
    
    H-minima变换可以抑制深度小于h的局部极小值，减少过度分割。
    适合密集培养的细胞核或细胞分割。
    """
    from skimage.morphology import h_minima, reconstruction
    from skimage.segmentation import watershed
    
    seg_img = _segmentation_image(green, red, cfg.segmentation_channel)
    
    # 二值化
    if cfg.threshold == "otsu":
        thr = threshold_otsu(seg_img)
    else:
        thr = float(cfg.threshold)
    
    binary = seg_img > thr
    binary = closing(binary, disk(3))
    binary = remove_small_objects(binary, max_size=max(1, cfg.minimum_cell_area // 4))
    binary = ndi.binary_fill_holes(binary)
    
    # 距离变换
    distance = ndi.distance_transform_edt(binary)
    
    # H-minima变换抑制浅的局部极小值
    # h值可调：较大的h会产生更少的marker（更少的细胞），适合抑制噪声
    h_value = distance.max() * 0.3  # 可以调整这个比例
    h_min = h_minima(distance, h=h_value)
    
    # 找到markers（extended minima）
    markers = label(h_min)
    
    # 分水岭
    labels_raw = watershed(-distance, markers, mask=binary)
    
    return _filter_labeled_cells(labels_raw, green, cfg, clear_border_cells=True)


def segment_whole_cells_morphological_opening(
    green: np.ndarray,
    red: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict]]:
    """Morphological Opening: 形态学开运算分离（简单直接，适合轻度粘连）
    
    Reference:
    - ImageJ Process > Binary > Watershed
    - Opening操作可以断开细胞间的细窄连接
    
    原理：开运算(Opening) = 先腐蚀再膨胀
    - 腐蚀可以断开粘连点
    - 膨胀恢复细胞大小
    适合粘连不严重、有明显细窄连接的情况
    """
    from skimage.morphology import opening
    
    seg_img = _segmentation_image(green, red, cfg.segmentation_channel)
    
    # 二值化
    if cfg.threshold == "otsu":
        thr = threshold_otsu(seg_img)
    else:
        thr = float(cfg.threshold)
    
    binary = seg_img > thr
    
    # 使用较大的结构元素进行开运算，断开粘连
    # disk大小可以根据粘连程度调整
    selem = disk(4)  # 可调
    binary = opening(binary, selem)
    
    binary = remove_small_objects(binary, max_size=max(1, cfg.minimum_cell_area // 4))
    binary = ndi.binary_fill_holes(binary)
    
    labels_raw = label(binary)
    return _filter_labeled_cells(labels_raw, green, cfg, clear_border_cells=True)


def segment_whole_cells_combined_markers(
    green: np.ndarray,
    red: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict]]:
    """Combined markers watershed: 结合距离变换和梯度的双重markers
    
    这是一个改进方法，结合了：
    - 距离变换识别细胞中心
    - 梯度图识别细胞边界
    适合复杂场景的粘连分割
    """
    from skimage.feature import peak_local_max
    from skimage.filters import sobel
    from skimage.segmentation import watershed
    
    seg_img = _segmentation_image(green, red, cfg.segmentation_channel)
    
    # 二值化
    if cfg.threshold == "otsu":
        thr = threshold_otsu(seg_img)
    else:
        thr = float(cfg.threshold)
    
    binary = seg_img > thr
    binary = closing(binary, disk(3))
    binary = remove_small_objects(binary, max_size=max(1, cfg.minimum_cell_area // 4))
    binary = ndi.binary_fill_holes(binary)
    
    # 距离变换找中心
    distance = ndi.distance_transform_edt(binary)
    min_distance = max(5, int(np.sqrt(cfg.minimum_cell_area / np.pi) * 0.5))
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary)
    
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = label(mask)
    
    # 梯度图
    gradient = sobel(seg_img)
    
    # 结合距离和梯度：在梯度图上进行分水岭，使用距离变换的markers
    # 这样既利用了细胞中心信息，又考虑了边界清晰度
    combined = gradient - distance * 0.1  # 权衡梯度和距离
    
    labels_raw = watershed(combined, markers, mask=binary)
    
    return _filter_labeled_cells(labels_raw, green, cfg, clear_border_cells=True)


def _to_cellpose_input(image: np.ndarray) -> np.ndarray:
    """Convert float [0,1] image to uint16 range Cellpose handles well."""
    img = np.asarray(image, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)
    return (img * 65535.0).astype(np.uint16)


def segment_whole_cells_cellpose(
    green: np.ndarray,
    red: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict]]:
    """Cellpose whole-cell segmentation on EGFP (or EGFP+DiI max).

    Requires optional dependency: pip install cellpose
    """
    if not cellpose_available():
        raise ImportError(
            "segmentation_method='cellpose' requires cellpose. "
            "Install with: pip install cellpose"
        )

    from cellpose import models

    seg_img = _segmentation_image(green, red, cfg.segmentation_channel)
    cp_img = _to_cellpose_input(seg_img)

    diameter = None if cfg.cellpose_diameter <= 0 else float(cfg.cellpose_diameter)
    model_type = (cfg.cellpose_model or "cyto2").strip()

    # Cellpose 2.x / 3.x compatibility
    masks: np.ndarray
    try:
        model = models.CellposeModel(gpu=bool(cfg.cellpose_gpu), model_type=model_type)
    except TypeError:
        # Newer cellpose may use pretrained_model=
        model = models.CellposeModel(gpu=bool(cfg.cellpose_gpu), pretrained_model=model_type)

    eval_kwargs: dict[str, Any] = {
        "diameter": diameter,
        "flow_threshold": float(cfg.cellpose_flow_threshold),
        "cellprob_threshold": float(cfg.cellpose_cellprob_threshold),
    }

    try:
        # channels=[0,0] => grayscale / single channel
        result = model.eval(cp_img, channels=[0, 0], **eval_kwargs)
    except TypeError:
        # Some versions dropped channels= in favor of channel_axis=None
        result = model.eval(cp_img, channel_axis=None, **eval_kwargs)

    if isinstance(result, (tuple, list)):
        masks = np.asarray(result[0])
    else:
        masks = np.asarray(result)

    masks = masks.astype(np.int32, copy=False)
    return _filter_labeled_cells(masks, green, cfg, clear_border_cells=True)


def segment_whole_cells(
    green: np.ndarray,
    red: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict]]:
    """Dispatch to different segmentation backends.
    
    Available methods:
    - otsu: 经典Otsu阈值法（默认）
    - watershed_distance: 距离变换+分水岭（适合圆形粘连细胞）
    - watershed_gradient: 基于梯度的分水岭（适合边界清晰的细胞）
    - hminima_watershed: H-minima抑制+分水岭（适合密集细胞）
    - morphological_opening: 形态学开运算（适合轻度粘连）
    - combined_markers: 结合距离和梯度的双重markers
    - cellpose: 深度学习分割（需要安装cellpose）
    """
    method = (cfg.segmentation_method or "otsu").strip().lower()
    
    if method == "cellpose":
        return segment_whole_cells_cellpose(green, red, cfg)
    elif method == "otsu":
        return segment_whole_cells_otsu(green, red, cfg)
    elif method == "watershed_distance":
        return segment_whole_cells_watershed_distance(green, red, cfg)
    elif method == "watershed_gradient":
        return segment_whole_cells_watershed_gradient(green, red, cfg)
    elif method == "hminima_watershed":
        return segment_whole_cells_hminima_watershed(green, red, cfg)
    elif method == "morphological_opening":
        return segment_whole_cells_morphological_opening(green, red, cfg)
    elif method == "combined_markers":
        return segment_whole_cells_combined_markers(green, red, cfg)
    else:
        raise ValueError(
            f"Unknown segmentation_method: {method}. "
            f"Available: otsu, watershed_distance, watershed_gradient, "
            f"hminima_watershed, morphological_opening, combined_markers, cellpose"
        )


# Back-compat alias used by older tests / imports
def segment_whole_cells_legacy(
    green: np.ndarray,
    red: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, list[dict]]:
    return segment_whole_cells_otsu(green, red, cfg)


def membrane_ring(labels: np.ndarray, ring_width: int = 3) -> np.ndarray:
    """Build fixed-width membrane ring: whole_cell - eroded_cell."""
    if ring_width < 1:
        raise ValueError("ring_width must be >= 1")

    selem = disk(ring_width)
    membrane = np.zeros_like(labels, dtype=np.int32)
    for cid in np.unique(labels):
        if cid == 0:
            continue
        cell = labels == cid
        eroded = erosion(cell, selem)
        ring = cell & ~eroded
        if not np.any(ring):
            ring = cell
        membrane[ring] = cid
    return membrane


def cytoplasm_from_ring(labels: np.ndarray, membrane: np.ndarray) -> np.ndarray:
    """Cytoplasm = whole cell minus membrane ring."""
    cyto = np.zeros_like(labels, dtype=np.int32)
    for cid in np.unique(labels):
        if cid == 0:
            continue
        mask = (labels == cid) & (membrane != cid)
        cyto[mask] = cid
    return cyto


def build_cell_masks(
    green: np.ndarray,
    red: np.ndarray,
    cfg: Config,
) -> CellMasks:
    """Run whole-cell segmentation and derive membrane / cytoplasm ROIs."""
    method = (cfg.segmentation_method or "otsu").strip().lower()
    labels, rejected = segment_whole_cells(green, red, cfg)
    membrane = membrane_ring(labels, ring_width=cfg.ring_width)
    cytoplasm = cytoplasm_from_ring(labels, membrane)
    kept_ids = [int(i) for i in np.unique(labels) if i != 0]
    return CellMasks(
        labels=labels,
        membrane=membrane,
        cytoplasm=cytoplasm,
        kept_ids=kept_ids,
        rejected=rejected,
        method=method,
    )
