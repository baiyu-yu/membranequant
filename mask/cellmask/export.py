"""导出筛选后的 mask / overlay / 清单，文件名跟随源图。"""

from __future__ import annotations

import json
import struct
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import tifffile

if TYPE_CHECKING:
    from .review_gui import ReviewItem


def _safe_stem(image_id: str) -> str:
    return image_id.replace("/", "_").replace("\\", "_").strip()


def export_item(
    item: "ReviewItem",
    out_dir: Path,
    *,
    renumber: bool = True,
    save_overlay: bool = True,
    save_rejected_mask: bool = False,
    dtype: str = "uint16",
) -> dict[str, Path]:
    """导出单张视野的结果。

    输出（以 image_id=104d1-1 为例）::

        out_dir/
          masks/104d1-1_mask.tif          # 保留细胞，1..N 连续编号
          overlays/104d1-1_overlay.png    # 轮廓叠加预览
          meta/104d1-1_meta.json          # 筛选元信息
    """
    from .review_gui import build_rgb_overlay
    from .segment import renumber_labels

    out_dir = Path(out_dir)
    mask_dir = out_dir / "masks"
    overlay_dir = out_dir / "overlays"
    meta_dir = out_dir / "meta"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    stem = _safe_stem(item.image_id)
    kept = item.kept_ids()
    if renumber:
        labels_out = renumber_labels(item.labels, kept)
    else:
        labels_out = item.labels.copy()
        for rid in item.rejected:
            labels_out[labels_out == rid] = 0

    if dtype == "uint16":
        mask_arr = labels_out.astype(np.uint16)
    else:
        mask_arr = labels_out.astype(np.int32)

    mask_path = mask_dir / f"{stem}_mask.tif"
    tifffile.imwrite(str(mask_path), mask_arr, compression="zlib")

    paths: dict[str, Path] = {"mask": mask_path}

    # 1. 导出彩图版 mask PNG (解决 Windows 默认看图查看是纯黑问题)
    colored_mask_path = mask_dir / f"{stem}_mask_colored.png"
    export_colored_mask_png(labels_out, colored_mask_path)
    paths["mask_colored"] = colored_mask_path

    # 2. 导出 Fiji / ImageJ 直接拖拽可用的 ROI.zip 文件 (严格保存在 rois/ 子目录中，使用 ZIP_STORED)
    rois_dir = out_dir / "rois"
    roi_zip_path_sub = rois_dir / f"{stem}_rois.zip"
    try:
        exported_zip = export_imagej_rois_zip(labels_out, roi_zip_path_sub)
        if exported_zip and exported_zip.exists():
            paths["rois_zip"] = exported_zip
            print(f"  [export] 已导出 ImageJ ROI: {exported_zip}")
    except Exception as e:
        print(f"[warn] ImageJ ROI.zip 导出失败 {stem}: {e}")

    # 3. 导出 Cellpose 原生 _seg.npy 文件
    seg_npy_path = out_dir / f"{stem}_seg.npy"
    try:
        np.save(
            str(seg_npy_path),
            {
                "masks": mask_arr,
                "filename": str(item.red_path) if item.red_path else item.image_id,
                "chan_choose": [0, 0],
            },
        )
        paths["seg_npy"] = seg_npy_path
    except Exception as e:
        print(f"[warn] _seg.npy 导出失败 {stem}: {e}")

    if save_overlay:
        try:
            overlay = build_rgb_overlay(item.image, item.labels, item.rejected)
            overlay_path = overlay_dir / f"{stem}_overlay.png"
            overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
            try:
                from PIL import Image

                Image.fromarray(overlay_uint8).save(str(overlay_path))
            except ImportError:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                plt.imsave(str(overlay_path), overlay)
            paths["overlay"] = overlay_path
        except Exception as e:
            print(f"[warn] overlay 保存失败 {stem}: {e}")

    if save_rejected_mask:
        rej = np.zeros_like(item.labels, dtype=np.uint16)
        for i, rid in enumerate(sorted(item.rejected), start=1):
            rej[item.labels == rid] = i
        rej_path = mask_dir / f"{stem}_rejected.tif"
        tifffile.imwrite(str(rej_path), rej, compression="zlib")
        paths["rejected"] = rej_path

    meta = {
        "image_id": item.image_id,
        "channel_used": item.channel_used,
        "model": item.model,
        "diameter": item.diameter,
        "n_cells_raw": len(item.all_ids()),
        "n_cells_kept": len(kept),
        "kept_ids_original": kept,
        "rejected_ids_original": sorted(item.rejected),
        "border_ids": list(item.border_ids),
        "scalebar_ids": list(item.scalebar_ids),
        "reviewed": item.reviewed,
        "notes": item.notes,
        "red_path": str(item.red_path) if item.red_path else None,
        "green_path": str(item.green_path) if item.green_path else None,
        "merge_path": str(item.merge_path) if item.merge_path else None,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "mask_file": mask_path.name,
    }
    meta_path = meta_dir / f"{stem}_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["meta"] = meta_path
    return paths


def export_all(
    items: list["ReviewItem"],
    out_dir: Path,
    *,
    only_reviewed: bool = False,
    renumber: bool = True,
    save_overlay: bool = True,
) -> list[Path]:
    """导出多张，并写总清单 summary.json / kept_list.csv。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported_masks: list[Path] = []
    rows: list[dict] = []

    for item in items:
        if only_reviewed and not item.reviewed:
            continue
        paths = export_item(
            item,
            out_dir,
            renumber=renumber,
            save_overlay=save_overlay,
        )
        exported_masks.append(paths["mask"])
        rows.append(
            {
                "image_id": item.image_id,
                "n_raw": len(item.all_ids()),
                "n_kept": len(item.kept_ids()),
                "n_rejected": len(item.rejected),
                "reviewed": item.reviewed,
                "mask": paths["mask"].name,
                "channel_used": item.channel_used,
                "model": item.model,
            }
        )

    summary = {
        "n_images": len(rows),
        "n_cells_kept_total": sum(r["n_kept"] for r in rows),
        "n_cells_raw_total": sum(r["n_raw"] for r in rows),
        "items": rows,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "Segmentation results were manually inspected and corrected. "
            "Exported masks contain only kept cells, renumbered from 1."
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 简易 CSV
    csv_path = out_dir / "kept_summary.csv"
    headers = [
        "image_id",
        "n_raw",
        "n_kept",
        "n_rejected",
        "reviewed",
        "mask",
        "channel_used",
        "model",
    ]
    lines = [",".join(headers)]
    for r in rows:
        lines.append(",".join(str(r[h]) for h in headers))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"导出完成 → {out_dir}")
    print(f"  masks: {len(exported_masks)}  summary: {summary_path.name}")
    return exported_masks


def _build_imagej_roi_bytes(name: str, contour: np.ndarray) -> bytes:
    """构建 ImageJ 二进制 Polygon ROI 文件内容。contour 为 Nx2 [y, x] 坐标。"""
    ys = contour[:, 0]
    xs = contour[:, 1]
    top, left = int(np.floor(ys.min())), int(np.floor(xs.min()))
    bottom, right = int(np.ceil(ys.max())), int(np.ceil(xs.max()))

    rel_xs = (xs - left).round().astype(np.float32).astype(np.int16)
    rel_ys = (ys - top).round().astype(np.float32).astype(np.int16)
    n_pts = len(rel_xs)

    # 64 字节 ImageJ ROI Header (全部使用大端字节序 >)
    header = bytearray(64)
    struct.pack_into(">4s", header, 0, b"Iout")
    struct.pack_into(">H", header, 4, 227)     # version 227
    struct.pack_into(">H", header, 6, 0)       # type 0 = Polygon
    struct.pack_into(">h", header, 8, top)     # top
    struct.pack_into(">h", header, 10, left)   # left
    struct.pack_into(">h", header, 12, bottom) # bottom
    struct.pack_into(">h", header, 14, right)  # right
    struct.pack_into(">H", header, 16, n_pts)  # nCoordinates (必须在偏移量 16!)

    # ImageJ 要求坐标数据为大端 int16
    x_bytes = rel_xs.astype(">i2").tobytes()
    y_bytes = rel_ys.astype(">i2").tobytes()

    return bytes(header) + x_bytes + y_bytes


def export_imagej_rois_zip(labels: np.ndarray, zip_path: Path) -> Path | None:
    """生成 Fiji / ImageJ 直接支持拖拽导入 ROI Manager 的 .rois.zip 文件。"""
    import zipfile
    from skimage.measure import find_contours

    unique_ids = [int(x) for x in np.unique(labels) if x != 0]
    if not unique_ids:
        return None

    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_STORED) as zf:
        for idx, cid in enumerate(unique_ids, start=1):
            mask = labels == cid
            contours = find_contours(mask.astype(float), 0.5)
            if not contours:
                continue
            c = max(contours, key=len)
            if len(c) < 3:
                continue
            roi_bytes = _build_imagej_roi_bytes(f"Cell-{cid}", c)
            roi_filename = f"{idx:03d}-cell{cid:04d}.roi"
            zf.writestr(roi_filename, roi_bytes)

    return zip_path


def export_colored_mask_png(labels: np.ndarray, png_path: Path) -> Path | None:
    """导出伪彩标注版 mask PNG，解决 Windows 看图软件直接打开显示纯黑的问题。"""
    h, w = labels.shape
    unique_ids = [int(x) for x in np.unique(labels) if x != 0]
    if not unique_ids:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        np.random.seed(42)
        max_id = max(unique_ids)
        colors = np.random.randint(60, 255, size=(max_id + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]
        rgb = colors[labels]

    try:
        from PIL import Image

        png_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(str(png_path))
    except Exception:
        pass
    return png_path

