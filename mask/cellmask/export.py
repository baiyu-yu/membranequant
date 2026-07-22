"""导出筛选后的 mask / overlay / 清单，文件名跟随源图。"""

from __future__ import annotations

import json
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

    if save_overlay:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            overlay = build_rgb_overlay(item.image, item.labels, item.rejected)
            overlay_path = overlay_dir / f"{stem}_overlay.png"
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
