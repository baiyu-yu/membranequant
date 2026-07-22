"""批处理流水线：扫描 → Cellpose 分割 → 人工筛选 → 导出。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .export import export_all, export_item
from .io_scan import FieldImages, load_field_channels, scan_fields
from .review_gui import MaskReviewer, ReviewItem
from .segment import SegmentResult, cellpose_status, segment_field


def _cache_path(cache_dir: Path, image_id: str) -> Path:
    safe = image_id.replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe}_seg.npz"


def _save_cache(path: Path, item: ReviewItem) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        image_id=item.image_id,
        image=item.image.astype(np.float32),
        labels=item.labels.astype(np.int32),
        channel_used=item.channel_used,
        model=item.model,
        diameter=-1.0 if item.diameter is None else float(item.diameter),
        border_ids=np.array(item.border_ids, dtype=np.int32),
        scalebar_ids=np.array(item.scalebar_ids, dtype=np.int32),
        scale_box=np.array(item.scale_box if item.scale_box else (-1, -1, -1, -1), dtype=np.int32),
        notes=item.notes or "",
        red_path=str(item.red_path) if item.red_path else "",
        green_path=str(item.green_path) if item.green_path else "",
        merge_path=str(item.merge_path) if item.merge_path else "",
    )


def _load_cache(path: Path) -> ReviewItem | None:
    if not path.is_file():
        return None
    try:
        data = np.load(str(path), allow_pickle=False)
        diam = float(data["diameter"])
        box = tuple(int(x) for x in data["scale_box"].tolist())
        scale_box = None if box[0] < 0 else box
        return ReviewItem(
            image_id=str(data["image_id"]),
            image=np.asarray(data["image"], dtype=np.float32),
            labels=np.asarray(data["labels"], dtype=np.int32),
            red_path=Path(str(data["red_path"])) if str(data["red_path"]) else None,
            green_path=Path(str(data["green_path"])) if str(data["green_path"]) else None,
            merge_path=Path(str(data["merge_path"])) if str(data["merge_path"]) else None,
            channel_used=str(data["channel_used"]),
            model=str(data["model"]),
            diameter=None if diam < 0 else diam,
            border_ids=[int(x) for x in data["border_ids"].tolist()],
            scalebar_ids=[int(x) for x in data["scalebar_ids"].tolist()],
            scale_box=scale_box,
            notes=str(data["notes"]),
        )
    except Exception as e:
        print(f"  [cache] 读取失败 {path.name}: {e}")
        return None


def segment_all_fields(
    fields: list[FieldImages],
    *,
    model: str = "cyto3",
    diameter: float | None = None,
    gpu: bool = True,
    use_channel: str = "red",
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    mask_scalebar: bool = True,
    scalebar_width_frac: float = 0.22,
    scalebar_height_frac: float = 0.12,
    progress: bool = True,
    cache_dir: Path | None = None,
    reuse_cache: bool = True,
) -> list[ReviewItem]:
    """对所有视野跑 Cellpose，返回可供筛选的 ReviewItem 列表。"""
    items: list[ReviewItem] = []
    n = len(fields)
    for i, fi in enumerate(fields, start=1):
        if progress:
            print(f"[{i}/{n}] 分割 {fi.image_id} ...", flush=True)

        if cache_dir is not None and reuse_cache:
            cached = _load_cache(_cache_path(cache_dir, fi.image_id))
            if cached is not None:
                items.append(cached)
                if progress:
                    print(
                        f"    → 缓存命中，{len(cached.all_ids())} 个细胞",
                        flush=True,
                    )
                continue

        try:
            red, green, note = load_field_channels(fi)
            result: SegmentResult = segment_field(
                red,
                green=green,
                model=model,
                diameter=diameter,
                gpu=gpu,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                use_channel=use_channel,
                mask_scalebar=mask_scalebar,
                scalebar_width_frac=scalebar_width_frac,
                scalebar_height_frac=scalebar_height_frac,
            )
            item = ReviewItem(
                image_id=fi.image_id,
                image=result.image_for_display,
                labels=result.labels,
                red_path=fi.red_path,
                green_path=fi.green_path,
                merge_path=fi.merge_path,
                channel_used=result.channel_used,
                model=result.model,
                diameter=result.diameter,
                border_ids=list(result.border_ids),
                scalebar_ids=list(result.scalebar_ids),
                scale_box=result.meta.get("scale_box"),
                notes=note,
            )
            items.append(item)
            if cache_dir is not None:
                _save_cache(_cache_path(cache_dir, fi.image_id), item)
            if progress:
                print(
                    f"    → {result.n_cells_raw} 个细胞 "
                    f"(触边{len(result.border_ids)}, 标尺区{len(result.scalebar_ids)}) "
                    f"[{result.channel_used}]",
                    flush=True,
                )
        except Exception as e:
            print(f"    ✗ 失败: {e}", flush=True)
    return items


def run_pipeline(
    input_dir: str | Path,
    out_dir: str | Path,
    *,
    model: str = "cyto3",
    diameter: float | None = None,
    gpu: bool = True,
    use_channel: str = "red",
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    mask_scalebar: bool = True,
    scalebar_width_frac: float = 0.22,
    scalebar_height_frac: float = 0.12,
    auto_flag: bool = True,
    skip_review: bool = False,
    limit: int | None = None,
    image_ids: list[str] | None = None,
    reuse_cache: bool = True,
    gui: str = "web",
    port: int = 8080,
) -> dict[str, Any]:
    """完整流程入口。

    Parameters
    ----------
    input_dir : 实验根目录
    out_dir : 导出目录
    model : cyto3 / cyto2
    diameter : None 或 <=0 为 Auto
    use_channel : red | green | max
    skip_review : True 时只分割+自动剔除触边/标尺，不弹 GUI
    limit : 只处理前 N 个视野（调试用）
    image_ids : 只处理指定 image_id 列表
    reuse_cache : 复用 out_dir/cache 中的分割结果
    gui : web (默认, 调起 Web 前端) | matplotlib
    port : Web GUI 尝试端口 (默认 8080)
    """
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"

    status = cellpose_status()
    print(status["message"])
    if not status["available"] and not skip_review:
        # 分割一定需要 cellpose；skip_review 同样需要
        pass
    if not status["available"]:
        raise ImportError(status["message"])

    report = scan_fields(input_dir, require_red=True)
    print(
        f"扫描完成: {report.all_tifs} 个 TIF, 解析 {report.parsed}, "
        f"有效视野 {len(report.fields)}"
    )
    if report.unparsed:
        print(f"  未识别文件名 ({len(report.unparsed)}): 例如 {report.unparsed[:5]}")
    if report.skipped_no_red:
        print(f"  跳过无红/Merge ({len(report.skipped_no_red)}): {report.skipped_no_red[:5]}")

    fields = report.fields
    if image_ids:
        want = set(image_ids)
        fields = [f for f in fields if f.image_id in want]
        print(f"按 image_id 过滤后: {len(fields)} 个视野")
    if limit is not None and limit > 0:
        fields = fields[:limit]
        print(f"limit={limit}, 实际处理 {len(fields)} 个视野")

    if not fields:
        raise RuntimeError("没有可处理的视野，请检查目录与命名。")

    items = segment_all_fields(
        fields,
        model=model,
        diameter=diameter,
        gpu=gpu,
        use_channel=use_channel,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        mask_scalebar=mask_scalebar,
        scalebar_width_frac=scalebar_width_frac,
        scalebar_height_frac=scalebar_height_frac,
        cache_dir=cache_dir,
        reuse_cache=reuse_cache,
    )
    if not items:
        raise RuntimeError("分割结果为空（全部失败）。")

    def _export_one(item: ReviewItem) -> Path | None:
        paths = export_item(item, out_dir, renumber=True, save_overlay=True)
        return paths.get("mask")

    def _export_all(all_items: list[ReviewItem]) -> list[Path]:
        return export_all(all_items, out_dir, only_reviewed=False, renumber=True)

    if skip_review:
        for it in items:
            if auto_flag:
                it.apply_auto_flags(border=True, scalebar=True)
            it.reviewed = True
        paths = export_all(items, out_dir, only_reviewed=False)
        return {
            "n_fields": len(items),
            "out_dir": str(out_dir),
            "exported": [str(p) for p in paths],
            "reviewed": False,
        }

    print("\n打开人工筛选窗口……")
    print("论文写法参考: Segmentation results were manually inspected and corrected.\n")

    if (gui or "web").lower() == "matplotlib":
        # 提前配置中文字体，避免窗口标题/状态栏 DejaVu 缺字警告
        from .fonts import setup_chinese_font

        setup_chinese_font(silent=False)
        reviewer = MaskReviewer(
            items,
            on_export_one=_export_one,
            on_export_all=_export_all,
            auto_flag_on_start=auto_flag,
        )
        reviewed_items = reviewer.run()
    else:
        from .web_gui import WebReviewServer

        web_server = WebReviewServer(
            items,
            on_export_one=_export_one,
            on_export_all=_export_all,
            auto_flag_on_start=auto_flag,
            port=port,
        )
        reviewed_items = web_server.start_and_wait()

    # 关闭窗口后若尚未导出，再导出一次
    paths = export_all(reviewed_items, out_dir, only_reviewed=False)
    return {
        "n_fields": len(reviewed_items),
        "out_dir": str(out_dir),
        "exported": [str(p) for p in paths],
        "reviewed": True,
        "n_kept_total": sum(len(it.kept_ids()) for it in reviewed_items),
    }

