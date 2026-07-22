"""Web 前端交互界面：双屏同步对比 + 0 延迟极速 Canvas 交互。

功能亮点：
1. 双屏同步对比 (Dual View Synchronized Mode)：
   - 一键开启双屏左右对比 (支持按 d 快捷键切换)。
   - 左屏显示 Mask 掩码与边框编辑工具，右屏同步显示 100% 纯荧光原图。
   - 滚轮放大、按住平移自动像素级 1:1 双屏联动！
2. 0 延迟即时勾选与状态同步：点击/取消勾选瞬间响应 (0ms)，后台异步极速同步。
3. 删除/合并分割边框 (Merge Border)：点击交界分割线，消除边框合并细胞。
4. 明确的导出与路径反馈：点击导出弹窗展示生成的 `.tif` 掩码及文件存放路径。
5. 搜索定位：侧边栏搜索框快速筛选 Cell ID。
"""

from __future__ import annotations

import base64
import io
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

import numpy as np
from skimage.measure import find_contours, regionprops

from .review_gui import ReviewItem
from .segment import find_adjacent_cells_at


def _normalize_img(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return (a - lo) / (hi - lo)


def _to_base64_png_rgb(rgb_u8: np.ndarray) -> str:
    """将 uint8 RGB (H, W, 3) 图像转为 Base64 PNG 字符串。"""
    buf = io.BytesIO()
    try:
        from PIL import Image

        if rgb_u8.ndim == 2:
            Image.fromarray(rgb_u8, mode="L").save(buf, format="PNG")
        else:
            Image.fromarray(rgb_u8, mode="RGB").save(buf, format="PNG")
    except ImportError:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.imsave(buf, rgb_u8, format="png")

    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def _make_fluorescent_pseudo_color(arr_2d: np.ndarray, channel: str = "red") -> np.ndarray:
    """将单通道 2D 归一化浮点图转为 RGB 荧光伪彩 uint8 数组。

    channel:
      - 'red': 经典红色荧光伪彩 (DiI 膜标记)
      - 'green': 经典绿色荧光伪彩 (EGFP 绿荧光)
    """
    norm = np.clip(np.asarray(arr_2d, dtype=np.float32), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    h, w = u8.shape[:2]

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if channel == "red":
        rgb[..., 0] = u8
    elif channel == "green":
        rgb[..., 1] = u8
    else:
        rgb[..., 0] = u8
        rgb[..., 1] = u8
        rgb[..., 2] = u8
    return rgb


class WebReviewServer:
    """CellMask 本地 Web 服务器（双屏同步对比版）。"""

    def __init__(
        self,
        items: list[ReviewItem],
        *,
        on_export_one: Callable[[ReviewItem], Path | None] | None = None,
        on_export_all: Callable[[list[ReviewItem]], list[Path]] | None = None,
        auto_flag_on_start: bool = True,
        port: int = 8080,
    ):
        if not items:
            raise ValueError("没有可筛选的图像")

        self.items = items
        self.index = 0
        self.on_export_one = on_export_one
        self.on_export_all = on_export_all
        self.port = port
        self.server: HTTPServer | None = None
        self.server_thread: threading.Thread | None = None
        self._finished = False

        self._bg_cache: dict[str, dict[str, str]] = {}
        self._geom_cache: dict[str, dict] = {}

        if auto_flag_on_start:
            for it in self.items:
                it.apply_auto_flags(border=True, scalebar=True)

        # 启动后台异步预加载线程 (预先计算所有视野背景与几何轮廓，切换视野 0ms 零延迟)
        def _preload_all():
            for it in self.items:
                try:
                    self._get_bg_images(it)
                    self._get_geom_metadata(it)
                except Exception:
                    pass

        threading.Thread(target=_preload_all, daemon=True).start()

    def _get_bg_images(self, item: ReviewItem) -> dict[str, str | None]:
        key = item.image_id
        if key in self._bg_cache:
            return self._bg_cache[key]

        import tifffile
        from .io_scan import _as_2d

        # 1. 红色荧光原图 (DiI 膜标记，RGB 红色荧光伪彩)
        red_arr = None
        if item.red_path and item.red_path.is_file():
            try:
                raw_r = tifffile.imread(str(item.red_path))
                red_arr = _normalize_img(_as_2d(raw_r))
            except Exception:
                pass
        if red_arr is None:
            red_arr = _normalize_img(item.image)

        rgb_red = _make_fluorescent_pseudo_color(red_arr, "red")
        bg_red = _to_base64_png_rgb(rgb_red)

        # 2. 绿色荧光原图 (EGFP 绿荧光，RGB 绿色荧光伪彩)
        green_arr = None
        bg_green = None
        if item.green_path and item.green_path.is_file():
            try:
                raw_g = tifffile.imread(str(item.green_path))
                green_arr = _normalize_img(_as_2d(raw_g))
                rgb_green = _make_fluorescent_pseudo_color(green_arr, "green")
                bg_green = _to_base64_png_rgb(rgb_green)
            except Exception:
                pass

        # 3. Merge 荧光合成彩图
        bg_merge = None
        if item.merge_path and item.merge_path.is_file():
            try:
                m_raw = tifffile.imread(str(item.merge_path))
                m_arr = np.asarray(m_raw)
                if m_arr.ndim == 3:
                    if m_arr.shape[0] in (2, 3, 4) and m_arr.shape[0] < min(m_arr.shape[1], m_arr.shape[2]):
                        m_arr = np.moveaxis(m_arr, 0, -1)
                    rgb_m = np.zeros((*m_arr.shape[:2], 3), dtype=np.uint8)
                    for c in range(min(3, m_arr.shape[-1])):
                        rgb_m[..., c] = (_normalize_img(m_arr[..., c]) * 255).astype(np.uint8)
                    bg_merge = _to_base64_png_rgb(rgb_m)
                else:
                    norm_m = _normalize_img(m_arr)
                    bg_merge = _to_base64_png_rgb(_make_fluorescent_pseudo_color(norm_m, "red"))
            except Exception:
                pass

        if bg_merge is None and green_arr is not None:
            # 自动基于红绿通道合成 RGB 荧光彩图
            r_u8 = (red_arr * 255).astype(np.uint8)
            g_u8 = (green_arr * 255).astype(np.uint8)
            synth_merge = np.stack([r_u8, g_u8, np.zeros_like(r_u8)], axis=-1)
            bg_merge = _to_base64_png_rgb(synth_merge)

        res = {"bg_red": bg_red, "bg_green": bg_green, "bg_merge": bg_merge}
        self._bg_cache[key] = res
        return res

    def _get_geom_metadata(self, item: ReviewItem, force_recompute: bool = False) -> dict:
        key = f"{item.image_id}_{hash(item.labels.tobytes())}"
        if not force_recompute and key in self._geom_cache:
            return self._geom_cache[key]

        labels = item.labels.astype(np.int32)
        border_set = set(item.border_ids)
        scalebar_set = set(item.scalebar_ids)

        props_all = regionprops(labels)
        cells_info = []
        contours_dict = {}

        for prop in props_all:
            cid = int(prop.label)
            cy, cx = prop.centroid
            area = int(prop.area)

            # 1 像素 0-padding 确保 find_contours 永远生成完整封闭、极致光滑的完美细胞轮廓，无断裂折线
            sub_mask = labels[prop.slice] == cid
            padded_mask = np.pad(sub_mask, 1, mode="constant", constant_values=0)
            raw_contours = find_contours(padded_mask.astype(float), 0.5)
            ymin, xmin = prop.slice[0].start - 1, prop.slice[1].start - 1

            formatted_contours = []
            for c in raw_contours:
                formatted_contours.append(
                    [[round(float(pt[1] + xmin), 1), round(float(pt[0] + ymin), 1)] for pt in c]
                )

            contours_dict[str(cid)] = formatted_contours
            cells_info.append(
                {
                    "id": cid,
                    "centroid": [round(float(cx), 1), round(float(cy), 1)],
                    "area": area,
                    "is_border": cid in border_set,
                    "is_scalebar": cid in scalebar_set,
                }
            )

        res = {
            "cells": cells_info,
            "contours": contours_dict,
            "width": int(labels.shape[1]),
            "height": int(labels.shape[0]),
        }
        self._geom_cache[key] = res
        return res

    def _get_item_payload(self, idx: int) -> dict:
        if idx < 0 or idx >= len(self.items):
            return {"error": "索引越界"}
        item = self.items[idx]
        item.reviewed = True

        bgs = self._get_bg_images(item)
        geom = self._get_geom_metadata(item)

        summary = [
            {
                "index": i,
                "image_id": it.image_id,
                "reviewed": it.reviewed,
                "exported": getattr(it, "exported", False),
            }
            for i, it in enumerate(self.items)
        ]
        return {
            "index": idx,
            "total": len(self.items),
            "image_id": item.image_id,
            "channel_used": item.channel_used,
            "model": item.model,
            "diameter": item.diameter if item.diameter else "Auto",
            "notes": item.notes,
            "scale_box": item.scale_box,
            "rejected_ids": sorted(list(item.rejected)),
            "bg_red": bgs["bg_red"],
            "bg_green": bgs["bg_green"],
            "bg_merge": bgs["bg_merge"],
            "geom": geom,
            "items_summary": summary,
        }

    def start_and_wait(self) -> list[ReviewItem]:
        reviewer_self = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def _send_json(self, data: dict, status: int = 200):
                body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _send_html(self, html: str):
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)

                if path in ("/", "/index.html"):
                    self._send_html(HTML_TEMPLATE)
                    return

                if path == "/api/items":
                    summary = [
                        {
                            "index": i,
                            "image_id": it.image_id,
                            "n_kept": len(it.kept_ids()),
                            "n_total": len(it.all_ids()),
                            "reviewed": it.reviewed,
                        }
                        for i, it in enumerate(reviewer_self.items)
                    ]
                    self._send_json({"items": summary, "current_index": reviewer_self.index})
                    return

                if path == "/api/item":
                    idx = int(query.get("index", [reviewer_self.index])[0])
                    reviewer_self.index = idx
                    self._send_json(reviewer_self._get_item_payload(idx))
                    return

                self.send_error(404, "Not Found")

            def do_POST(self):
                parsed = urlparse(self.path)
                path = parsed.path
                length = int(self.headers.get("Content-Length", 0))
                raw_data = self.rfile.read(length).decode("utf-8")
                data = json.loads(raw_data) if raw_data else {}

                idx = int(data.get("index", reviewer_self.index))
                item = reviewer_self.items[idx]
                item.reviewed = True

                if path in ("/api/toggle_cell", "/api/set_cell_status"):
                    cid = int(data.get("cell_id", 0))
                    if path == "/api/toggle_cell":
                        if cid in item.rejected:
                            item.rejected.remove(cid)
                        else:
                            item.rejected.add(cid)
                    else:
                        keep = bool(data.get("keep", True))
                        if keep and cid in item.rejected:
                            item.rejected.remove(cid)
                        elif not keep:
                            item.rejected.add(cid)

                    self._send_json({"success": True, "rejected_ids": sorted(list(item.rejected))})
                    return

                if path == "/api/merge_border":
                    x = float(data.get("x", 0))
                    y = float(data.get("y", 0))
                    cids = find_adjacent_cells_at(item.labels, y, x, radius=5)
                    merged = False
                    if len(cids) >= 2:
                        id1, id2 = cids[0], cids[1]
                        merged = item.merge_cells(id1, id2)
                    elif len(cids) == 1:
                        cids_wider = find_adjacent_cells_at(item.labels, y, x, radius=12)
                        if len(cids_wider) >= 2:
                            id1, id2 = cids_wider[0], cids_wider[1]
                            merged = item.merge_cells(id1, id2)

                    if merged:
                        reviewer_self._get_geom_metadata(item, force_recompute=True)

                    self._send_json(
                        {
                            "success": merged,
                            "cids_found": cids,
                            "item": reviewer_self._get_item_payload(idx),
                        }
                    )
                    return

                if path == "/api/action":
                    action = data.get("action")
                    if action == "keep_all":
                        item.rejected.clear()
                    elif action == "reject_border":
                        item.rejected.update(item.border_ids)
                    elif action == "reject_scalebar":
                        item.rejected.update(item.scalebar_ids)
                    elif action == "reset":
                        item.reset_flags()

                    self._send_json({"success": True, "rejected_ids": sorted(list(item.rejected))})
                    return

                if path == "/api/export_one":
                    exported_path = None
                    try:
                        if reviewer_self.on_export_one:
                            exported_path = reviewer_self.on_export_one(item)
                        item.exported = True
                        self._send_json(
                            {
                                "success": True,
                                "image_id": item.image_id,
                                "exported_path": str(exported_path) if exported_path else "已导出",
                            }
                        )
                    except Exception as err:
                        self._send_json({"success": False, "error": str(err)}, status=500)
                    return

                if path == "/api/export_all":
                    paths_out = []
                    try:
                        if reviewer_self.on_export_all:
                            paths_out = reviewer_self.on_export_all(reviewer_self.items)
                        for it in reviewer_self.items:
                            it.exported = True
                        reviewer_self._finished = True
                        self._send_json(
                            {
                                "success": True,
                                "count": len(paths_out),
                                "exported": [str(p) for p in paths_out],
                            }
                        )
                        threading.Thread(target=reviewer_self.stop).start()
                    except Exception as err:
                        self._send_json({"success": False, "error": str(err)}, status=500)
                    return

                self.send_error(404, "Not Found")

        port = self.port
        for p in range(port, port + 20):
            try:
                self.server = HTTPServer(("127.0.0.1", p), Handler)
                self.port = p
                break
            except OSError:
                continue

        if not self.server:
            raise RuntimeError("无法绑定本地 HTTP 端口")

        url = f"http://127.0.0.1:{self.port}"
        print(f"\n==============================================")
        print(f"  CellMask Web 双屏同步对比界面已启动!")
        print(f"  访问地址: {url}")
        print(f"==============================================\n")

        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

        try:
            webbrowser.open(url)
        except Exception:
            pass

        try:
            while not self._finished:
                self.server_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            print("\n正在停止 Web 服务器...")
            self.stop()

        return self.items

    def stop(self):
        self._finished = True
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CellMask 极速审核与双屏同步对比</title>
    <style>
        :root {
            --bg-dark: #121418;
            --panel-bg: #1c2026;
            --panel-border: #2e3642;
            --accent: #2563eb;
            --accent-hover: #3b82f6;
            --keep-color: #22c55e;
            --reject-color: #ef4444;
            --text-main: #f3f4f6;
            --text-sub: #9ca3af;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg-dark); color: var(--text-main); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

        header {
            background: var(--panel-bg);
            border-bottom: 1px solid var(--panel-border);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 56px;
        }
        .header-title { font-weight: 600; font-size: 16px; display: flex; align-items: center; gap: 10px; }
        .badge { background: rgba(37, 99, 235, 0.2); color: #60a5fa; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: normal; }

        .top-controls { display: flex; align-items: center; gap: 12px; }
        .btn {
            background: #2a313d; color: var(--text-main); border: 1px solid var(--panel-border);
            padding: 6px 14px; border-radius: 6px; font-size: 13px; cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
            transition: all 0.15s ease;
        }
        .btn:hover { background: #374151; border-color: #4b5563; }
        .btn-primary { background: var(--accent); border-color: var(--accent); }
        .btn-primary:hover { background: var(--accent-hover); }
        .btn-success { background: #16a34a; border-color: #16a34a; }
        .btn-success:hover { background: #22c55e; }

        main { flex: 1; display: flex; overflow: hidden; position: relative; }

        /* Workspace & Panels */
        .canvas-workspace {
            flex: 1; display: flex; position: relative; background: #0a0c0e; overflow: hidden; gap: 2px;
        }
        .canvas-panel {
            flex: 1; position: relative; height: 100%; overflow: hidden; cursor: grab; user-select: none; background: #0a0c0e;
        }
        .canvas-panel:active { cursor: grabbing; }
        canvas { position: absolute; top: 0; left: 0; transform-origin: 0 0; }

        .panel-tag {
            position: absolute; top: 12px; right: 16px; background: rgba(18, 21, 25, 0.85); backdrop-filter: blur(6px);
            padding: 4px 10px; border-radius: 4px; font-size: 11px; color: #d1d5db; border: 1px solid rgba(255,255,255,0.1);
            pointer-events: none; z-index: 8; display: flex; align-items: center; gap: 6px;
        }

        .floating-tools {
            position: absolute; top: 16px; left: 16px; background: rgba(28, 32, 38, 0.85); backdrop-filter: blur(8px);
            border: 1px solid var(--panel-border); border-radius: 8px; padding: 6px; display: flex; flex-direction: column; gap: 6px; z-index: 10;
        }
        .tool-btn {
            width: 38px; height: 38px; border-radius: 6px; border: none; background: transparent; color: var(--text-sub);
            display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 16px; transition: all 0.15s;
        }
        .tool-btn:hover { background: rgba(255, 255, 255, 0.1); color: #fff; }
        .tool-btn.active { background: var(--accent); color: #fff; }

        aside {
            width: 320px; background: var(--panel-bg); border-left: 1px solid var(--panel-border);
            display: flex; flex-direction: column; height: 100%; z-index: 5;
        }
        .sidebar-section { padding: 12px 14px; border-bottom: 1px solid var(--panel-border); }
        .section-title { font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-sub); margin-bottom: 8px; letter-spacing: 0.5px; }

        .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 4px; }
        .stat-card { background: #121519; padding: 6px; border-radius: 6px; text-align: center; border: 1px solid rgba(255,255,255,0.05); }
        .stat-num { font-size: 15px; font-weight: bold; margin-top: 2px; }
        .stat-num.keep { color: var(--keep-color); }
        .stat-num.rej { color: var(--reject-color); }

        .btn-group-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }

        .cell-list-wrapper { flex: 1; overflow-y: auto; padding: 8px 10px; }
        .cell-item {
            display: flex; align-items: center; justify-content: space-between; padding: 6px 10px; border-radius: 6px;
            background: #15181d; border: 1px solid transparent; margin-bottom: 4px; cursor: pointer; transition: all 0.1s;
        }
        .cell-item:hover { border-color: rgba(255,255,255,0.15); background: #1e232a; }
        .cell-info { display: flex; align-items: center; gap: 8px; }
        .cell-tag { font-size: 10px; padding: 1px 4px; border-radius: 3px; background: #374151; color: #d1d5db; }
        .cell-tag.border { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }
        .cell-tag.scalebar { background: rgba(6, 182, 212, 0.2); color: #67e8f9; }

        .toggle-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; font-size: 13px; color: #d1d5db; }
        .switch { position: relative; display: inline-block; width: 34px; height: 18px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #374151; transition: .2s; border-radius: 18px; }
        .slider:before { position: absolute; content: ""; height: 14px; width: 14px; left: 2px; bottom: 2px; background-color: white; transition: .2s; border-radius: 50%; }
        input:checked + .slider { background-color: var(--accent); }
        input:checked + .slider:before { transform: translateX(16px); }

        #toast {
            position: absolute; bottom: 24px; left: 50%; transform: translateX(-50%); background: rgba(15, 23, 42, 0.95); color: #fff;
            padding: 10px 20px; border-radius: 20px; font-size: 13px; border: 1px solid #3b82f6; display: none; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }

        .search-box {
            width: 100%; background: #121519; border: 1px solid var(--panel-border); color: #fff;
            padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-bottom: 8px;
        }
    </style>
</head>
<body>

    <header>
        <div class="header-title">
            <span>🔬 CellMask 交互审核</span>
            <span class="badge" id="image-id-tag">载入中...</span>
        </div>
        <div class="top-controls">
            <button class="btn" onclick="navField(-1)">◀ 上一张 (p)</button>
            <select id="field-select" onchange="loadField(parseInt(this.value))" style="background:#2a313d; color:#fff; border:1px solid #374151; border-radius:6px; padding:4px 8px; font-size:13px; cursor:pointer; max-width: 250px;">
                <option value="0">1 / 1</option>
            </select>
            <button class="btn" onclick="navField(1)">下一张 (n) ▶</button>
            <button class="btn" style="background:#059669; color:#fff; border-color:#059669;" onclick="jumpToNextUnexported()" title="按 u 键跳转下一张未导出的视野">⏭️ 跳至未导出 (u)</button>
            <span class="badge" id="export-status-badge" style="background: #374151; font-size:12px; padding:4px 8px;">导出进度: 0/0</span>
            <div style="width: 1px; height: 20px; background: var(--panel-border);"></div>
            <button class="btn btn-primary" onclick="exportOne()">导出本张</button>
            <button class="btn btn-success" onclick="exportAll()">完成并导出全部</button>
        </div>
    </header>

    <main>
        <div class="floating-tools">
            <button class="tool-btn active" id="tool-toggle" onclick="setTool('toggle')" title="切换保留/剔除模式 (点击细胞)">👆</button>
            <button class="tool-btn" id="tool-merge" onclick="setTool('merge')" title="删除分割边框 / 合并细胞模式 (点击分割线)">✂️</button>
            <button class="tool-btn" id="tool-dual" onclick="toggleDualView()" title="开启/关闭 左右双屏同步对比 (按 d 键)">🖼️</button>
            <button class="tool-btn" onclick="resetZoom()" title="重置视角 (Fit Screen)">🔍</button>
        </div>

        <div class="canvas-workspace" id="canvas-workspace">
            <div class="canvas-panel" id="panel-left">
                <div class="panel-tag" id="tag-left">✏️ 编辑视窗 (Mask 掩码)</div>
                <canvas id="main-canvas"></canvas>
            </div>
            <div class="canvas-panel" id="panel-right" style="display: none; border-left: 2px solid var(--panel-border);">
                <div class="panel-tag" id="tag-right">🖼️ 纯原图对比 (无掩码)</div>
                <canvas id="raw-canvas"></canvas>
            </div>
        </div>

        <aside>
            <div class="sidebar-section">
                <div class="section-title">视野统计</div>
                <div class="stat-grid">
                    <div class="stat-card">
                        <div style="font-size:10px; color:#9ca3af;">总数</div>
                        <div class="stat-num" id="stat-total">0</div>
                    </div>
                    <div class="stat-card">
                        <div style="font-size:10px; color:#9ca3af;">保留</div>
                        <div class="stat-num keep" id="stat-keep">0</div>
                    </div>
                    <div class="stat-card">
                        <div style="font-size:10px; color:#9ca3af;">剔除</div>
                        <div class="stat-num rej" id="stat-rej">0</div>
                    </div>
                </div>
            </div>

            <div class="sidebar-section">
                <div class="section-title">快捷筛选操作</div>
                <div class="btn-group-grid">
                    <button class="btn" onclick="sendAction('reject_border')">剔除触边 (b)</button>
                    <button class="btn" onclick="sendAction('reject_scalebar')">剔除标尺区 (s)</button>
                    <button class="btn" onclick="sendAction('keep_all')">全部保留 (a)</button>
                    <button class="btn" onclick="sendAction('reset')">重置标记 (r)</button>
                </div>
            </div>

            <div class="sidebar-section">
                <div class="section-title">显示与双屏对比设置</div>
                <div class="toggle-row">
                    <span>双屏同步对比 (d)</span>
                    <label class="switch">
                        <input type="checkbox" id="dual-view-toggle" onchange="toggleDualView()">
                        <span class="slider"></span>
                    </label>
                </div>
                <div class="toggle-row">
                    <span>背景衬底</span>
                    <select id="channel-select" onchange="changeChannel()" style="background:#2a313d; color:#fff; border:1px solid #374151; border-radius:4px; padding:2px 6px;">
                        <option value="red">🔴 红 (Cellpose 分割源图)</option>
                        <option value="green">🟢 绿 (对比衬底)</option>
                        <option value="merge">🟣 Merge (对比衬底)</option>
                    </select>
                </div>
                <div class="toggle-row">
                    <span>显示编号 (i)</span>
                    <label class="switch">
                        <input type="checkbox" id="show-ids" checked onchange="renderAllCanvases()">
                        <span class="slider"></span>
                    </label>
                </div>
                <div class="toggle-row">
                    <span>显示剔除轮廓 (x)</span>
                    <label class="switch">
                        <input type="checkbox" id="show-rejected" checked onchange="renderAllCanvases()">
                        <span class="slider"></span>
                    </label>
                </div>
                <div class="toggle-row">
                    <span>Mask 透明度</span>
                    <input type="range" id="opacity-slider" min="0.1" max="0.8" step="0.05" value="0.3" oninput="renderAllCanvases()" style="width:90px;">
                </div>
            </div>

            <div style="padding: 10px 14px 0 14px; display:flex; flex-direction:column; flex:1; overflow:hidden;">
                <div class="section-title">细胞清单 (勾选保留)</div>
                <input type="text" class="search-box" id="cell-search" placeholder="🔍 搜索 Cell ID..." oninput="filterCellList()">
                <div class="cell-list-wrapper" id="cell-list"></div>
            </div>
        </aside>
    </main>

    <div id="toast">已成功导出</div>

    <script>
        let currentItem = null;
        let rejectedSet = new Set();
        let currentTool = 'toggle';
        let currentChannel = 'red';
        let isDualView = false;

        let zoom = 1.0;
        let panX = 0, panY = 0;
        let isDragging = false;
        let dragStartX = 0, dragStartY = 0;

        const panelLeft = document.getElementById('panel-left');
        const panelRight = document.getElementById('panel-right');
        const canvasLeft = document.getElementById('main-canvas');
        const canvasRight = document.getElementById('raw-canvas');
        const ctxLeft = canvasLeft.getContext('2d');
        const ctxRight = canvasRight.getContext('2d');

        const bgImages = { red: new Image(), green: new Image(), merge: new Image() };

        async function loadField(index = 0) {
            try {
                const res = await fetch(`/api/item?index=${index}`);
                const data = await res.json();
                if (data.error) return alert(data.error);

                currentItem = data;
                rejectedSet = new Set(data.rejected_ids || []);

                const badgeText = `${data.image_id} [源: 红通道]`;
                document.getElementById('image-id-tag').innerText = badgeText;

                const selectEl = document.getElementById('field-select');
                if (data.items_summary) {
                    selectEl.innerHTML = '';
                    data.items_summary.forEach((it, idx) => {
                        const opt = document.createElement('option');
                        opt.value = idx;
                        const icon = it.exported ? '✅' : '⚪';
                        const tag = it.exported ? '[已导出]' : '[未导出]';
                        opt.innerText = `[${idx + 1}/${data.total}] ${icon} ${tag} ${it.image_id}`;
                        if (idx === data.index) opt.selected = true;
                        selectEl.appendChild(opt);
                    });
                    updateExportBadge(data.items_summary);
                }

                if (data.bg_red) {
                    bgImages.red.onload = () => renderAllCanvases();
                    bgImages.red.src = data.bg_red;
                }
                if (data.bg_green) {
                    bgImages.green.onload = () => renderAllCanvases();
                    bgImages.green.src = data.bg_green;
                } else {
                    bgImages.green.src = '';
                }
                if (data.bg_merge) {
                    bgImages.merge.onload = () => renderAllCanvases();
                    bgImages.merge.src = data.bg_merge;
                } else {
                    bgImages.merge.src = '';
                }

                resetZoom();
                updateSidebarStats();
                renderCellList();
                renderAllCanvases();
            } catch (err) {
                console.error("加载失败", err);
                alert("切换视野失败: " + err);
            }
        }

        function toggleDualView() {
            isDualView = !isDualView;
            document.getElementById('dual-view-toggle').checked = isDualView;
            document.getElementById('tool-dual').classList.toggle('active', isDualView);
            panelRight.style.display = isDualView ? 'block' : 'none';

            showToast(isDualView ? '已开启双屏同步对比 (左右放大/平移自动像素级联动)' : '已关闭双屏对比');
            resetZoom();
            renderAllCanvases();
        }

        function setTool(tool) {
            currentTool = tool;
            document.getElementById('tool-toggle').classList.toggle('active', tool === 'toggle');
            document.getElementById('tool-merge').classList.toggle('active', tool === 'merge');
            showToast(tool === 'merge' ? '删除边框模式：点击或划过两个细胞交界边框合并' : '切换模式：点击细胞保留/剔除');
        }

        function changeChannel() {
            currentChannel = document.getElementById('channel-select').value;
            renderAllCanvases();
        }

        function resetZoom() {
            if (!currentItem) return;
            const w = currentItem.geom.width;
            const h = currentItem.geom.height;
            const activeContainer = panelLeft;
            const cw = activeContainer.clientWidth;
            const ch = activeContainer.clientHeight;

            const scale = Math.min(cw / w, ch / h) * 0.92;
            zoom = scale;
            panX = (cw - w * scale) / 2;
            panY = (ch - h * scale) / 2;
            renderAllCanvases();
        }

        function updateSidebarStats() {
            if (!currentItem) return;
            const total = currentItem.geom.cells.length;
            const rejCount = rejectedSet.size;
            const keepCount = total - rejCount;

            document.getElementById('stat-total').innerText = total;
            document.getElementById('stat-keep').innerText = keepCount;
            document.getElementById('stat-rej').innerText = rejCount;
        }

        function renderCellList() {
            const listEl = document.getElementById('cell-list');
            listEl.innerHTML = '';
            if (!currentItem) return;

            const filterVal = document.getElementById('cell-search').value.trim();

            currentItem.geom.cells.forEach(c => {
                if (filterVal && !c.id.toString().includes(filterVal)) return;

                const isKept = !rejectedSet.has(c.id);
                const itemEl = document.createElement('div');
                itemEl.className = 'cell-item';
                itemEl.id = `cell-row-${c.id}`;
                itemEl.onclick = (e) => {
                    if (e.target.tagName !== 'INPUT') toggleCellLocal(c.id);
                };

                let tagsHtml = '';
                if (c.is_border) tagsHtml += '<span class="cell-tag border">触边</span>';
                if (c.is_scalebar) tagsHtml += '<span class="cell-tag scalebar">标尺</span>';

                itemEl.innerHTML = `
                    <div class="cell-info">
                        <input type="checkbox" id="chk-${c.id}" ${isKept ? 'checked' : ''} onclick="event.stopPropagation(); setCellStatusLocal(${c.id}, this.checked)">
                        <span id="label-${c.id}" style="font-weight:600; font-size:13px; color:${isKept ? '#34d399' : '#f87171'}">#${c.id}</span>
                        ${tagsHtml}
                    </div>
                    <span style="font-size:11px; color:#9ca3af;">${c.area} px</span>
                `;
                listEl.appendChild(itemEl);
            });
        }

        function filterCellList() {
            renderCellList();
        }

        function updateCellRowUI(cellId) {
            const isKept = !rejectedSet.has(cellId);
            const chk = document.getElementById(`chk-${cellId}`);
            if (chk) chk.checked = isKept;
            const lbl = document.getElementById(`label-${cellId}`);
            if (lbl) lbl.style.color = isKept ? '#34d399' : '#f87171';
        }

        function toggleCellLocal(cellId) {
            if (rejectedSet.has(cellId)) {
                rejectedSet.delete(cellId);
            } else {
                rejectedSet.add(cellId);
            }
            updateCellRowUI(cellId);
            updateSidebarStats();
            renderAllCanvases();
            fetch('/api/toggle_cell', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: currentItem.index, cell_id: cellId })
            });
        }

        function setCellStatusLocal(cellId, keep) {
            if (keep) {
                rejectedSet.delete(cellId);
            } else {
                rejectedSet.add(cellId);
            }
            updateCellRowUI(cellId);
            updateSidebarStats();
            renderAllCanvases();
            fetch('/api/set_cell_status', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: currentItem.index, cell_id: cellId, keep: keep })
            });
        }

        function renderAllCanvases() {
            if (!currentItem) return;

            const w = currentItem.geom.width;
            const h = currentItem.geom.height;
            const img = bgImages[currentChannel] && bgImages[currentChannel].complete ? bgImages[currentChannel] : bgImages.red;

            // 1. 渲染左屏 (编辑屏：背景 + Mask 填充 + 轮廓 + ID)
            canvasLeft.width = panelLeft.clientWidth;
            canvasLeft.height = panelLeft.clientHeight;

            ctxLeft.clearRect(0, 0, canvasLeft.width, canvasLeft.height);
            ctxLeft.save();
            ctxLeft.translate(panX, panY);
            ctxLeft.scale(zoom, zoom);

            if (img.complete) {
                ctxLeft.drawImage(img, 0, 0, w, h);
            }

            if (currentItem.scale_box) {
                const [y0, y1, x0, x1] = currentItem.scale_box;
                ctxLeft.strokeStyle = 'cyan';
                ctxLeft.lineWidth = 1 / zoom;
                ctxLeft.setLineDash([4 / zoom, 4 / zoom]);
                ctxLeft.strokeRect(x0, y0, x1 - x0, y1 - y0);
                ctxLeft.setLineDash([]);
            }

            const showIds = document.getElementById('show-ids').checked;
            const showRejected = document.getElementById('show-rejected').checked;
            const opacity = parseFloat(document.getElementById('opacity-slider').value);

            currentItem.geom.cells.forEach(c => {
                const isKept = !rejectedSet.has(c.id);
                if (!isKept && !showRejected) return;

                const contours = currentItem.geom.contours[c.id.toString()] || [];
                const color = isKept ? '#22c55e' : '#ef4444';

                contours.forEach(poly => {
                    if (poly.length === 0) return;
                    ctxLeft.beginPath();
                    ctxLeft.moveTo(poly[0][0], poly[0][1]);
                    for (let i = 1; i < poly.length; i++) {
                        ctxLeft.lineTo(poly[i][0], poly[i][1]);
                    }
                    ctxLeft.closePath();

                    ctxLeft.fillStyle = isKept ? `rgba(34, 197, 94, ${opacity})` : `rgba(239, 68, 68, ${opacity * 0.7})`;
                    ctxLeft.fill();

                    ctxLeft.strokeStyle = color;
                    ctxLeft.lineWidth = (isKept ? 1.8 : 1.2) / zoom;
                    ctxLeft.stroke();
                });

                if (showIds) {
                    const [cx, cy] = c.centroid;
                    ctxLeft.font = `bold ${Math.max(10, 12 / zoom)}px sans-serif`;
                    ctxLeft.fillStyle = isKept ? '#a7f3d0' : '#fca5a5';
                    ctxLeft.textAlign = 'center';
                    ctxLeft.textBaseline = 'middle';
                    ctxLeft.fillText(c.id.toString(), cx, cy);
                }
            });

            // 渲染鼠标悬停提示高亮 (Hover Highlight)
            if (hoveredCell) {
                const isKept = !rejectedSet.has(hoveredCell.id);
                const contours = currentItem.geom.contours[hoveredCell.id.toString()] || [];
                contours.forEach(poly => {
                    if (poly.length === 0) return;
                    ctxLeft.beginPath();
                    ctxLeft.moveTo(poly[0][0], poly[0][1]);
                    for (let i = 1; i < poly.length; i++) {
                        ctxLeft.lineTo(poly[i][0], poly[i][1]);
                    }
                    ctxLeft.closePath();

                    ctxLeft.fillStyle = 'rgba(251, 191, 36, 0.4)';
                    ctxLeft.fill();

                    ctxLeft.strokeStyle = '#fbbf24';
                    ctxLeft.lineWidth = 3.0 / zoom;
                    ctxLeft.stroke();
                });

                const [cx, cy] = hoveredCell.centroid;
                const tooltipText = isKept ? `点击 剔除 #${hoveredCell.id}` : `点击 保留 #${hoveredCell.id}`;
                ctxLeft.font = `bold ${Math.max(11, 13 / zoom)}px sans-serif`;
                ctxLeft.fillStyle = '#fef08a';
                ctxLeft.textAlign = 'center';
                ctxLeft.textBaseline = 'bottom';
                ctxLeft.fillText(tooltipText, cx, cy - 10 / zoom);
            }
            ctxLeft.restore();

            // 2. 渲染右屏 (纯原图对比屏：100% 同步视角，无 Mask 掩码)
            if (isDualView) {
                canvasRight.width = panelRight.clientWidth;
                canvasRight.height = panelRight.clientHeight;

                ctxRight.clearRect(0, 0, canvasRight.width, canvasRight.height);
                ctxRight.save();
                ctxRight.translate(panX, panY);
                ctxRight.scale(zoom, zoom);

                if (img.complete) {
                    ctxRight.drawImage(img, 0, 0, w, h);
                }

                if (currentItem.scale_box) {
                    const [y0, y1, x0, x1] = currentItem.scale_box;
                    ctxRight.strokeStyle = 'cyan';
                    ctxRight.lineWidth = 1 / zoom;
                    ctxRight.setLineDash([4 / zoom, 4 / zoom]);
                    ctxRight.strokeRect(x0, y0, x1 - x0, y1 - y0);
                    ctxRight.setLineDash([]);
                }
                ctxRight.restore();
            }
        }

        let hoveredCell = null;
        let mouseDownPos = { x: 0, y: 0 };

        panelLeft.addEventListener('mousemove', (e) => {
            if (isDragging) return;
            const rect = panelLeft.getBoundingClientRect();
            const canvasX = (e.clientX - rect.left - panX) / zoom;
            const canvasY = (e.clientY - rect.top - panY) / zoom;

            const cell = findCellAt(canvasX, canvasY);
            if (cell !== hoveredCell) {
                hoveredCell = cell;
                panelLeft.style.cursor = cell ? 'pointer' : 'grab';
                renderAllCanvases();
            }
        });

        panelLeft.addEventListener('mouseleave', () => {
            if (hoveredCell) {
                hoveredCell = null;
                renderAllCanvases();
            }
        });

        // 双屏像素级同步滚轮与平移事件绑定
        [panelLeft, panelRight].forEach(targetPanel => {
            targetPanel.addEventListener('wheel', (e) => {
                e.preventDefault();
                const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
                const rect = targetPanel.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;

                panX = mouseX - (mouseX - panX) * zoomFactor;
                panY = mouseY - (mouseY - panY) * zoomFactor;
                zoom *= zoomFactor;

                renderAllCanvases();
            });

            targetPanel.addEventListener('mousedown', (e) => {
                isDragging = true;
                mouseDownPos = { x: e.clientX, y: e.clientY };
                dragStartX = e.clientX - panX;
                dragStartY = e.clientY - panY;
            });

            targetPanel.addEventListener('mouseup', (e) => {
                if (e.button === 0 && targetPanel === panelLeft) {
                    const distMoved = Math.hypot(e.clientX - mouseDownPos.x, e.clientY - mouseDownPos.y);
                    if (distMoved < 6) { // 纯点击触发圈内直接剔除/保留
                        const rect = panelLeft.getBoundingClientRect();
                        const canvasX = (e.clientX - rect.left - panX) / zoom;
                        const canvasY = (e.clientY - rect.top - panY) / zoom;

                        if (currentTool === 'toggle') {
                            const clickedCell = findCellAt(canvasX, canvasY);
                            if (clickedCell) {
                                toggleCellLocal(clickedCell.id);
                                showToast(rejectedSet.has(clickedCell.id) ? `❌ 已剔除细胞 #${clickedCell.id}` : `✅ 已保留细胞 #${clickedCell.id}`);
                            }
                        } else if (currentTool === 'merge') {
                            mergeBorderAt(canvasX, canvasY);
                        }
                    }
                }
                isDragging = false;
            });

            targetPanel.addEventListener('mousemove', (e) => {
                if (isDragging) {
                    panX = e.clientX - dragStartX;
                    panY = e.clientY - dragStartY;
                    renderAllCanvases();
                }
            });
        });

        window.addEventListener('mouseup', () => { isDragging = false; });
        window.addEventListener('resize', () => { renderAllCanvases(); });
        panelLeft.addEventListener('contextmenu', e => e.preventDefault());
        panelRight.addEventListener('contextmenu', e => e.preventDefault());

        function findCellAt(x, y) {
            if (!currentItem || !currentItem.geom) return null;

            // 1. 精确多边形内部 Ray-casting 判定
            for (const c of currentItem.geom.cells) {
                const contours = currentItem.geom.contours[c.id.toString()] || [];
                for (const poly of contours) {
                    if (pointInPoly(x, y, poly)) return c;
                }
            }

            // 2. 智能容错判定：点击质心/编号/附近 20px 范围内的细胞
            let bestCell = null;
            let minDistance = Infinity;
            for (const c of currentItem.geom.cells) {
                const [cx, cy] = c.centroid;
                const dist = Math.hypot(x - cx, y - cy);
                const approxRadius = Math.max(16, Math.sqrt(c.area / Math.PI) * 1.25);
                if (dist <= approxRadius && dist < minDistance) {
                    minDistance = dist;
                    bestCell = c;
                }
            }
            return bestCell;
        }

        function pointInPoly(x, y, poly) {
            let inside = false;
            for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
                const xi = poly[i][0], yi = poly[i][1];
                const xj = poly[j][0], yj = poly[j][1];
                const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
                if (intersect) inside = !inside;
            }
            return inside;
        }

        async function mergeBorderAt(x, y) {
            showToast('正在消除边框合并细胞...');
            const res = await fetch('/api/merge_border', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: currentItem.index, x: x, y: y })
            });
            const data = await res.json();
            if (data.success) {
                currentItem = data.item;
                rejectedSet = new Set(data.item.rejected_ids || []);
                showToast('✅ 已消除交界边框，成功合并细胞!');
                updateSidebarStats();
                renderCellList();
                renderAllCanvases();
            } else {
                showToast('⚠️ 未检测到要消除的共享边框');
            }
        }

        async function sendAction(actionName) {
            const res = await fetch('/api/action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: currentItem.index, action: actionName })
            });
            const data = await res.json();
            if (data.success) {
                rejectedSet = new Set(data.rejected_ids || []);
                updateSidebarStats();
                renderCellList();
                renderAllCanvases();
            }
        }

        function navField(dir) {
            if (!currentItem) return;
            const target = currentItem.index + dir;
            if (target < 0) {
                showToast('已经是第一张视野');
            } else if (target >= currentItem.total) {
                showToast('已经是最后一张视野！点击“完成并导出全部”即可完成');
            } else {
                loadField(target);
            }
        }

        function updateExportBadge(summary) {
            if (!summary) return;
            const total = summary.length;
            const exportedCount = summary.filter(x => x.exported).length;
            const unexportedCount = total - exportedCount;
            const badgeEl = document.getElementById('export-status-badge');
            
            if (!badgeEl) return;
            if (unexportedCount === 0) {
                badgeEl.style.background = '#059669';
                badgeEl.style.color = '#fff';
                badgeEl.innerText = `🎉 全部 ${total} 张已导出`;
            } else {
                badgeEl.style.background = '#d97706';
                badgeEl.style.color = '#fff';
                badgeEl.innerText = `已导出 ${exportedCount}/${total} (剩余 ${unexportedCount} 张未导出)`;
            }
        }

        function jumpToNextUnexported() {
            if (!currentItem || !currentItem.items_summary) return;
            const summary = currentItem.items_summary;
            const currIdx = currentItem.index;
            
            let targetIdx = -1;
            for (let i = currIdx + 1; i < summary.length; i++) {
                if (!summary[i].exported) { targetIdx = i; break; }
            }
            if (targetIdx === -1) {
                for (let i = 0; i < currIdx; i++) {
                    if (!summary[i].exported) { targetIdx = i; break; }
                }
            }

            if (targetIdx !== -1) {
                showToast(`⏭️ 已跳转到第 ${targetIdx + 1} 张未导出视野: ${summary[targetIdx].image_id}`);
                loadField(targetIdx);
            } else {
                showToast('🎉 完美！所有视野均已成功导出！');
            }
        }

        async function exportOne() {
            try {
                showToast('正在导出当前视野...');
                const res = await fetch('/api/export_one', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ index: currentItem.index })
                });
                const data = await res.json();
                if (data.success) {
                    showToast(`✅ 成功导出视野 [${data.image_id}]`);
                    if (currentItem && currentItem.items_summary) {
                        currentItem.items_summary[currentItem.index].exported = true;
                        updateExportBadge(currentItem.items_summary);
                        loadField(currentItem.index);
                    }
                } else {
                    showToast(`❌ 导出失败: ${data.error || '未知错误'}`);
                }
            } catch (err) {
                showToast(`❌ 导出请求失败: ${err}`);
            }
        }

        async function exportAll() {
            try {
                showToast('正在导出全部视野...');
                const res = await fetch('/api/export_all', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                const data = await res.json();
                if (data.success) {
                    showToast(`🎉 已成功导出全部 ${data.count} 个视野！`);
                    if (currentItem && currentItem.items_summary) {
                        currentItem.items_summary.forEach(x => x.exported = true);
                        updateExportBadge(currentItem.items_summary);
                        loadField(currentItem.index);
                    }
                } else {
                    showToast(`❌ 导出失败: ${data.error || '未知错误'}`);
                }
            } catch (err) {
                showToast(`❌ 导出请求失败: ${err}`);
            }
        }

        function showToast(msg) {
            const t = document.getElementById('toast');
            t.innerText = msg;
            t.style.display = 'block';
            setTimeout(() => { t.style.display = 'none'; }, 2500);
        }

        window.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
            const k = e.key.toLowerCase();
            if (k === 'd') toggleDualView();
            if (k === 'u') jumpToNextUnexported();
            if (k === 'n' || e.key === 'ArrowRight') navField(1);
            if (k === 'p' || e.key === 'ArrowLeft') navField(-1);
            if (k === 'a') sendAction('keep_all');
            if (k === 'b') sendAction('reject_border');
            if (k === 's') sendAction('reject_scalebar');
            if (k === 'r') sendAction('reset');
            if (k === 'i') {
                const el = document.getElementById('show-ids');
                el.checked = !el.checked;
                renderAllCanvases();
            }
            if (k === 'x') {
                const el = document.getElementById('show-rejected');
                el.checked = !el.checked;
                renderAllCanvases();
            }
        });

        loadField(0);
    </script>
</body>
</html>
"""
