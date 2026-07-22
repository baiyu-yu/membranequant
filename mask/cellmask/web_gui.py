"""Web 前端交互界面：极速响应版 REST API + Canvas 交互。

性能与体验优化：
1. 0 延迟本地即时更新：点击/勾选取消细胞瞬间完成（0ms 延迟），无网络卡顿。
2. 背景图与轮廓缓存：避免重复生成 Base64 PNG 和计算 find_contours，大幅降低 CPU 开销。
3. 状态异步同步：取消勾选时仅传输几字节 ID 状态，彻底解决连续勾选被重置的竞争问题。
4. 明确的导出提示：点击“导出本张”弹出明确的提示框与导出路径。
5. 搜索过滤：支持在侧边栏搜索框快速过滤 Cell ID。
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


def _arr_to_base64_png(arr: np.ndarray) -> str:
    """将 [0, 1] float 图像转为 Base64 PNG 字符串。"""
    img = np.asarray(arr, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)
    img_uint8 = (img * 255).astype(np.uint8)

    try:
        from PIL import Image

        buf = io.BytesIO()
        if img_uint8.ndim == 2:
            Image.fromarray(img_uint8, mode="L").save(buf, format="PNG")
        else:
            Image.fromarray(img_uint8).save(buf, format="PNG")
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
    except ImportError:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        buf = io.BytesIO()
        plt.imsave(buf, img, format="png")
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


class WebReviewServer:
    """CellMask 本地 Web 服务器（极速缓存版）。"""

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

        # 内存缓存
        self._bg_cache: dict[str, dict[str, str]] = {}
        self._geom_cache: dict[str, dict] = {}

        if auto_flag_on_start:
            for it in self.items:
                it.apply_auto_flags(border=True, scalebar=True)

    def _get_bg_images(self, item: ReviewItem) -> dict[str, str | None]:
        key = item.image_id
        if key in self._bg_cache:
            return self._bg_cache[key]

        bg_red = _arr_to_base64_png(item.image)
        bg_green = None
        bg_merge = None

        if item.green_path and item.green_path.is_file():
            try:
                import tifffile

                g_arr = tifffile.imread(str(item.green_path)).astype(np.float32)
                g_arr = (g_arr - g_arr.min()) / (g_arr.max() - g_arr.min() + 1e-6)
                bg_green = _arr_to_base64_png(g_arr)
            except Exception:
                pass

        if item.merge_path and item.merge_path.is_file():
            try:
                import tifffile

                m_arr = tifffile.imread(str(item.merge_path)).astype(np.float32)
                m_arr = (m_arr - m_arr.min()) / (m_arr.max() - m_arr.min() + 1e-6)
                bg_merge = _arr_to_base64_png(m_arr)
            except Exception:
                pass

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
            mask = labels == cid

            raw_contours = find_contours(mask.astype(float), 0.5)
            formatted_contours = []
            for c in raw_contours:
                formatted_contours.append([[round(float(pt[1]), 1), round(float(pt[0]), 1)] for pt in c[::2]])

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

                # 轻量级切换接口：仅更新并返回 rejected 集合，不重发大图片
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
                        # 重新计算 geom 缓存
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
        print(f"  CellMask Web 极速交互界面已启动!")
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
    <title>CellMask 极速审核与边框编辑</title>
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

        .canvas-container {
            flex: 1; position: relative; background: #0a0c0e; overflow: hidden; cursor: grab; user-select: none;
        }
        .canvas-container:active { cursor: grabbing; }
        canvas { position: absolute; top: 0; left: 0; transform-origin: 0 0; }

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
            <span id="page-indicator" style="font-size:13px; color:#9ca3af;">1 / 1</span>
            <button class="btn" onclick="navField(1)">下一张 (n) ▶</button>
            <div style="width: 1px; height: 20px; background: var(--panel-border);"></div>
            <button class="btn btn-primary" onclick="exportOne()">导出本张</button>
            <button class="btn btn-success" onclick="exportAll()">完成并导出全部</button>
        </div>
    </header>

    <main>
        <div class="floating-tools">
            <button class="tool-btn active" id="tool-toggle" onclick="setTool('toggle')" title="切换保留/剔除模式 (点击细胞)">👆</button>
            <button class="tool-btn" id="tool-merge" onclick="setTool('merge')" title="删除分割边框 / 合并细胞模式 (点击分割线)">✂️</button>
            <button class="tool-btn" onclick="resetZoom()" title="重置视角 (Fit Screen)">🔍</button>
        </div>

        <div class="canvas-container" id="canvas-container">
            <canvas id="main-canvas"></canvas>
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
                <div class="section-title">显示设置</div>
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
                        <input type="checkbox" id="show-ids" checked onchange="renderCanvas()">
                        <span class="slider"></span>
                    </label>
                </div>
                <div class="toggle-row">
                    <span>显示剔除轮廓 (x)</span>
                    <label class="switch">
                        <input type="checkbox" id="show-rejected" checked onchange="renderCanvas()">
                        <span class="slider"></span>
                    </label>
                </div>
                <div class="toggle-row">
                    <span>Mask 透明度</span>
                    <input type="range" id="opacity-slider" min="0.1" max="0.8" step="0.05" value="0.3" oninput="renderCanvas()" style="width:90px;">
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
        let rejectedSet = new Set(); // 本地权威 剔除 Set，0 延迟更新
        let currentTool = 'toggle';
        let currentChannel = 'red';
        let zoom = 1.0;
        let panX = 0, panY = 0;
        let isDragging = false;
        let dragStartX = 0, dragStartY = 0;

        const container = document.getElementById('canvas-container');
        const canvas = document.getElementById('main-canvas');
        const ctx = canvas.getContext('2d');
        const bgImages = { red: new Image(), green: new Image(), merge: new Image() };

        async function loadField(index = 0) {
            try {
                const res = await fetch(`/api/item?index=${index}`);
                const data = await res.json();
                if (data.error) return alert(data.error);

                currentItem = data;
                rejectedSet = new Set(data.rejected_ids || []);

                if (data.bg_red) {
                    bgImages.red.onload = () => renderCanvas();
                    bgImages.red.src = data.bg_red;
                }
                if (data.bg_green) {
                    bgImages.green.onload = () => renderCanvas();
                    bgImages.green.src = data.bg_green;
                } else {
                    bgImages.green.src = '';
                }
                if (data.bg_merge) {
                    bgImages.merge.onload = () => renderCanvas();
                    bgImages.merge.src = data.bg_merge;
                } else {
                    bgImages.merge.src = '';
                }

                resetZoom();
                updateSidebarStats();
                renderCellList();
                renderCanvas();
            } catch (err) {
                console.error("加载失败", err);
            }
        }

        function setTool(tool) {
            currentTool = tool;
            document.getElementById('tool-toggle').classList.toggle('active', tool === 'toggle');
            document.getElementById('tool-merge').classList.toggle('active', tool === 'merge');
            showToast(tool === 'merge' ? '删除边框模式：点击或划过两个细胞交界边框合并' : '切换模式：点击细胞保留/剔除');
        }

        function changeChannel() {
            currentChannel = document.getElementById('channel-select').value;
            renderCanvas();
        }

        function resetZoom() {
            if (!currentItem) return;
            const w = currentItem.geom.width;
            const h = currentItem.geom.height;
            const cw = container.clientWidth;
            const ch = container.clientHeight;

            const scale = Math.min(cw / w, ch / h) * 0.92;
            zoom = scale;
            panX = (cw - w * scale) / 2;
            panY = (ch - h * scale) / 2;
            renderCanvas();
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

        // 0ms 纯前端即时更新
        function toggleCellLocal(cellId) {
            if (rejectedSet.has(cellId)) {
                rejectedSet.delete(cellId);
            } else {
                rejectedSet.add(cellId);
            }
            updateCellRowUI(cellId);
            updateSidebarStats();
            renderCanvas();
            // 后台异步同步给后端
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
            renderCanvas();
            // 后台异步同步
            fetch('/api/set_cell_status', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: currentItem.index, cell_id: cellId, keep: keep })
            });
        }

        function renderCanvas() {
            if (!currentItem) return;

            const w = currentItem.geom.width;
            const h = currentItem.geom.height;
            canvas.width = container.clientWidth;
            canvas.height = container.clientHeight;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.save();
            ctx.translate(panX, panY);
            ctx.scale(zoom, zoom);

            const img = bgImages[currentChannel] && bgImages[currentChannel].complete ? bgImages[currentChannel] : bgImages.red;
            if (img.complete) {
                ctx.drawImage(img, 0, 0, w, h);
            }

            if (currentItem.scale_box) {
                const [y0, y1, x0, x1] = currentItem.scale_box;
                ctx.strokeStyle = 'cyan';
                ctx.lineWidth = 1 / zoom;
                ctx.setLineDash([4 / zoom, 4 / zoom]);
                ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
                ctx.setLineDash([]);
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
                    ctx.beginPath();
                    ctx.moveTo(poly[0][0], poly[0][1]);
                    for (let i = 1; i < poly.length; i++) {
                        ctx.lineTo(poly[i][0], poly[i][1]);
                    }
                    ctx.closePath();

                    ctx.fillStyle = isKept ? `rgba(34, 197, 94, ${opacity})` : `rgba(239, 68, 68, ${opacity * 0.7})`;
                    ctx.fill();

                    ctx.strokeStyle = color;
                    ctx.lineWidth = (isKept ? 1.8 : 1.2) / zoom;
                    ctx.stroke();
                });

                if (showIds) {
                    const [cx, cy] = c.centroid;
                    ctx.font = `bold ${Math.max(10, 12 / zoom)}px sans-serif`;
                    ctx.fillStyle = isKept ? '#a7f3d0' : '#fca5a5';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(c.id.toString(), cx, cy);
                }
            });

            ctx.restore();
        }

        container.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
            const mouseX = e.clientX - container.getBoundingClientRect().left;
            const mouseY = e.clientY - container.getBoundingClientRect().top;

            panX = mouseX - (mouseX - panX) * zoomFactor;
            panY = mouseY - (mouseY - panY) * zoomFactor;
            zoom *= zoomFactor;

            renderCanvas();
        });

        container.addEventListener('mousedown', (e) => {
            if (e.button === 1 || e.button === 2 || e.shiftKey) {
                isDragging = true;
                dragStartX = e.clientX - panX;
                dragStartY = e.clientY - panY;
            } else if (e.button === 0) {
                const rect = container.getBoundingClientRect();
                const canvasX = (e.clientX - rect.left - panX) / zoom;
                const canvasY = (e.clientY - rect.top - panY) / zoom;

                if (currentTool === 'toggle') {
                    const clickedCell = findCellAt(canvasX, canvasY);
                    if (clickedCell) toggleCellLocal(clickedCell.id);
                } else if (currentTool === 'merge') {
                    mergeBorderAt(canvasX, canvasY);
                }
            }
        });

        container.addEventListener('mousemove', (e) => {
            if (isDragging) {
                panX = e.clientX - dragStartX;
                panY = e.clientY - dragStartY;
                renderCanvas();
            }
        });

        window.addEventListener('mouseup', () => { isDragging = false; });
        container.addEventListener('contextmenu', e => e.preventDefault());

        function findCellAt(x, y) {
            if (!currentItem) return null;
            for (const c of currentItem.geom.cells) {
                const contours = currentItem.geom.contours[c.id.toString()] || [];
                for (const poly of contours) {
                    if (pointInPoly(x, y, poly)) return c;
                }
            }
            return null;
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
                renderCanvas();
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
                renderCanvas();
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
                    const msg = `🎉 成功导出当前视野 [${data.image_id}]!\n\n导出路径: ${data.exported_path}`;
                    alert(msg);
                    showToast('✅ 导出成功!');
                } else {
                    alert(`导出失败: ${data.error || '未知错误'}`);
                }
            } catch (err) {
                alert(`导出请求失败: ${err}`);
            }
        }

        async function exportAll() {
            if (!confirm("确认导出所有已确认视野并结束？")) return;
            try {
                showToast('正在导出全部视野...');
                const res = await fetch('/api/export_all', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                const data = await res.json();
                if (data.success) {
                    alert(`🎉 已成功导出全部 ${data.count} 个视野文件！您现在可以关闭本页面。`);
                    window.close();
                } else {
                    alert(`导出失败: ${data.error || '未知错误'}`);
                }
            } catch (err) {
                alert(`导出请求失败: ${err}`);
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
            if (k === 'n' || e.key === 'ArrowRight') navField(1);
            if (k === 'p' || e.key === 'ArrowLeft') navField(-1);
            if (k === 'a') sendAction('keep_all');
            if (k === 'b') sendAction('reject_border');
            if (k === 's') sendAction('reject_scalebar');
            if (k === 'r') sendAction('reset');
            if (k === 'i') {
                const el = document.getElementById('show-ids');
                el.checked = !el.checked;
                renderCanvas();
            }
            if (k === 'x') {
                const el = document.getElementById('show-rejected');
                el.checked = !el.checked;
                renderCanvas();
            }
        });

        loadField(0);
    </script>
</body>
</html>
"""
