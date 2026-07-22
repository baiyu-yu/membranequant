"""Web 前端交互界面：基于 Python 标准库 http.server 提供 REST API 与现代 Canvas Web 界面。

功能特色：
1. 图像平移/缩放：鼠标滚轮放大缩小、按住拖拽平移。
2. 消除细胞边框（合并）：点击或划过两个细胞的交界边框，消除分割线并合并细胞。
3. 点击与侧边栏勾选：点击 Canvas 细胞或在侧边栏 Checkbox 勾选/取消勾选切换保留/剔除。
4. 快速筛选与批量操作：剔除触边细胞、剔除标尺区、全选保留、重置。
5. 通道与显示控制：红/绿/Merge 通道切换，Mask 透明度调节，ID/轮廓/标尺框显隐。
6. 一键导出：导出当前或导出全部，打包生成 masks, overlays, meta 等文件。
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
    """将 [0, 1] float 或 uint8 图像转为 Base64 PNG 字符串。"""
    import matplotlib.pyplot as plt

    img = np.asarray(arr, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)
    buf = io.BytesIO()
    plt.imsave(buf, img, format="png")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def _extract_contours_and_metadata(item: ReviewItem) -> dict:
    """提取当前 ReviewItem 的轮廓、形心、面积与标签元数据。"""
    labels = item.labels.astype(np.int32)
    kept_set = set(item.kept_ids())
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

        # 提取轮廓坐标 [[x, y], ...]
        raw_contours = find_contours(mask.astype(float), 0.5)
        formatted_contours = []
        for c in raw_contours:
            # c 是 (row, col) = (y, x)，前端 Canvas 需要 [x, y]
            formatted_contours.append([[round(float(pt[1]), 1), round(float(pt[0]), 1)] for pt in c[::2]])

        contours_dict[str(cid)] = formatted_contours
        cells_info.append(
            {
                "id": cid,
                "centroid": [round(float(cx), 1), round(float(cy), 1)],
                "area": area,
                "is_kept": cid in kept_set,
                "is_border": cid in border_set,
                "is_scalebar": cid in scalebar_set,
            }
        )

    return {
        "cells": cells_info,
        "contours": contours_dict,
        "width": int(labels.shape[1]),
        "height": int(labels.shape[0]),
    }


class WebReviewServer:
    """CellMask 本地 Web 服务器。"""

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

        if auto_flag_on_start:
            for it in self.items:
                it.apply_auto_flags(border=True, scalebar=True)

    def _get_item_payload(self, idx: int) -> dict:
        if idx < 0 or idx >= len(self.items):
            return {"error": "索引越界"}
        item = self.items[idx]
        item.reviewed = True

        # 生成通道背景图
        display_img = item.image
        bg_red = _arr_to_base64_png(display_img)
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

        geom = _extract_contours_and_metadata(item)
        return {
            "index": idx,
            "total": len(self.items),
            "image_id": item.image_id,
            "channel_used": item.channel_used,
            "model": item.model,
            "diameter": item.diameter if item.diameter else "Auto",
            "notes": item.notes,
            "scale_box": item.scale_box,
            "bg_red": bg_red,
            "bg_green": bg_green,
            "bg_merge": bg_merge,
            "geom": geom,
        }

    def start_and_wait(self) -> list[ReviewItem]:
        """启动 HTTP 服务并自动在浏览器中打开页面，阻塞直到用户导出或关闭。"""
        reviewer_self = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # 禁用标准日志输出

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

                if path == "/api/toggle_cell":
                    cid = int(data.get("cell_id", 0))
                    if cid in item.rejected:
                        item.rejected.remove(cid)
                    else:
                        item.rejected.add(cid)
                    self._send_json({"success": True, "item": reviewer_self._get_item_payload(idx)})
                    return

                if path == "/api/set_cell_status":
                    cid = int(data.get("cell_id", 0))
                    keep = bool(data.get("keep", True))
                    if keep and cid in item.rejected:
                        item.rejected.remove(cid)
                    elif not keep:
                        item.rejected.add(cid)
                    self._send_json({"success": True, "item": reviewer_self._get_item_payload(idx)})
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
                        # 尝试扩大半径搜索相邻细胞
                        cids_wider = find_adjacent_cells_at(item.labels, y, x, radius=12)
                        if len(cids_wider) >= 2:
                            id1, id2 = cids_wider[0], cids_wider[1]
                            merged = item.merge_cells(id1, id2)

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

                    self._send_json({"success": True, "item": reviewer_self._get_item_payload(idx)})
                    return

                if path == "/api/export_one":
                    path_out = None
                    if reviewer_self.on_export_one:
                        path_out = reviewer_self.on_export_one(item)
                    self._send_json(
                        {
                            "success": True,
                            "exported_path": str(path_out) if path_out else None,
                        }
                    )
                    return

                if path == "/api/export_all":
                    paths_out = []
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
                    # 延时关闭服务器
                    threading.Thread(target=reviewer_self.stop).start()
                    return

                self.send_error(404, "Not Found")

        # 寻找可用端口
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
        print(f"  CellMask Web 交互界面已启动!")
        print(f"  访问地址: {url}")
        print(f"  在浏览器中可进行放大、平移、剔除及删除边框合并")
        print(f"==============================================\n")

        # 在新线程运行服务器
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

        # 自动打开浏览器
        try:
            webbrowser.open(url)
        except Exception:
            pass

        # 阻塞等待完成
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
    <title>CellMask 人工筛选与边框编辑</title>
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

        /* Top Header */
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
        .btn-active { background: #3b82f6; border-color: #60a5fa; color: #fff; }

        /* Main Workspace */
        main { flex: 1; display: flex; overflow: hidden; position: relative; }

        /* Left Canvas Container */
        .canvas-container {
            flex: 1; position: relative; background: #0a0c0e; overflow: hidden; cursor: grab; user-select: none;
        }
        .canvas-container:active { cursor: grabbing; }
        canvas { position: absolute; top: 0; left: 0; transform-origin: 0 0; }

        /* Floating Toolbar */
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

        /* Right Sidebar */
        aside {
            width: 320px; background: var(--panel-bg); border-left: 1px solid var(--panel-border);
            display: flex; flex-direction: column; height: 100%; z-index: 5;
        }
        .sidebar-section { padding: 14px; border-bottom: 1px solid var(--panel-border); }
        .section-title { font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-sub); margin-bottom: 10px; letter-spacing: 0.5px; }

        .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 8px; }
        .stat-card { background: #121519; padding: 8px; border-radius: 6px; text-align: center; border: 1px solid rgba(255,255,255,0.05); }
        .stat-num { font-size: 16px; font-weight: bold; margin-top: 2px; }
        .stat-num.keep { color: var(--keep-color); }
        .stat-num.rej { color: var(--reject-color); }

        .btn-group-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }

        /* Cell List */
        .cell-list-wrapper { flex: 1; overflow-y: auto; padding: 10px; }
        .cell-item {
            display: flex; align-items: center; justify-content: space-between; padding: 8px 10px; border-radius: 6px;
            background: #15181d; border: 1px solid transparent; margin-bottom: 6px; cursor: pointer; transition: all 0.12s;
        }
        .cell-item:hover { border-color: rgba(255,255,255,0.15); background: #1e232a; }
        .cell-item.selected { border-color: var(--accent); }
        .cell-info { display: flex; align-items: center; gap: 10px; }
        .cell-tag { font-size: 10px; padding: 1px 5px; border-radius: 3px; background: #374151; color: #d1d5db; }
        .cell-tag.border { background: rgba(239, 68, 68, 0.2); color: #fca5a5; }
        .cell-tag.scalebar { background: rgba(6, 182, 212, 0.2); color: #67e8f9; }

        /* Switch Toggle */
        .toggle-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; font-size: 13px; color: #d1d5db; }
        .switch { position: relative; display: inline-block; width: 34px; height: 18px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #374151; transition: .2s; border-radius: 18px; }
        .slider:before { position: absolute; content: ""; height: 14px; width: 14px; left: 2px; bottom: 2px; background-color: white; transition: .2s; border-radius: 50%; }
        input:checked + .slider { background-color: var(--accent); }
        input:checked + .slider:before { transform: translateX(16px); }

        /* Status Toast */
        #toast {
            position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.85); color: #fff;
            padding: 8px 16px; border-radius: 20px; font-size: 13px; border: 1px solid #374151; display: none; z-index: 100;
        }
    </style>
</head>
<body>

    <header>
        <div class="header-title">
            <span>🔬 CellMask 交互审核与边框编辑</span>
            <span class="badge" id="image-id-tag">载入中...</span>
        </div>
        <div class="top-controls">
            <button class="btn" onclick="navField(-1)">◀ 上一张 (p)</button>
            <span id="page-indicator" style="font-size:13px; color:#9ca3af;">1 / 1</span>
            <button class="btn" onclick="navField(1)">下一张 (n) ▶</button>
            <div style="width: 1px; height: 20px; background: var(--panel-border);"></div>
            <button class="btn" onclick="exportOne()">导出本张</button>
            <button class="btn btn-success" onclick="exportAll()">完成并导出全部</button>
        </div>
    </header>

    <main>
        <!-- Floating Tool Menu -->
        <div class="floating-tools">
            <button class="tool-btn active" id="tool-toggle" onclick="setTool('toggle')" title="切换保留/剔除模式 (点击细胞)">👆</button>
            <button class="tool-btn" id="tool-merge" onclick="setTool('merge')" title="删除分割边框 / 合并细胞模式 (点击分割线)">✂️</button>
            <button class="tool-btn" onclick="resetZoom()" title="重置视角 (Fit Screen)">🔍</button>
        </div>

        <div class="canvas-container" id="canvas-container">
            <canvas id="main-canvas"></canvas>
        </div>

        <aside>
            <!-- Field Info & Stats -->
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

            <!-- Quick Actions -->
            <div class="sidebar-section">
                <div class="section-title">快捷筛选操作</div>
                <div class="btn-group-grid">
                    <button class="btn" onclick="sendAction('reject_border')">剔除触边 (b)</button>
                    <button class="btn" onclick="sendAction('reject_scalebar')">剔除标尺区 (s)</button>
                    <button class="btn" onclick="sendAction('keep_all')">全部保留 (a)</button>
                    <button class="btn" onclick="sendAction('reset')">重置标记 (r)</button>
                </div>
            </div>

            <!-- Visual Settings -->
            <div class="sidebar-section">
                <div class="section-title">显示设置</div>
                <div class="toggle-row">
                    <span>通道背景</span>
                    <select id="channel-select" onchange="changeChannel()" style="background:#2a313d; color:#fff; border:1px solid #374151; border-radius:4px; padding:2px 6px;">
                        <option value="red">红 (DiI)</option>
                        <option value="green">绿 (EGFP)</option>
                        <option value="merge">Merge</option>
                    </select>
                </div>
                <div class="toggle-row">
                    <span>显示细胞编号 (i)</span>
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
                    <span>Mask 不透明度</span>
                    <input type="range" id="opacity-slider" min="0.1" max="0.8" step="0.05" value="0.3" oninput="renderCanvas()" style="width:90px;">
                </div>
            </div>

            <!-- Cell Checklist -->
            <div class="section-title" style="padding: 10px 14px 0 14px;">细胞清单 (勾选保留)</div>
            <div class="cell-list-wrapper" id="cell-list">
                <!-- Cell items rendered dynamically -->
            </div>
        </aside>
    </main>

    <div id="toast">保存成功</div>

    <script>
        let currentItem = null;
        let currentTool = 'toggle'; // 'toggle' | 'merge'
        let currentChannel = 'red';
        let zoom = 1.0;
        let panX = 0, panY = 0;
        let isDragging = false;
        let dragStartX = 0, dragStartY = 0;

        const container = document.getElementById('canvas-container');
        const canvas = document.getElementById('main-canvas');
        const ctx = canvas.getContext('2d');

        const bgImages = { red: new Image(), green: new Image(), merge: new Image() };

        // 载入当前视野
        async function loadField(index = 0) {
            try {
                const res = await fetch(`/api/item?index=${index}`);
                const data = await res.json();
                if (data.error) return alert(data.error);

                currentItem = data;
                document.getElementById('image-id-tag').innerText = data.image_id;
                document.getElementById('page-indicator').innerText = `${data.index + 1} / ${data.total}`;

                if (data.bg_red) bgImages.red.src = data.bg_red;
                if (data.bg_green) bgImages.green.src = data.bg_green;
                if (data.bg_merge) bgImages.merge.src = data.bg_merge;

                bgImages.red.onload = () => {
                    if (index === 0 && zoom === 1.0) resetZoom();
                    renderCanvas();
                };

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
            showToast(tool === 'merge' ? '删除边框模式：点击或划过两个细胞的交界边框进行合并' : '切换模式：点击细胞保留/剔除');
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
            const cells = currentItem.geom.cells;
            const total = cells.length;
            const keep = cells.filter(c => c.is_kept).length;
            const rej = total - keep;

            document.getElementById('stat-total').innerText = total;
            document.getElementById('stat-keep').innerText = keep;
            document.getElementById('stat-rej').innerText = rej;
        }

        function renderCellList() {
            const listEl = document.getElementById('cell-list');
            listEl.innerHTML = '';
            if (!currentItem) return;

            currentItem.geom.cells.forEach(c => {
                const itemEl = document.createElement('div');
                itemEl.className = 'cell-item';
                itemEl.onclick = (e) => {
                    if (e.target.tagName !== 'INPUT') toggleCell(c.id);
                };

                let tagsHtml = '';
                if (c.is_border) tagsHtml += '<span class="cell-tag border">触边</span>';
                if (c.is_scalebar) tagsHtml += '<span class="cell-tag scalebar">标尺</span>';

                itemEl.innerHTML = `
                    <div class="cell-info">
                        <input type="checkbox" ${c.is_kept ? 'checked' : ''} onclick="event.stopPropagation(); setCellStatus(${c.id}, this.checked)">
                        <span style="font-weight:600; font-size:13px; color:${c.is_kept ? '#34d399' : '#f87171'}">#${c.id}</span>
                        ${tagsHtml}
                    </div>
                    <span style="font-size:11px; color:#9ca3af;">${c.area} px</span>
                `;
                listEl.appendChild(itemEl);
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

            // 1. 绘制背景图
            const img = bgImages[currentChannel] && bgImages[currentChannel].complete ? bgImages[currentChannel] : bgImages.red;
            if (img.complete) {
                ctx.drawImage(img, 0, 0, w, h);
            }

            // 2. 绘制标尺框
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

            // 3. 绘制细胞 Mask 填充与轮廓
            currentItem.geom.cells.forEach(c => {
                if (!c.is_kept && !showRejected) return;

                const contours = currentItem.geom.contours[c.id.toString()] || [];
                const color = c.is_kept ? '#22c55e' : '#ef4444';

                contours.forEach(poly => {
                    if (poly.length === 0) return;
                    ctx.beginPath();
                    ctx.moveTo(poly[0][0], poly[0][1]);
                    for (let i = 1; i < poly.length; i++) {
                        ctx.lineTo(poly[i][0], poly[i][1]);
                    }
                    ctx.closePath();

                    // Fill
                    ctx.fillStyle = c.is_kept ? `rgba(34, 197, 94, ${opacity})` : `rgba(239, 68, 68, ${opacity * 0.7})`;
                    ctx.fill();

                    // Contour Border
                    ctx.strokeStyle = color;
                    ctx.lineWidth = (c.is_kept ? 1.8 : 1.2) / zoom;
                    ctx.stroke();
                });

                // Draw ID
                if (showIds) {
                    const [cx, cy] = c.centroid;
                    ctx.font = `bold ${Math.max(10, 12 / zoom)}px sans-serif`;
                    ctx.fillStyle = c.is_kept ? '#a7f3d0' : '#fca5a5';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(c.id.toString(), cx, cy);
                }
            });

            ctx.restore();
        }

        // Canvas 交互
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
                // 中键/右键/Shift 平移
                isDragging = true;
                dragStartX = e.clientX - panX;
                dragStartY = e.clientY - panY;
            } else if (e.button === 0) {
                const rect = container.getBoundingClientRect();
                const canvasX = (e.clientX - rect.left - panX) / zoom;
                const canvasY = (e.clientY - rect.top - panY) / zoom;

                if (currentTool === 'toggle') {
                    const clickedCell = findCellAt(canvasX, canvasY);
                    if (clickedCell) toggleCell(clickedCell.id);
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

        // 后端 API 调用
        async function toggleCell(cellId) {
            const res = await fetch('/api/toggle_cell', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: currentItem.index, cell_id: cellId })
            });
            const data = await res.json();
            if (data.success) {
                currentItem = data.item;
                updateSidebarStats();
                renderCellList();
                renderCanvas();
            }
        }

        async function setCellStatus(cellId, keep) {
            const res = await fetch('/api/set_cell_status', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: currentItem.index, cell_id: cellId, keep: keep })
            });
            const data = await res.json();
            if (data.success) {
                currentItem = data.item;
                updateSidebarStats();
                renderCellList();
                renderCanvas();
            }
        }

        async function mergeBorderAt(x, y) {
            const res = await fetch('/api/merge_border', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: currentItem.index, x: x, y: y })
            });
            const data = await res.json();
            if (data.success) {
                currentItem = data.item;
                showToast('已合并相邻细胞，消除边框!');
                updateSidebarStats();
                renderCellList();
                renderCanvas();
            } else {
                showToast('未检测到要消除的共享边框');
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
                currentItem = data.item;
                updateSidebarStats();
                renderCellList();
                renderCanvas();
            }
        }

        function navField(dir) {
            if (!currentItem) return;
            const target = currentItem.index + dir;
            if (target >= 0 && target < currentItem.total) {
                loadField(target);
            }
        }

        async function exportOne() {
            const res = await fetch('/api/export_one', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ index: currentItem.index })
            });
            const data = await res.json();
            if (data.success) showToast('已成功导出当前视野到 output 目录');
        }

        async function exportAll() {
            if (!confirm("确认导出所有已确认视野并退出 Web 审核？")) return;
            const res = await fetch('/api/export_all', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });
            const data = await res.json();
            if (data.success) {
                alert(`已成功导出 ${data.count} 个文件！您可以关闭本页面。`);
                window.close();
            }
        }

        function showToast(msg) {
            const t = document.getElementById('toast');
            t.innerText = msg;
            t.style.display = 'block';
            setTimeout(() => { t.style.display = 'none'; }, 2000);
        }

        // 键盘快捷键
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

        // 启动
        loadField(0);
    </script>
</body>
</html>
"""
