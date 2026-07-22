"""人工筛选 GUI：点击切换保留/剔除细胞，支持批量标记与导出。

操作说明（窗口打开后）
----------------------
鼠标
  左键点击细胞轮廓内  →  切换 保留 / 剔除
  右键点击细胞        →  仅剔除（取消）

键盘
  n / →     下一张（先保存当前筛选状态到内存）
  p / ←     上一张
  a         全部保留（取消所有剔除）
  b         剔除所有触边细胞
  s         剔除与右下角标尺重叠的细胞
  r         撤销：恢复到分割后的初始状态
  e         导出当前这张（写入磁盘）
  w         导出全部
  q / Esc   结束并导出全部已确认的结果
  h         在终端打印帮助
  i         开关细胞编号
  x         开关已剔除轮廓的显示

显示
  绿色轮廓 = 保留
  红色轮廓 = 已剔除
  青色虚线框 = 右下角标尺屏蔽区
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.patches import Rectangle
from skimage.measure import find_contours, regionprops
from skimage.segmentation import find_boundaries  # used by build_rgb_overlay

from .fonts import apply_font_to_axes, get_chinese_font_prop, setup_chinese_font


@dataclass
class ReviewItem:
    """单张图的分割 + 筛选状态。"""

    image_id: str
    image: np.ndarray  # 显示用 float [0,1]
    labels: np.ndarray  # 原始 label（分割后不变）
    red_path: Path | None = None
    green_path: Path | None = None
    merge_path: Path | None = None
    channel_used: str = "red"
    model: str = "cyto3"
    diameter: float | None = None
    border_ids: list[int] = field(default_factory=list)
    scalebar_ids: list[int] = field(default_factory=list)
    scale_box: tuple[int, int, int, int] | None = None
    # 筛选状态
    rejected: set[int] = field(default_factory=set)
    reviewed: bool = False
    notes: str = ""

    def all_ids(self) -> list[int]:
        return [int(x) for x in np.unique(self.labels) if x != 0]

    def kept_ids(self) -> list[int]:
        return [i for i in self.all_ids() if i not in self.rejected]

    def apply_auto_flags(self, border: bool = True, scalebar: bool = True) -> None:
        if border:
            self.rejected.update(self.border_ids)
        if scalebar:
            self.rejected.update(self.scalebar_ids)

    def reset_flags(self) -> None:
        self.rejected = set()
        self.apply_auto_flags(border=True, scalebar=True)

    def merge_cells(self, id1: int, id2: int) -> bool:
        """合并两个细胞：将 id2 合并进 id1，消除分隔边框。"""
        id1, id2 = int(id1), int(id2)
        if id1 == id2 or id1 <= 0 or id2 <= 0:
            return False
        if not np.any(self.labels == id1) or not np.any(self.labels == id2):
            return False
        self.labels[self.labels == id2] = id1
        if id2 in self.rejected:
            self.rejected.remove(id2)
        if id2 in self.border_ids and id1 not in self.border_ids:
            self.border_ids.append(id1)
        if id2 in self.border_ids:
            self.border_ids = [x for x in self.border_ids if x != id2]
        if id2 in self.scalebar_ids and id1 not in self.scalebar_ids:
            self.scalebar_ids.append(id1)
        if id2 in self.scalebar_ids:
            self.scalebar_ids = [x for x in self.scalebar_ids if x != id2]
        return True

    def filtered_labels(self) -> np.ndarray:
        from .segment import renumber_labels

        return renumber_labels(self.labels, self.kept_ids())



class MaskReviewer:
    """交互式人工筛选器。"""

    def __init__(
        self,
        items: list[ReviewItem],
        *,
        on_export_one: Callable[[ReviewItem], Path | None] | None = None,
        on_export_all: Callable[[list[ReviewItem]], list[Path]] | None = None,
        auto_flag_on_start: bool = True,
        title_prefix: str = "CellMask 人工筛选",
    ):
        if not items:
            raise ValueError("没有可筛选的图像")

        # 必须在创建 Figure 之前配置中文字体，否则标题/状态栏会缺字刷屏
        setup_chinese_font(silent=False)
        self._font_prop = get_chinese_font_prop()

        self.items = items
        self.index = 0
        self.on_export_one = on_export_one
        self.on_export_all = on_export_all
        self.title_prefix = title_prefix
        self._cid_click = None
        self._cid_key = None
        self._finished = False
        self._show_ids = True
        self._show_rejected = True

        if auto_flag_on_start:
            for it in self.items:
                it.apply_auto_flags(border=True, scalebar=True)

        self.fig, self.ax = plt.subplots(figsize=(11, 9))
        try:
            self.fig.canvas.manager.set_window_title(title_prefix)
        except Exception:
            pass
        self._overlay_artists: list = []
        self._status_text = None
        self._help_shown = False

    def _text_kwargs(self, **extra) -> dict:
        """带中文字体的 text/title 参数。"""
        kw = dict(extra)
        if self._font_prop is not None:
            kw["fontproperties"] = self._font_prop
        return kw

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self) -> list[ReviewItem]:
        """阻塞运行，关闭窗口或按 q 后返回全部 items。"""
        self._cid_click = self.fig.canvas.mpl_connect(
            "button_press_event", self._on_click
        )
        self._cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._draw()
        print(self._help_text())
        plt.show()
        # 关闭后标记当前为已审阅
        if 0 <= self.index < len(self.items):
            self.items[self.index].reviewed = True
        return self.items

    def _help_text(self) -> str:
        return (
            "\n========== CellMask 人工筛选 ==========\n"
            "左键: 切换保留/剔除   右键: 剔除\n"
            "n/→: 下一张   p/←: 上一张\n"
            "a: 全部保留   b: 剔触边   s: 剔标尺区\n"
            "r: 重置本张   e: 导出本张   w: 导出全部\n"
            "i: 开关编号   x: 显隐剔除轮廓   q: 结束并导出全部\n"
            "========================================\n"
        )

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------
    def _current(self) -> ReviewItem:
        return self.items[self.index]

    def _draw(self) -> None:
        item = self._current()
        self.ax.clear()
        self._overlay_artists.clear()

        img = item.image
        if img.ndim == 2:
            self.ax.imshow(img, cmap="gray", interpolation="nearest")
        else:
            self.ax.imshow(img, interpolation="nearest")

        # 标尺区域提示
        if item.scale_box is not None:
            y0, y1, x0, x1 = item.scale_box
            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=1.0,
                edgecolor="cyan",
                facecolor="none",
                linestyle="--",
                alpha=0.7,
                label="scale bar zone",
            )
            self.ax.add_patch(rect)

        labels = item.labels
        kept = set(item.kept_ids())
        rejected = set(item.rejected)

        # 轮廓 + 编号（按连通域一次 regionprops）
        props_all = regionprops(labels.astype(np.int32))
        centroids = {int(p.label): p.centroid for p in props_all}

        for cid in item.all_ids():
            is_kept = cid in kept
            if not is_kept and not self._show_rejected:
                continue
            mask = labels == cid
            if not np.any(mask):
                continue
            color = "#33f24a" if is_kept else "#ff3333"
            lw = 1.5 if is_kept else 1.1
            ls = "-" if is_kept else "--"
            for contour in find_contours(mask.astype(float), 0.5):
                self.ax.plot(
                    contour[:, 1],
                    contour[:, 0],
                    color=color,
                    linewidth=lw,
                    linestyle=ls,
                    alpha=0.95,
                )

            if self._show_ids and cid in centroids:
                cy, cx = centroids[cid]
                txt_color = "lime" if is_kept else "tomato"
                self.ax.text(
                    cx,
                    cy,
                    str(cid),
                    color=txt_color,
                    fontsize=8,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    alpha=0.95,
                    **self._text_kwargs(),
                )

        n_keep = len(kept)
        n_rej = len(rejected)
        n_all = len(item.all_ids())
        title = (
            f"{self.title_prefix}  [{self.index + 1}/{len(self.items)}]  "
            f"{item.image_id}\n"
            f"通道={item.channel_used}  模型={item.model}  "
            f"直径={item.diameter if item.diameter else 'Auto'}  "
            f"保留 {n_keep}/{n_all}  剔除 {n_rej}"
        )
        if item.notes:
            title += f"\n{item.notes}"
        self.ax.set_title(title, fontsize=11, **self._text_kwargs())
        self.ax.set_axis_off()

        status = (
            "左键切换 | 右键剔除 | n下一张 p上一张 | b触边 s标尺 | "
            "a全保留 r重置 | e导出本张 w全部 | q结束"
        )
        # supxlabel 对 fontproperties 支持因版本而异，写完再强制套字体
        self.fig.supxlabel(status, fontsize=9, color="0.35")
        apply_font_to_axes(self.ax, self.fig)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------
    def _label_at(self, x: float, y: float) -> int:
        item = self._current()
        h, w = item.labels.shape
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            return 0
        return int(item.labels[yi, xi])

    def _on_click(self, event: MouseEvent) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        cid = self._label_at(event.xdata, event.ydata)
        if cid == 0:
            return
        item = self._current()
        if event.button == 1:
            if cid in item.rejected:
                item.rejected.discard(cid)
            else:
                item.rejected.add(cid)
        elif event.button == 3:
            item.rejected.add(cid)
        item.reviewed = True
        self._draw()

    def _on_key(self, event: KeyEvent) -> None:
        key = (event.key or "").lower()
        item = self._current()

        if key in ("n", "right"):
            item.reviewed = True
            if self.index < len(self.items) - 1:
                self.index += 1
                self._draw()
            else:
                print("已经是最后一张。按 q 结束，或 E 导出全部。")
        elif key in ("p", "left"):
            item.reviewed = True
            if self.index > 0:
                self.index -= 1
                self._draw()
            else:
                print("已经是第一张。")
        elif key == "a":
            item.rejected.clear()
            item.reviewed = True
            self._draw()
            print(f"[{item.image_id}] 已全部保留")
        elif key == "b":
            item.rejected.update(item.border_ids)
            item.reviewed = True
            self._draw()
            print(f"[{item.image_id}] 已剔除触边细胞: {item.border_ids}")
        elif key == "s":
            item.rejected.update(item.scalebar_ids)
            item.reviewed = True
            self._draw()
            print(f"[{item.image_id}] 已剔除标尺区细胞: {item.scalebar_ids}")
        elif key == "r":
            item.reset_flags()
            item.reviewed = True
            self._draw()
            print(f"[{item.image_id}] 已重置为自动标记状态")
        elif key == "i":
            self._show_ids = not self._show_ids
            self._draw()
        elif key == "e":
            item.reviewed = True
            if self.on_export_one:
                path = self.on_export_one(item)
                print(f"已导出: {path}" if path else "导出失败")
            else:
                print("未设置单张导出回调")
        elif key in ("w", "shift+e"):
            if self.on_export_all:
                paths = self.on_export_all(self.items)
                print(f"已导出 {len(paths)} 个文件")
            else:
                print("未设置全部导出回调")
        elif key in ("q", "escape"):
            item.reviewed = True
            if self.on_export_all:
                paths = self.on_export_all(self.items)
                print(f"结束。已导出 {len(paths)} 个文件。")
            self._finished = True
            plt.close(self.fig)
        elif key == "h":
            print(self._help_text())
        elif key == "x":
            # 切换是否显示已剔除轮廓
            self._show_rejected = not self._show_rejected
            self._draw()


def build_rgb_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    rejected: set[int] | None = None,
) -> np.ndarray:
    """生成 RGB 叠加预览图（用于保存 overlay）。"""
    rejected = rejected or set()
    base = np.asarray(image, dtype=np.float32)
    if base.ndim == 2:
        rgb = np.stack([base, base, base], axis=-1)
    else:
        rgb = base[..., :3].copy()
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
    rgb = np.clip(rgb, 0, 1)

    boundaries_keep = np.zeros(base.shape[:2], dtype=bool)
    boundaries_rej = np.zeros(base.shape[:2], dtype=bool)
    for cid in np.unique(labels):
        cid = int(cid)
        if cid == 0:
            continue
        mask = labels == cid
        b = find_boundaries(mask, mode="outer")
        if cid in rejected:
            boundaries_rej |= b
        else:
            boundaries_keep |= b

    out = rgb.copy()
    out[boundaries_keep] = (0.15, 1.0, 0.25)
    out[boundaries_rej] = (1.0, 0.2, 0.2)
    return np.clip(out, 0, 1)
