"""为 Matplotlib 配置中文字体，避免 DejaVu Sans 缺字警告。"""

from __future__ import annotations

import os
import warnings
from functools import lru_cache
from pathlib import Path

# 优先顺序：微软雅黑 / 黑体 / 宋体 / 等线 / Noto
_FONT_CANDIDATES: list[tuple[str, str]] = [
    # (显示名提示, 文件名)
    ("Microsoft YaHei", "msyh.ttc"),
    ("Microsoft YaHei", "msyh.ttf"),
    ("SimHei", "simhei.ttf"),
    ("SimSun", "simsun.ttc"),
    ("DengXian", "Deng.ttf"),
    ("Noto Sans SC", "NotoSansSC-VF.ttf"),
    ("Noto Sans CJK SC", "NotoSansCJKsc-Regular.otf"),
    ("Source Han Sans SC", "SourceHanSansSC-Regular.otf"),
    ("STXihei", "STXIHEI.TTF"),
    ("STSong", "STSONG.TTF"),
]

_CONFIGURED = False
_FONT_NAME: str | None = None
_FONT_PATH: str | None = None


def _windows_font_dirs() -> list[Path]:
    dirs: list[Path] = []
    windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or r"C:\Windows"
    dirs.append(Path(windir) / "Fonts")
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
    if local.is_dir():
        dirs.append(local)
    return dirs


def _find_font_file() -> Path | None:
    for _name, filename in _FONT_CANDIDATES:
        for d in _windows_font_dirs():
            p = d / filename
            if p.is_file():
                return p
    # 再扫一遍：任意含 YaHei / SimHei 的文件
    keywords = ("msyh", "simhei", "simsun", "noto", "sourcehan", "deng")
    for d in _windows_font_dirs():
        try:
            for p in d.iterdir():
                low = p.name.lower()
                if p.suffix.lower() in {".ttf", ".ttc", ".otf"} and any(
                    k in low for k in keywords
                ):
                    return p
        except OSError:
            continue
    return None


@lru_cache(maxsize=1)
def get_chinese_font_prop():
    """返回可用的 FontProperties；找不到中文字体则返回 None。"""
    from matplotlib import font_manager

    path = _find_font_file()
    if path is None:
        return None
    try:
        font_manager.fontManager.addfont(str(path))
    except Exception:
        # 部分旧版 matplotlib 对 ttc 的 addfont 支持不完整
        pass
    try:
        return font_manager.FontProperties(fname=str(path))
    except Exception:
        return None


def setup_chinese_font(*, silent: bool = False) -> str | None:
    """配置全局 rcParams，使标题/标签能显示中文。

    Returns
    -------
    成功时返回字体名或路径；失败返回 None。
    """
    global _CONFIGURED, _FONT_NAME, _FONT_PATH
    if _CONFIGURED:
        return _FONT_NAME or _FONT_PATH

    import matplotlib
    from matplotlib import font_manager

    # 负号用 ASCII，避免与中文字体冲突变成方块
    matplotlib.rcParams["axes.unicode_minus"] = False

    path = _find_font_file()
    if path is None:
        if not silent:
            warnings.warn(
                "未找到中文字体（微软雅黑/黑体等），界面中文可能显示为方块。"
                "请确认 C:\\Windows\\Fonts 下存在 msyh.ttc 或 simhei.ttf。",
                UserWarning,
                stacklevel=2,
            )
        _CONFIGURED = True
        return None

    _FONT_PATH = str(path)
    prop = None
    try:
        font_manager.fontManager.addfont(_FONT_PATH)
        prop = font_manager.FontProperties(fname=_FONT_PATH)
        _FONT_NAME = prop.get_name()
    except Exception:
        # 回退：按家族名
        _FONT_NAME = path.stem
        prop = None

    # 把中文字体放到 sans-serif 最前
    families = list(matplotlib.rcParams.get("font.sans-serif", []))
    prefer: list[str] = []
    if _FONT_NAME:
        prefer.append(_FONT_NAME)
    prefer.extend(
        [
            "Microsoft YaHei",
            "Microsoft YaHei UI",
            "SimHei",
            "SimSun",
            "DengXian",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
    )
    # 去重保序
    seen: set[str] = set()
    merged: list[str] = []
    for f in prefer + families:
        if f and f not in seen:
            seen.add(f)
            merged.append(f)
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = merged

    # 明确指定默认字体文件（最稳）
    if prop is not None:
        try:
            matplotlib.rcParams["font.sans-serif"] = [_FONT_NAME] + [
                x for x in merged if x != _FONT_NAME
            ]
        except Exception:
            pass

    # 压掉残余缺字警告（字体已配好时极少触发；配失败时也少刷屏）
    warnings.filterwarnings(
        "ignore",
        message=r".*Glyph .* missing from font.*",
        category=UserWarning,
    )

    _CONFIGURED = True
    if not silent:
        print(f"[字体] 已启用中文字体: {_FONT_NAME or path.name} ({path})")
    return _FONT_NAME or _FONT_PATH


def apply_font_to_axes(ax, fig=None) -> None:
    """对已有 Axes/Figure 上的文字强制套用中文字体。"""
    prop = get_chinese_font_prop()
    if prop is None:
        return

    for label in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]:
        try:
            label.set_fontproperties(prop)
        except Exception:
            pass
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        try:
            t.set_fontproperties(prop)
        except Exception:
            pass
    if fig is not None:
        for text in fig.texts:
            try:
                text.set_fontproperties(prop)
            except Exception:
                pass
        # supxlabel / suptitle
        for attr in ("_suptitle", "_supxlabel", "_supylabel"):
            obj = getattr(fig, attr, None)
            if obj is not None:
                try:
                    obj.set_fontproperties(prop)
                except Exception:
                    pass
