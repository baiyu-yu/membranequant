"""扫描实验目录、解析文件名、配对红/绿通道。

命名约定
--------
``[前缀_]{实验}{药物}{组别}-{视野}[-{通道}].tif``

示例::

    C2_104d1-1.tif      # Merge（可选）
    C2_104d1-1-1.tif    # 红通道 DiI / 膜标记
    C2_104d1-1-2.tif    # 绿通道 EGFP
    C2_we2-3.tif        # 实验 w，药物 e，组 2，视野 3
    C2_we2-3-1.tif      # 同上红通道
    C2_we2-3-2.tif      # 同上绿通道

通道码：``-1`` = 红，``-2`` = 绿，无后缀 = Merge。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile

# C2_104d1-1-1  /  C2_wd1-3-2  /  we2-3  /  104d1-1
FILENAME_RE = re.compile(
    r"^(?P<prefix>[A-Za-z0-9]+_)?"
    r"(?P<experiment>[A-Za-z0-9]+?)"
    r"(?P<drug>[A-Za-z])"
    r"(?P<group>\d+)"
    r"-(?P<field>\d+)"
    r"(?:-(?P<channel>\d+))?$",
    re.IGNORECASE,
)

IMAGE_EXTS = {".tif", ".tiff", ".TIF", ".TIFF"}


@dataclass(frozen=True)
class ParsedName:
    stem: str
    experiment: str
    drug: str
    group: str
    field: int
    channel: str | None  # "red" | "green" | "merge" | "chN"
    path: Path

    @property
    def condition_id(self) -> str:
        return f"{self.experiment}{self.drug}{self.group}"

    @property
    def base_key(self) -> str:
        """同一视野共用键，如 104d1-1 / we2-3。"""
        return f"{self.condition_id}-{self.field}"

    @property
    def image_id(self) -> str:
        return self.base_key


@dataclass
class FieldImages:
    """一个视野的红/绿/Merge 路径。"""

    experiment: str
    drug: str
    group: str
    field: int
    red_path: Path | None = None
    green_path: Path | None = None
    merge_path: Path | None = None

    @property
    def image_id(self) -> str:
        return f"{self.experiment}{self.drug}{self.group}-{self.field}"

    @property
    def condition_id(self) -> str:
        return f"{self.experiment}{self.drug}{self.group}"

    def has_red(self) -> bool:
        return self.red_path is not None

    def has_any(self) -> bool:
        return any(p is not None for p in (self.red_path, self.green_path, self.merge_path))

    def describe(self) -> str:
        parts = [self.image_id]
        if self.red_path:
            parts.append("R")
        if self.green_path:
            parts.append("G")
        if self.merge_path:
            parts.append("M")
        return " ".join(parts)


@dataclass
class ScanReport:
    fields: list[FieldImages]
    all_tifs: int
    parsed: int
    unparsed: list[str] = field(default_factory=list)
    skipped_no_red: list[str] = field(default_factory=list)


def parse_filename(path: Path) -> ParsedName | None:
    stem = path.stem
    match = FILENAME_RE.match(stem)
    if not match:
        return None

    ch_code = match.group("channel")
    if ch_code is None:
        channel = "merge"
    elif ch_code == "1":
        channel = "red"
    elif ch_code == "2":
        channel = "green"
    else:
        channel = f"ch{ch_code}"

    return ParsedName(
        stem=stem,
        experiment=match.group("experiment"),
        drug=match.group("drug").lower(),
        group=match.group("group"),
        field=int(match.group("field")),
        channel=channel,
        path=path,
    )


def iter_tifs(root: Path) -> list[Path]:
    """递归收集 .tif / .tiff（仅文件，跳过 MetaData 等）。"""
    root = Path(root)
    found: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in IMAGE_EXTS:
            continue
        # 跳过明显的非实验图
        name_lower = p.name.lower()
        if "metadata" in str(p.parent).lower():
            continue
        if name_lower.endswith(("lut.png", "logo.jpg", "logo.png")):
            continue
        found.append(p)

    seen: set[Path] = set()
    unique: list[Path] = []
    for p in sorted(found, key=lambda x: str(x).lower()):
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def scan_fields(root: Path, require_red: bool = True) -> ScanReport:
    """扫描目录并按视野配对通道。

    Parameters
    ----------
    root : 实验根目录，如 ``D:\\课题同步\\...\\3B_7.20``
    require_red : 若 True，只保留至少有红通道（或可从 merge 提取）的视野。
                  Merge 单独存在时也会保留，分割时会尝试拆通道。
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root}")

    tifs = iter_tifs(root)
    by_key: dict[str, FieldImages] = {}
    parsed_count = 0
    unparsed: list[str] = []

    for path in tifs:
        parsed = parse_filename(path)
        if parsed is None:
            try:
                rel = str(path.relative_to(root))
            except ValueError:
                rel = path.name
            unparsed.append(rel)
            continue

        parsed_count += 1
        key = parsed.base_key
        if key not in by_key:
            by_key[key] = FieldImages(
                experiment=parsed.experiment,
                drug=parsed.drug,
                group=parsed.group,
                field=parsed.field,
            )
        fi = by_key[key]
        if parsed.channel == "red":
            fi.red_path = path
        elif parsed.channel == "green":
            fi.green_path = path
        elif parsed.channel == "merge":
            fi.merge_path = path
        # chN 忽略或当额外通道

    fields = sorted(
        by_key.values(),
        key=lambda f: (f.experiment, f.drug, f.group, f.field),
    )

    skipped: list[str] = []
    kept: list[FieldImages] = []
    for fi in fields:
        if require_red and not fi.red_path and not fi.merge_path:
            skipped.append(fi.image_id)
            continue
        if not fi.has_any():
            skipped.append(fi.image_id)
            continue
        kept.append(fi)

    return ScanReport(
        fields=kept,
        all_tifs=len(tifs),
        parsed=parsed_count,
        unparsed=unparsed,
        skipped_no_red=skipped,
    )


def _as_2d(arr: np.ndarray) -> np.ndarray:
    """将 tifffile 读出的数组规范为 2D 灰度。"""
    a = np.asarray(arr)
    while a.ndim > 2:
        # (C, H, W) or (H, W, C) or (Z, H, W)
        if a.shape[0] <= 4 and a.shape[0] < a.shape[-1]:
            a = a[0]
        elif a.shape[-1] <= 4:
            a = a[..., 0]
        else:
            a = a[0]
    return np.squeeze(a)


def _normalize_uint16_like(img: np.ndarray) -> np.ndarray:
    """保留动态范围，转为 float32 [0, 1]。"""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    img = img.astype(np.float32)
    lo, hi = float(np.min(img)), float(np.max(img))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return (img - lo) / (hi - lo)


def load_grayscale(path: Path) -> np.ndarray:
    """加载单通道 TIF → float32 [0,1]，形状 (H, W)。"""
    raw = tifffile.imread(str(path))
    return _normalize_uint16_like(_as_2d(raw))


def load_multichannel_merge(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """尝试从 merge TIF 拆出 (red, green)。

    常见布局：
      - (2, H, W) 或 (3, H, W)：axis0 = 通道
      - (H, W, 2) 或 (H, W, 3)：axis-1 = 通道
      - 单通道：返回 (gray, None)
    """
    raw = tifffile.imread(str(path))
    a = np.asarray(raw)

    if a.ndim == 2:
        g = _normalize_uint16_like(a)
        return g, None

    # (C, H, W)
    if a.ndim == 3 and a.shape[0] in (2, 3, 4) and a.shape[0] < min(a.shape[1], a.shape[2]):
        red = _normalize_uint16_like(a[0])
        green = _normalize_uint16_like(a[1]) if a.shape[0] >= 2 else None
        return red, green

    # (H, W, C)
    if a.ndim == 3 and a.shape[-1] in (2, 3, 4):
        red = _normalize_uint16_like(a[..., 0])
        green = _normalize_uint16_like(a[..., 1]) if a.shape[-1] >= 2 else None
        return red, green

    # 其他：取第一层
    return _normalize_uint16_like(_as_2d(a)), None


def load_field_channels(
    field: FieldImages,
    prefer_separate: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """加载一个视野的分割用通道。

    Returns
    -------
    red : 红通道（或用于分割的主通道）float32 [0,1]
    green : 绿通道，可能为 None
    source_note : 说明文字
    """
    red: np.ndarray | None = None
    green: np.ndarray | None = None
    notes: list[str] = []

    if prefer_separate and field.red_path is not None:
        red = load_grayscale(field.red_path)
        notes.append(f"红={field.red_path.name}")
    if prefer_separate and field.green_path is not None:
        green = load_grayscale(field.green_path)
        notes.append(f"绿={field.green_path.name}")

    if red is None and field.merge_path is not None:
        m_red, m_green = load_multichannel_merge(field.merge_path)
        red = m_red
        if green is None:
            green = m_green
        notes.append(f"Merge={field.merge_path.name}")

    if red is None and green is not None:
        # 仅有绿：退而用之
        red = green
        notes.append("无红通道，使用绿通道分割")

    if red is None:
        raise FileNotFoundError(f"{field.image_id}: 找不到可用图像")

    return red, green, "; ".join(notes)


def mask_scale_bar_region(
    image: np.ndarray,
    width_frac: float = 0.22,
    height_frac: float = 0.12,
    fill_value: float | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """将右下角标尺区域置为背景中位数，降低被当成细胞的概率。

    Returns
    -------
    masked_image
    box : (y0, y1, x0, x1) 被屏蔽的像素范围
    """
    h, w = image.shape[:2]
    bw = max(1, int(w * width_frac))
    bh = max(1, int(h * height_frac))
    y0, y1 = h - bh, h
    x0, x1 = w - bw, w

    out = image.copy()
    if fill_value is None:
        # 用图像中部区域中位数作背景估计
        cy0, cy1 = h // 4, 3 * h // 4
        cx0, cx1 = w // 4, 3 * w // 4
        fill_value = float(np.median(out[cy0:cy1, cx0:cx1]))
    out[y0:y1, x0:x1] = fill_value
    return out, (y0, y1, x0, x1)
