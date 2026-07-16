"""File scanning and filename parsing for MembraneQuant experiments.

Filename rule (user convention)
-------------------------------
``C2_{实验}{药物}{组别}-{视野}[-{通道}].tif``

Examples::

    C2_104d1-1-1.tif
        实验=104, 药物=d, 组别=1, 视野=1, 通道=红(DiI)

    C2_104d1-1-2.tif
        实验=104, 药物=d, 组别=1, 视野=1, 通道=绿(EGFP)

    C2_wd1-3-1.tif
        实验=w,  药物=d, 组别=1, 视野=3, 通道=红

    C2_104d1-1.tif
        同上，无通道后缀 = Merge（可选）

Channel codes: ``-1`` 红 DiI，``-2`` 绿 EGFP，无后缀 = Merge.

Folder layouts (both supported, recursive search)::

    3B_7.15/C2_104d1-1-1/C2_104d1-1-1.tif   # one folder per image
    Experiment/任意子目录/*.tif              # flat under groups
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import tifffile

from .utils import normalize_to_unit

# C2_104d1-1-1  /  C2_wd1-3-2  /  C2_104e2-4
# prefix_? + experiment + drug(letter) + group(digits) - field [- channel]
FILENAME_RE = re.compile(
    r"^(?P<prefix>[A-Za-z0-9]+_)?"
    r"(?P<experiment>[A-Za-z0-9]+?)"
    r"(?P<drug>[A-Za-z])"
    r"(?P<group>\d+)"
    r"-(?P<field>\d+)"
    r"(?:-(?P<channel>\d+))?$",
    re.IGNORECASE,
)

CHANNEL_CN = {
    "red": "红(DiI)",
    "green": "绿(EGFP)",
    "merge": "Merge合成",
}


@dataclass(frozen=True)
class ParsedName:
    stem: str
    experiment: str  # 实验：104 / w / …
    drug: str  # 药物：单个字母 d / e / …
    group: str  # 组别：数字 1 / 2 / …
    field: int  # 视野
    channel: str | None  # merge | red | green | chN
    path: Path

    @property
    def condition_id(self) -> str:
        """实验+药物+组别，如 104d1、wd1。"""
        return f"{self.experiment}{self.drug}{self.group}"

    @property
    def base_key(self) -> str:
        """同一视野红/绿/Merge 共用键：104d1-1。"""
        return f"{self.condition_id}-{self.field}"

    def describe_cn(self) -> str:
        ch = CHANNEL_CN.get(self.channel or "", self.channel or "?")
        return (
            f"实验 {self.experiment}，药物 {self.drug}，"
            f"组别 {self.group}，视野 {self.field}，通道 {ch}"
        )


@dataclass
class FieldPair:
    """Paired Red (DiI) + Green (EGFP) images for one field of view."""

    experiment: str
    drug: str
    group: str  # 组别（来自文件名，不是文件夹名）
    field: int
    red_path: Path
    green_path: Path
    merge_path: Path | None = None

    @property
    def condition_id(self) -> str:
        return f"{self.experiment}{self.drug}{self.group}"

    @property
    def image_id(self) -> str:
        """稳定 ID，用于输出文件名：104d1-1 或 wd1-3。"""
        return f"{self.condition_id}-{self.field}"

    def describe_cn(self) -> str:
        merge = "，含 Merge" if self.merge_path else ""
        return (
            f"实验 **{self.experiment}**，药物 **{self.drug}**，"
            f"组别 **{self.group}**，视野 **{self.field}**{merge}"
        )


@dataclass
class ScanReport:
    """Diagnostic info from a scan (for Web UI preview)."""

    pairs: list[FieldPair]
    all_tifs: int
    parsed: int
    unpaired: list[str]


def parse_filename(path: Path, group: str | None = None) -> ParsedName | None:
    """Parse a TIF filename. ``group`` arg is ignored (组别 always from name).

    Kept for call-site compatibility; 组别 only comes from the filename rule.
    """
    del group  # 组别以文件名为准，不用文件夹名覆盖
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
    """Recursively collect .tif / .tiff files under root (files only)."""
    root = Path(root)
    found: list[Path] = []
    for pattern in ("**/*.tif", "**/*.tiff", "**/*.TIF", "**/*.TIFF"):
        found.extend(p for p in root.glob(pattern) if p.is_file())
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in sorted(found, key=lambda x: str(x).lower()):
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def scan_pairs(root: Path) -> list[FieldPair]:
    """Scan experiment directory (recursively) and pair Red/Green."""
    return scan_with_report(root).pairs


def scan_with_report(root: Path) -> ScanReport:
    """Recursive scan with diagnostics for the Web UI."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Experiment root not found: {root}")

    tifs = iter_tifs(root)
    by_key: dict[str, dict[str, ParsedName]] = {}
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
        channel_map = by_key.setdefault(key, {})
        if parsed.channel and parsed.channel not in channel_map:
            channel_map[parsed.channel] = parsed

    pairs: list[FieldPair] = []
    unpaired: list[str] = []

    for key, channels in sorted(by_key.items()):
        red = channels.get("red")
        green = channels.get("green")
        merge = channels.get("merge")
        if red is not None and green is not None:
            pairs.append(
                FieldPair(
                    experiment=red.experiment,
                    drug=red.drug,
                    group=red.group,
                    field=red.field,
                    red_path=red.path,
                    green_path=green.path,
                    merge_path=merge.path if merge else None,
                )
            )
        else:
            have = ", ".join(
                CHANNEL_CN.get(c, c) for c in sorted(channels.keys())
            )
            sample = next(iter(channels.values()))
            unpaired.append(
                f"{sample.base_key}（{sample.describe_cn()}）仅有: {have}，"
                f"需要同时有 红(-1) 与 绿(-2)"
            )

    if unparsed:
        show = unparsed[:8]
        note = f"有 {len(unparsed)} 个文件名无法解析（示例: " + "; ".join(show) + ")"
        if len(unparsed) > 8:
            note += " …"
        unpaired.insert(0, note)

    pairs.sort(key=lambda p: (p.experiment, p.drug, p.group, p.field))
    return ScanReport(
        pairs=pairs,
        all_tifs=len(tifs),
        parsed=parsed_count,
        unpaired=unpaired,
    )


def load_image(path: Path) -> "np.ndarray":
    """Load TIF via tifffile and normalize to float32 [0, 1]."""
    import numpy as np

    raw = tifffile.imread(str(path))
    return normalize_to_unit(np.asarray(raw))


def iter_pairs(root: Path) -> Iterator[FieldPair]:
    yield from scan_pairs(root)


def discover_groups(root: Path) -> list[Path]:
    """Dirs that contain at least one TIF (recursive)."""
    root = Path(root)
    dirs: set[Path] = set()
    for p in iter_tifs(root):
        dirs.add(p.parent)
    return sorted(dirs)
