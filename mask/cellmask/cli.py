"""命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cellmask",
        description=(
            "Cellpose 全细胞分割 + 人工筛选导出。\n"
            "优先使用红色膜标记通道 (cyto3)，支持触边/标尺区自动标记与点击剔除。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 扫描 + 分割 + 人工筛选
  python -m cellmask --input-dir "D:\\课题同步\\实验结果图\\共定位-荧光\\3B_7.20"

  # 细胞特别小时用 cyto2，直径约 50
  python -m cellmask -i "D:\\...\\3B_7.20" --model cyto2 --diameter 50

  # 只处理前 3 张调试
  python -m cellmask -i "D:\\...\\3B_7.20" --limit 3

  # 仅扫描不分割
  python -m cellmask -i "D:\\...\\3B_7.20" --scan-only

  # 不弹 GUI，仅自动剔除触边/标尺区后导出
  python -m cellmask -i "D:\\...\\3B_7.20" --skip-review
""",
    )
    p.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="实验根目录，例如 D:\\课题同步\\实验结果图\\共定位-荧光\\3B_7.20",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        default=None,
        help="导出目录（默认: <input-dir>/cellmask_output）",
    )
    p.add_argument(
        "--model",
        default="cyto3",
        choices=["cyto3", "cyto2", "cyto"],
        help="Cellpose 模型（默认 cyto3；细胞很小时试 cyto2）。不要用 nuclei。",
    )
    p.add_argument(
        "--diameter",
        type=float,
        default=0,
        help="细胞直径（pixel）。0 或省略 = Auto。也可填 40–80。",
    )
    p.add_argument(
        "--channel",
        default="red",
        choices=["red", "green", "max"],
        help="分割用通道：red=红膜标记(默认), green=绿, max=红绿取大",
    )
    p.add_argument("--no-gpu", action="store_true", help="强制 CPU")
    p.add_argument(
        "--flow-threshold",
        type=float,
        default=0.4,
        help="Cellpose flow_threshold（默认 0.4）",
    )
    p.add_argument(
        "--cellprob-threshold",
        type=float,
        default=0.0,
        help="Cellpose cellprob_threshold（默认 0.0）",
    )
    p.add_argument(
        "--no-scalebar-mask",
        action="store_true",
        help="不对右下角标尺区域做屏蔽",
    )
    p.add_argument(
        "--scalebar-width",
        type=float,
        default=0.22,
        help="右下角屏蔽区宽度占图像比例（默认 0.22）",
    )
    p.add_argument(
        "--scalebar-height",
        type=float,
        default=0.12,
        help="右下角屏蔽区高度占图像比例（默认 0.12）",
    )
    p.add_argument(
        "--no-auto-flag",
        action="store_true",
        help="启动筛选时不自动剔除触边/标尺区细胞",
    )
    p.add_argument(
        "--skip-review",
        action="store_true",
        help="跳过人工 GUI，仅自动标记后导出",
    )
    p.add_argument(
        "--scan-only",
        action="store_true",
        help="只扫描并打印配对结果，不分割",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 个视野（调试）",
    )
    p.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="只处理指定 image_id，例如 104d1-1 we2-3",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="忽略已有分割缓存，强制重新跑 Cellpose",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    # 尽早配置中文字体（筛选 GUI / 任何 matplotlib 中文）
    try:
        from .fonts import setup_chinese_font

        setup_chinese_font(silent=True)
    except Exception:
        pass

    args = build_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"错误: 目录不存在 → {input_dir}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else input_dir / "cellmask_output"

    if args.scan_only:
        from .io_scan import scan_fields

        report = scan_fields(input_dir, require_red=True)
        print(f"根目录: {input_dir}")
        print(f"TIF 文件: {report.all_tifs}")
        print(f"解析成功: {report.parsed}")
        print(f"有效视野: {len(report.fields)}")
        for fi in report.fields:
            print(f"  {fi.describe()}")
            if fi.red_path:
                print(f"      red:   {fi.red_path}")
            if fi.green_path:
                print(f"      green: {fi.green_path}")
            if fi.merge_path:
                print(f"      merge: {fi.merge_path}")
        if report.unparsed:
            print(f"\n未识别 ({len(report.unparsed)}):")
            for u in report.unparsed[:30]:
                print(f"  ? {u}")
        if report.skipped_no_red:
            print(f"\n跳过无红/Merge: {report.skipped_no_red}")
        return 0

    from .pipeline import run_pipeline
    from .segment import cellpose_status

    st = cellpose_status()
    print(st["message"])
    if not st["available"]:
        print(
            "\n请先激活 conda 环境 mem 并安装 cellpose:\n"
            "  conda activate mem\n"
            "  pip install cellpose\n",
            file=sys.stderr,
        )
        return 1

    diameter = None if args.diameter is None or args.diameter <= 0 else float(args.diameter)

    try:
        result = run_pipeline(
            input_dir=input_dir,
            out_dir=out_dir,
            model=args.model,
            diameter=diameter,
            gpu=not args.no_gpu,
            use_channel=args.channel,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            mask_scalebar=not args.no_scalebar_mask,
            scalebar_width_frac=args.scalebar_width,
            scalebar_height_frac=args.scalebar_height,
            auto_flag=not args.no_auto_flag,
            skip_review=args.skip_review,
            limit=args.limit,
            image_ids=args.ids,
            reuse_cache=not args.no_cache,
        )
    except Exception as e:
        print(f"运行失败: {e}", file=sys.stderr)
        return 1

    print("\n===== 完成 =====")
    print(f"视野数: {result['n_fields']}")
    print(f"输出目录: {result['out_dir']}")
    if result.get("n_kept_total") is not None:
        print(f"保留细胞总数: {result['n_kept_total']}")
    return 0
