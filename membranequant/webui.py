"""MembraneQuant 中文 Web 界面（Gradio）。

核心操作：
  1. 选择实验数据文件夹
  2. 调整参数后点击「开始分析」
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from membranequant.config import Config, load_config
from membranequant.dual_backend import describe_runtime, dualcellquant_status
from membranequant.io import scan_with_report
from membranequant.main import run_pipeline


def _default_cfg(config_path: Path | None = None) -> Config:
    return load_config(config_path)


def _preview_pairs(input_dir: str) -> str:
    path = Path(input_dir.strip().strip('"').strip("'")) if input_dir else None
    if not path or not path.is_dir():
        return (
            "⚠️ **输入路径无效**\n\n"
            "请填写已存在的实验文件夹完整路径，例如：\n"
            r"`D:\神秘文件夹\实验结果图\共定位-荧光\3B_7.15`"
            "\n\n支持中文路径；可直接从资源管理器地址栏复制粘贴。"
        )
    try:
        report = scan_with_report(path)
    except Exception as exc:
        return f"⚠️ **扫描失败**：{exc}"

    pairs = report.pairs
    header = (
        f"📂 扫描目录：`{path}`\n\n"
        f"- 找到 TIF 文件：**{report.all_tifs}** 个（含子文件夹，递归搜索）\n"
        f"- 文件名解析成功：**{report.parsed}** 个\n"
        f"- 红绿配对成功：**{len(pairs)}** 组视野\n"
    )

    if not pairs:
        tips = (
            "\n⚠️ **未找到可分析的红绿图片对**\n\n"
            "程序会**递归**搜索子文件夹。你的常见结构是：\n"
            "```\n"
            "3B_7.15/\n"
            "  C2_104d1-1-1/C2_104d1-1-1.tif   ← 红 DiI（-1）\n"
            "  C2_104d1-1-2/C2_104d1-1-2.tif   ← 绿 EGFP（-2）\n"
            "  C2_104d1-1/C2_104d1-1.tif       ← Merge（可选）\n"
            "```\n"
            "每个视野必须**同时**有 `-1` 和 `-2` 才会进入分析。\n"
        )
        if report.unpaired:
            tips += "\n**未配对 / 解析问题：**\n" + "\n".join(f"- {u}" for u in report.unpaired[:20])
        return header + tips

    lines = [
        header,
        f"✅ **已识别 {len(pairs)} 组视野（红+绿配对成功）**\n",
        "命名规则：`C2_{实验}{药物}{组别}-{视野}[-{通道}]`，"
        "其中 **通道** `-1`=红(DiI)，`-2`=绿(EGFP)，无后缀=Merge。\n",
        "下面是将要分析的列表（最多显示 40 条）：\n",
    ]
    for p in pairs[:40]:
        lines.append(
            f"- **{p.image_id}** — {p.describe_cn()}\n"
            f"  - 红(DiI): `{p.red_path.name}`\n"
            f"  - 绿(EGFP): `{p.green_path.name}`"
        )
    if len(pairs) > 40:
        lines.append(f"\n… 还有 **{len(pairs) - 40}** 组未全部列出")
    if report.unpaired:
        lines.append("\n**部分未配对（不会分析）：**")
        for u in report.unpaired[:12]:
            lines.append(f"- {u}")
    lines.append("\n确认无误后，点击 **开始分析**。")
    return "\n".join(lines)


def _df_for_gradio(df: pd.DataFrame | None, max_rows: int = 200) -> pd.DataFrame:
    """Make a dataframe safe for Gradio (no NaN-only display issues, limited rows)."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.head(max_rows).copy()
    # Gradio + NaN can confuse some versions; keep numbers finite for display
    out = out.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
    return out


def _load_overlay_images(output_dir: Path, limit: int = 8) -> list:
    """Load overlay PNGs as RGB arrays for Gradio Gallery (avoids Chinese-path bugs)."""
    ov_dir = output_dir / "overlays"
    if not ov_dir.is_dir():
        return []
    paths = sorted(ov_dir.glob("*_overlay.png"))[:limit]
    images: list = []
    for p in paths:
        try:
            # Prefer PIL if available; else matplotlib/imageio via skimage
            from PIL import Image

            img = Image.open(p).convert("RGB")
            images.append(np.asarray(img))
        except Exception:
            try:
                import matplotlib.image as mpimg

                images.append(mpimg.imread(str(p)))
            except Exception:
                continue
    return images


def _load_summary_tables(
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """加载结果表格、叠加图和统计图表"""
    results_path = output_dir / "csv" / "results.csv"
    summary_path = output_dir / "csv" / "summary.csv"
    results = pd.read_csv(results_path) if results_path.is_file() else pd.DataFrame()
    summary = pd.read_csv(summary_path) if summary_path.is_file() else pd.DataFrame()
    overlays = _load_overlay_images(output_dir, limit=8)
    
    # 加载统计图表
    plots_dir = output_dir / "plots"
    plot_images = []
    if plots_dir.is_dir():
        plot_files = sorted(plots_dir.glob("*.png"))
        for p in plot_files[:12]:  # 最多加载12张图（含多指标/共定位）
            try:
                from PIL import Image
                img = Image.open(p).convert("RGB")
                plot_images.append(np.asarray(img))
            except Exception:
                continue
    
    return results, summary, overlays, plot_images


def run_from_ui(
    input_dir: str,
    output_dir: str,
    seg_method: str,
    ring_width: int,
    min_area: int,
    max_area: int,
    min_red_cov: float,
    rolling_ball: int,
    gaussian_sigma: float,
    enable_denoise: bool,
    seg_channel: str,
    cellpose_model: str,
    cellpose_diameter: float,
    cellpose_gpu: bool,
    save_overlay: bool,
    save_mask: bool,
    save_graphpad: bool,
    compute_pearson: bool,
    config_path: str | None,
    cellpose_flow_threshold: float = 0.4,
    cellpose_cellprob_threshold: float = 0.0,
    progress=None,
) -> tuple[str, Any, Any, list[str], list[str]]:
    """从 Web 界面执行完整分析流程。"""
    logs: list[str] = []

    def _log(msg: str) -> None:
        logs.append(msg)

    empty_df = pd.DataFrame()
    empty_gallery = []
    in_path = Path(input_dir.strip().strip('"').strip("'")) if input_dir else None
    if not in_path or not in_path.is_dir():
        return (
            "❌ **请先填写有效的「输入实验文件夹」路径。**\n\n"
            "例如：`D:\\神秘文件夹\\实验结果图\\共定位-荧光\\3B_7.15`\n"
            "该目录下可以是「每个视野一个子文件夹」，程序会递归找到里面的 TIF。",
            empty_df,
            empty_df,
            empty_gallery,
            empty_gallery,
        )

    if output_dir and str(output_dir).strip():
        out_path = Path(str(output_dir).strip().strip('"').strip("'"))
    else:
        out_path = in_path / "Results"

    cfg_file = (
        Path(str(config_path).strip().strip('"').strip("'"))
        if config_path and str(config_path).strip()
        else None
    )
    overrides: dict[str, Any] = {
        "segmentation_method": "dualcellquant",
        "ring_width": int(ring_width),
        "minimum_cell_area": int(min_area),
        "maximum_cell_area": int(max_area),
        "minimum_red_coverage": float(min_red_cov),
        "rolling_ball_radius": int(rolling_ball),
        "dual_bg_radius": int(rolling_ball),
        "gaussian_sigma": float(gaussian_sigma),
        "enable_denoise": bool(enable_denoise),
        "segmentation_channel": seg_channel,
        "cellpose_model": cellpose_model or "cyto2",
        "cellpose_diameter": float(cellpose_diameter),
        "dual_diameter": float(cellpose_diameter),
        "cellpose_gpu": bool(cellpose_gpu),
        "dual_use_gpu": bool(cellpose_gpu),
        "dual_auto_gpu": False,  # respect checkbox exactly
        "cellpose_flow_threshold": float(cellpose_flow_threshold),
        "dual_flow_threshold": float(cellpose_flow_threshold),
        "cellpose_cellprob_threshold": float(cellpose_cellprob_threshold),
        "dual_cellprob_threshold": float(cellpose_cellprob_threshold),
        "save_overlay": bool(save_overlay),
        "save_mask": bool(save_mask),
        "save_graphpad": bool(save_graphpad),
        "compute_pearson": bool(compute_pearson),
    }
    # legacy UI still exposes seg_method; Dual path ignores non-dual choices
    _ = seg_method

    try:
        cfg = load_config(cfg_file, overrides=overrides)
    except Exception as exc:
        return f"❌ **参数配置无效**：{exc}", empty_df, empty_df, empty_gallery, empty_gallery

    rt = describe_runtime(cfg)
    if not rt.get("available") or not rt.get("cellpose_available"):
        return (
            "❌ **DualCellQuant / Cellpose 未就绪（已取消静默降级）**\n\n"
            f"{rt.get('message')}\n\n"
            "请在运行分析的机器上安装：\n"
            '```\npip install "git+https://github.com/fuji3to4/DualCellQuant.git"\n'
            "pip install cellpose torch\n"
            "# GPU: 安装与驱动匹配的 CUDA 版 PyTorch\n```",
            empty_df,
            empty_df,
            empty_gallery,
            empty_gallery,
        )
    _log(f"Runtime: {rt['summary']}")
    _log(f"DualCellQuant 状态：{rt['message']}")
    if not rt.get("will_use_gpu"):
        _log(
            "⚠️ Cellpose 未走 GPU：GPU 占用会很低。请勾选「使用 GPU 加速」，"
            "并确认 torch.cuda.is_available()=True。"
        )
    else:
        _log(
            f"✅ Cellpose 将使用 GPU（{rt.get('cuda_device_name') or 'CUDA'}）。"
            " 注意：EDT / mask / 背景扣除 / 定量仍是 CPU，GPU 不会全程打满。"
        )

    def on_progress(message: str, fraction: float) -> None:
        _log(f"[{fraction * 100:5.1f}%] {message}")
        if progress is not None:
            try:
                progress(fraction, desc=message)
            except TypeError:
                try:
                    progress(fraction)
                except Exception:
                    pass

    try:
        _log(f"输入目录：{in_path}")
        _log(f"输出目录：{out_path}")
        _log(f"Backend：DualCellQuant | dual_use_gpu={cfg.dual_use_gpu} | bg_mode={cfg.dual_bg_mode}")
        results_path = run_pipeline(
            in_path.resolve(), out_path.resolve(), cfg, progress=on_progress
        )
        results_df, summary_df, overlays, plot_images = _load_summary_tables(out_path.resolve())

        n_rows = len(results_df)
        n_pass = 0
        qc_reasons = ""
        if not results_df.empty and "QC" in results_df.columns:
            n_pass = int((results_df["QC"] == "pass").sum())
            if n_pass == 0 and "QC_Reason" in results_df.columns:
                top = results_df["QC_Reason"].fillna("").value_counts().head(5)
                qc_reasons = "\n".join(f"- `{k}`: {v} 个" for k, v in top.items() if k)

        rt_done = describe_runtime(cfg)
        seg_info = "DualCellQuant → Cellpose-SAM (`cpdino`)"
        hardware_info = rt_done["hardware"]
        if rt_done.get("will_use_gpu"):
            hardware_info += "（仅分割阶段；EDT/mask/定量为 CPU）"
        backend_col = ""
        if not results_df.empty and "Backend" in results_df.columns:
            backends = sorted({str(x) for x in results_df["Backend"].dropna().unique()})
            backend_col = ", ".join(backends) if backends else "n/a"
        else:
            backend_col = "dualcellquant (expected)"

        warn_qc = ""
        if n_rows > 0 and n_pass == 0:
            warn_qc = (
                "\n### ⚠️ 警告：没有细胞通过质控\n\n"
                "常见原因：\n"
                "1. 绿色通道读成 0（RGB 导出时曾只取 R 平面，现已修复，请**重新分析**）\n"
                "2. 最低 Red Coverage / AND 像素阈值过严 — 可在左侧略调低\n"
            )
            if qc_reasons:
                warn_qc += "\n本次失败原因统计：\n" + qc_reasons + "\n"

        report = (
            f"## ✅ 分析完成\n\n"
            f"| 项目 | 内容 |\n|------|------|\n"
            f"| 结果文件 | `{results_path}` |\n"
            f"| 检出细胞数 | **{n_rows}** 行（每个细胞一行） |\n"
            f"| 通过质控 | **{n_pass}** 个细胞 |\n"
            f"| 图像分析后端 | {seg_info} |\n"
            f"| results Backend 列 | `{backend_col}` |\n"
            f"| 运行硬件 | {hardware_info} |\n"
            f"| GPU 请求 / CUDA | `{cfg.dual_use_gpu}` / `{rt_done.get('cuda_available')}` |\n"
            f"| 背景模式 | `{cfg.dual_bg_mode}`（rolling 很慢；dark_subtract 快） |\n"
            f"| 输出文件夹 | `{out_path.resolve()}` |\n"
            f"{warn_qc}\n"
            f"**输出说明：**\n"
            f"- `csv/results.csv`：单细胞明细（**Ratio_T_over_R**、**RatioOfMeans_T_R**、**Enrichment_Membrane_vs_Whole** 等）\n"
            f"- `csv/summary.csv`：按条件汇总（默认主指标 **Ratio_T_over_R**）\n"
            f"- `csv/graphpad_*.csv`：GraphPad 宽表（多指标）\n"
            f"- `overlays/`：分割叠加图 + 红绿共定位散点图\n"
            f"- `plots/`：📊 **300 dpi 统计图（可直接放 PPT）**\n"
            f"- `masks/`：标签图；`qc/`：剔除原因；`logs/pipeline.log`：分阶段耗时\n\n"
            f"💡 也可用可视化前端：`python membranequant/visualizer_app.py` 上传 results.csv 出图。\n\n"
            f"### 运行日志（末尾）\n```\n" + "\n".join(logs[-80:]) + "\n```"
        )
        # Gradio: return limited dataframes + in-memory images (not Chinese file paths)
        return (
            report,
            _df_for_gradio(results_df, 300),
            _df_for_gradio(summary_df, 100),
            overlays,
            plot_images,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        return (
            f"## ❌ 分析失败\n\n**错误：** `{exc}`\n\n"
            f"```\n{tb}\n```",
            empty_df,
            empty_df,
            empty_gallery,
            empty_gallery,
        )


def build_app(config_path: Path | None = None):
    import gradio as gr

    cfg = _default_cfg(config_path)
    cp = dualcellquant_status()
    rt0 = describe_runtime(cfg)
    if cp["available"] and cp.get("cellpose_available"):
        if rt0.get("will_use_gpu"):
            cuda_status = (
                f"\n\n🚀 **将使用 GPU 做 Cellpose 分割**（{cp.get('cuda_device_name') or 'CUDA'}）。"
                " 后处理（EDT / mask / 定量）仍是 CPU，任务管理器里 GPU 不会全程 100%。"
            )
        elif cp.get("cuda_available"):
            cuda_status = (
                "\n\n⚠️ **检测到 CUDA 可用，但当前配置未启用 GPU**。"
                "请勾选下方「使用 GPU 加速」，否则会很慢且 GPU 空闲。"
            )
        else:
            cuda_status = (
                "\n\n⚠️ **CUDA 不可用**：当前 PyTorch 为 CPU 模式"
                "（`torch.cuda.is_available()=False`）。“有 DualCellQuant”≠ 已用 GPU。"
            )
        cp_banner = (
            f"✅ **DualCellQuant 已安装**（版本: {cp.get('version') or '?'}；"
            f"分割模型: **{cp.get('model_name', 'cpdino')} / Cellpose-SAM**）"
            f"{cuda_status}\n\n"
            f"`{rt0['summary']}`"
        )
    elif cp["available"]:
        cp_banner = (
            f"⚠️ **DualCellQuant 已装但 Cellpose/torch 不可用**：{cp.get('message')}\n\n"
            "分析会**直接失败**（已取消静默降级）。"
        )
    else:
        cp_banner = (
            "❌ **DualCellQuant 未安装** — 分析会直接失败（已取消静默降级）。\n"
            '请运行 `pip install "git+https://github.com/fuji3to4/DualCellQuant.git"` 并安装 cellpose + torch。'
        )

    # Prefer GPU when CUDA is present
    gpu_default = bool(cfg.dual_use_gpu)
    if cp.get("cuda_available") and cfg.dual_auto_gpu:
        gpu_default = True

    with gr.Blocks(title="MembraneQuant 膜定位定量") as demo:
        gr.Markdown(
            f"""
# MembraneQuant · 膜定位定量工具

{cp_banner}

用于分析 **EGFP 标记蛋白的细胞膜定位**，并用 **DiI 膜染色** 做质量检查。

### 架构（当前）

1. **图像分析后端固定为 DualCellQuant**（Cellpose-SAM `cpdino` → EDT 径向膜区 → mask → T/R 定量）。
2. **只有分割阶段吃 GPU**；背景扣除 / EDT / 定量是 CPU。所以“有 CUDA”≠“GPU 打满”。
3. Dual 用的是 **cpdino（比经典 cyto2 重得多）**；`dual_bg_mode=rolling` 时 rolling-ball 会极慢。

### 设计原则

1. **细胞边界** 主要由 **绿色 EGFP** 分割得到。
2. **膜区域** 由 Dual 的 **EDT 径向百分比**（默认 85–100%）得到。
3. **DiI（红）** 参与 mask / T/R 与质控。

### 分析流程（自动）

读取目录 → 配对红/绿 → Dual 分割 → 径向膜 ROI → mask → 定量 → 质控 → CSV / 叠加图
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                # ----- 1. 文件夹 -----
                gr.Markdown(
                    """
### ① 选择数据文件夹

**这一步必做。** 告诉程序实验数据在哪、结果写到哪。
"""
                )
                input_dir = gr.Textbox(
                    label="输入：实验数据文件夹（必填）",
                    placeholder=r"例如 D:\神秘文件夹\实验结果图\共定位-荧光\3B_7.15",
                    info=(
                        "填写实验根目录完整路径（支持中文）。程序会递归搜索所有子文件夹里的 TIF。\n"
                        "支持两种结构：\n"
                        "① 每张图一个子文件夹：3B_7.15/C2_104d1-1-1/C2_104d1-1-1.tif\n"
                        "② 传统扁平：Experiment/GroupA/*.tif\n"
                        "文件名规则：C2_{实验}{药物}{组别}-{视野}[-通道]\n"
                        "例：C2_104d1-1-1 → 实验104、药物d、组别1、视野1、红；"
                        "C2_wd1-3-2 → 实验w、药物d、组别1、视野3、绿。\n"
                        "通道：-1 红(DiI)，-2 绿(EGFP)，无后缀 Merge(可选)。"
                    ),
                )
                output_dir = gr.Textbox(
                    label="输出：结果保存文件夹（可选）",
                    placeholder="留空则自动使用「输入路径/Results」",
                    info=(
                        "分析结果会写到这里：csv 表格、叠加图、mask、质控日志。\n"
                        "留空时默认保存在输入文件夹下的 Results 子目录。"
                    ),
                )
                preview_btn = gr.Button(
                    "预览配对（先检查能不能读到图）",
                    variant="secondary",
                )
                preview_md = gr.Markdown(
                    "💡 填好输入路径后，建议先点 **预览配对**，确认红绿通道都识别到了，再点「开始分析」。"
                )

                # ----- 2. 分割 -----
                gr.Markdown(
                    """
### ② 细胞分割（如何认出每个细胞）

程序需要先把每个细胞从背景里分出来，得到「全细胞轮廓」。  

**🎯 针对细胞粘连问题，现提供多种方案：**

1. **Otsu 阈值法**（默认）- 简单快速，适合分离良好的细胞
2. **距离变换+分水岭** - ImageJ经典方法，适合圆形粘连细胞
3. **梯度+分水岭** - 利用边界强度分割
4. **H-minima+分水岭** - 文献推荐，适合密集细胞（抑制过度分割）
5. **形态学开运算** - 通过腐蚀膨胀断开细窄连接
6. **距离+梯度双重markers** - 综合方法，效果更稳定

💡 **粘连严重时建议**：先试 **距离变换分水岭**，效果不好再试 **H-minima分水岭** 或 **双重markers**
"""
                )
                seg_method = gr.Dropdown(
                    choices=[
                        ("Otsu 阈值法（默认，推荐先用这个）", "otsu"),
                        ("距离变换+分水岭（适合圆形粘连细胞）", "watershed_distance"),
                        ("梯度+分水岭（适合边界清晰的细胞）", "watershed_gradient"),
                        ("H-minima+分水岭（适合密集粘连细胞）", "hminima_watershed"),
                        ("形态学开运算（适合轻度粘连）", "morphological_opening"),
                        ("距离+梯度双重markers（综合方法）", "combined_markers"),
                        ("Cellpose 深度学习（需额外安装）", "cellpose"),
                    ],
                    value=(
                        cfg.segmentation_method
                        if cfg.segmentation_method in (
                            "otsu", "cellpose", "watershed_distance", 
                            "watershed_gradient", "hminima_watershed",
                            "morphological_opening", "combined_markers"
                        )
                        else "otsu"
                    ),
                    label="细胞分割方法",
                    info=(
                        "🔬 **针对粘连细胞的多种分割方案**（参考ImageJ和文献方法）：\n\n"
                        "• **Otsu 阈值法**：简单快速，适合分离良好的细胞\n"
                        "• **距离变换+分水岭**：经典方法，适合圆形/椭圆形粘连细胞（ImageJ Watershed插件原理）\n"
                        "• **梯度+分水岭**：利用边界强度，适合边界清晰但粘连的细胞\n"
                        "• **H-minima+分水岭**：文献常用，抑制过度分割，适合密集培养细胞\n"
                        "• **形态学开运算**：通过腐蚀-膨胀断开细窄连接，适合轻度粘连\n"
                        "• **距离+梯度双重markers**：综合方法，结合中心和边界信息\n"
                        "• **Cellpose**：深度学习（需 pip install cellpose），最强但最慢\n\n"
                        "💡 **建议**：粘连严重时依次尝试：距离变换分水岭 → H-minima分水岭 → 双重markers"
                    ),
                )
                with gr.Accordion("Cellpose 专用选项（仅在上面选了 Cellpose 时生效）", open=False):
                    gr.Markdown(
                        """
**什么时候改这里？**  
只有当你选择了「Cellpose 深度学习」时这些参数才有用。  
模型与直径会影响识别到的细胞大小与数量；不确定时保持默认即可。
"""
                    )
                    cellpose_model = gr.Textbox(
                        label="Cellpose 模型名称",
                        value=cfg.cellpose_model,
                        info=(
                            "常用：cyto2 / cyto3（细胞质模型，适合本实验）、nuclei（核）。\n"
                            "一般用 cyto2 即可。改错名字会导致加载失败。"
                        ),
                    )
                    cellpose_diameter = gr.Number(
                        label="细胞直径（像素）；0 = 自动估计",
                        value=float(cfg.cellpose_diameter),
                        precision=1,
                        info=(
                            "填 0 让 Cellpose 自动估细胞大小（推荐）。\n"
                            "若已知细胞大约多少像素宽，可手动填写，有时更稳。"
                        ),
                    )
                    cellpose_gpu = gr.Checkbox(
                        label="使用 GPU 加速（Dual Cellpose 分割）",
                        value=gpu_default,
                        info=(
                            "必须勾选且 torch.cuda.is_available()=True 才会用 GPU。"
                            "前端显示“CUDA 可用”不等于已加速。"
                            "仅分割阶段用 GPU；后处理是 CPU。"
                        ),
                    )
                    cellpose_flow_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=float(cfg.cellpose_flow_threshold),
                        step=0.05,
                        label="Cellpose 流量阈值 (flow_threshold)",
                        info="控制识别的细胞形状与流向匹配程度。默认 0.4。调高会检测到更多细胞但形状可能不规则，调低会更严格。",
                    )
                    cellpose_cellprob_threshold = gr.Slider(
                        minimum=-6.0,
                        maximum=6.0,
                        value=float(cfg.cellpose_cellprob_threshold),
                        step=0.5,
                        label="Cellpose 细胞概率阈值 (cellprob_threshold)",
                        info="控制检测细胞的置信度。默认 0.0。调小（如 -2.0）会检测到更多弱信号细胞，调大（如 2.0）会只保留强信号细胞。",
                    )

                seg_channel = gr.Radio(
                    choices=[
                        ("仅绿色 EGFP（推荐）", "green"),
                        ("绿色 + 红色取最大（EGFP+DiI）", "green_red"),
                    ],
                    value=cfg.segmentation_channel,
                    label="用哪个通道来分割「全细胞」",
                    info=(
                        "• 仅绿色：完全按设计，用 EGFP 定细胞边界（推荐，DiI 断裂时也不怕）。\n"
                        "• 绿+红取最大：细胞边缘 EGFP 很弱但 DiI 较好时可试；仍不是「只用 DiI 分割」。"
                    ),
                )

                # ----- 3. 膜环与过滤 -----
                gr.Markdown(
                    """
### ③ 膜环与细胞过滤

膜区域 = 全细胞轮廓向内「挖」一圈固定宽度的环。  
环太窄 → 像素少、噪声大；环太宽 → 会掺进胞质。常用 **2～5 像素**。
"""
                )
                ring_width = gr.Slider(
                    1,
                    10,
                    value=int(cfg.ring_width),
                    step=1,
                    label="膜环宽度（像素）",
                    info=(
                        "从细胞边界向内侵蚀后的「壳」厚度。设计文档推荐 2 / 3 / 4 / 5。\n"
                        "默认 3。改这个会直接影响 M/C（膜/质荧光比）的数值，写论文时要固定并报告。"
                    ),
                )
                min_area = gr.Number(
                    label="最小细胞面积（像素）",
                    value=int(cfg.minimum_cell_area),
                    precision=0,
                    info="面积小于此值的连通域会被丢掉（碎屑、噪声）。默认 500。图分辨率很低时可适当减小。",
                )
                max_area = gr.Number(
                    label="最大细胞面积（像素）",
                    value=int(cfg.maximum_cell_area),
                    precision=0,
                    info="面积过大的区域常是粘连团或整片背景误分割，会被丢掉。默认 50000。",
                )
                min_red_cov = gr.Slider(
                    0.0,
                    1.0,
                    value=float(cfg.minimum_red_coverage),
                    step=0.05,
                    label="最低红色覆盖率 Red Coverage（质控阈值）",
                    info=(
                        "膜环上有多少比例像素被认为「有 DiI 信号」。\n"
                        "低于此值 → 该细胞 QC 记为 fail（可能 DiI 染色断裂/缺失）。\n"
                        "默认 0.5（50%）。染色普遍偏弱时可略降，但过低会放过差质量细胞。"
                    ),
                )

                # ----- 4. 预处理与导出 -----
                with gr.Accordion("④ 预处理与导出选项（一般保持默认即可）", open=False):
                    gr.Markdown(
                        """
**背景校正 / 降噪**：对应 ImageJ 里 rolling ball + 高斯平滑，让阈值分割更稳。  
**导出开关**：控制是否生成叠加图、mask、GraphPad 表等；关掉可略快一点。
"""
                    )
                    rolling_ball = gr.Number(
                        label="Rolling ball 背景半径（像素）",
                        value=int(cfg.rolling_ball_radius),
                        precision=0,
                        info=(
                            "滚动球背景扣除，半径约等于背景不均匀的尺度。\n"
                            "默认 50，与常见 ImageJ 流程一致。背景很慢变时可略增大。"
                        ),
                    )
                    gaussian_sigma = gr.Number(
                        label="高斯平滑 σ（像素）",
                        value=float(cfg.gaussian_sigma),
                        precision=2,
                        info="轻度去噪。默认 1。越大越糊，边界越圆滑，但细节会丢。",
                    )
                    enable_denoise = gr.Checkbox(
                        label="启用高斯降噪",
                        value=bool(cfg.enable_denoise),
                        info="取消勾选则只做背景校正、不做高斯模糊。",
                    )
                    save_overlay = gr.Checkbox(
                        label="保存叠加图（overlays）",
                        value=bool(cfg.save_overlay),
                        info="生成带细胞编号的彩色叠加 PNG，方便肉眼检查分割是否合理。强烈建议开启。",
                    )
                    save_mask = gr.Checkbox(
                        label="保存分割 mask（masks）",
                        value=bool(cfg.save_mask),
                        info="导出全细胞 / 膜 / 胞质的标签 TIFF，便于后期用其他软件复核。",
                    )
                    save_graphpad = gr.Checkbox(
                        label="导出 GraphPad 宽表 CSV",
                        value=bool(cfg.save_graphpad),
                        info="生成 graphpad_MC.csv：每一列是一个组别的 M/C 值，可直接粘贴进 GraphPad Prism。",
                    )
                    compute_pearson = gr.Checkbox(
                        label="计算膜上 Pearson 相关（辅助指标）",
                        value=bool(cfg.compute_pearson),
                        info=(
                            "在膜环像素上算绿/红相关，作为辅助，不是主指标。\n"
                            "主结论请看 M/C（膜绿均值 / 胞质绿均值）和 MembraneFraction。"
                        ),
                    )
                    config_file = gr.Textbox(
                        label="自定义 config.yaml 路径（可选）",
                        value=str(config_path) if config_path else "",
                        placeholder="留空 = 使用软件自带默认配置",
                        info="高级用户可指向自己的 yaml；界面上改过的参数仍会覆盖文件中的对应项。",
                    )

                run_btn = gr.Button("开始分析", variant="primary", size="lg")
                gr.Markdown(
                    """
---
**主指标速查（膜定位 / 共定位）**

| 名称 | 含义 | 用途 |
|------|------|------|
| **M/C_DiI** | DiI 引导膜上绿均值 ÷ 胞质绿均值 | **推荐主指标** |
| **MEI** | (膜−质)/(膜+质)，∈(−1,1) | 稳健富集指数 |
| **Manders_M1** | Costes 阈值下绿与 DiI 共现比例 | 标准共定位 |
| **EdgeCenterRatio** | 几何边缘/中心绿强度比 | DiI 不均时备份 |
| M/C / MembraneFraction | 固定几何膜环 | 传统/对照 |
| **Red Coverage** | 膜环 DiI 覆盖率 | **仅质控** |
"""
                )

            with gr.Column(scale=1):
                gr.Markdown(
                    """
### ⑤ 分析结果

点 **开始分析** 后，这里会显示进度摘要、按组汇总表、每个细胞的详细表，以及叠加图预览。
"""
                )
                report = gr.Markdown(
                    "尚未运行。请先在左侧填写 **输入实验文件夹**，建议先 **预览配对**，再点 **开始分析**。"
                )
                summary_table = gr.Dataframe(
                    label="组别汇总 summary.csv（Mean M/C、SD、SEM、细胞数等）",
                    interactive=False,
                )
                results_table = gr.Dataframe(
                    label="单细胞明细 results.csv（预览；完整文件在输出目录 csv/ 下）",
                    interactive=False,
                )
                gallery = gr.Gallery(
                    label="叠加图预览（绿=细胞边界，红=膜环，蓝=胞质；图上有 Cell 编号）",
                    columns=2,
                    height=420,
                )
                plots_gallery = gr.Gallery(
                    label="📊 统计图表（M/C对比、箱线图、质控统计、相关性等）",
                    columns=2,
                    height=500,
                )
                gr.Markdown(
                    """
#### 结果怎么看？

1. 先看 **叠加图**：细胞有没有漏分/粘连、膜环是否贴边。  
2. 再看 **summary**：各组 M/C 均值是否符合预期。  
3. 打开 `qc/` 日志：被剔细胞的原因（面积过小、贴边、Red Coverage 低等）。  
4. 用 `graphpad_MC.csv` 做统计图；写方法时注明膜环宽度与分割方法。
"""
                )

        preview_btn.click(fn=_preview_pairs, inputs=[input_dir], outputs=[preview_md])

        run_btn.click(
            fn=run_from_ui,
            inputs=[
                input_dir,
                output_dir,
                seg_method,
                ring_width,
                min_area,
                max_area,
                min_red_cov,
                rolling_ball,
                gaussian_sigma,
                enable_denoise,
                seg_channel,
                cellpose_model,
                cellpose_diameter,
                cellpose_gpu,
                save_overlay,
                save_mask,
                save_graphpad,
                compute_pearson,
                config_file,
                cellpose_flow_threshold,
                cellpose_cellprob_threshold,
            ],
            outputs=[report, results_table, summary_table, gallery, plots_gallery],
        )

    return demo


def launch_webui(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    config_path: Path | None = None,
) -> None:
    """启动 Gradio 服务（阻塞直到退出）。"""
    try:
        import gradio as gr  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Web 界面需要安装 gradio，请执行：\n  pip install gradio\n"
        ) from exc

    demo = build_app(config_path=config_path)
    print(f"MembraneQuant 中文界面 → http://{host}:{port}")
    print("浏览器打不开时，请手动复制上面的地址。")

    import os
    import string
    allowed_paths = []
    if os.name == "nt":
        try:
            import ctypes
            bitmask = ctypes.windll.kernel32.GetLogicalDrives()
            for letter in string.ascii_uppercase:
                if bitmask & 1:
                    allowed_paths.append(f"{letter}:\\")
                bitmask >>= 1
        except Exception:
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    allowed_paths.append(drive)
    else:
        allowed_paths.append("/")

    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=True,
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":
    import sys

    _repo = Path(__file__).resolve().parent.parent
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    launch_webui()
