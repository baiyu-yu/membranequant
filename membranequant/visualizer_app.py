"""MembraneQuant 结果可视化前端（Gradio）。

功能：
  - 上传 results.csv 一键出图
  - 选择分组列 / 分析指标
  - 多面板预览（柱状、箱线、小提琴、共定位、多指标趋势、QC）
  - 一键保存全部 PNG 到指定目录（300 dpi，直接放 PPT）
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure the parent directory of membranequant is in sys.path
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from membranequant.plots import (
    METRIC_LABELS,
    build_summary_from_results,
    generate_all_plots,
    plot_coloc_dashboard,
    plot_correlation_heatmap,
    plot_mc_comparison_bar,
    plot_metric_boxplot,
    plot_metric_violin,
    plot_multi_metric_bars,
    plot_qc_statistics,
    plot_scatter_membrane_vs_cyto,
    plot_batch_effect_comparison,
)

METRIC_CHOICES = [
    ("RatioOfMeans_T_R — T/R 均值比 (Dual 膜区)【主终点·DualCellQuant论文标准】", "RatioOfMeans_T_R"),
    ("Enrichment_Membrane_vs_Whole — 膜富集倍数【辅助终点·膜相对全细胞】", "Enrichment_Membrane_vs_Whole"),
    ("Ratio_T_over_R — T/R 像素比 (Dual 膜区)【探索性指标】", "Ratio_T_over_R"),
    ("MembraneGreen — 膜区 EGFP 均值", "MembraneGreen"),
    ("MembraneRed — 膜区 DiI 均值", "MembraneRed"),
    ("MembraneFraction — 膜区 EGFP 积分占比", "MembraneFraction"),
    ("WholeGreen — 全细胞 EGFP 均值", "WholeGreen"),
    ("WholeRed — 全细胞 DiI 均值", "WholeRed"),
    ("RedCoverage — DiI 覆盖率", "RedCoverage"),
    ("M/C_DiI — DiI引导膜/质比（旧版）", "M/C_DiI"),
    ("MEI — 膜富集指数（旧版）", "MEI"),
    ("Manders_M1 — 绿与DiI共现比例（旧版）", "Manders_M1"),
    ("EdgeCenterRatio — 边缘/中心强度比（旧版）", "EdgeCenterRatio"),
    ("PearsonWhole — 全细胞 Pearson r（旧版）", "PearsonWhole"),
    ("M/C — 几何膜环膜/质比（旧版）", "M/C"),
]


def _load_csv(file_obj) -> pd.DataFrame:
    if file_obj is None:
        raise ValueError("请先上传 results.csv")
    path = getattr(file_obj, "name", None) or str(file_obj)
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
    return df


def _pick_metric(df: pd.DataFrame, selected: str) -> str:
    if selected in df.columns and df[selected].notna().any():
        return selected
    for m in (
        "RatioOfMeans_T_R",
        "Enrichment_Membrane_vs_Whole",
        "Ratio_T_over_R",
        "MembraneGreen",
        "M/C_DiI",
        "MEI",
        "Manders_M1",
        "M/C",
        "MembraneFraction",
    ):
        if m in df.columns and df[m].notna().any():
            return m
    # last resort: first numeric
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    if not nums:
        raise ValueError("CSV 中没有可用的数值指标列")
    return nums[0]


def _prepare_df(df: pd.DataFrame, group_col: str, metric: str, filter_qc: bool) -> tuple[pd.DataFrame, str, str]:
    group_col = (group_col or "Condition").strip()
    if group_col not in df.columns:
        for fb in ("Condition", "Group", "Drug", "Experiment", "Image"):
            if fb in df.columns:
                group_col = fb
                break
        else:
            df = df.copy()
            df["Condition"] = "All"
            group_col = "Condition"

    metric = _pick_metric(df, metric)
    work = df.copy()
    work["Condition"] = work[group_col].astype(str)
    if filter_qc and "QC" in work.columns:
        passed = work[work["QC"] == "pass"].copy()
        if not passed.empty:
            work = passed
    return work, group_col, metric


def process_upload(file_obj, group_col, selected_metric, filter_qc, save_dir):
    """生成图表并可选保存到用户目录。"""
    try:
        df = _load_csv(file_obj)
    except Exception as e:
        return (None,) * 9 + (f"❌ 读取失败：{e}", None)

    try:
        work, used_group, metric = _prepare_df(df, group_col, selected_metric, filter_qc)
    except Exception as e:
        return (None,) * 9 + (f"❌ 数据准备失败：{e}", None)

    temp_dir = Path(tempfile.gettempdir()) / "mq_plots_viz"
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary_from_results(work, metric=metric, group_col="Condition", filter_qc=False)

    paths = {
        "bar": temp_dir / "01_bar.png",
        "box": temp_dir / "02_box.png",
        "violin": temp_dir / "03_violin.png",
        "multi": temp_dir / "04_multi_metric.png",
        "coloc": temp_dir / "05_coloc_dashboard.png",
        "scatter": temp_dir / "06_scatter.png",
        "qc": temp_dir / "07_qc.png",
        "heat": temp_dir / "08_heatmap.png",
        "batch_effect": temp_dir / "11_batch_effect.png",
    }

    label = METRIC_LABELS.get(metric, metric)
    try:
        plot_mc_comparison_bar(
            summary,
            paths["bar"],
            results_df=work,
            title=f"各组 {label} 均值比较",
            ylabel=label,
        )
        plot_metric_boxplot(work, paths["box"], metric=metric, title=f"{label} 箱线图")
        plot_metric_violin(work, paths["violin"], metric=metric, title=f"{label} 小提琴图")
        plot_multi_metric_bars(work, paths["multi"])
        plot_coloc_dashboard(work, paths["coloc"])
        plot_scatter_membrane_vs_cyto(work, paths["scatter"])
        plot_qc_statistics(df, paths["qc"])  # QC 用全量数据
        plot_correlation_heatmap(work, paths["heat"])
        plot_batch_effect_comparison(work, paths["batch_effect"], metric=metric, title=f"批次效应对比 ({label})")
    except Exception as e:
        import traceback

        return (None,) * 9 + (f"❌ 作图失败：{e}\n\n```\n{traceback.format_exc()}\n```", None)

    # Optional save to user folder
    saved_note = ""
    save_path_msg = None
    if save_dir and str(save_dir).strip():
        out = Path(str(save_dir).strip().strip('"').strip("'"))
        try:
            out.mkdir(parents=True, exist_ok=True)
            # Also run full plot suite into that folder
            generate_all_plots(work, summary, out, metric=metric)
            # Copy interactive-session plots with clear names
            ppt_dir = out / "ppt_figures"
            ppt_dir.mkdir(exist_ok=True)
            for key, p in paths.items():
                if p.is_file():
                    shutil.copy2(p, ppt_dir / p.name)
            # Export a short stats table
            summary.to_csv(ppt_dir / f"summary_{metric.replace('/', '_')}.csv", index=False)
            work.to_csv(ppt_dir / "cells_used_for_plot.csv", index=False)
            saved_note = f"\n\n💾 **已保存到** `{ppt_dir.resolve()}`（300 dpi PNG + CSV，可直接拖进 PPT）"
            save_path_msg = str(ppt_dir.resolve())
        except Exception as e:
            saved_note = f"\n\n⚠️ 保存目录失败：{e}（图表仍可在上方预览/右键另存）"

    # Stats text
    lines = [
        f"📊 **分析完成**",
        f"- 分组列: `{used_group}` → 映射为 Condition",
        f"- 主指标: `{metric}`（{label}）",
        f"- 细胞数: **{len(work)}**" + ("（仅 QC=pass）" if filter_qc else ""),
        "",
        f"### 各组 {metric}（Mean ± SEM）",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- **{row['Condition']}** (n={int(row['N_Cells'])}): "
            f"**{row['Mean_M/C']:.4f}** ± {row['SEM']:.4f}"
        )

    # Quick trend hint
    if len(summary) >= 2 and summary["Mean_M/C"].notna().all():
        spread = float(summary["Mean_M/C"].max() - summary["Mean_M/C"].min())
        overall = float(summary["Mean_M/C"].mean()) if summary["Mean_M/C"].mean() else 1.0
        rel = spread / max(abs(overall), 1e-6)
        if rel < 0.05:
            lines.append(
                "\n⚠️ **组间差异很小**（相对跨度 <5%）。"
                "建议同时看 `MEI`、`Manders_M1`、`EdgeCenterRatio` 是否一致；"
                "并检查分割 overlay 与 RedCoverage QC。"
            )
        else:
            lines.append(f"\n✅ 组间均值跨度约 **{spread:.4f}**（相对 {rel*100:.1f}%），可进一步做统计检验。")

    lines.append(saved_note)
    lines.append(
        "\n---\n"
        "**指标怎么读**\n"
        "- `Ratio_T_over_R` / `RatioOfMeans_T_R` ↑ → EGFP 蛋白更富集于 DiI 标注的膜区\n"
        "- `Enrichment_Membrane_vs_Whole` ↑ → 膜区荧光强度显著高于全细胞平均水平\n"
        "- `RedCoverage` → 评估 Reference 膜标注连续度与质控\n"
        "- 对角线以上的膜-胞质散点 → 膜富集"
    )

    return (
        str(paths["bar"]),
        str(paths["box"]),
        str(paths["violin"]),
        str(paths["multi"]),
        str(paths["coloc"]),
        str(paths["scatter"]),
        str(paths["qc"]),
        str(paths["heat"]),
        str(paths["batch_effect"]),
        "\n".join(lines),
        save_path_msg,
    )


def build_gui():
    sns.set_style("whitegrid")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    with gr.Blocks(title="MembraneQuant 结果可视化") as demo:
        gr.Markdown(
            """
# 📊 MembraneQuant 结果可视化（PPT 出图）

上传分析得到的 `results.csv`，自动生成**发表/答辩级**统计图（300 dpi）。

**你的生物学问题**：绿色荧光蛋白的**膜定位是否变化**；红色 DiI 是亲脂性**膜标志**。

| 推荐指标 | 含义 | 何时用 |
|---------|------|--------|
| **RatioOfMeans_T_R** *(主终点)* | Dual 膜区 Target均值 / Reference均值 | DualCellQuant 论文标准膜归一化比值（首选主指标） |
| **Enrichment_Membrane_vs_Whole** *(辅助终点)* | 膜区 EGFP 均值 / 全细胞 EGFP 均值 | 评估 EGFP 相对全细胞/胞浆的膜富集倍数 |
| **Ratio_T_over_R** *(探索指标)* | Dual 膜区 T/R 像素级强度比值 | 像素级比值均值（像素细粒度补充） |
| **RedCoverage** | 膜区 DiI 信号覆盖率 | 质控 / 评估膜标志质量 |

### 🔬 统计学显著性与数据过滤标准

为了保证实验结论的科学严谨性，可视化系统自动运行以下学术级标准流程：

#### 1. 异常数据全局过滤规则 (Outlier Filtering)
* **筛选指标**：`EdgeCenterRatio` (几何外边缘与中心均值比) 
* **过滤阈值**：**`EdgeCenterRatio <= 10`**
* **过滤逻辑**：极暗的假阳性细胞（荧光接近纯噪点）因中心均值接近零，会导致该比值虚高几千倍。全局过滤此指标可系统清除全部图表中的低信噪比噪点异常值（保留真实的单细胞水平分布）。

#### 2. 统计检验模型 (Statistical Testing)
* **分析模型**：采用双样本双尾非等方差 **Welch's t-test** 进行组间两两对比。该检验不要求两组样本数或方差相等，是细胞生物学成像最稳健的分析方法。

#### 3. 显著性等级符号定义 (Significance Stars)
根据国际顶级期刊（如 Nature, Cell, GraphPad Prism）标准定义：
* **`****`**：$p < 0.0001$（极度显著差异）
* **`***`** ：$p < 0.001$（极度显著差异）
* **`**`  ：$p < 0.01$（高度显著差异）
* **`*`**   ：$p < 0.05$（显著差异）
* **`ns`**  ：$p \ge 0.05$（no significance，无显著差异）
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="上传 results.csv", file_types=[".csv"])
                group_col = gr.Textbox(
                    label="分组列",
                    value="Condition",
                    placeholder="Condition / Group / Drug / Experiment",
                    info="按此列分组上色与统计",
                )
                metric_select = gr.Dropdown(
                    choices=METRIC_CHOICES,
                    value="RatioOfMeans_T_R",
                    label="主分析指标",
                )
                filter_qc = gr.Checkbox(label="仅统计 QC=pass 细胞", value=True)
                save_dir = gr.Textbox(
                    label="保存目录（可选，用于 PPT）",
                    placeholder=r"例如 D:\论文图\membrane_ppt",
                    info="填写后点击生成，会把全部 300dpi PNG + CSV 写入该目录下的 ppt_figures/",
                )
                submit_btn = gr.Button("生成可视化图表", variant="primary", size="lg")
                info_output = gr.Markdown("💡 上传 `results.csv` 后点击生成。")
                save_status = gr.Textbox(label="实际保存路径", interactive=False)

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("柱状图"):
                        bar_img = gr.Image(label="组间均值 ± SEM", type="filepath")
                    with gr.Tab("箱线图"):
                        box_img = gr.Image(label="单细胞分布", type="filepath")
                    with gr.Tab("小提琴图"):
                        violin_img = gr.Image(label="分布形态", type="filepath")
                    with gr.Tab("多指标趋势"):
                        multi_img = gr.Image(label="多指标是否一致", type="filepath")
                    with gr.Tab("共定位面板"):
                        coloc_img = gr.Image(label="Manders / Pearson / MEI", type="filepath")
                    with gr.Tab("膜-质散点"):
                        scatter_img = gr.Image(label="膜 vs 胞质", type="filepath")
                    with gr.Tab("质控"):
                        qc_img = gr.Image(label="QC", type="filepath")
                    with gr.Tab("相关性热图"):
                        heat_img = gr.Image(label="指标相关", type="filepath")
                    with gr.Tab("批次效应对比"):
                        batch_effect_img = gr.Image(label="104 vs w 批次效应", type="filepath")

        submit_btn.click(
            fn=process_upload,
            inputs=[file_input, group_col, metric_select, filter_qc, save_dir],
            outputs=[
                bar_img,
                box_img,
                violin_img,
                multi_img,
                coloc_img,
                scatter_img,
                qc_img,
                heat_img,
                batch_effect_img,
                info_output,
                save_status,
            ],
        )

        gr.Markdown(
            """
### 使用提示
1. 先跑主分析 WebUI / CLI 得到 `Results/csv/results.csv`
2. 本页上传 CSV → 选指标 → 填保存路径 → 生成
3. 图片在预览区可右键另存；填了保存目录则自动落盘
4. 若**所有指标都无趋势**：优先回看 `overlays/` 分割是否粘连/漏分割，以及 `RedCoverage` 是否过低
"""
        )

    return demo


if __name__ == "__main__":
    app = build_gui()
    app.launch(share=False, inbrowser=True, server_port=7861)
