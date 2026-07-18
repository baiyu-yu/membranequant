"""数据可视化和统计图表生成模块（PPT 级 300 dpi PNG）

生成膜定位 / 共定位分析结果图：
- 分组柱状图、箱线图+蜂群、小提琴图
- 多指标并排对比
- 质控统计、相关性热图
- 膜 vs 胞质散点、共定位指标分布
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import ensure_dir

sns.set_style("whitegrid")
sns.set_palette("husl")

# 设置中文字体和样式 (必须在 sns.set_style 之后，否则会被其覆盖重置)
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

PPT_DPI = 300

METRIC_LABELS = {
    "M/C_DiI": "M/C (DiI引导膜/质比)",
    "MEI": "膜富集指数 MEI",
    "Manders_M1": "Manders M1 (绿∩红)",
    "Manders_M2": "Manders M2 (红∩绿)",
    "EdgeCenterRatio": "边缘/中心强度比",
    "M/C": "M/C (几何膜环)",
    "MembraneFraction": "膜荧光占比",
    "MembraneFraction_DiI": "DiI膜荧光占比",
    "PearsonWhole": "Pearson r (全细胞)",
    "PearsonMem": "Pearson r (膜环)",
    "PearsonDiI": "Pearson r (DiI膜)",
    "RedCoverage": "DiI覆盖率",
}


def _pass_df(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    df = results_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    if "QC" in df.columns:
        df = df[df["QC"] == "pass"].copy()
    # 全局自动排除 EdgeCenterRatio > 10 的亮背景噪点异常细胞
    if "EdgeCenterRatio" in df.columns:
        df = df[df["EdgeCenterRatio"] <= 10].copy()
    return df


def _ensure_condition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Condition" not in out.columns or out["Condition"].isna().all():
        for alt in ("Group", "Drug", "Experiment"):
            if alt in out.columns and out[alt].notna().any():
                out["Condition"] = out[alt].astype(str)
                break
        else:
            out["Condition"] = "All"
    out["Condition"] = out["Condition"].astype(str)
    return out


def _empty_fig(output_path: Path, msg: str = "无数据") -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=16)
    ax.set_axis_off()
    fig.savefig(output_path, dpi=PPT_DPI, bbox_inches="tight")
    plt.close(fig)


def _add_significance_brackets(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    conditions: list[str],
    is_boxplot: bool = False
) -> None:
    """自动在图上添加显著性标识括号及 p 值符号 (*, **, ***, ns)。"""
    import re
    from scipy import stats

    # 解析 Condition，如 104d1 -> Exp='104', Drug='d', Group='1'
    pattern = re.compile(r"^([a-zA-Z0-9]+?)([a-zA-Z]+)([0-9]+)$")

    cond_meta = {}
    for cond in conditions:
        subset = df[df["Condition"] == cond]
        if not subset.empty:
            row = subset.iloc[0]
            exp = row.get("Experiment")
            drug = row.get("Drug")
            grp = row.get("Group")
            if pd.notna(exp) and pd.notna(drug) and pd.notna(grp):
                cond_meta[cond] = (str(exp), str(drug), str(grp))
                continue
        # Fallback to regex
        m = pattern.match(cond)
        if m:
            cond_meta[cond] = m.groups()

    # 确定需要比较的配对
    pairs = []
    for i, c1 in enumerate(conditions):
        for j, c2 in enumerate(conditions):
            if i >= j:
                continue
            if c1 not in cond_meta or c2 not in cond_meta:
                continue
            e1, d1, g1 = cond_meta[c1]
            e2, d2, g2 = cond_meta[c2]

            # 条件：必须在同一个 Experiment 内
            if e1 != e2:
                continue

            # 配对类型 1：相同 Group，不同 Drug (如 wd1 vs we1, 104d1 vs 104e1)
            # 配对类型 2：相同 Drug，不同 Group (如 wd1 vs wd3, wd2 vs wd3, wd1 vs wd2)
            is_drug_comparison = (g1 == g2 and d1 != d2)
            is_group_comparison = (d1 == d2 and g1 != g2)

            if is_drug_comparison or is_group_comparison:
                pairs.append((i, j, c1, c2))

    if not pairs:
        return

    # 计算每个 x 位置的初始最高 y 值
    x_offset = 1 if is_boxplot else 0

    y_maxs = {}
    for idx, cond in enumerate(conditions):
        vals = df[df["Condition"] == cond][metric].dropna().values
        if len(vals) > 0:
            # 取 95 分位数作为图表主体的高度，防止被个别异常大值拉得太高
            y_maxs[idx] = float(np.percentile(vals, 95))
        else:
            y_maxs[idx] = 1.0

    # 图表整体的 y 范围
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    spacing = 0.06 * y_range
    bracket_h = 0.015 * y_range

    # 跟踪当前每个 x 位置的最高 y 占用
    top_y = {idx: y_maxs[idx] for idx in range(len(conditions))}

    # 对 pairs 按照 x 间距从小到大排序，这样跨度小的括号在下面，跨度大的在上面
    pairs.sort(key=lambda p: p[1] - p[0])

    for idx1, idx2, c1, c2 in pairs:
        vals1 = df[df["Condition"] == c1][metric].dropna().values
        vals2 = df[df["Condition"] == c2][metric].dropna().values
        if len(vals1) < 3 or len(vals2) < 3:
            continue

        # 计算 p 值
        _, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)
        if p_val < 0.0001:
            text = "****"
        elif p_val < 0.001:
            text = "***"
        elif p_val < 0.01:
            text = "**"
        elif p_val < 0.05:
            text = "*"
        else:
            text = "ns"

        # 确定括号 of y 轴高度：横跨 idx1 到 idx2 之间的最大高度加上 spacing
        span_y = [top_y[i] for i in range(idx1, idx2 + 1)]
        y_coord = max(span_y) + spacing

        # 绘制括号
        x1 = idx1 + x_offset
        x2 = idx2 + x_offset
        ax.plot([x1, x1, x2, x2], [y_coord - bracket_h, y_coord, y_coord, y_coord - bracket_h], lw=1.2, c="black", zorder=5)

        # 绘制文本
        ax.text((x1 + x2) * 0.5, y_coord + 0.01 * y_range, text, ha="center", va="bottom", fontsize=9, fontweight="bold", zorder=6)

        # 更新该跨度内所有 x 位置的最高 y 占用，为下一个更高的括号留出空间
        for i in range(idx1, idx2 + 1):
            top_y[i] = y_coord + 2 * bracket_h

    # 调整 y 轴上限，防止括号超出绘图区
    new_y_max = max(top_y.values()) + spacing
    ax.set_ylim(y_min, new_y_max)


def plot_mc_comparison_bar(
    summary_df: pd.DataFrame,
    output_path: Path,
    results_df: pd.DataFrame | None = None,
    title: str = "膜定位指数对比",
    ylabel: str | None = None,
) -> None:
    """柱状图对比各组均值，带 SEM 误差线。"""
    ensure_dir(output_path.parent)

    if summary_df is None or summary_df.empty:
        _empty_fig(output_path)
        return

    df = summary_df.copy()
    mean_col = "Mean_M/C" if "Mean_M/C" in df.columns else ("Mean" if "Mean" in df.columns else None)
    if mean_col is None:
        _empty_fig(output_path, "无均值列")
        return

    if "Condition" not in df.columns or df["Condition"].isna().all():
        df["Condition"] = df.get("Group", "Unknown")

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.9), 6))

    x = np.arange(len(df))
    means = df[mean_col].values.astype(float)
    sems = df["SEM"].values.astype(float) if "SEM" in df.columns else np.zeros_like(means)

    bars = ax.bar(
        x, means, yerr=sems, capsize=5, alpha=0.85, edgecolor="black", linewidth=1.5
    )
    colors = sns.color_palette("Set2", len(df))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel("组别", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel or "指标均值 ± SEM", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Condition"].astype(str), rotation=45, ha="right")

    y_top = float(np.nanmax(means + sems)) if len(means) else 1.0
    for i, (mean, sem) in enumerate(zip(means, sems)):
        ax.text(
            i,
            mean + sem + max(0.02 * y_top, 0.02),
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    if "N_Cells" in df.columns:
        for i, n in enumerate(df["N_Cells"]):
            ax.text(i, 0.0, f"n={int(n)}", ha="center", va="bottom", fontsize=8, style="italic")

    # 添加显著性标记
    if results_df is not None:
        metric = summary_df["Metric"].iloc[0] if "Metric" in summary_df.columns else "M/C_DiI"
        conditions = list(df["Condition"].astype(str).values)
        clean_results = _ensure_condition(_pass_df(results_df))
        if "EdgeCenterRatio" in clean_results.columns:
            clean_results = clean_results[clean_results["EdgeCenterRatio"] <= 10]
        _add_significance_brackets(ax, clean_results, metric, conditions, is_boxplot=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PPT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_metric_boxplot(
    results_df: pd.DataFrame,
    output_path: Path,
    metric: str = "M/C_DiI",
    title: str | None = None,
) -> None:
    """箱线图 + 单细胞散点。"""
    ensure_dir(output_path.parent)
    df = _ensure_condition(_pass_df(results_df))
    if df.empty or metric not in df.columns:
        _empty_fig(output_path, f"无指标 {metric}")
        return

    df = df.dropna(subset=[metric])
    if df.empty:
        _empty_fig(output_path, "无有效数值")
        return

    conditions = list(df["Condition"].unique())
    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.2), 6))

    data = [df[df["Condition"] == c][metric].values for c in conditions]
    bp = ax.boxplot(
        data,
        tick_labels=conditions,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
        boxprops=dict(facecolor="lightblue", alpha=0.65),
        medianprops=dict(color="darkblue", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    colors = sns.color_palette("Set2", len(conditions))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    rng = np.random.default_rng(42)
    for i, condition in enumerate(conditions, 1):
        y = df[df["Condition"] == condition][metric].values
        x = rng.normal(i, 0.06, size=len(y))
        ax.scatter(x, y, alpha=0.45, s=22, color="0.3", edgecolors="white", linewidth=0.4, zorder=3)

    # 添加显著性标记
    clean_df = df.copy()
    if "EdgeCenterRatio" in clean_df.columns:
        clean_df = clean_df[clean_df["EdgeCenterRatio"] <= 10]
    _add_significance_brackets(ax, clean_df, metric, conditions, is_boxplot=True)

    label = METRIC_LABELS.get(metric, metric)
    ax.set_xlabel("组别", fontsize=12, fontweight="bold")
    ax.set_ylabel(label, fontsize=12, fontweight="bold")
    ax.set_title(title or f"{label} 分布", fontsize=14, fontweight="bold", pad=20)
    ax.tick_params(axis="x", rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PPT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_mc_boxplot(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "膜定位指数(M/C)分布",
) -> None:
    """兼容旧接口：优先 M/C_DiI，否则 M/C。"""
    df = results_df if results_df is not None else pd.DataFrame()
    metric = "M/C_DiI" if "M/C_DiI" in df.columns else "M/C"
    # 允许调用方把任意指标拷到 M/C 列
    if "M/C" in df.columns and title and "M/C" not in title:
        metric = "M/C"
    plot_metric_boxplot(df, output_path, metric=metric if metric in df.columns else "M/C", title=title)


def plot_metric_violin(
    results_df: pd.DataFrame,
    output_path: Path,
    metric: str = "M/C_DiI",
    title: str | None = None,
) -> None:
    """小提琴图（发表常用）。"""
    ensure_dir(output_path.parent)
    df = _ensure_condition(_pass_df(results_df))
    if df.empty or metric not in df.columns:
        _empty_fig(output_path, f"无指标 {metric}")
        return
    df = df.dropna(subset=[metric])
    if df.empty:
        _empty_fig(output_path)
        return

    fig, ax = plt.subplots(figsize=(max(8, df["Condition"].nunique() * 1.2), 6))
    sns.violinplot(
        data=df,
        x="Condition",
        y=metric,
        hue="Condition",
        inner="quartile",
        cut=0,
        ax=ax,
        palette="Set2",
        legend=False,
    )
    sns.stripplot(
        data=df,
        x="Condition",
        y=metric,
        color="0.25",
        alpha=0.35,
        size=3,
        ax=ax,
    )
    # 添加显著性标记
    conditions = list(df["Condition"].unique())
    clean_df = df.copy()
    if "EdgeCenterRatio" in clean_df.columns:
        clean_df = clean_df[clean_df["EdgeCenterRatio"] <= 10]
    _add_significance_brackets(ax, clean_df, metric, conditions, is_boxplot=False)

    label = METRIC_LABELS.get(metric, metric)
    ax.set_xlabel("组别", fontsize=12, fontweight="bold")
    ax.set_ylabel(label, fontsize=12, fontweight="bold")
    ax.set_title(title or f"{label} 小提琴图", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PPT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_multi_metric_bars(
    results_df: pd.DataFrame,
    output_path: Path,
    metrics: list[str] | None = None,
) -> None:
    """多指标均值 ± SEM 分组柱状图（一张图看趋势是否一致）。"""
    ensure_dir(output_path.parent)
    df = _ensure_condition(_pass_df(results_df))
    metrics = metrics or ["M/C_DiI", "MEI", "Manders_M1", "EdgeCenterRatio", "PearsonWhole"]
    metrics = [m for m in metrics if m in df.columns]
    if df.empty or not metrics:
        _empty_fig(output_path, "无可用指标")
        return

    rows = []
    for m in metrics:
        for cond, sub in df.groupby("Condition"):
            vals = sub[m].dropna()
            if vals.empty:
                continue
            n = len(vals)
            mean = float(vals.mean())
            sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            rows.append({"Condition": cond, "Metric": METRIC_LABELS.get(m, m), "Mean": mean, "SEM": sem, "N": n})
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        _empty_fig(output_path)
        return

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 1.8), 6))
    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Mean",
        hue="Condition",
        errorbar=None,
        ax=ax,
        palette="Set2",
        edgecolor="black",
        linewidth=0.8,
    )
    # Manual SEM error bars
    conditions = list(plot_df["Condition"].unique())
    metric_labels = list(plot_df["Metric"].unique())
    n_hue = len(conditions)
    width = 0.8 / max(n_hue, 1)
    for i, met in enumerate(metric_labels):
        for j, cond in enumerate(conditions):
            row = plot_df[(plot_df["Metric"] == met) & (plot_df["Condition"] == cond)]
            if row.empty:
                continue
            mean = float(row["Mean"].iloc[0])
            sem = float(row["SEM"].iloc[0])
            x = i - 0.4 + width / 2 + j * width
            ax.errorbar(x, mean, yerr=sem, fmt="none", ecolor="black", capsize=3, linewidth=1)

    ax.set_xlabel("分析指标", fontsize=12, fontweight="bold")
    ax.set_ylabel("均值 ± SEM", fontsize=12, fontweight="bold")
    ax.set_title("多指标膜定位/共定位趋势对比", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="组别", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PPT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_qc_statistics(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """质控统计图：通过率和失败原因分布。"""
    ensure_dir(output_path.parent)

    if results_df is None or results_df.empty or "QC" not in results_df.columns:
        _empty_fig(output_path, "无QC数据")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    qc_counts = results_df["QC"].value_counts()
    labels = []
    values = []
    for key, lab in [("pass", "通过"), ("fail", "失败")]:
        if key in qc_counts.index:
            labels.append(lab)
            values.append(int(qc_counts[key]))
    # Include any other QC labels
    for key in qc_counts.index:
        if key not in ("pass", "fail"):
            labels.append(str(key))
            values.append(int(qc_counts[key]))

    if values:
        colors = sns.color_palette("Set2", len(values))
        ax1.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": 12, "fontweight": "bold"},
        )
    ax1.set_title(
        f"质控通过率\n总计 {len(results_df)} 个细胞",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    if "QC_Reason" in results_df.columns:
        failed = results_df[results_df["QC"] == "fail"]
        if not failed.empty and failed["QC_Reason"].notna().any():
            reasons = failed["QC_Reason"].value_counts().head(8)
            reason_map = {
                "area_too_small": "面积过小",
                "area_too_large": "面积过大",
                "touch_border": "接触边界",
                "eccentricity": "离心率过高",
                "green_saturation": "绿色饱和",
                "red_coverage": "红色覆盖不足",
                "membrane_pixels": "膜环像素不足",
                "membrane_pixels_low": "膜环像素不足",
            }
            reasons.index = [reason_map.get(r, r) for r in reasons.index]
            ax2.barh(
                range(len(reasons)),
                reasons.values,
                color="coral",
                alpha=0.85,
                edgecolor="black",
                linewidth=1,
            )
            ax2.set_yticks(range(len(reasons)))
            ax2.set_yticklabels(reasons.index, fontsize=10)
            ax2.set_xlabel("细胞数", fontsize=11, fontweight="bold")
            ax2.set_title("质控失败原因", fontsize=13, fontweight="bold", pad=15)
            ax2.invert_yaxis()
            for i, v in enumerate(reasons.values):
                ax2.text(v + 0.5, i, str(v), va="center", fontsize=9, fontweight="bold")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.grid(axis="x", alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "无失败细胞", ha="center", va="center", fontsize=14, transform=ax2.transAxes)
            ax2.set_axis_off()
    else:
        ax2.text(0.5, 0.5, "无QC_Reason数据", ha="center", va="center", fontsize=14, transform=ax2.transAxes)
        ax2.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=PPT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """指标相关性热图。"""
    ensure_dir(output_path.parent)
    df = _pass_df(results_df)
    if df.empty:
        _empty_fig(output_path)
        return

    numeric_cols = [
        "M/C_DiI",
        "MEI",
        "Manders_M1",
        "EdgeCenterRatio",
        "M/C",
        "MembraneFraction",
        "PearsonWhole",
        "RedCoverage",
        "Area",
        "MembraneGreen_DiI",
        "CytoGreen_DiI",
    ]
    available_cols = [col for col in numeric_cols if col in df.columns]
    if len(available_cols) < 2:
        _empty_fig(output_path, "数值列不足")
        return

    corr = df[available_cols].corr()
    labels = [METRIC_LABELS.get(col, col) for col in available_cols]

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("相关系数", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.iloc[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="black" if abs(val) < 0.55 else "white",
                fontsize=8,
                fontweight="bold",
            )
    ax.set_title("指标相关性热图", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PPT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_membrane_vs_cyto(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """散点图：膜绿色 vs 胞质绿色（优先 DiI 引导）。"""
    ensure_dir(output_path.parent)
    df = _ensure_condition(_pass_df(results_df))

    xcol = "CytoGreen_DiI" if "CytoGreen_DiI" in df.columns else "CytoGreen"
    ycol = "MembraneGreen_DiI" if "MembraneGreen_DiI" in df.columns else "MembraneGreen"
    if df.empty or xcol not in df.columns or ycol not in df.columns:
        _empty_fig(output_path)
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    conditions = list(df["Condition"].unique())
    colors = sns.color_palette("Set2", len(conditions))
    for condition, color in zip(conditions, colors):
        mask = df["Condition"] == condition
        ax.scatter(
            df.loc[mask, xcol],
            df.loc[mask, ycol],
            label=condition,
            alpha=0.65,
            s=40,
            color=color,
            edgecolors="black",
            linewidth=0.4,
        )

    max_val = max(float(df[xcol].max()), float(df[ycol].max()), 1e-6)
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.35, linewidth=2, label="M/C=1")
    ax.set_xlabel("胞质绿色荧光强度", fontsize=12, fontweight="bold")
    ax.set_ylabel("膜绿色荧光强度", fontsize=12, fontweight="bold")
    ax.set_title("膜 vs 胞质荧光强度（对角线以上 = 膜富集）", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="best", frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PPT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_area_distribution(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """细胞面积分布直方图。"""
    ensure_dir(output_path.parent)
    df = _ensure_condition(_pass_df(results_df))
    if df.empty or "Area" not in df.columns:
        _empty_fig(output_path)
        return

    conditions = list(df["Condition"].unique())
    n_conditions = len(conditions)
    n_show = min(n_conditions, 4)
    fig, axes = plt.subplots(1, n_show, figsize=(min(n_show * 4, 16), 4))
    if n_show == 1:
        axes = [axes]

    for i, (condition, ax) in enumerate(zip(conditions[:n_show], axes)):
        data = df[df["Condition"] == condition]["Area"].values
        ax.hist(
            data,
            bins=30,
            color=sns.color_palette("Set2", n_conditions)[i],
            alpha=0.75,
            edgecolor="black",
        )
        ax.axvline(data.mean(), color="red", linestyle="--", linewidth=2, label=f"均值: {data.mean():.0f}")
        ax.set_xlabel("面积 (像素)", fontsize=10, fontweight="bold")
        ax.set_ylabel("细胞数", fontsize=10, fontweight="bold")
        ax.set_title(f"{condition}\nn={len(data)}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("细胞面积分布", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PPT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_coloc_dashboard(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """共定位三联图：Manders M1 / Pearson / MEI。"""
    ensure_dir(output_path.parent)
    df = _ensure_condition(_pass_df(results_df))
    metrics = [m for m in ("Manders_M1", "PearsonWhole", "MEI") if m in df.columns]
    if df.empty or not metrics:
        _empty_fig(output_path, "无共定位指标")
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sns.boxplot(
            data=df.dropna(subset=[metric]),
            x="Condition",
            y=metric,
            hue="Condition",
            ax=ax,
            palette="Set2",
            showfliers=False,
            legend=False,
        )
        sns.stripplot(
            data=df.dropna(subset=[metric]),
            x="Condition",
            y=metric,
            ax=ax,
            color="0.25",
            alpha=0.4,
            size=3,
        )
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=40)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("共定位 / 膜富集指标面板（PPT可用）", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=PPT_DPI, bbox_inches="tight")
    plt.close(fig)


def build_summary_from_results(
    results_df: pd.DataFrame,
    metric: str = "M/C_DiI",
    group_col: str = "Condition",
    filter_qc: bool = True,
) -> pd.DataFrame:
    """从 results.csv 构建 summary（可视化前端用）。"""
    df = results_df.copy().replace([np.inf, -np.inf], np.nan)
    if filter_qc and "QC" in df.columns:
        df = df[df["QC"] == "pass"]
    # 汇总计算时全局自动排除 EdgeCenterRatio > 10 的噪点异常细胞，确保均值不受偏离影响
    if "EdgeCenterRatio" in df.columns:
        df = df[df["EdgeCenterRatio"] <= 10]
    if group_col not in df.columns:
        df = _ensure_condition(df)
        group_col = "Condition"
    if metric not in df.columns:
        for fallback in ("M/C_DiI", "M/C", "Manders_M1", "MEI"):
            if fallback in df.columns:
                metric = fallback
                break
    rows = []
    for g, sub in df.groupby(group_col):
        vals = sub[metric].dropna() if metric in sub.columns else pd.Series(dtype=float)
        if vals.empty:
            continue
        n = int(vals.count())
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else 0.0
        sem = sd / np.sqrt(n) if n > 1 else 0.0
        rows.append(
            {
                "Condition": str(g),
                "Mean_M/C": mean,
                "Mean": mean,
                "SD": sd,
                "SEM": sem,
                "N_Cells": n,
                "Metric": metric,
            }
        )
    return pd.DataFrame(rows)


def generate_all_plots(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    metric: str = "M/C_DiI",
) -> list[Path]:
    """生成所有统计图表，返回生成的文件路径列表。"""
    plots_dir = ensure_dir(Path(output_dir) / "plots")
    saved: list[Path] = []

    # If summary empty or wrong metric, rebuild
    if summary_df is None or summary_df.empty or "Mean_M/C" not in summary_df.columns:
        summary_df = build_summary_from_results(results_df, metric=metric)

    jobs: list[tuple[str, Any]] = [
        ("01_metric_bar.png", lambda p: plot_mc_comparison_bar(
            summary_df, p, results_df=results_df, title=f"{METRIC_LABELS.get(metric, metric)} 组间比较", ylabel=METRIC_LABELS.get(metric, metric)
        )),
        ("02_metric_boxplot.png", lambda p: plot_metric_boxplot(results_df, p, metric=metric)),
        ("03_qc_statistics.png", lambda p: plot_qc_statistics(results_df, p)),
        ("04_correlation_heatmap.png", lambda p: plot_correlation_heatmap(results_df, p)),
        ("05_membrane_vs_cyto_scatter.png", lambda p: plot_scatter_membrane_vs_cyto(results_df, p)),
        ("06_area_distribution.png", lambda p: plot_area_distribution(results_df, p)),
        ("07_multi_metric_bars.png", lambda p: plot_multi_metric_bars(results_df, p)),
        ("08_coloc_dashboard.png", lambda p: plot_coloc_dashboard(results_df, p)),
        ("09_mei_violin.png", lambda p: plot_metric_violin(results_df, p, metric="MEI" if "MEI" in results_df.columns else metric)),
        ("10_manders_boxplot.png", lambda p: plot_metric_boxplot(results_df, p, metric="Manders_M1" if "Manders_M1" in results_df.columns else metric)),
        ("11_batch_effect_comparison.png", lambda p: plot_batch_effect_comparison(results_df, p, metric=metric)),
    ]

    print("📊 生成统计图表（300 dpi，可用于 PPT）...")
    for name, fn in jobs:
        path = plots_dir / name
        try:
            fn(path)
            saved.append(path)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ⚠️ {name} 生成失败: {e}")

    print(f"\n📁 所有图表已保存到: {plots_dir}")
    return saved


def plot_batch_effect_comparison(
    results_df: pd.DataFrame,
    output_path: Path,
    metric: str = "M/C_DiI",
    title: str | None = None,
) -> None:
    """绘制批次效应对比图（实验 104 vs 实验 w）。

    自变量X轴为药物（d, e），颜色（hue）为批次（104, w）。
    并在图上标出 104 与 w 在同种药物下的显著性差异，直观展示系统性偏差（批次效应）。
    """
    import numpy as np
    import pandas as pd
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns

    ensure_dir(output_path.parent)
    df = _ensure_condition(_pass_df(results_df))
    if df.empty or metric not in df.columns:
        _empty_fig(output_path, f"无指标 {metric}")
        return

    # 排除极低表达/噪点异常细胞 (EdgeCenterRatio > 10)
    if "EdgeCenterRatio" in df.columns:
        df = df[df["EdgeCenterRatio"] <= 10]

    df = df.dropna(subset=[metric]).copy()
    if df.empty:
        _empty_fig(output_path, "无有效数值")
        return

    df["Exp"] = df["Experiment"].astype(str).replace({"104.0": "104", "104": "Batch 104", "w": "Batch w"})
    df["Drug_Name"] = df["Drug"].astype(str).replace({"d": "Drug d", "e": "Drug e"})

    # 仅保留主要的实验和药物组
    plot_df = df[df["Exp"].isin(["Batch 104", "Batch w"]) & df["Drug_Name"].isin(["Drug d", "Drug e"])].copy()
    if plot_df.empty:
        _empty_fig(output_path, "没有匹配的批次/药物组别进行对比")
        return

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))

    # 定义高端学术色卡：Batch 104 (青绿色), Batch w (珊瑚色)
    custom_colors = ["#4db6ac", "#ff8a65"]

    # 绘制分组箱线图
    sns.boxplot(
        data=plot_df,
        x="Drug_Name",
        y=metric,
        hue="Exp",
        hue_order=["Batch 104", "Batch w"],
        order=["Drug d", "Drug e"],
        ax=ax,
        palette=custom_colors,
        width=0.55,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=4, zorder=4),
        boxprops=dict(alpha=0.8, linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        medianprops=dict(linewidth=1.8, color="#222222"),
    )

    # 绘制抖动单细胞散点（以和箱子对齐）- 显著减弱散点视觉占比，更干净
    rng = np.random.default_rng(42)
    positions = {
        ("Drug d", "Batch 104"): -0.15,
        ("Drug d", "Batch w"): 0.15,
        ("Drug e", "Batch 104"): 0.85,
        ("Drug e", "Batch w"): 1.15,
    }

    for (drug, exp), x_center in positions.items():
        sub = plot_df[(plot_df["Drug_Name"] == drug) & (plot_df["Exp"] == exp)]
        y_vals = sub[metric].values
        if len(y_vals) > 0:
            x_vals = rng.normal(x_center, 0.04, size=len(y_vals))
            ax.scatter(x_vals, y_vals, alpha=0.15, s=8, color="0.3", edgecolors="none", linewidth=0, zorder=3)

    # 统计学检验与画括号
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    spacing = 0.06 * y_range
    bracket_h = 0.015 * y_range

    # 定义所有的比较对 (x1, x2, val1, val2, description)
    comparisons = [
        # 1. d组: Batch 104 vs Batch w
        ((-0.15, 0.15), 
         plot_df[(plot_df["Drug_Name"] == "Drug d") & (plot_df["Exp"] == "Batch 104")][metric].values,
         plot_df[(plot_df["Drug_Name"] == "Drug d") & (plot_df["Exp"] == "Batch w")][metric].values,
         "Batch 104 vs Batch w (Drug d)"),
         
        # 2. e组: Batch 104 vs Batch w
        ((0.85, 1.15),
         plot_df[(plot_df["Drug_Name"] == "Drug e") & (plot_df["Exp"] == "Batch 104")][metric].values,
         plot_df[(plot_df["Drug_Name"] == "Drug e") & (plot_df["Exp"] == "Batch w")][metric].values,
         "Batch 104 vs Batch w (Drug e)"),
         
        # 3. Batch 104: Drug d vs Drug e
        ((-0.15, 0.85),
         plot_df[(plot_df["Drug_Name"] == "Drug d") & (plot_df["Exp"] == "Batch 104")][metric].values,
         plot_df[(plot_df["Drug_Name"] == "Drug e") & (plot_df["Exp"] == "Batch 104")][metric].values,
         "Drug d vs Drug e (Batch 104)"),
         
        # 4. Batch w: Drug d vs Drug e (排除 wd3 异常)
        ((0.15, 1.15),
         plot_df[(plot_df["Drug_Name"] == "Drug d") & (plot_df["Exp"] == "Batch w") & (plot_df["Condition"] != "wd3")][metric].values,
         plot_df[(plot_df["Drug_Name"] == "Drug e") & (plot_df["Exp"] == "Batch w")][metric].values,
         "Drug d vs Drug e (Batch w)"),
    ]

    x_coords = [-0.15, 0.15, 0.85, 1.15]
    y_maxs = {}
    for idx, (drug, exp) in enumerate([
        ("Drug d", "Batch 104"),
        ("Drug d", "Batch w"),
        ("Drug e", "Batch 104"),
        ("Drug e", "Batch w")
    ]):
        vals = plot_df[(plot_df["Drug_Name"] == drug) & (plot_df["Exp"] == exp)][metric].dropna().values
        y_maxs[idx] = float(np.percentile(vals, 95)) if len(vals) > 0 else 1.0

    top_y = {idx: y_maxs[idx] for idx in range(4)}

    # 按间距排序：横向的大括号(1.0)在上方，垂直的小括号(0.3)在下方
    comparisons.sort(key=lambda c: c[0][1] - c[0][0])

    y_limits = []
    for (x1, x2), vals1, vals2, label_text in comparisons:
        if len(vals1) < 3 or len(vals2) < 3:
            continue

        _, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)
        if p_val < 0.0001:
            text = "****"
        elif p_val < 0.001:
            text = "***"
        elif p_val < 0.01:
            text = "**"
        elif p_val < 0.05:
            text = "*"
        else:
            text = "ns"

        idx1 = x_coords.index(x1)
        idx2 = x_coords.index(x2)

        span_y = [top_y[i] for i in range(idx1, idx2 + 1)]
        y_coord = max(span_y) + spacing

        # 绘制括号
        ax.plot([x1, x1, x2, x2], [y_coord - bracket_h, y_coord, y_coord, y_coord - bracket_h], lw=1.0, c="0.3", zorder=5)
        # 绘制标注文本 (含星号与简要描述)
        annot = f"{text} (p={p_val:.2e})" if p_val < 0.05 else "ns"
        ax.text((x1 + x2) * 0.5, y_coord + 0.005 * y_range, annot, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="0.2", zorder=6)

        # 更新最高位置
        for i in range(idx1, idx2 + 1):
            top_y[i] = y_coord + 2.5 * bracket_h
        y_limits.append(y_coord + 3 * bracket_h)

    # 调整 Y 轴上限
    if y_limits:
        ax.set_ylim(y_min, max(y_limits) + 0.05 * y_range)

    label = METRIC_LABELS.get(metric, metric)
    ax.set_xlabel("药物处理", fontsize=11, fontweight="bold")
    ax.set_ylabel(label, fontsize=11, fontweight="bold")
    ax.set_title(title or f"批次效应与加药对照分析: {label}", fontsize=12, fontweight="bold", pad=20)
    
    # 极简 Nature 样式
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    
    # 轻量级水平网格线
    ax.grid(True, axis="y", linestyle=":", alpha=0.5, color="#b0bec5")
    ax.legend(title="实验批次", frameon=True, facecolor="white", edgecolor="none")

    fig.tight_layout()
    fig.savefig(output_path, dpi=PPT_DPI, bbox_inches="tight")
    plt.close(fig)
