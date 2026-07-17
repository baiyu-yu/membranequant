"""数据可视化和统计图表生成模块

生成各种统计图表用于展示膜定位分析结果：
- 分组M/C比较（柱状图、箱线图、散点图）
- 质控统计图
- 分割方法对比
- 相关性分析
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

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_mc_comparison_bar(
    summary_df: pd.DataFrame,
    output_path: Path,
    title: str = "膜定位指数(M/C)对比",
) -> None:
    """柱状图对比各组的M/C均值，带误差线（SEM）
    
    Args:
        summary_df: 汇总数据，需包含 Condition, Mean_M/C, SEM 列
        output_path: 输出图片路径
        title: 图表标题
    """
    ensure_dir(output_path.parent)
    
    if summary_df.empty or "Mean_M/C" not in summary_df.columns:
        # 创建空白图表
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=16)
        ax.set_axis_off()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    
    df = summary_df.copy()
    if "Condition" not in df.columns or df["Condition"].isna().all():
        df["Condition"] = df.get("Group", "Unknown")
    
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.8), 6))
    
    x = np.arange(len(df))
    means = df["Mean_M/C"].values
    sems = df["SEM"].values if "SEM" in df.columns else np.zeros_like(means)
    
    bars = ax.bar(x, means, yerr=sems, capsize=5, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # 颜色渐变
    colors = sns.color_palette("husl", len(df))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel("组别", fontsize=12, fontweight='bold')
    ax.set_ylabel("M/C (膜/胞质荧光比)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Condition"], rotation=45, ha='right')
    
    # 添加数值标签
    for i, (mean, sem) in enumerate(zip(means, sems)):
        ax.text(i, mean + sem + 0.05, f'{mean:.2f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 添加细胞数标注
    if "N_Cells" in df.columns:
        for i, n in enumerate(df["N_Cells"]):
            ax.text(i, 0.05, f'n={n}', 
                    ha='center', va='bottom', fontsize=8, style='italic')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_mc_boxplot(
    results_df: pd.DataFrame,
    output_path: Path,
    title: str = "膜定位指数(M/C)分布",
) -> None:
    """箱线图展示各组M/C分布，叠加散点显示单细胞数据
    
    Args:
        results_df: 单细胞结果数据
        output_path: 输出图片路径
        title: 图表标题
    """
    ensure_dir(output_path.parent)
    
    if results_df.empty or "M/C" not in results_df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=16)
        ax.set_axis_off()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    
    df = results_df[results_df["QC"] == "pass"].copy() if "QC" in results_df.columns else results_df.copy()
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "无通过QC的细胞", ha='center', va='center', fontsize=16)
        ax.set_axis_off()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    
    if "Condition" not in df.columns or df["Condition"].isna().all():
        df["Condition"] = df.get("Group", "Unknown")
    
    fig, ax = plt.subplots(figsize=(max(8, len(df["Condition"].unique()) * 1.2), 6))
    
    # 箱线图
    bp = ax.boxplot(
        [df[df["Condition"] == c]["M/C"].values for c in df["Condition"].unique()],
        labels=df["Condition"].unique(),
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=6),
        boxprops=dict(facecolor='lightblue', alpha=0.6),
        medianprops=dict(color='darkblue', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )
    
    # 叠加散点
    for i, condition in enumerate(df["Condition"].unique(), 1):
        y = df[df["Condition"] == condition]["M/C"].values
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.4, s=20, color='gray', edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel("组别", fontsize=12, fontweight='bold')
    ax.set_ylabel("M/C (膜/胞质荧光比)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_qc_statistics(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """质控统计图：通过率和失败原因分布
    
    Args:
        results_df: 单细胞结果数据
        output_path: 输出图片路径
    """
    ensure_dir(output_path.parent)
    
    if results_df.empty or "QC" not in results_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "无QC数据", ha='center', va='center', fontsize=16)
        ax.set_axis_off()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：通过率饼图
    qc_counts = results_df["QC"].value_counts()
    colors = ['#66c2a5', '#fc8d62']
    explode = (0.05, 0)
    
    ax1.pie(qc_counts.values, labels=['通过', '失败'], autopct='%1.1f%%',
            startangle=90, colors=colors, explode=explode,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title(f"质控通过率\n总计 {len(results_df)} 个细胞", 
                  fontsize=13, fontweight='bold', pad=15)
    
    # 右图：失败原因柱状图
    if "QC_Reason" in results_df.columns:
        failed = results_df[results_df["QC"] == "fail"]
        if not failed.empty and failed["QC_Reason"].notna().any():
            reasons = failed["QC_Reason"].value_counts().head(8)
            
            # 中文化原因
            reason_map = {
                "area_too_small": "面积过小",
                "area_too_large": "面积过大",
                "touch_border": "接触边界",
                "eccentricity": "离心率过高",
                "green_saturation": "绿色饱和",
                "red_coverage": "红色覆盖不足",
                "membrane_pixels": "膜环像素不足",
            }
            reasons.index = [reason_map.get(r, r) for r in reasons.index]
            
            bars = ax2.barh(range(len(reasons)), reasons.values, color='coral', 
                            alpha=0.8, edgecolor='black', linewidth=1)
            ax2.set_yticks(range(len(reasons)))
            ax2.set_yticklabels(reasons.index, fontsize=10)
            ax2.set_xlabel("细胞数", fontsize=11, fontweight='bold')
            ax2.set_title("质控失败原因", fontsize=13, fontweight='bold', pad=15)
            ax2.invert_yaxis()
            
            # 添加数值标签
            for i, v in enumerate(reasons.values):
                ax2.text(v + 0.5, i, str(v), va='center', fontsize=9, fontweight='bold')
            
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.grid(axis='x', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "无失败细胞", ha='center', va='center', 
                     fontsize=14, transform=ax2.transAxes)
            ax2.set_axis_off()
    else:
        ax2.text(0.5, 0.5, "无QC_Reason数据", ha='center', va='center', 
                 fontsize=14, transform=ax2.transAxes)
        ax2.set_axis_off()
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_correlation_heatmap(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """相关性热图：M/C, MembraneFraction, RedCoverage等指标的相关性
    
    Args:
        results_df: 单细胞结果数据
        output_path: 输出图片路径
    """
    ensure_dir(output_path.parent)
    
    df = results_df[results_df["QC"] == "pass"].copy() if "QC" in results_df.columns else results_df.copy()
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=16)
        ax.set_axis_off()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    
    # 选择数值列
    numeric_cols = [
        "M/C", "MembraneFraction", "RedCoverage", "Area",
        "MembraneGreen", "CytoGreen", "MembraneRed"
    ]
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "数值列不足", ha='center', va='center', fontsize=16)
        ax.set_axis_off()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    
    # 计算相关性
    corr = df[available_cols].corr()
    
    # 中文标签
    label_map = {
        "M/C": "M/C比",
        "MembraneFraction": "膜占比",
        "RedCoverage": "红色覆盖",
        "Area": "面积",
        "MembraneGreen": "膜绿色",
        "CytoGreen": "胞质绿色",
        "MembraneRed": "膜红色",
    }
    labels = [label_map.get(col, col) for col in available_cols]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("相关系数", fontsize=11, fontweight='bold')
    
    # 设置刻度
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    
    # 添加数值
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.5 else "white",
                          fontsize=9, fontweight='bold')
    
    ax.set_title("指标相关性热图", fontsize=14, fontweight='bold', pad=20)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_scatter_membrane_vs_cyto(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """散点图：膜绿色 vs 胞质绿色，按组着色
    
    Args:
        results_df: 单细胞结果数据
        output_path: 输出图片路径
    """
    ensure_dir(output_path.parent)
    
    df = results_df[results_df["QC"] == "pass"].copy() if "QC" in results_df.columns else results_df.copy()
    
    if df.empty or "MembraneGreen" not in df.columns or "CytoGreen" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=16)
        ax.set_axis_off()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    
    if "Condition" not in df.columns or df["Condition"].isna().all():
        df["Condition"] = "All"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    conditions = df["Condition"].unique()
    colors = sns.color_palette("husl", len(conditions))
    
    for condition, color in zip(conditions, colors):
        mask = df["Condition"] == condition
        ax.scatter(df.loc[mask, "CytoGreen"], 
                  df.loc[mask, "MembraneGreen"],
                  label=condition, alpha=0.6, s=40, color=color,
                  edgecolors='black', linewidth=0.5)
    
    # 对角线 (M/C = 1)
    max_val = max(df["CytoGreen"].max(), df["MembraneGreen"].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=2, label='M/C=1')
    
    ax.set_xlabel("胞质绿色荧光强度", fontsize=12, fontweight='bold')
    ax.set_ylabel("膜绿色荧光强度", fontsize=12, fontweight='bold')
    ax.set_title("膜 vs 胞质荧光强度散点图", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, shadow=True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_area_distribution(
    results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """细胞面积分布直方图，按组分面板
    
    Args:
        results_df: 单细胞结果数据
        output_path: 输出图片路径
    """
    ensure_dir(output_path.parent)
    
    df = results_df[results_df["QC"] == "pass"].copy() if "QC" in results_df.columns else results_df.copy()
    
    if df.empty or "Area" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=16)
        ax.set_axis_off()
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return
    
    if "Condition" not in df.columns or df["Condition"].isna().all():
        df["Condition"] = "All"
    
    conditions = df["Condition"].unique()
    n_conditions = len(conditions)
    
    fig, axes = plt.subplots(1, min(n_conditions, 4), figsize=(min(n_conditions * 4, 16), 4))
    if n_conditions == 1:
        axes = [axes]
    
    for i, (condition, ax) in enumerate(zip(conditions[:4], axes)):
        data = df[df["Condition"] == condition]["Area"].values
        ax.hist(data, bins=30, color=sns.color_palette("husl", n_conditions)[i], 
                alpha=0.7, edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {data.mean():.0f}')
        ax.set_xlabel("面积 (像素)", fontsize=10, fontweight='bold')
        ax.set_ylabel("细胞数", fontsize=10, fontweight='bold')
        ax.set_title(f"{condition}\nn={len(data)}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle("细胞面积分布", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_all_plots(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """生成所有统计图表
    
    Args:
        results_df: 单细胞结果数据
        summary_df: 汇总数据
        output_dir: 输出目录
    """
    plots_dir = output_dir / "plots"
    ensure_dir(plots_dir)
    
    print("📊 生成统计图表...")
    
    try:
        plot_mc_comparison_bar(summary_df, plots_dir / "01_mc_comparison_bar.png")
        print("  ✅ M/C对比柱状图")
    except Exception as e:
        print(f"  ⚠️ M/C柱状图生成失败: {e}")
    
    try:
        plot_mc_boxplot(results_df, plots_dir / "02_mc_boxplot.png")
        print("  ✅ M/C箱线图")
    except Exception as e:
        print(f"  ⚠️ M/C箱线图生成失败: {e}")
    
    try:
        plot_qc_statistics(results_df, plots_dir / "03_qc_statistics.png")
        print("  ✅ 质控统计图")
    except Exception as e:
        print(f"  ⚠️ 质控统计图生成失败: {e}")
    
    try:
        plot_correlation_heatmap(results_df, plots_dir / "04_correlation_heatmap.png")
        print("  ✅ 相关性热图")
    except Exception as e:
        print(f"  ⚠️ 相关性热图生成失败: {e}")
    
    try:
        plot_scatter_membrane_vs_cyto(results_df, plots_dir / "05_membrane_vs_cyto_scatter.png")
        print("  ✅ 膜-胞质散点图")
    except Exception as e:
        print(f"  ⚠️ 散点图生成失败: {e}")
    
    try:
        plot_area_distribution(results_df, plots_dir / "06_area_distribution.png")
        print("  ✅ 面积分布图")
    except Exception as e:
        print(f"  ⚠️ 面积分布图生成失败: {e}")
    
    print(f"\n📁 所有图表已保存到: {plots_dir}")
