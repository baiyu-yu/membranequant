# MembraneQuant 🔬
**高通量双通道荧光显微图像细胞分割与质膜定量分析系统**

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![UI Framework](https://img.shields.io/badge/UI-Gradio%204.x-orange.svg)](https://gradio.app/)
[![Deep Learning](https://img.shields.io/badge/Backend-Cellpose%20%2F%20DualCellQuant-brightgreen.svg)](https://github.com/fuji3to4/DualCellQuant)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

---

## 📖 项目简介 (Overview)

**MembraneQuant** 是一款专为双通道荧光显微成像（如红色通道 DiI/膜标记与绿色通道 EGFP/目标蛋白表达）设计的高通量图像定量分析与可视化系统。系统能够自动扫描实验文件夹、智能配对红绿通道视野，利用深度学习模型分割细胞，结合欧氏距离变换（EDT）生成**周质膜富集区域/固定宽度膜带 ROI（peripheral membrane-enriched ROI / fixed-width membrane band ROI）**与细胞质 ROI，并自动完成多维度荧光强度定量、质控过滤（QC）、 GraphPad Prism 兼容数据导出以及可视化图表生成。

> 💡 **学术表达规范**：受光学衍射极限限制，宽场及常规共聚焦显微图像无法直接分辨 ~7 nm 的真实物理质膜。本系统遵照科研论文（如 *SoftwareX 2023*）的标准表达范式，将提取的膜区域定义为**外周膜富集 ROI (peripheral membrane-enriched ROI)**。

---

## 📚 文献映射与学术引用指南 (Literature Mapping & Citation Guide)

为方便在发表学术论文（Methods & Citations 章节）时直接引用，本项目将各个分析步骤与其底层算子、经典论文及国际通用标准进行了映射归纳：

| 分析步骤 (Pipeline Step) | 对应算法 / 原理 (Algorithm) | 推荐引用文献 (Reference) | 规范度与成熟度 |
| :--- | :--- | :--- | :---: |
| **1. TIFF 图像加载** | 16-bit 深度保留加载 (`tifffile`) | Gohlke C. *tifffile: Read and write TIFF files.* | ⭐⭐⭐⭐⭐ |
| **2. 背景扣除** | Rolling Ball 背景扣除算法 | **Sternberg SR** (1983) *Biomedical Image Processing*; ImageJ/Fiji (*Schneider et al., 2012*) | ⭐⭐⭐⭐⭐ |
| **3. 细胞形态分割** | 基于深层流场的神经网络分割 (Cellpose) | **Stringer et al.** (2021) *Cellpose: a generalist algorithm for cellular segmentation.* **Nat Methods** | ⭐⭐⭐⭐⭐ |
| **4. 细胞边界提取** | 数学形态学二值轮廓 (Binary Contour) | **Soille P.** (1999) *Morphological Image Analysis*; *van der Walt et al.* (2014) scikit-image | ⭐⭐⭐⭐⭐ |
| **5. 质膜 ROI 生成** | 欧氏距离变换 (Euclidean Distance Transform, EDT) 径向距离场 | **Danielsson PE** (1980) *Euclidean Distance Mapping*; DualCellQuant (2026) | ⭐⭐⭐⭐⭐ |
| **6. 细胞质 ROI 提取** | 结构元素二值腐蚀 (Binary Erosion) / 减去外周膜带 | **Serra J.** (1982) *Image Analysis and Mathematical Morphology* | ⭐⭐⭐⭐⭐ |
| **7. 荧光强度测量** | 平均/积分荧光强度 (ROI Mean Intensity) | **Schneider CA et al.** (2012) *NIH Image to ImageJ: 25 years of image analysis.* **Nat Methods** | ⭐⭐⭐⭐⭐ |
| **8. 质控异常剔除** | 面积/极值荧光强度阈值过滤 (QC Filter) | **McQuin C et al.** (2018) *CellProfiler 3.0.* **PLoS Biol** | ⭐⭐⭐⭐⭐ |
| **9. 方法范式对比** | 膜易位与膜富集度测量 Workflow | 借鉴 *SoftwareX* (2023) *Find_plasma_membrane & measure_plasma_membrane* | ⭐⭐⭐⭐⭐ |

---

## 🛠️ 技术选型与架构设计 (Technology Stack)

项目的核心设计原则是**学术严谨性、硬件可扩展性、极佳的易用性与原生数据分析集成**。以下为项目的详细技术选型及其职责拆解：

```
+-----------------------------------------------------------------------------------+
|                                  MembraneQuant                                    |
+-----------------------------------------------------------------------------------+
|  [ Web UI / CLI 交互层 ]     - Gradio 4.x / Python Argparse                       |
|  [ 图像分割与 ROI 提取 ]     - DualCellQuant / Cellpose / PyTorch / EDT 算法      |
|  [ 图像处理与矩阵计算 ]       - OpenCV / scikit-image / SciPy / tifffile / NumPy  |
|  [ 数据分析与质控导出 ]       - Pandas / GraphPad Prism Exporter / QC Filter       |
|  [ 统计绘图与可视化 ]         - Matplotlib / Seaborn / PIL                         |
+-----------------------------------------------------------------------------------+
```

### 1. 深度学习与细胞分割引擎 (AI & Segmentation Engine)
* **[Cellpose](https://www.cellpose.org/) & [PyTorch](https://pytorch.org/)**: 选型基于其在密集、不规则细胞形态上的卓越泛化分割能力 (*Stringer et al., Nat Methods, 2021*)。通过 PyTorch 框架支持 **CUDA GPU 硬件加速**，大大提升高通量图像处理吞吐量；在无 GPU 环境下可无缝降级至 CPU 运行。
* **[DualCellQuant](https://github.com/fuji3to4/DualCellQuant)**: 专用的双通道图像分析后端，负责从背景扣除到结合 Cellpose 的双通道细胞掩模（Mask）生成，保障分割的高鲁棒性。

### 2. 图像处理与几何算法 (Image Processing & Geometric Algorithms)
* **ImageJ 兼容球滚背景扣除 (Rolling Ball Background Subtraction)**: 基于 `skimage.restoration.rolling_ball` 实现 ImageJ 经典的 **Rolling Ball 算法**（*Sternberg SR, 1983*），配合 ImageJ 缩放因子 (shrinkFactor) 优化逻辑，实现高鲁棒性的非均匀背景光斑/荧光背景扣除。
* **EDT 膜富集 ROI 提取 (Euclidean Distance Transform)**: 借助 OpenCV / SciPy 算法库，基于 **EDT 欧氏距离变换算法** (*Danielsson PE, 1980*)，在细胞 Mask 边界生成精准的固定宽度外周膜富集 ROI (peripheral membrane-enriched ROI) 与细胞质腐蚀 ROI (*Serra J, 1982*)，避免传统膨胀腐蚀导致的重叠失真。
* **[tifffile](https://github.com/cgohlke/tifffile)**: 专门用于加载科研级 TIF/TIFF 显微图像 (*Christoph Gohlke*)，完全保留 16-bit 原始动态范围与色彩深度，避免普通图像库加载导致的灰度信息截断。
* **OpenCV (`opencv-python`) & `scikit-image`**: 提供高效的高斯滤波、连通域分析（Connected Components）、数学形态学二值轮廓提取 (*Soille P, 1999*) 及 Mask 叠加渲染。

### 3. 数据分析与专业导出 (Data Analysis & Export Engine)
* **[Pandas](https://pandas.pydata.org/)**: 负责单细胞层级（Per-Cell）与视野层级（Per-Field）多维量化数据的清洗、条件聚合、长宽表转换与透视分析。
* **GraphPad Prism 适配导出 (Custom GraphPad Exporter)**: 内置专门定制的数据导出引擎，可直接将单细胞与视野统计数据转换为符合 GraphPad Prism 格式规范的数据矩阵，支持一键粘贴或导入直接进行双因素方差分析（Two-way ANOVA）及 $t$ 检验。
* **PyYAML**: 用于结构化配置文件的读取与动态管理，保障实验参数的高可扩展性与可复现性。

### 4. 数据可视化与交互界面 (Visualization & User Interface)
* **[Gradio (>=4.0)](https://gradio.app/)**: 构建零代码中文交互式 Web UI 界面。支持文件夹递归扫描预览、实时参数配置、分析进度条提示、图像 Mask 叠加快照查看以及结果图表下载。
* **[Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)**: 用于自动生成论文级统计渲染图表（*Hunter JD, 2007*），包括提琴图（Violin Plot）、带散点的条形图（Strip/Bar Plot）、红绿通道共定位散点图（Colocalization Scatter）等。

---

## 💡 核心功能特性 (Key Features)

1. **智能目录扫描与通道配对**: 自动递归搜索实验文件夹，基于文件名规则识别红通道（如 DiI 膜标记）与绿通道（如 EGFP 蛋白），完成双通道自动配对。
2. **多 ROI 精确量化**: 自动提取并计算每个细胞的：
   - 全细胞平均/总荧光强度 ($I_{\text{cell}}$)
   - 外周膜富集 ROI 平均/总荧光强度 ($I_{\text{membrane}}$)
   - 细胞质 ROI 平均/总荧光强度 ($I_{\text{cytoplasm}}$)
   - 膜/质比值 ($T/R = \text{Membrane} / \text{Cytoplasm}$)
3. **质控过滤 (Quality Control)**: 参考 CellProfiler 质控机制（*McQuin et al., 2018*），可自定义细胞面积上/下限、荧光强度极值阈值，自动剔除离群伪影。
4. **多格式结果导出**: 自动输出单细胞明细 CSV、视野汇总 CSV、各指标数据矩阵，以及原生适配 GraphPad Prism 的格式文件。
5. **双界面模式**: 既支持简单易用的 Web UI 界面，也支持批处理脚本和 CLI 命令行工具。

---

## 📂 项目结构 (Directory Structure)

```
membranequant/
├── main.py              # CLI 命令行入口与 Pipeline 流程控制
├── webui.py             # 基于 Gradio 的中文 Web 界面
├── visualizer_app.py    # 独立的可视化交互分析查看器
├── config.py            # 配置参数类定义与 YAML 加载逻辑
├── config.yaml          # 默认配置文件
├── dual_backend.py      # DualCellQuant / Cellpose 后端接口封装
├── segmentation.py      # 细胞分割与距离变换 (EDT) ROI 算法
├── measurements.py      # 单细胞/视野荧光强度与共定位指标计算
├── qc.py                # 质量控制 (QC) 过滤模块
├── export.py            # CSV 及 GraphPad Prism 数据导出器
├── plots.py             # Matplotlib / Seaborn 统计图表生成
├── visualization.py     # Mask 叠加图与散点图绘制
├── io.py                # 图像加载与文件对查找扫描
├── utils.py             # 辅助工具函数（日志、路径创建等）
└── requirements.txt     # 项目依赖配置文件
```

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备 (Installation)

推荐使用 Python 3.9+ 环境。克隆/下载项目后，在根目录安装依赖：

```bash
pip install -r requirements.txt
```

> **GPU 加载说明**：如需使用 GPU 加速，请确保已根据硬件版本安装匹配的 `torch` 与 `torchvision` (CUDA 版本)。

### 2. 启动 Web 界面 (Recommended)

在终端运行以下命令启动中文 Web 界面：

```bash
# 启动 Web UI
python main.py --webui --port 7860
```

或在 Windows 环境下直接双击运行脚本：
- `run_webui.cmd`

启动后在浏览器中打开 `http://localhost:7860` 即可使用。

### 3. 命令行批量处理 (CLI Mode)

通过命令行对指定实验数据目录进行分析：

```bash
python main.py --input-dir "D:/ExperimentData/3B_7.15" --out-dir "./output"
```

---

## 📄 开源协议 (License)

本项目基于 [MIT License](LICENSE) 协议开源。
