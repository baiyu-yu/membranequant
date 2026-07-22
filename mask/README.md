# CellMask — Cellpose 全细胞分割 + 人工筛选

针对共定位荧光实验图的**全细胞 mask**工具：自动扫描目录 → Cellpose 分割 → **人工点选剔除** → 按源图名导出。

> 论文可写：*Segmentation results were manually inspected and corrected.*

---

## 功能概览

| 步骤 | 内容 |
|------|------|
| 1. 扫描 | 递归读取指定目录下 `.tif`，解析 `104d1-1` / `we2-3` 命名，配对红/绿 |
| 2. 分割 | **Cellpose `cyto3`**（可选 `cyto2`），**优先红通道**膜标记，不用 nuclei |
| 3. 筛选 | 鼠标点选取消错误/重叠/切边/粘连细胞；触边与右下角标尺区可自动预标记 |
| 4. 导出 | `masks/{image_id}_mask.tif`、叠加预览、JSON/CSV 清单 |

---

## 环境

在已有 conda 环境 **`mem`** 中使用：

```bash
conda activate mem
cd D:\chaos\github\membranequant\mask
pip install -r requirements.txt
pip install cellpose   # 若尚未安装
```

GPU 需匹配的 PyTorch（CUDA 版）。无 GPU 可加 `--no-gpu`。

---

## 输入目录与命名

例如：

```text
D:\课题同步\实验结果图\共定位-荧光\3B_7.20\
  C2_104d1-1\C2_104d1-1.tif        # Merge（可选）
  C2_104d1-1-1\C2_104d1-1-1.tif    # 红 DiI
  C2_104d1-1-2\C2_104d1-1-2.tif    # 绿 EGFP
  C2_we2-3-1\...
```

命名规则：`[前缀_]{实验}{药物}{组别}-{视野}[-{通道}]`

| 示例 | 含义 |
|------|------|
| `104d1-1` | 实验 104，药物 d，组 1，视野 1 |
| `we2-3` | 实验 w，药物 e，组 2，视野 3 |
| `…-1` | 红通道 |
| `…-2` | 绿通道 |
| 无后缀 | Merge |

分割默认用**红通道**；若红更糊可改 `--channel green` 或 `max`。

---

## 快速开始

```bash
conda activate mem
cd D:\chaos\github\membranequant\mask

# 仅扫描配对（不跑模型）
python -m cellmask -i "D:\课题同步\实验结果图\共定位-荧光\3B_7.20" --scan-only

# 完整流程：分割 + 人工筛选窗口
python -m cellmask -i "D:\课题同步\实验结果图\共定位-荧光\3B_7.20"

# 先用 3 张试参数
python -m cellmask -i "D:\课题同步\实验结果图\共定位-荧光\3B_7.20" --limit 3

# 细胞偏小
python -m cellmask -i "..." --model cyto2 --diameter 50

# 直径 Auto（默认）
python -m cellmask -i "..." --diameter 0
```

或双击/命令行：

```bat
run_cellmask.cmd "D:\课题同步\实验结果图\共定位-荧光\3B_7.20" --limit 3
```

默认输出：`<input-dir>/cellmask_output/`，可用 `-o` 改。

---

## Web 前端交互审核与边框消除

运行 `python -m cellmask` 时默认自动拉起本地 **Web 交互界面**（在浏览器中打开 `http://127.0.0.1:8080`）：

### 核心操作与工具
1. **缩放与平移 (Zoom & Pan)**：
   - 鼠标滚轮向上/向下滚动可自由放大、缩小图像；
   - 按住鼠标中键 / 右键 / Shift + 左键拖拽平移视角；左上角 🔍 按钮一键重置自适应屏幕。
2. **消除/删除细胞边框 (合并细胞)**：
   - 点击左上角 ✂️ **“删除边框/合并”** 工具模式；
   - 点击或划过过度分割的两个细胞交界的分割边框线条，系统自动消去该边框并将它们合并为一个完整的细胞 Mask。
3. **点选与侧边栏勾选 (Checkbox List)**：
   - 👆 模式下，左键点击 Canvas 上的细胞即可快速切换 **保留 (绿色)** / **剔除 (红色)**；
   - 右侧侧边栏提供完整细胞列表，带 Checkbox 勾选框，可直接勾选/取消勾选；
   - 侧边栏提供“剔除触边”、“剔除标尺区”、“全部保留”、“重置”快捷按钮。
4. **通道与显示调节**：
   - 实时切换底层红通道 (DiI)、绿通道 (EGFP)、Merge 合成背景图；
   - 调节 Mask 透明度 slider，一键显隐 Cell ID 编号 (按 `i`) 与剔除轮廓 (按 `x`)。

---

## 快捷键一览

| 快捷键 | 作用 |
|------|------|
| **左键点击** | 👆 模式：切换保留/剔除；✂️ 模式：点击交界边框消除并合并细胞 |
| `n` / `→` | 下一张视野 |
| `p` / `←` | 上一张视野 |
| `a` | 本张全部保留 |
| `b` | 剔除所有触边细胞 |
| `s` | 剔除标尺区细胞 |
| `r` | 重置为初始自动标记状态 |
| `i` | 开关细胞编号显示 |
| `x` | 开关已剔除轮廓显示 |


---

## 右下角标尺

默认会：

1. 分割前把右下角约 **22%×12%** 区域填成背景中位数，降低被当成细胞的概率；
2. 仍与该区域重叠较多的细胞标为 `scalebar_ids`，进入筛选时默认剔除。

若标尺位置不同，可调：

```bash
python -m cellmask -i "..." --scalebar-width 0.25 --scalebar-height 0.15
```

完全关闭：`--no-scalebar-mask`（仍可在 GUI 里自己点掉）。

---

## 导出结构

以 `image_id = 104d1-1` 为例：

```text
cellmask_output/
  masks/
    104d1-1_mask.tif       # 仅保留细胞，ID 从 1 连续编号
  overlays/
    104d1-1_overlay.png    # 绿=保留 / 红=剔除 轮廓预览
  meta/
    104d1-1_meta.json      # 保留/剔除 ID、路径、参数
  summary.json
  kept_summary.csv
```

文件名与视野 ID 一致，便于和原图对应。

---

## 推荐参数（与常见 Cellpose 用法一致）

| 参数 | 建议 |
|------|------|
| Model | **cyto3**；细胞特别小可试 **cyto2** |
| Diameter | **Auto**，或约 **40–80** pixel |
| 通道 | 膜标记更清楚时用**红**；不要用 **nuclei** |
| 筛选 | 删错分、重叠、切边、粘连未分开的，只留好的 |

---

## 常用参数一览

```text
-i / --input-dir     实验根目录（必填）
-o / --out-dir       输出目录
--model cyto3|cyto2  分割模型
--diameter 0         0=Auto
--channel red|green|max
--no-gpu
--limit N
--ids 104d1-1 we2-3
--scan-only
--skip-review        不弹 GUI，仅自动标记后导出
--no-auto-flag
--no-scalebar-mask
```

---

## 目录结构

```text
mask/
  cellmask/
    __init__.py
    __main__.py
    cli.py
    io_scan.py      # 扫描与读图
    segment.py      # Cellpose
    review_gui.py   # 人工筛选
    export.py       # 导出
    pipeline.py     # 串联流程
  requirements.txt
  run_cellmask.cmd
  README.md
```
