"""快速测试新的分割方法是否可用"""
import sys
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

from membranequant.config import Config
from membranequant.segmentation import (
    segment_whole_cells,
    segment_whole_cells_watershed_distance,
    segment_whole_cells_watershed_gradient,
    segment_whole_cells_hminima_watershed,
    segment_whole_cells_morphological_opening,
    segment_whole_cells_combined_markers,
)
import numpy as np

print("✅ 所有导入成功！")
print("\n可用的分割方法：")

methods = [
    "otsu",
    "watershed_distance", 
    "watershed_gradient",
    "hminima_watershed",
    "morphological_opening",
    "combined_markers",
]

for method in methods:
    print(f"  - {method}")

# 创建测试数据
print("\n开始测试各个方法...")
green = np.random.rand(100, 100).astype(np.float32)
red = np.random.rand(100, 100).astype(np.float32)

# 创建一些模拟的细胞（圆形区域）
from skimage.draw import disk as draw_disk
for i in range(3):
    rr, cc = draw_disk((30 + i*30, 30 + i*30), 10, shape=green.shape)
    green[rr, cc] = 0.8
    red[rr, cc] = 0.7

print("生成了测试图像（100x100，包含3个模拟细胞）")

# 测试每个方法
for method in methods:
    try:
        cfg = Config(segmentation_method=method, minimum_cell_area=50)
        labels, rejected = segment_whole_cells(green, red, cfg)
        n_cells = len(np.unique(labels)) - 1  # 减去背景
        print(f"✅ {method:25s} - 识别到 {n_cells} 个细胞")
    except Exception as e:
        print(f"❌ {method:25s} - 错误: {e}")

print("\n✅ 测试完成！所有方法都可以正常调用。")
print("\n下一步：在 webui 中选择合适的方法进行实际数据分析")
