"""
将深度估计 .pt 模型导出为 ONNX（支持动态矩形输入）。
深度模型是全卷积结构，dynamic=True 后可喂任意 [1,3,H,W]（H/W 需为 32 倍数）。
输出为 [1,1,H,W] 的逐像素深度图（米）。
"""
from ultralytics import YOLO

# 1. 加载训练好的深度模型
model = YOLO("/home/nvidia/code-main/chapter6/YoloLearn/runs/depth/train/weights/best.pt")

# 2. 导出动态 ONNX
model.export(
    format="onnx",
    imgsz=640,          # tracing 示例尺寸，dynamic 时不限制运行时尺寸
    opset=12,
    simplify=True,
    dynamic=True,       # batch/H/W 均动态，支持 n x m
)

# 3. 验证输入/输出维度是否为动态
print("导出完成。验证维度：")
import onnx
m = onnx.load("best.onnx")
for inp in m.graph.input:
    dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in inp.type.tensor_type.shape.dim]
    print("  输入", inp.name, dims)
for out in m.graph.output:
    dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in out.type.tensor_type.shape.dim]
    print("  输出", out.name, dims)
# 期望：输入 images ['batch',3,'height','width']；输出 output0 ['batch',1,'height','width']
