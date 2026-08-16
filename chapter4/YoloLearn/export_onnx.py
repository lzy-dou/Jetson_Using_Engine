"""
导出支持动态矩形输入 (n x m) 的 ONNX 模型。

关键点：
1. dynamic=True 会让 batch / height / width 三个维度全部变为动态，
   导出后可喂任意 [1, 3, H, W]（矩形也行），例如 [1,3,480,640]、[1,3,320,512]。
2. 运行时 H、W 必须各自是 32 的整数倍（YOLO 最大 stride=32），
   否则特征图对不齐会报错或检测错位。常见可用尺寸：320/352/384/416/448/480/512/640 等。
3. imgsz 只作为 tracing 示例尺寸，dynamic=True 时不限制运行时尺寸；可不动。
4. 若 simplify 在动态图上报错，ultralytics 会自动回退到非简化图，不影响使用。
"""
from ultralytics import YOLO

# 1. 加载你训练好的 .pt 模型（替换为你的实际路径）
model = YOLO("/home/nvidia/code-main/chapter4/YoloLearn/runs/detect/car_det_run/weights/best.pt")

# 2. 导出动态 ONNX
model.export(
    format="onnx",
    imgsz=640,          # 仅 tracing 示例尺寸，开启 dynamic 后不限制运行时输入
    opset=12,           # opset 版本，12 及以上即可
    simplify=True,      # 简化计算图（动态图若简化失败会自动回退）
    dynamic=True,       # 【关键】开启动态输入：batch / H / W 均动态，支持 n x m
)

print("导出完成。可用以下代码验证输入维度是否为动态：")

import onnx
m = onnx.load("best.onnx")
for inp in m.graph.input:
    dims = [d.dim_value if d.HasField("dim_value") else d.dim_param or "?" for d in inp.type.tensor_type.shape.dim]
    print(inp.name, dims)
# 期望输出类似: images ['batch', 3, 'height', 'width']  -> 说明 H/W 已是动态
