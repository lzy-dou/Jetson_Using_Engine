from ultralytics import YOLO

# 加载你训练好的最佳模型
model = YOLO("runs/classify/car_cls_run/weights/best.pt")

# 导出为 ONNX
model.export(
    format="onnx",          # 指定导出格式
    imgsz=64,               # 输入图像尺寸（与训练时一致）
    batch=1,                # 批量大小（推理通常设为1）
    opset=12,               # ONNX opset 版本（默认12，兼容性好）
    simplify=True,          # 简化 ONNX 模型（推荐开启，减小体积）
    dynamic=False           # 是否允许动态输入尺寸（通常保持False）
)