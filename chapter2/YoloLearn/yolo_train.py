from ultralytics import YOLO

# 加载预训练分类模型（YOLOv8-cls）
model = YOLO("/home/nvidia/code-main/chapter2/yolov8n-cls.pt")  # 也可换 yolov8s-cls.pt 等

# 训练
model.train(
    data="car_classification_split",  # 根目录，内部需有 car/ 和 background/ 子文件夹
    epochs=50,
    imgsz=64,          # 与原模型一致的输入尺寸
    batch=32,
    device="cpu",      # 或 "0" 使用 GPU
    workers=2,
    name="car_cls_run"
)
print("分类训练完成！")