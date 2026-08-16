from ultralytics import YOLO

model = YOLO("/workspace/chapter5/YoloLearn/yolo11n-seg.pt")  # 分割预训练模型
model.train(
    data="/workspace/chapter5/YoloLearn/dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device="cuda",       # 或 "cpu"
    workers=0,
    name="car_seg_run"
)
print("分割训练完成")
