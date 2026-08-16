from ultralytics import YOLO

model = YOLO("/workspace/chapter4/YoloLearn/yolo11n.pt")  # 也可换 yolo11s.pt
print("train start---------")
model.train(
    data="/workspace/chapter4/YoloLearn/dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device="cuda",       # 或 "cpu"
    workers=4,
    name="car_det_run"
)
print("检测训练完成")
