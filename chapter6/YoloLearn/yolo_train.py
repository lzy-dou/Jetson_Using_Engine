from ultralytics import YOLO

model = YOLO("/workspace/chapter6/YoloLearn/yolo26s-depth.pt")
# 直接用内置的 nyu-depth.yaml，首次运行自动下载数据集
model.train(
	data="/workspace/chapter6/YoloLearn/nyu-depth/nyu-depth.yaml", 
	epochs=100, 
	imgsz=640,
	batch=12,
	device="cuda",
	workers=0,
        optimizer="AdamW", 
        lr0=1e-4)
            
 # model.train(
 #   data="/workspace/chapter5/YoloLearn/dataset/data.yaml",
 #   epochs=100,
 #   imgsz=640,
 #   batch=8,
 #   device="cuda",       # 或 "cpu"
 #   workers=0,
 #   name="car_seg_run"
#)
