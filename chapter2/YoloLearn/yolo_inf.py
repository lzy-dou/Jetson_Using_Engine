from ultralytics import YOLO

# 加载最佳模型
model = YOLO("runs/classify/car_cls_run/weights/best.pt")

# 对一张图片进行预测
results = model("../chapter2/car_classification_split/val/background/000760.jpg")
# results = model("chapter2/car_classification_split/val/background_val/000492.jpg")
probs = results[0].probs  # 获取概率
print(probs.top1)         # 预测的类别索引
print(probs.top1conf)     # 置信度
print(probs.top5)         # 前5个类别