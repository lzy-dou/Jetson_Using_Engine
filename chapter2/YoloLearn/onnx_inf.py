import onnxruntime as ort
import numpy as np
import cv2
import os

# 获取当前脚本的绝对路径所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接相对于脚本目录的模型路径（假设 runs 与脚本同级）
model_path = os.path.join(script_dir, "runs/classify/car_cls_run/weights/best.onnx")
picture_path=os.path.join(script_dir, "car_classification_split/val/car/000346.jpg")
# 加载 ONNX 模型
session = ort.InferenceSession(model_path)

# 准备输入（示例：单张 64x64 RGB 图）
# img = cv2.imread("/home/nvidia/code-main/chapter2/YoloLearn/car_classification_split/val/car/000459.jpg")
img = cv2.imread(picture_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (64, 64))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
img = np.expand_dims(img, axis=0)   # 增加 batch 维度

# 推理
outputs = session.run(None, {"images": img})  # 输入名通常为 "images"
probs = outputs[0]  # 分类概率
print(probs)