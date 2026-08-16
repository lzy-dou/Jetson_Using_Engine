import cv2
import os
from ultralytics import YOLO
from pathlib import Path

# ========== 配置区域 ==========
MODEL_PATH = "/home/nvidia/code-main/chapter4/YoloLearn/runs/detect/car_det_run/weights/best.pt"   
# 替换为您的模型路径
IMAGE_FOLDER = "/home/nvidia/code-main/data/camera"# 替换为您的图片文件夹路径
OUTPUT_FOLDER = None        # 若需要保存结果图，设置输出文件夹路径；否则为 None
WAIT_KEY = False             # 是否等待按键再显示下一张（若为 False，每张图显示 0.5 秒后自动继续）
# ===============================

# 1. 加载模型
model = YOLO(MODEL_PATH)
print(f"✅ 模型加载成功: {MODEL_PATH}")

# 获取所有图片文件
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
image_paths = []
for ext in image_extensions:
    image_paths.extend(Path(IMAGE_FOLDER).glob(f'*{ext}'))
    image_paths.extend(Path(IMAGE_FOLDER).glob(f'*{ext.upper()}'))

# 按文件名升序排列（确保顺序）
image_paths = sorted(image_paths, key=lambda p: p.name)

# 可选：如果希望按数字顺序（例如 image_1.jpg, image_2.jpg），
# 可以使用自然排序（需额外库 natsort），或自定义 key 提取数字。
# 简单情况下文件名排序通常已满足要求。

if not image_paths:
    print(f"❌ 在 {IMAGE_FOLDER} 中没有找到图片文件！")
    exit()

print(f"📸 共找到 {len(image_paths)} 张图片，按文件名顺序推理...")

# 遍历图片（此时顺序已固定）
image_paths=image_paths[2500:]
for i, img_path in enumerate(image_paths, 1):
    print(f"\n[{i}/{len(image_paths)}] 处理: {img_path.name}")
    
    # 推理
    results = model(img_path, conf=0.6,imgsz=320)   # conf 阈值可调
    
    # 获取带标注的图像（BGR 格式，可直接用于 OpenCV 显示）
    annotated_img = results[0].plot()      # 返回 numpy 数组 (H,W,3) BGR
    
    # 显示图像
    cv2.imshow("YOLO Detection - Press any key for next, ESC to exit", annotated_img)
    key = cv2.waitKey(1)   # 若 WAIT_KEY=True 则无限等待按键，否则自动 500ms
    if key == 27:   # ESC 键退出
        print("🚪 用户中断，退出...")
        break
    
    # 保存结果（如果需要）
    if OUTPUT_FOLDER:
        save_path = os.path.join(OUTPUT_FOLDER, img_path.name)
        cv2.imwrite(save_path, annotated_img)
        print(f"💾 已保存: {save_path}")

# 5. 关闭所有窗口
cv2.destroyAllWindows()
print("🏁 推理完成！")

