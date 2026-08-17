"""
Chapter 5 YOLO11-seg 推理脚本
=============================
遍历文件夹中的图片，逐张推理并展示分割结果。

配置方式：直接修改下方"配置区域"的变量即可。
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ========== 配置区域（改这里就行） ==========
MODEL_PATH = "/home/nvidia/code-main/chapter5/YoloLearn/runs/segment/car_seg_run/weights/best.pt"  # 模型路径
IMAGE_FOLDER = "/home/nvidia/code-main/data/camera"                   # 图片文件夹
OUTPUT_FOLDER = ""                             # 保存目录，None=不保存

DISPLAY_TIME = 100    # 每张图展示毫秒数（2000=2秒，0=按键才继续，100=快进）
CONF_THRES = 0.25      # 置信度阈值
SAVE_RESULTS = False    # 是否保存结果图
INFER_SIZE = 320       # 推理输入尺寸（320更快，640更准，支持320/480/640/960等）
DISPLAY_MAX_W = 1280   # 显示窗口最大宽度（图片太大会缩放显示，不影响保存的原图尺寸）
# ==========================================

# 19个类别的颜色 (Cityscapes格式)
COLORS = [
    (128, 64, 128),   (244, 35, 232),  (70, 70, 70),    (102, 102, 156),
    (190, 153, 153),  (153, 153, 153), (250, 170, 30),  (220, 220, 0),
    (107, 142, 35),   (152, 251, 152), (70, 130, 180),  (220, 20, 60),
    (255, 0, 0),      (0, 0, 142),     (0, 0, 70),      (0, 60, 100),
    (0, 80, 100),     (0, 0, 230),     (119, 11, 32),
]
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle"
]


def draw_masks(image, masks, class_ids, alpha=0.5):
    """在图像上绘制半透明分割mask（不画框）"""
    overlay = image.copy()

    for mask, cid in zip(masks, class_ids):
        cid = int(cid)
        color = COLORS[cid % len(COLORS)]
        overlay[mask] = color

    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return result


def main():
    # 加载模型
    model = YOLO(MODEL_PATH)
    print(f"模型加载成功: {MODEL_PATH}")

    # 收集图片
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    image_paths = []
    for ext in exts:
        image_paths.extend(Path(IMAGE_FOLDER).glob(f'*{ext}'))
        image_paths.extend(Path(IMAGE_FOLDER).glob(f'*{ext.upper()}'))
    image_paths = sorted(image_paths, key=lambda p: p.name)

    if not image_paths:
        print(f"在 {IMAGE_FOLDER} 中没有找到图片！")
        return
    print(f"共 {len(image_paths)} 张图片 | 每张展示 {DISPLAY_TIME}ms | 推理尺寸={INFER_SIZE} | 保存={SAVE_RESULTS}")

    # 创建保存目录
    if SAVE_RESULTS and OUTPUT_FOLDER:
        Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_paths, 1):
        # 推理
        results = model.predict(
            source=str(img_path),
            conf=CONF_THRES,
            imgsz=INFER_SIZE,
            verbose=False,
        )

        # 读取原图
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[{i}/{len(image_paths)}] {img_path.name} 读取失败，跳过")
            continue

        r = results[0]

        if r.masks is not None and len(r.masks) > 0:
            masks = r.masks.data.cpu().numpy()
            masks = masks.astype(bool)
            class_ids = r.boxes.cls.cpu().numpy()

            # mask尺寸可能与原图不同，需要resize
            h, w = image.shape[:2]
            resized_masks = np.zeros((len(masks), h, w), dtype=bool)
            for j, m in enumerate(masks):
                resized_masks[j] = cv2.resize(m.astype(np.uint8), (w, h)) > 0

            annotated = draw_masks(image, resized_masks, class_ids)
            print(f"[{i}/{len(image_paths)}] {img_path.name}  检测到 {len(masks)} 个实例")
        else:
            annotated = image
            print(f"[{i}/{len(image_paths)}] {img_path.name}  未检测到目标")

        # 显示（图片过大时缩放显示，不影响保存）
        display = annotated
        h, w = display.shape[:2]
        if w > DISPLAY_MAX_W:
            scale = DISPLAY_MAX_W / w
            display = cv2.resize(display, (DISPLAY_MAX_W, int(h * scale)))

        cv2.imshow("YOLO11 Seg - ESC退出 空格暂停", display)
        key = cv2.waitKey(DISPLAY_TIME)

        if key == 27:  # ESC
            print("用户中断")
            break
        elif key == 32:  # 空格 暂停，再按任意键继续
            cv2.waitKey(0)

        # 保存
        if SAVE_RESULTS and OUTPUT_FOLDER:
            save_path = str(Path(OUTPUT_FOLDER) / img_path.name)
            cv2.imwrite(save_path, annotated)

    cv2.destroyAllWindows()
    print("推理完成！")


if __name__ == "__main__":
    main()
