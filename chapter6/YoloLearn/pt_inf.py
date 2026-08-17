
# =============================================
# 用户配置区域（直接修改这里）
# =============================================
SOURCE_PATH = "/home/nvidia/code-main/chapter6/YoloLearn/nyu-depth/images/val"   # 图片文件夹路径或单张图片路径
IMGSZ = 320                                                # 推理输入尺寸
SAVE_RESULTS = False                                        # 是否保存带框的结果图
WAIT_MS = 1000                                                # 每张图显示后等待毫秒数（0为等待按键）
DEVICE = "cpu"                                            # 推理设备 "cuda" 或 "cpu"
WEIGHTS_PATH = "/home/nvidia/code-main/chapter6/YoloLearn/runs/depth/train/weights/best.pt"  # 模型权重路径
# =============================================



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 推理脚本（窗口常驻，实时更新显示）
- 单窗口显示，每推理一张立即更新
- 自定义尺寸、保存、等待时间
"""

import os
import cv2
import glob
from ultralytics import YOLO

# =============================================
# 用户配置区域（直接修改这里）
# =============================================
SOURCE_PATH = "/home/nvidia/code-main/chapter6/YoloLearn/nyu-depth/images/val"   # 图片文件夹路径或单张图片路径
IMGSZ = 320                                                # 推理输入尺寸
SAVE_RESULTS = False                                        # 是否保存带框的结果图
WAIT_MS = 1                                                # 每张图显示后等待毫秒数（0为等待按键）
DEVICE = "cpu"                                            # 推理设备 "cuda" 或 "cpu"
WEIGHTS_PATH = "/home/nvidia/code-main/chapter6/YoloLearn/runs/depth/train/weights/best.pt"  # 模型权重路径
# =============================================



def main():
    # 加载模型
    print(f"加载模型: {WEIGHTS_PATH}")
    model = YOLO(WEIGHTS_PATH)

    # 获取图片列表
    if os.path.isdir(SOURCE_PATH):
        img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        img_files = []
        for ext in img_extensions:
            img_files.extend(glob.glob(os.path.join(SOURCE_PATH, ext)))
        img_files.sort()
        print(f"找到 {len(img_files)} 张图片")
    else:
        img_files = [SOURCE_PATH]

    if not img_files:
        print("未找到任何图片，请检查路径。")
        return

    # 创建保存目录
    save_dir = None
    if SAVE_RESULTS:
        base_dir = SOURCE_PATH if os.path.isdir(SOURCE_PATH) else os.path.dirname(SOURCE_PATH)
        save_dir = os.path.join(base_dir, 'yolo_results')
        os.makedirs(save_dir, exist_ok=True)
        print(f"结果将保存至: {save_dir}")

    # 创建一个窗口（只创建一次）
    window_name = "YOLO Detection (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 遍历推理
    for idx, img_path in enumerate(img_files, 1):
        print(f"\n正在处理 [{idx}/{len(img_files)}]: {img_path}")

        results = model.predict(
            source=img_path,
            imgsz=IMGSZ,
            device=DEVICE,
            save=False,
            show=False,
            verbose=False
        )

        result = results[0]
        img_with_boxes = result.plot()  # RGB
        img_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)

        # 更新窗口显示
        cv2.imshow(window_name, img_bgr)

        # 保存结果
        if SAVE_RESULTS and save_dir:
            save_path = os.path.join(save_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, img_bgr)
            print(f"已保存: {save_path}")

        # 等待控制
        if WAIT_MS == 0:
            # 等待按键（按任意键继续，按 'q' 退出整个程序）
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("用户终止推理")
                break
        else:
            # 等待指定毫秒，若期间按下 'q' 则退出
            key = cv2.waitKey(WAIT_MS) & 0xFF
            if key == ord('q'):
                print("用户终止推理")
                break

    # 关闭窗口（循环结束后）
    cv2.destroyAllWindows()
    print("\n所有图片处理完毕！")


if __name__ == "__main__":
    main()