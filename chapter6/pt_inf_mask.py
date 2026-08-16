#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 深度估计 / 掩膜推理脚本（仅显示深度图或掩膜，不显示原图/边界框）
- 适用于 YOLO 深度模型（result.depth）或分割模型（result.masks）
- 支持批量处理文件夹
- 输入推理尺寸、显示尺寸可自定义
- 窗口常驻，实时更新显示
- 支持保存结果

主要修复点（相对原版）：
1. 深度数据取 result.depth.data（torch.Tensor, (H,W)），而非 result.depth（DepthMap 包装对象）
2. INPUT_SIZE 改为 640，与训练 imgsz 一致，保证深度质量
3. 深度图缩放改用 INTER_LINEAR，避免块状感
4. mask_to_image 兼容 DepthMap / Tensor / ndarray，并做维度规整
"""

import os
import cv2
import numpy as np
import glob
from ultralytics import YOLO

# =============================================
# 用户配置区域
# =============================================
SOURCE_PATH = "/home/nvidia/code-main/chapter6/YoloLearn/nyu-depth/images/val"
WEIGHTS_PATH = "/home/nvidia/code-main/chapter6/YoloLearn/runs/depth/train/weights/best.pt"

INPUT_SIZE = 640        # 推理尺寸，需与训练 imgsz 一致(640)以获最佳深度质量；降可提速但损失精度
DISPLAY_WIDTH = 640     # 显示窗口宽，None 则保持推理尺寸
DISPLAY_HEIGHT = 640    # 显示窗口高，None 则保持推理尺寸

SAVE_RESULTS = False
WAIT_MS = 0             # 每张图显示等待毫秒数（0 为等待按键）
DEVICE = "cpu"       # Jetson 有 CUDA 用 cuda:0，无则 cpu
# =============================================


def mask_to_image(mask_data, mask_type='depth'):
    """
    将模型输出转换为可显示的 BGR 图像。
    mask_data: DepthMap / torch.Tensor / np.ndarray
    mask_type: 'depth' -> 伪彩色(JET)；'segmentation' -> 灰度转 BGR
    """
    # 1) 若传入 DepthMap 包装对象，取其 .data 张量
    if hasattr(mask_data, 'data') and not isinstance(mask_data, np.ndarray):
        mask_data = mask_data.data
    # 2) Tensor -> numpy
    if hasattr(mask_data, 'cpu'):
        mask_data = mask_data.cpu().numpy()
    if not isinstance(mask_data, np.ndarray):
        raise TypeError(f"无法识别的掩膜数据类型: {type(mask_data)}")

    # 3) 维度规整到 (H, W)
    mask_data = np.squeeze(mask_data)
    if mask_data.ndim != 2:
        raise ValueError(f"期望 2D 深度/掩膜，得到 shape {mask_data.shape}")

    # 4) 归一化到 0-255
    if mask_data.dtype != np.uint8:
        mn, mx = float(mask_data.min()), float(mask_data.max())
        mask_data = ((mask_data - mn) / (mx - mn + 1e-6) * 255.0).astype(np.uint8)

    # 5) 着色
    if mask_type == 'depth':
        return cv2.applyColorMap(mask_data, cv2.COLORMAP_JET)
    else:
        return cv2.cvtColor(mask_data, cv2.COLOR_GRAY2BGR)


def main():
    print(f"加载模型: {WEIGHTS_PATH}")
    model = YOLO(WEIGHTS_PATH)

    # 收集图片
    if os.path.isdir(SOURCE_PATH):
        img_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'):
            img_files.extend(glob.glob(os.path.join(SOURCE_PATH, ext)))
        img_files.sort()
        print(f"找到 {len(img_files)} 张图片")
    else:
        img_files = [SOURCE_PATH]
    if not img_files:
        print("未找到任何图片，请检查路径。")
        return

    # 保存目录
    save_dir = None
    if SAVE_RESULTS:
        base_dir = SOURCE_PATH if os.path.isdir(SOURCE_PATH) else os.path.dirname(SOURCE_PATH)
        save_dir = os.path.join(base_dir, 'mask_results')
        os.makedirs(save_dir, exist_ok=True)
        print(f"结果将保存至: {save_dir}")

    display_size = None
    if DISPLAY_WIDTH is not None and DISPLAY_HEIGHT is not None:
        display_size = (DISPLAY_WIDTH, DISPLAY_HEIGHT)

    window_name = "Depth/Mask Result (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    first_depth = True
    for idx, img_path in enumerate(img_files, 1):
        print(f"\n正在处理 [{idx}/{len(img_files)}]: {img_path}")

        results = model.predict(
            source=img_path,
            imgsz=INPUT_SIZE,
            device=DEVICE,
            save=False,
            show=False,
            verbose=False,
        )
        result = results[0]

        # ---- 优先深度模型；其次分割模型 ----
        if getattr(result, 'depth', None) is not None:
            # 深度模型：result.depth.data 是 torch.Tensor, shape (H,W), 单位米
            depth_tensor = result.depth.data
            if first_depth:
                d_np = depth_tensor.cpu().numpy() if hasattr(depth_tensor, 'cpu') else np.asarray(depth_tensor)
                print(f"  深度图 shape={d_np.shape}, 范围={d_np.min():.3f}~{d_np.max():.3f} m")
                first_depth = False
            mask_image = mask_to_image(depth_tensor, mask_type='depth')
        elif getattr(result, 'masks', None) is not None and result.masks.data.numel() > 0:
            # 分割模型：合并所有掩膜
            masks = result.masks.data  # (N, H, W) 或 (N,1,H,W)
            mask_np = masks.cpu().numpy()
            if mask_np.ndim == 4:
                mask_np = mask_np.squeeze(1)
            combined = np.max(mask_np, axis=0)  # (H, W)
            mask_image = mask_to_image(combined, mask_type='segmentation')
        else:
            print("  警告：模型未输出深度或掩膜数据，跳过该图片。")
            continue

        # ---- 缩放显示（深度用 LINEAR 更平滑）----
        if display_size is not None:
            mask_image = cv2.resize(mask_image, display_size, interpolation=cv2.INTER_LINEAR)

        cv2.imshow(window_name, mask_image)

        if SAVE_RESULTS and save_dir:
            save_path = os.path.join(save_dir, os.path.basename(img_path))
            cv2.imwrite(save_path, mask_image)
            print(f"  已保存: {save_path}")

        key = cv2.waitKey(WAIT_MS) & 0xFF
        if key == ord('q'):
            print("用户终止推理")
            break

    cv2.destroyAllWindows()
    print("\n所有图片处理完毕！")


if __name__ == "__main__":
    main()
