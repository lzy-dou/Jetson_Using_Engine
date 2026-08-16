#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度估计 ONNX 推理（Python）
- 预处理：letterbox -> BGR2RGB -> /255 -> CHW（与 ultralytics 训练对齐）
- 后处理：读 [1,1,H,W] 深度图 -> squeeze -> letterbox 反映射(裁掉填充区) -> resize 回原图 -> 归一化上色
- 支持动态矩形输入（H/W 需为 32 倍数）
"""
import os
import glob
import cv2
import numpy as np
import onnxruntime as ort

# ============================ 配置 ============================
MODEL_PATH   = "/home/nvidia/code-main/chapter6/YoloLearn/runs/depth/train/weights/best.onnx"
IMAGE_FOLDER = "/home/nvidia/code-main/data/camera"
OUTPUT_FOLDER = ""            # 空字符串则不保存

INPUT_H = 320                 # 推理高（32 倍数）
INPUT_W = 320                 # 推理宽（32 倍数）
WAIT_MS = 0                   # 每张显示等待 ms，0=按键继续
# =============================================================


def letterbox(img, target_h, target_w):
    h, w = img.shape[:2]
    r = min(target_h / h, target_w / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = (target_w - new_w) / 2.0
    dh = (target_h - new_h) / 2.0
    top = int(round(dh - 0.1))
    left = int(round(dw - 0.1))
    bottom = target_h - new_h - top
    right = target_w - new_w - left
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, (114, 114, 114))
    return out, new_h, new_w, top, left


def preprocess(lb, H, W):
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32, copy=False) / 255.0
    x = x.transpose(2, 0, 1)[None]          # (1,3,H,W)
    return np.ascontiguousarray(x)


def main():
    if INPUT_H % 32 != 0 or INPUT_W % 32 != 0:
        raise SystemExit(f"INPUT_H/W 必须是 32 倍数，当前 {INPUT_H}x{INPUT_W}")

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(MODEL_PATH, providers=providers)
    in_name = sess.get_inputs()[0].name
    print(f"模型: {MODEL_PATH}")
    print(f"输入: {sess.get_inputs()[0].shape}  输出: {sess.get_outputs()[0].shape}")
    print(f"使用的 Provider: {sess.get_providers()}")

    # 收集图片
    if os.path.isdir(IMAGE_FOLDER):
        files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'):
            files.extend(glob.glob(os.path.join(IMAGE_FOLDER, ext)))
        files.sort()
    else:
        files = [IMAGE_FOLDER]
    print(f"共 {len(files)} 张图片")

    window = "Depth ONNX (q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    first = True

    for i, f in enumerate(files, 1):
        img = cv2.imread(f)
        if img is None:
            print(f"[{i}] 读取失败: {f}")
            continue
        print(f"[{i}/{len(files)}] {f}")

        lb, new_h, new_w, top, left = letterbox(img, INPUT_H, INPUT_W)
        x = preprocess(lb, INPUT_H, INPUT_W)

        out = sess.run(None, {in_name: x})[0]   # (1,1,Hd,Wd) 或类似
        depth = np.squeeze(out)                  # (Hd,Wd)
        if depth.ndim != 2:
            depth = out.reshape(out.shape[-2], out.shape[-1])
        if first:
            print(f"  输出张量 shape={out.shape} -> 深度图 {depth.shape}, "
                  f"范围 {depth.min():.3f}~{depth.max():.3f} m")
            first = False

        # letterbox 反映射：裁掉填充区，再 resize 回原图
        hd, wd = depth.shape
        sy, sx = hd / INPUT_H, wd / INPUT_W
        t = int(round(top * sy)); l = int(round(left * sx))
        hh = int(round(new_h * sy)); ww = int(round(new_w * sx))
        hh = max(1, min(hh, hd - t)); ww = max(1, min(ww, wd - l))
        depth_valid = depth[t:t + hh, l:l + ww]
        depth_full = cv2.resize(depth_valid, (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

        # 归一化 + 伪彩色
        mn, mx = float(depth_full.min()), float(depth_full.max())
        d8 = ((depth_full - mn) / (mx - mn + 1e-6) * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(d8, cv2.COLORMAP_JET)

        cv2.imshow(window, colored)
        if OUTPUT_FOLDER:
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, os.path.basename(f)), colored)

        if (cv2.waitKey(WAIT_MS) & 0xFF) == ord('q'):
            break

    cv2.destroyAllWindows()
    print("完成")


if __name__ == "__main__":
    main()
