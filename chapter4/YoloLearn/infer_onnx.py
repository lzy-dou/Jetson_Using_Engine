"""
ONNX 推理脚本（与 ultralytics YOLO11 的 letterbox 预处理 / 解码-NMS 后处理对齐）
适用：ultralytics 默认导出的检测 ONNX 模型（不含 NMS，输出 output0 = [1, 4+nc, anchors]）

【自定义输入尺寸说明（重要）】
ONNX 的输入尺寸在“导出时”就固定了。默认导出(dynamic=False)锁死为 640x640，
运行时无法改成其它尺寸，否则 onnxruntime 会报形状不匹配错误。
要使用自定义尺寸加速推理，必须用 dynamic=True 重新导出一次：

    from ultralytics import YOLO
    YOLO("best.pt").export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=True)

导出后模型输入 H/W 变为动态，即可把下方 INPUT_SIZE 改成 320/480/512 等任意值。
尺寸越小推理越快（640->320 约快 4 倍），但对小目标检测精度会下降，按需权衡。
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# ========== 配置区域 ==========
MODEL_PATH = "/home/nvidia/code-main/chapter4/YoloLearn/runs/detect/car_det_run/weights/best.onnx"          # 你的 ONNX 模型路径（替换为实际路径）
IMAGE_FOLDER = "/home/nvidia/code-main/data/camera"           # 图片文件夹路径（替换为实际路径）
OUTPUT_FOLDER = None              # 需要保存结果图则填目录路径；None 则不保存
FACTOR=10
CONF_THRES = 0.6                 # 置信度阈值
IOU_THRES = 0.45                  # NMS 的 IoU 阈值
INPUT_SIZE = 32*FACTOR            # 推理输入尺寸（需模型 dynamic=True 才能生效；默认导出固定640会忽略此项）
WAIT_KEY = False                  # False: 每张图显示 10ms 自动继续；True: 等待按键
# ===============================

# 类别名（与训练时 data.yaml 中 names 对应）
CLASS_NAMES = ["vehicle"]


def letterbox(im, new_shape=640, color=(114, 114, 114)):
    """等比缩放并填充到 new_shape×new_shape，返回 (填充后图像, 缩放比r, (left, top)填充量)。"""
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))  # (w, h) 缩放后尺寸
    dw = (new_shape - new_unpad[0]) / 2.0
    dh = (new_shape - new_unpad[1]) / 2.0
    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 与 ultralytics 一致：左右/上下按 round 偏置分配
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def preprocess(im, size):
    """BGR -> RGB, HWC -> CHW, /255, 加 batch 维 -> [1,3,size,size]。"""
    im_lb, r, pad = letterbox(im, size)
    im_t = im_lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    im_t = np.ascontiguousarray(im_t[None])  # [1, 3, size, size]
    return im_t, r, pad


def resolve_input_size(sess, desired):
    """根据模型实际输入形状决定推理尺寸。
    - 模型动态输入(dynamic=True)：使用 desired。
    - 模型固定输入：忽略 desired，使用模型固定尺寸并给出提示。
    """
    shape = sess.get_inputs()[0].shape  # [B, 3, H, W]
    hw = shape[2:]
    dynamic = any(not isinstance(d, int) for d in hw)
    if dynamic:
        print(f"模型为动态输入，使用自定义尺寸 {desired}x{desired}。")
        return desired
    fixed = int(hw[0])
    if fixed != desired:
        print(f"⚠️ 模型输入固定为 {fixed}x{fixed}，当前 INPUT_SIZE={desired} 无效，将按 {fixed} 推理。")
        print("   如需自定义输入尺寸，请用 dynamic=True 重新导出 ONNX（见文件顶部说明）。")
    return fixed


def postprocess(output, r, pad, orig_shape, conf_thres, iou_thres):
    """
    解码 + 置信度过滤 + NMS，并把框映射回原图坐标。
    output[0]: [1, 4+nc, anchors] -> [anchors, 4+nc]
    前 4 列 = cx,cy,w,h（640 输入空间），其余 = 各类置信度。
    """
    pred = output[0]
    pred = np.squeeze(pred, 0).T  # [anchors, 4+nc]

    boxes_xywh = pred[:, :4]
    scores_all = pred[:, 4:]                     # [anchors, nc]
    class_ids = scores_all.argmax(axis=1)
    confidences = scores_all.max(axis=1)

    # 置信度过滤
    mask = confidences > conf_thres
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    if len(boxes_xywh) == 0:
        return [], [], []

    # NMS（cv2.dnn.NMSBoxes 接收 xywh 与对应分数）
    indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), confidences.tolist(),
                               conf_thres, iou_thres)
    if len(indices) == 0:
        return [], [], []
    indices = np.asarray(indices).reshape(-1).astype(int)
    boxes_xywh = boxes_xywh[indices]
    confidences = confidences[indices]
    class_ids = class_ids[indices]

    # xywh -> xyxy，并按缩放比/填充量映射回原图
    left, top = pad
    x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0 - left) / r
    y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0 - top) / r
    x2 = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0 - left) / r
    y2 = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0 - top) / r
    x1 = np.clip(x1, 0, orig_shape[1])
    y1 = np.clip(y1, 0, orig_shape[0])
    x2 = np.clip(x2, 0, orig_shape[1])
    y2 = np.clip(y2, 0, orig_shape[0])
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    return boxes, confidences, class_ids


def draw(im, boxes, confidences, class_ids):
    for (x1, y1, x2, y2), conf, cid in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{CLASS_NAMES[int(cid)]} {float(conf):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(im, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(im, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return im


def main():
    # 1. 加载 ONNX 模型
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if "CUDAExecutionProvider" in ort.get_available_providers() \
        else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(MODEL_PATH, providers=providers)
    inp = sess.get_inputs()[0]
    in_name = inp.name
    print(f"模型加载成功: {MODEL_PATH}")
    print(f"输入名: {in_name} | 输入形状: {inp.shape} | Provider: {sess.get_providers()}")
    print(f"输出节点: {[o.name for o in sess.get_outputs()]}  形状: {[o.shape for o in sess.get_outputs()]}")

    # 解析实际可用的推理尺寸（固定模型会忽略 INPUT_SIZE）
    infer_size = resolve_input_size(sess, INPUT_SIZE)

    # 2. 收集图片
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_paths = []
    for ext in exts:
        image_paths.extend(Path(IMAGE_FOLDER).glob(f'*{ext}'))
        image_paths.extend(Path(IMAGE_FOLDER).glob(f'*{ext.upper()}'))
    image_paths = sorted(image_paths, key=lambda p: p.name)
    if not image_paths:
        print(f"在 {IMAGE_FOLDER} 中没有找到图片！")
        return
    print(f"共找到 {len(image_paths)} 张图片，开始推理...")
    image_paths=image_paths[2500:]
    shape_printed = False
    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {img_path.name}")
        im0 = cv2.imread(str(img_path))
        if im0 is None:
            print("  读取失败，跳过")
            continue

        # 预处理
        im_t, r, pad = preprocess(im0, infer_size)

        # 推理
        output = sess.run(None, {in_name: im_t})
        if not shape_printed:
            print(f"  输出 output0 实际形状: {output[0].shape} "
                  f"(格式 [1, 4+nc, anchors]，anchor 数随输入尺寸变化)")
            shape_printed = True

        # 后处理
        boxes, confidences, class_ids = postprocess(
            output, r, pad, im0.shape[:2], CONF_THRES, IOU_THRES)

        # 画框
        annotated = im0.copy()
        if len(boxes):
            annotated = draw(annotated, boxes, confidences, class_ids)
            for (x1, y1, x2, y2), conf in zip(boxes, confidences):
                print(f"  框: ({int(x1)},{int(y1)},{int(x2)},{int(y2)})  conf={float(conf):.3f}")
        else:
            print("  未检测到目标")

        cv2.imshow("ONNX Detection - ESC to exit", annotated)
        key = cv2.waitKey(1)
        if key == 27:
            print("用户中断，退出...")
            break

        if OUTPUT_FOLDER:
            save_path = str(Path(OUTPUT_FOLDER) / img_path.name)
            cv2.imwrite(save_path, annotated)
            print(f"  已保存: {save_path}")

    cv2.destroyAllWindows()
    print("推理完成！")


if __name__ == "__main__":
    main()
