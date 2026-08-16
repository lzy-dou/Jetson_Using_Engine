"""
Chapter 5 YOLO11-seg ONNX Runtime 推理脚本
===========================================
不依赖 ultralytics，仅用 onnxruntime + opencv + numpy 完成:
letterbox预处理 → 推理 → 置信度过滤 → NMS → mask解码 → 坐标还原 → 可视化/保存

依赖:
    pip install onnxruntime opencv-python numpy
    (有NVIDIA GPU且想用GPU推理: pip install onnxruntime-gpu 并把 PROVIDERS 里加 CUDAExecutionProvider)

配置方式：直接修改下方"配置区域"的变量即可。
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# ========== 配置区域（改这里就行） ==========
ONNX_PATH = "/home/nvidia/code-main/chapter5/YoloLearn/runs/segment/car_seg_run/weights/best.onnx"
IMAGE_FOLDER = "/home/nvidia/code-main/data/camera"
OUTPUT_FOLDER = ""          # 保存目录，空字符串="" 表示不保存

IMGSZ = 320                  # 必须和导出ONNX时的imgsz一致
NUM_CLASSES = 19
CONF_THRES = 0.25
IOU_THRES = 0.45
MASK_ONLY = True             # True: 只显示彩色掩膜(黑底)；False: 掩膜叠加在原图上
DISPLAY_TIME = 1             # 每张图展示毫秒数，0=按键才继续
DISPLAY_MAX_W = 1280

PROVIDERS = ["CPUExecutionProvider"]   # GPU: ["CUDAExecutionProvider", "CPUExecutionProvider"]
# ==========================================

COLORS = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
    (0, 80, 100), (0, 0, 230), (119, 11, 32),
]
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


def letterbox(image, new_shape=640, color=(114, 114, 114)):
    """等比缩放+padding到正方形，返回缩放后的图、缩放比例、padding量"""
    h0, w0 = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w0, h0) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (left, top)


def preprocess(image, imgsz):
    img, r, (padw, padh) = letterbox(image, imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None]  # NCHW
    return np.ascontiguousarray(img), r, (padw, padh)


def xywh2xyxy(box):
    out = np.empty_like(box)
    out[..., 0] = box[..., 0] - box[..., 2] / 2
    out[..., 1] = box[..., 1] - box[..., 3] / 2
    out[..., 2] = box[..., 0] + box[..., 2] / 2
    out[..., 3] = box[..., 1] + box[..., 3] / 2
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def postprocess(pred, proto, r, pad, orig_shape, imgsz, conf_thres, iou_thres, num_classes):
    """
    pred:  [4+nc+32, num_anchors]  (已去掉batch维)
    proto: [32, mh, mw]
    返回: boxes(xyxy, 原图坐标), scores, class_ids, masks(bool, 原图尺寸)
    """
    pred = pred.T  # [num_anchors, 4+nc+32]
    boxes = pred[:, :4]
    scores_all = pred[:, 4:4 + num_classes]
    mask_coefs = pred[:, 4 + num_classes:]

    class_ids = np.argmax(scores_all, axis=1)
    scores = scores_all[np.arange(len(scores_all)), class_ids]

    keep = scores > conf_thres
    if not np.any(keep):
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int), np.empty((0, *orig_shape[:2]), dtype=bool)

    boxes, scores, class_ids, mask_coefs = boxes[keep], scores[keep], class_ids[keep], mask_coefs[keep]
    boxes_xyxy = xywh2xyxy(boxes)

    idxs = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), scores.tolist(), conf_thres, iou_thres
    )
    if len(idxs) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int), np.empty((0, *orig_shape[:2]), dtype=bool)
    idxs = np.array(idxs).reshape(-1)

    boxes_xyxy = boxes_xyxy[idxs]
    scores = scores[idxs]
    class_ids = class_ids[idxs]
    mask_coefs = mask_coefs[idxs]

    # 坐标从letterbox后的imgsz空间还原到原图空间
    padw, padh = pad
    boxes_xyxy[:, [0, 2]] -= padw
    boxes_xyxy[:, [1, 3]] -= padh
    boxes_xyxy /= r
    h0, w0 = orig_shape[:2]
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, w0)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, h0)

    # mask解码: (num_det, 32) @ (32, mh*mw) -> (num_det, mh, mw)
    c, mh, mw = proto.shape
    proto_flat = proto.reshape(c, mh * mw)
    masks = sigmoid(mask_coefs @ proto_flat).reshape(-1, mh, mw)

    # proto尺寸(mh,mw)对应letterbox后的imgsz空间(通常缩小4倍)，先resize到imgsz再按pad/r还原到原图
    out_masks = np.zeros((len(masks), h0, w0), dtype=bool)
    for i in range(len(masks)):
        m = cv2.resize(masks[i], (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        m = m[int(padh):imgsz - int(padh) if padh > 0 else imgsz,
              int(padw):imgsz - int(padw) if padw > 0 else imgsz]
        if m.size == 0:
            continue
        m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_LINEAR)
        out_masks[i] = m > 0.5

    return boxes_xyxy, scores, class_ids, out_masks


def draw_masks(image, masks, class_ids, alpha=0.5, mask_only=False):
    if mask_only:
        result = np.zeros_like(image)
        for mask, cid in zip(masks, class_ids):
            result[mask] = COLORS[int(cid) % len(COLORS)]
        return result

    overlay = image.copy()
    for mask, cid in zip(masks, class_ids):
        overlay[mask] = COLORS[int(cid) % len(COLORS)]
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def main():
    print(f"加载ONNX模型: {ONNX_PATH}")
    session = ort.InferenceSession(ONNX_PATH, providers=PROVIDERS)
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    print(f"输入: {input_name} | 输出: {output_names} | providers: {session.get_providers()}")

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif")
    image_paths = []
    for ext in exts:
        image_paths.extend(Path(IMAGE_FOLDER).glob(f"*{ext}"))
        image_paths.extend(Path(IMAGE_FOLDER).glob(f"*{ext.upper()}"))
    image_paths = sorted(image_paths, key=lambda p: p.name)

    if not image_paths:
        print(f"在 {IMAGE_FOLDER} 中没有找到图片！")
        return
    print(f"共 {len(image_paths)} 张图片")

    if OUTPUT_FOLDER:
        Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_paths, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[{i}/{len(image_paths)}] {img_path.name} 读取失败，跳过")
            continue

        inp, r, pad = preprocess(image, IMGSZ)
        outputs = session.run(output_names, {input_name: inp})
        pred, proto = outputs[0][0], outputs[1][0]  # 去掉batch维

        boxes, scores, class_ids, masks = postprocess(
            pred, proto, r, pad, image.shape, IMGSZ, CONF_THRES, IOU_THRES, NUM_CLASSES
        )

        if len(masks) > 0:
            annotated = draw_masks(image, masks, class_ids, mask_only=MASK_ONLY)
            print(f"[{i}/{len(image_paths)}] {img_path.name}  检测到 {len(masks)} 个实例")
        else:
            annotated = image
            print(f"[{i}/{len(image_paths)}] {img_path.name}  未检测到目标")

        display = annotated
        h, w = display.shape[:2]
        if w > DISPLAY_MAX_W:
            scale = DISPLAY_MAX_W / w
            display = cv2.resize(display, (DISPLAY_MAX_W, int(h * scale)))

        cv2.imshow("YOLO11-seg ONNX - ESC退出 空格暂停", display)
        key = cv2.waitKey(DISPLAY_TIME)
        if key == 27:
            print("用户中断")
            break
        elif key == 32:
            cv2.waitKey(0)

        if OUTPUT_FOLDER:
            cv2.imwrite(str(Path(OUTPUT_FOLDER) / img_path.name), annotated)

    cv2.destroyAllWindows()
    print("推理完成！")


if __name__ == "__main__":
    main()
