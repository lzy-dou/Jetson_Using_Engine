"""
PT → ONNX 模型导出脚本
=======================
将训练好的 YOLO11-seg .pt 权重导出为 .onnx 模型，供 Python(onnxruntime) /
C++(ONNX Runtime) 推理使用。

配置方式：直接修改下方"配置区域"的变量即可。
"""
from ultralytics import YOLO

# ========== 配置区域（改这里就行） ==========
PT_PATH = "/home/nvidia/code-main/chapter5/YoloLearn/runs/segment/car_seg_run/weights/best.pt"

IMGSZ = 640          # 导出尺寸，需与训练/推理时保持一致（正方形，或[h, w]）
OPSET = 12           # ONNX opset版本，12/13/17 都比较稳妥，onnxruntime版本较老可用12
DYNAMIC = False       # 是否使用动态batch/宽高，C++部署建议False，固定shape更快更简单
SIMPLIFY = True       # 是否用onnx-simplifier化简计算图（需要 pip install onnxslim 或 onnx-simplifier）
HALF = False          # 是否导出FP16，仅在GPU上导出/推理时才有意义，CPU上无效甚至更慢
NMS = False           # 是否把NMS一起导出进ONNX图（True时导出模型自带NMS，输出格式不同，
                       # 本项目配套的 infer_onnx_seg.py / .cpp 按 NMS=False 的标准输出格式解析，
                       # 若改成True，需要同步修改后处理代码，不建议随意开启）
# ==========================================


def main():
    print(f"加载PT模型: {PT_PATH}")
    model = YOLO(PT_PATH)

    onnx_path = model.export(
        format="onnx",
        imgsz=IMGSZ,
        opset=OPSET,
        dynamic=DYNAMIC,
        simplify=SIMPLIFY,
        half=HALF,
        nms=NMS,
    )

    print(f"ONNX模型导出成功: {onnx_path}")
    print("模型输出说明 (NMS=False, 分割模型标准导出格式):")
    print("  output0: [1, 4+nc+32, num_anchors]  → 每个anchor: cx,cy,w,h + nc个类别分数 + 32个mask系数")
    print("  output1: [1, 32, mh, mw]             → mask原型 (prototype masks)，通常 mh=mw=imgsz/4")
    print("请记下 nc(类别数)=19 与 imgsz，供 infer_onnx_seg.py / infer_onnx_seg.cpp 配置使用")


if __name__ == "__main__":
    main()
