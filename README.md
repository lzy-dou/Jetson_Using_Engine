关于jetson orin nx使用yolo训练二分类（汽车与背景）、汽车检测、实例分割、深度估计的代码
其中最外层的py用于训练pt和onnx模型并使用py推理，内层Onnx_cpp文件夹是使用cpp推理onnx模型
最重要的是Engine_cpp文件夹是使用cpp导出engine模型再使用cpp推理，目前没有整合项目


----------------------------------------------------------------------------------------------|
！数据集与创作模仿来自https://github.com/vision-adas/code，如有侵权请联系删除：1810775525@qq.com  |
----------------------------------------------------------------------------------------------|

using belike（chapter6）：

# 1.导出ONNX模型
cd chapter6

python yolo_train.py #训练

python export_onnx.py

# 2.. 编译
 cd chapter6/YoloLearn/Engine_cpp
 mkdir -p build
 cd build
 cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.6 ..

# 3. 从 ONNX 构建 engine
./build_engine \
    /home/nvidia/code-main/chapter6/YoloLearn/runs/depth/train/weights/best.onnx \
    /home/nvidia/code-main/chapter6/YoloLearn/runs/depth/train/weights/best.engine \
    fp16

./trt_infer

