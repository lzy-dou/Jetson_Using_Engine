关于jetson orin nx使用yolo训练二分类（汽车与背景）、汽车检测、实例分割、深度估计的代码
其中最外层的py用于训练pt和onnx模型并使用py推理，内层Onnx_cpp文件夹是使用cpp推理onnx模型
最重要的是Engine_cpp文件夹是使用cpp导出engine模型再使用cpp推理，目前没有整合项目
！数据集与创作模仿来自https://github.com/vision-adas/code，如有侵权请联系删除：1810775525@qq.com
