#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

int main() {
    // 使用绝对路径，避免相对路径带来的困扰
    std::string model_path = "/home/nvidia/code-main/chapter2/YoloLearn/runs/classify/car_cls_run/weights/best.onnx";
    std::string image_path = "/home/nvidia/code-main/chapter2/YoloLearn/car_classification_split/val/background/000088.jpg";

    // 读取图片
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    // 使用 OpenCV 的 blobFromImage 一步完成预处理
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0/255.0, cv::Size(64, 64),
                                          cv::Scalar(), true, false);
    // blob 形状: (1, 3, 64, 64)

    // 加载 ONNX 模型
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);

    // 获取输入输出名称
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> input_names = {input_name.get()};
    std::vector<const char*> output_names = {output_name.get()};

    // 准备输入张量
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, 3, 64, 64};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, blob.ptr<float>(), blob.total() * sizeof(float),
            input_shape.data(), input_shape.size());

    // 推理
    std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr},
                                                         input_names.data(), &input_tensor, 1,
                                                         output_names.data(), 1);

    // 提取结果
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    int64_t num_classes = output_info.GetShape()[1];

    std::cout << "Probabilities: ";
    for (int i = 0; i < num_classes; ++i) {
        std::cout << output_data[i] << " ";
    }
    std::cout << std::endl;

    // 预测类别
    int pred = std::max_element(output_data, output_data + num_classes) - output_data;
    std::cout << "Predicted class: " << pred << std::endl;

    return 0;
}