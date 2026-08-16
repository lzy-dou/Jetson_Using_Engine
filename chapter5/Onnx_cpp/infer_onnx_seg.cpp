// Chapter 5 YOLO11-seg ONNX Runtime C++ 推理程序
// =================================================
// 依赖: ONNX Runtime (>=1.12, 用到 GetInputNameAllocated / GetOutputNameAllocated),
//       OpenCV (>=4.x，用到 cv::dnn::NMSBoxes)
//
// 编译 (参考 CMakeLists.txt):
//   mkdir build && cd build
//   cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime
//   make -j
//
// 运行:
//   ./infer_onnx_seg <model.onnx> <image_folder> [output_folder]

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <vector>

namespace fs = std::filesystem;

// ========== 配置区域 ==========
constexpr int IMGSZ = 320;
constexpr int NUM_CLASSES = 19;
constexpr float CONF_THRES = 0.25f;
constexpr float IOU_THRES = 0.45f;
constexpr bool MASK_ONLY = true;  // true: 纯黑底彩色掩膜；false: 叠加在原图上
// ================================

static const std::vector<cv::Scalar> COLORS = {
    {128, 64, 128}, {244, 35, 232}, {70, 70, 70}, {102, 102, 156},
    {190, 153, 153}, {153, 153, 153}, {250, 170, 30}, {220, 220, 0},
    {107, 142, 35}, {152, 251, 152}, {70, 130, 180}, {220, 20, 60},
    {255, 0, 0}, {0, 0, 142}, {0, 0, 70}, {0, 60, 100},
    {0, 80, 100}, {0, 0, 230}, {119, 11, 32},
};

struct LetterboxInfo {
    float r;
    int padw, padh;
};

// 等比缩放+padding到正方形
static cv::Mat letterbox(const cv::Mat& img, int newSize, LetterboxInfo& info,
                          const cv::Scalar& color = cv::Scalar(114, 114, 114)) {
    int h0 = img.rows, w0 = img.cols;
    float r = std::min(static_cast<float>(newSize) / h0, static_cast<float>(newSize) / w0);
    int newUnpadW = static_cast<int>(std::round(w0 * r));
    int newUnpadH = static_cast<int>(std::round(h0 * r));

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);

    int dw = newSize - newUnpadW;
    int dh = newSize - newUnpadH;
    int top = dh / 2, bottom = dh - dh / 2;
    int left = dw / 2, right = dw - dw / 2;

    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    info.r = r;
    info.padw = left;
    info.padh = top;
    return out;
}

struct Detection {
    cv::Rect2f box;      // 原图坐标 xyxy
    float score;
    int classId;
    cv::Mat mask;         // CV_8U, 原图尺寸, 0/255
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "用法: " << argv[0] << " <model.onnx> <image_folder> [output_folder]\n";
        return 1;
    }
    std::string modelPath = argv[1];
    std::string imageFolder = argv[2];
    std::string outputFolder = (argc >= 4) ? argv[3] : "";

    // ---- 初始化 ONNX Runtime ----
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo11-seg");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // 如需GPU: 取消下一行注释，并确保链接的是 onnxruntime-gpu / 带CUDA EP的库
    // OrtCUDAProviderOptions cudaOpts; sessionOptions.AppendExecutionProvider_CUDA(cudaOpts);

#ifdef _WIN32
    std::wstring wModelPath(modelPath.begin(), modelPath.end());
    Ort::Session session(env, wModelPath.c_str(), sessionOptions);
#else
    Ort::Session session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    auto inputNameAlloc = session.GetInputNameAllocated(0, allocator);
    std::string inputName = inputNameAlloc.get();
    std::vector<std::string> outputNames;
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        auto nameAlloc = session.GetOutputNameAllocated(i, allocator);
        outputNames.emplace_back(nameAlloc.get());
    }
    std::cout << "输入: " << inputName << " | 输出数量: " << outputNames.size() << "\n";

    std::vector<const char*> inputNamesC = {inputName.c_str()};
    std::vector<const char*> outputNamesC;
    for (auto& n : outputNames) outputNamesC.push_back(n.c_str());

    if (!outputFolder.empty()) fs::create_directories(outputFolder);

    // ---- 遍历图片 ----
    std::vector<fs::path> imagePaths;
    for (auto& entry : fs::directory_iterator(imageFolder)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            imagePaths.push_back(entry.path());
        }
    }
    std::sort(imagePaths.begin(), imagePaths.end());
    std::cout << "共 " << imagePaths.size() << " 张图片\n";

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    int idx = 0;
    for (const auto& path : imagePaths) {
        ++idx;
        cv::Mat image = cv::imread(path.string());
        if (image.empty()) {
            std::cout << "[" << idx << "/" << imagePaths.size() << "] " << path.filename()
                      << " 读取失败，跳过\n";
            continue;
        }

        // ---- 预处理 ----
        LetterboxInfo lb;
        cv::Mat letterboxed = letterbox(image, IMGSZ, lb);
        cv::Mat rgb;
        cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

        // HWC -> CHW
        std::vector<float> inputTensorValues(3 * IMGSZ * IMGSZ);
        std::vector<cv::Mat> channels(3);
        for (int c = 0; c < 3; ++c) {
            channels[c] = cv::Mat(IMGSZ, IMGSZ, CV_32F, inputTensorValues.data() + c * IMGSZ * IMGSZ);
        }
        cv::split(rgb, channels);

        std::array<int64_t, 4> inputShape = {1, 3, IMGSZ, IMGSZ};
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, inputTensorValues.data(), inputTensorValues.size(),
            inputShape.data(), inputShape.size());

        // ---- 推理 ----
        auto outputs = session.Run(Ort::RunOptions{nullptr}, inputNamesC.data(), &inputTensor, 1,
                                    outputNamesC.data(), outputNamesC.size());

        // output0: [1, 4+nc+32, num_anchors]  output1: [1, 32, mh, mw]
        float* predData = outputs[0].GetTensorMutableData<float>();
        auto predShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();  // [1, C, N]
        int predC = static_cast<int>(predShape[1]);
        int numAnchors = static_cast<int>(predShape[2]);

        float* protoData = outputs[1].GetTensorMutableData<float>();
        auto protoShape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();  // [1, 32, mh, mw]
        int maskC = static_cast<int>(protoShape[1]);
        int mh = static_cast<int>(protoShape[2]);
        int mw = static_cast<int>(protoShape[3]);

        // ---- 解析检测框 (predData 是 [C, N]，按列取每个anchor) ----
        std::vector<cv::Rect> boxesForNMS;
        std::vector<float> scoresForNMS;
        std::vector<int> classIds;
        std::vector<std::vector<float>> maskCoefsAll;

        for (int a = 0; a < numAnchors; ++a) {
            float cx = predData[0 * numAnchors + a];
            float cy = predData[1 * numAnchors + a];
            float w = predData[2 * numAnchors + a];
            float h = predData[3 * numAnchors + a];

            int bestClass = -1;
            float bestScore = -1.f;
            for (int c = 0; c < NUM_CLASSES; ++c) {
                float s = predData[(4 + c) * numAnchors + a];
                if (s > bestScore) {
                    bestScore = s;
                    bestClass = c;
                }
            }
            if (bestScore < CONF_THRES) continue;

            float x1 = cx - w / 2.f, y1 = cy - h / 2.f;
            boxesForNMS.emplace_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                                               static_cast<int>(w), static_cast<int>(h)));
            scoresForNMS.push_back(bestScore);
            classIds.push_back(bestClass);

            std::vector<float> coef(maskC);
            for (int m = 0; m < maskC; ++m) {
                coef[m] = predData[(4 + NUM_CLASSES + m) * numAnchors + a];
            }
            maskCoefsAll.push_back(std::move(coef));
        }

        std::vector<int> keepIdx;
        cv::dnn::NMSBoxes(boxesForNMS, scoresForNMS, CONF_THRES, IOU_THRES, keepIdx);

        std::vector<Detection> detections;
        for (int i : keepIdx) {
            Detection det;
            // letterbox坐标 -> 原图坐标
            float x1 = (boxesForNMS[i].x - lb.padw) / lb.r;
            float y1 = (boxesForNMS[i].y - lb.padh) / lb.r;
            float x2 = (boxesForNMS[i].x + boxesForNMS[i].width - lb.padw) / lb.r;
            float y2 = (boxesForNMS[i].y + boxesForNMS[i].height - lb.padh) / lb.r;
            x1 = std::clamp(x1, 0.f, static_cast<float>(image.cols));
            y1 = std::clamp(y1, 0.f, static_cast<float>(image.rows));
            x2 = std::clamp(x2, 0.f, static_cast<float>(image.cols));
            y2 = std::clamp(y2, 0.f, static_cast<float>(image.rows));
            det.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
            det.score = scoresForNMS[i];
            det.classId = classIds[i];

            // ---- mask解码: coef(32) · proto(32, mh*mw) -> (mh, mw)，sigmoid ----
            cv::Mat maskSmall(mh, mw, CV_32F, cv::Scalar(0));
            const auto& coef = maskCoefsAll[i];
            for (int y = 0; y < mh; ++y) {
                for (int x = 0; x < mw; ++x) {
                    float sum = 0.f;
                    for (int c = 0; c < maskC; ++c) {
                        sum += coef[c] * protoData[c * mh * mw + y * mw + x];
                    }
                    maskSmall.at<float>(y, x) = 1.0f / (1.0f + std::exp(-sum));
                }
            }
            cv::Mat maskLetterbox;
            cv::resize(maskSmall, maskLetterbox, cv::Size(IMGSZ, IMGSZ), 0, 0, cv::INTER_LINEAR);
            cv::Rect cropRect(lb.padw, lb.padh, IMGSZ - 2 * lb.padw, IMGSZ - 2 * lb.padh);
            cropRect &= cv::Rect(0, 0, IMGSZ, IMGSZ);
            cv::Mat maskCropped = maskLetterbox(cropRect);
            cv::Mat maskFull;
            cv::resize(maskCropped, maskFull, image.size(), 0, 0, cv::INTER_LINEAR);
            cv::Mat maskBin;
            cv::threshold(maskFull, maskBin, 0.5, 255, cv::THRESH_BINARY);
            maskBin.convertTo(det.mask, CV_8U);

            detections.push_back(std::move(det));
        }

        // ---- 绘制 ----
        cv::Mat annotated = MASK_ONLY ? cv::Mat::zeros(image.size(), image.type()) : image.clone();
        for (const auto& det : detections) {
            cv::Scalar color = COLORS[det.classId % COLORS.size()];
            annotated.setTo(color, det.mask);
        }
        if (!MASK_ONLY) {
            cv::addWeighted(annotated, 0.5, image, 0.5, 0, annotated);
        }

        std::cout << "[" << idx << "/" << imagePaths.size() << "] " << path.filename()
                  << "  检测到 " << detections.size() << " 个实例\n";

        if (!outputFolder.empty()) {
            fs::path savePath = fs::path(outputFolder) / path.filename();
            cv::imwrite(savePath.string(), annotated);
        }

        cv::imshow("YOLO11-seg ONNX C++ - ESC退出", annotated);
        int key = cv::waitKey(1);
        if (key == 27) {
            std::cout << "用户中断\n";
            break;
        }
    }

    cv::destroyAllWindows();
    std::cout << "推理完成！\n";
    return 0;
}
