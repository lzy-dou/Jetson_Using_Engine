// =============================================================================
// 深度估计 ONNX C++ 推理（支持动态矩形输入 n x m）
// 依赖：ONNX Runtime(C++) + OpenCV
// 预处理：letterbox -> BGR2RGB -> /255 -> CHW
// 后处理：读 [1,1,Hd,Wd] 深度图 -> 裁掉 letterbox 填充区 -> resize 回原图 -> 归一化上色
// 编译见同目录 CMakeLists.txt。
// =============================================================================
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

// ============================ 配置 ============================
const std::string MODEL_PATH   = "/home/nvidia/code-main/chapter6/YoloLearn/runs/depth/train/weights/best.onnx";
const std::string IMAGE_FOLDER = "/home/nvidia/code-main/chapter6/YoloLearn/nyu-depth/images/val";
const std::string OUTPUT_FOLDER = "";            // 空则不保存

const int   INPUT_H = 320;                        // 32 倍数
const int   INPUT_W = 320;                        // 32 倍数
const int   WAIT_MS = 1;
// =============================================================

struct LBInfo { int new_h, new_w, top, left; };

static LBInfo letterbox(const cv::Mat& src, cv::Mat& dst, int target_h, int target_w) {
    int h = src.rows, w = src.cols;
    float r = std::min((float)target_h / h, (float)target_w / w);
    int new_w = (int)std::round(w * r);
    int new_h = (int)std::round(h * r);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    float dw = (target_w - new_w) / 2.0f;
    float dh = (target_h - new_h) / 2.0f;
    int top    = (int)std::round(dh - 0.1f);
    int left   = (int)std::round(dw - 0.1f);
    int bottom = (target_h - new_h) - top;
    int right  = (target_w - new_w) - left;
    cv::copyMakeBorder(resized, dst, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return {new_h, new_w, top, left};
}

static std::vector<float> preprocess(const cv::Mat& lb, int H, int W) {
    cv::Mat rgb, f32;
    cv::cvtColor(lb, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> chans(3);
    cv::split(f32, chans);  // R,G,B
    std::vector<float> input((size_t)3 * H * W);
    for (int c = 0; c < 3; ++c) {
        const cv::Mat& ch = chans[c];
        for (int y = 0; y < H; ++y) {
            const float* s = ch.ptr<float>(y);
            float* d = input.data() + (size_t)c * H * W + (size_t)y * W;
            std::memcpy(d, s, (size_t)W * sizeof(float));
        }
    }
    return input;
}

int main() {
    if (INPUT_H % 32 != 0 || INPUT_W % 32 != 0) {
        std::cerr << "INPUT_H/W 必须是 32 倍数: " << INPUT_H << "x" << INPUT_W << std::endl;
        return -1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "depth");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, MODEL_PATH.c_str(), opts);

    Ort::AllocatorWithDefaultOptions allocator;
    auto in_name_ptr  = session.GetInputNameAllocated(0, allocator);
    auto out_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* input_name  = in_name_ptr.get();
    const char* output_name = out_name_ptr.get();
    std::array<const char*, 1> in_names{input_name};
    std::array<const char*, 1> out_names{output_name};

    // 诊断打印 I/O 形状（动态模型部分 ORT 版本查询会抛异常，保护一下）
    try {
        auto in_info  = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto out_info = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto pshape = [](const std::string& t, const std::vector<int64_t>& s) {
            std::cout << t << " [";
            for (size_t i = 0; i < s.size(); ++i) std::cout << s[i] << (i + 1 < s.size() ? "," : "");
            std::cout << "]" << std::endl;
        };
        pshape("输入形状", in_info.GetShape());
        pshape("输出形状", out_info.GetShape());
    } catch (const std::exception& e) {
        std::cerr << "（类型信息查询跳过: " << e.what() << "）" << std::endl;
    }

    if (!std::filesystem::exists(IMAGE_FOLDER)) {
        std::cerr << "图片文件夹不存在: " << IMAGE_FOLDER << std::endl;
        return -1;
    }
    std::vector<std::string> files;
    for (const auto& e : std::filesystem::directory_iterator(IMAGE_FOLDER)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff")
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    std::cout << "共 " << files.size() << " 张图片，推理输入 1x3x" << INPUT_H << "x" << INPUT_W << std::endl;

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> in_shape{1, 3, INPUT_H, INPUT_W};
    bool first = true;

    for (size_t i = 0; i < files.size(); ++i) {
        std::cout << "[" << (i + 1) << "/" << files.size() << "] " << files[i] << std::endl;
        cv::Mat img = cv::imread(files[i]);
        if (img.empty()) { std::cerr << "  读取失败" << std::endl; continue; }

        try {
            cv::Mat lb;
            LBInfo lbi = letterbox(img, lb, INPUT_H, INPUT_W);
            std::vector<float> input = preprocess(lb, INPUT_H, INPUT_W);

            Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
                mem_info, input.data(), input.size(), in_shape.data(), in_shape.size());
            auto outputs = session.Run(Ort::RunOptions{nullptr},
                                       in_names.data(), &in_tensor, 1,
                                       out_names.data(), 1);

            auto& out = outputs[0];
            auto out_dims = out.GetTensorTypeAndShapeInfo().GetShape();
            const float* out_data = out.GetTensorMutableData<float>();
            // 取最后两个维度作为深度图 Hd x Wd
            int64_t total = 1;
            for (auto d : out_dims) total *= d;
            int Hd = (int)out_dims[out_dims.size() - 2];
            int Wd = (int)out_dims[out_dims.size() - 1];
            if (first) {
                std::cout << "  输出 shape: [";
                for (size_t k = 0; k < out_dims.size(); ++k)
                    std::cout << out_dims[k] << (k + 1 < out_dims.size() ? "," : "");
                std::cout << "] -> 深度图 " << Hd << "x" << Wd << std::endl;
                first = false;
            }

            // 深度图（共享输出缓冲区，不修改）
            cv::Mat depth(Hd, Wd, CV_32F, (void*)out_data);

            // letterbox 反映射：裁掉填充区
            float sy = (float)Hd / INPUT_H, sx = (float)Wd / INPUT_W;
            int t  = (int)std::round(lbi.top * sy);
            int l  = (int)std::round(lbi.left * sx);
            int hh = std::max(1, std::min((int)std::round(lbi.new_h * sy), Hd - t));
            int ww = std::max(1, std::min((int)std::round(lbi.new_w * sx), Wd - l));
            cv::Mat depth_valid = depth(cv::Rect(l, t, ww, hh)).clone();

            // resize 回原图
            cv::Mat depth_full;
            cv::resize(depth_valid, depth_full, cv::Size(img.cols, img.rows), 0, 0, cv::INTER_LINEAR);

            // 归一化 0-255
            double mn, mx;
            cv::minMaxLoc(depth_full, &mn, &mx);
            double scale = 255.0 / (mx - mn + 1e-6);
            cv::Mat d8;
            depth_full.convertTo(d8, CV_8U, scale, -mn * scale);
            // 伪彩色
            cv::Mat colored;
            cv::applyColorMap(d8, colored, cv::COLORMAP_JET);

            cv::imshow("Depth ONNX (ESC to exit)", colored);
            if (!OUTPUT_FOLDER.empty()) {
                std::filesystem::create_directories(OUTPUT_FOLDER);
                cv::imwrite(OUTPUT_FOLDER + "/" +
                            std::filesystem::path(files[i]).filename().string(), colored);
            }
            int key = cv::waitKey(WAIT_MS);
            if (key == 27) { std::cout << "用户中断" << std::endl; break; }
        } catch (const std::exception& e) {
            std::cerr << "  本张处理出错: " << e.what() << std::endl;
            continue;
        }
    }
    cv::destroyAllWindows();
    std::cout << "完成" << std::endl;
    return 0;
}
