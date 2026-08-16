// =============================================================================
// YOLO11 ONNX C++ 推理（支持动态矩形输入 n x m）
// 依赖：ONNX Runtime(C++) + OpenCV
// 与 Python 版的 letterbox 预处理 / 解码-NMS 后处理完全对齐。
// 模型需用 dynamic=True 导出（见 export_onnx_dynamic.py）。
// 编译见同目录 CMakeLists.txt。
// =============================================================================
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

// ============================ 配置区域 ============================
const std::string MODEL_PATH   = "/home/nvidia/code-main/chapter4/YoloLearn/runs/detect/car_det_run/weights/best.onnx";   // ONNX 模型路径
const std::string IMAGE_FOLDER = "/home/nvidia/code-main/data/camera";       // 图片文件夹
const std::string OUTPUT_FOLDER = "";            // 保存目录；空字符串则不保存

const int   INPUT_H = 320;                       // 推理输入高（必须 32 的倍数）
const int   INPUT_W = 320;                       // 推理输入宽（必须 32 的倍数）
const float CONF_THRES = 0.6f;
const float IOU_THRES  = 0.45f;
const int   WAIT_KEY   = 1;                     // 显示等待 ms，0 表示等待按键
const int   START_INDEX = 2500;   // 新增：起始索引（0-based）
// =================================================================

const std::vector<std::string> CLASS_NAMES = {"vehicle"};

struct Detection {
    float x1, y1, x2, y2, score;
    int   class_id;
};

// ---------------- letterbox：等比缩放并填充到 INPUT_H x INPUT_W ----------------
struct LBInfo { float ratio; int pad_top, pad_left; };

static LBInfo letterbox(const cv::Mat& src, cv::Mat& dst, int target_h, int target_w) {
    int h = src.rows, w = src.cols;
    float r = std::min((float)target_h / h, (float)target_w / w);
    int new_w = (int)std::round(w * r);
    int new_h = (int)std::round(h * r);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 与 ultralytics 一致：左右/上下按 round(±0.1) 分配
    float dw = (target_w - new_w) / 2.0f;
    float dh = (target_h - new_h) / 2.0f;
    int top    = (int)std::round(dh - 0.1f);
    int bottom = (target_h - new_h) - top;
    int left   = (int)std::round(dw - 0.1f);
    int right  = (target_w - new_w) - left;
    cv::copyMakeBorder(resized, dst, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return {r, top, left};
}

// ---------------- 预处理：BGR->RGB, HWC->CHW, /255 ----------------
static std::vector<float> preprocess(const cv::Mat& letterboxed, int H, int W) {
    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    cv::Mat f32;
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> chans(3);
    cv::split(f32, chans);  // RGB 顺序：chans[0]=R, [1]=G, [2]=B

    std::vector<float> input((size_t)3 * H * W);
    for (int c = 0; c < 3; ++c) {
        const cv::Mat& ch = chans[c];
        for (int y = 0; y < H; ++y) {
            const float* src_row = ch.ptr<float>(y);
            float* dst_row = input.data() + (size_t)c * H * W + (size_t)y * W;
            std::memcpy(dst_row, src_row, (size_t)W * sizeof(float));
        }
    }
    return input;
}

// ---------------- 后处理：解码 + 置信度过滤 + NMS + 映射回原图 ----------------
static std::vector<Detection> postprocess(const float* out, const std::vector<int64_t>& dims,
                                          float ratio, int pad_left, int pad_top,
                                          int orig_w, int orig_h,
                                          float conf_thres, float iou_thres) {
    // dims: [1, C, A] 或 [1, A, C]，C = 4 + nc
    if (dims.size() < 3) {
        std::cerr << "输出维度<3，跳过" << std::endl;
        return {};
    }
    // 合法性校验：防止异常/动态符号维度(-1)或超大值撑爆 vector
    for (int64_t d : dims) {
        if (d <= 0 || d > 100000000LL) {
            std::cerr << "输出维度异常，跳过: [";
            for (int64_t x : dims) std::cerr << x << " ";
            std::cerr << "]" << std::endl;
            return {};
        }
    }
    int dim1 = (int)dims[1], dim2 = (int)dims[2];
    int channels, num_anchors;
    bool transpose_needed;
    if (dim1 < dim2) { channels = dim1; num_anchors = dim2; transpose_needed = false; }
    else             { channels = dim2; num_anchors = dim1; transpose_needed = true;  }
    int nc = channels - 4;
    if (nc < 1) { std::cerr << "输出通道数异常: " << channels << std::endl; return {}; }

    auto at = [&](int c, int a) -> float {
        return transpose_needed ? out[(size_t)a * channels + c]
                                : out[(size_t)c * num_anchors + a];
    };

    std::vector<cv::Rect2d> boxes_xywh;
    std::vector<float>      scores;
    std::vector<int>        class_ids;

    for (int a = 0; a < num_anchors; ++a) {
        float cx = at(0, a), cy = at(1, a), w = at(2, a), h = at(3, a);
        float max_s = 0.f; int max_id = 0;
        for (int c = 0; c < nc; ++c) {
            float s = at(4 + c, a);
            if (s > max_s) { max_s = s; max_id = c; }
        }
        if (max_s > conf_thres) {
            boxes_xywh.emplace_back(cx - w / 2.0, cy - h / 2.0, (double)w, (double)h);
            scores.push_back(max_s);
            class_ids.push_back(max_id);
        }
    }

    std::vector<int> kept;
    if (!boxes_xywh.empty())
        cv::dnn::NMSBoxes(boxes_xywh, scores, conf_thres, iou_thres, kept);

    std::vector<Detection> result;
    result.reserve(kept.size());
    for (int idx : kept) {
        const auto& b = boxes_xywh[idx];
        float cx = (float)(b.x + b.width  / 2.0);
        float cy = (float)(b.y + b.height / 2.0);
        float bw = (float)b.width, bh = (float)b.height;
        // 输入空间 -> 原图空间
        float x1 = (cx - bw / 2.f - pad_left) / ratio;
        float y1 = (cy - bh / 2.f - pad_top)  / ratio;
        float x2 = (cx + bw / 2.f - pad_left) / ratio;
        float y2 = (cy + bh / 2.f - pad_top)  / ratio;
        x1 = std::clamp(x1, 0.f, (float)orig_w);
        y1 = std::clamp(y1, 0.f, (float)orig_h);
        x2 = std::clamp(x2, 0.f, (float)orig_w);
        y2 = std::clamp(y2, 0.f, (float)orig_h);
        result.push_back({x1, y1, x2, y2, scores[idx], class_ids[idx]});
    }
    return result;
}

static void draw(cv::Mat& im, const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        cv::rectangle(im, cv::Point((int)d.x1, (int)d.y1), cv::Point((int)d.x2, (int)d.y2),
                      cv::Scalar(0, 255, 0), 2);
        std::string label = CLASS_NAMES[d.class_id] + " " + cv::format("%.2f", d.score);
        int base = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
        cv::rectangle(im, cv::Point((int)d.x1, (int)d.y1 - ts.height - 6),
                      cv::Point((int)d.x1 + ts.width, (int)d.y1), cv::Scalar(0, 255, 0), -1);
        cv::putText(im, label, cv::Point((int)d.x1, (int)d.y1 - 4),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }
}

int main() {
    // 校验输入尺寸为 32 的倍数
    if (INPUT_H % 32 != 0 || INPUT_W % 32 != 0) {
        std::cerr << "INPUT_H/INPUT_W 必须是 32 的整数倍！当前 " << INPUT_H << "x" << INPUT_W << std::endl;
        return -1;
    }

    // 1. 初始化 ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo");
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

    std::cout << "模型加载成功: " << MODEL_PATH << std::endl;
    std::cout << "推理输入: 1x3x" << INPUT_H << "x" << INPUT_W << std::endl;

    // 打印模型 I/O 形状（动态维度会显示 -1）
    // 注意：部分 ORT C++ 版本查询动态模型类型信息时会抛 std::length_error，
    //       该段仅用于诊断打印，非推理必需，故加 try/catch 保护。
    try {
        auto in_info  = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto out_info = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto print_shape = [](const std::string& tag, const std::vector<int64_t>& s) {
            std::cout << tag << " [";
            for (size_t i = 0; i < s.size(); ++i) std::cout << s[i] << (i + 1 < s.size() ? "," : "");
            std::cout << "]" << std::endl;
        };
        print_shape("输入形状", in_info.GetShape());
        print_shape("输出形状", out_info.GetShape());
    } catch (const std::exception& e) {
        std::cerr << "（类型信息查询跳过: " << e.what() << "，不影响推理）" << std::endl;
    }

    // 2. 收集图片
    if (!std::filesystem::exists(IMAGE_FOLDER)) {
        std::cerr << "图片文件夹不存在: " << IMAGE_FOLDER << std::endl;
        return -1;
    }
    std::vector<std::string> files;
    for (const auto& e : std::filesystem::directory_iterator(IMAGE_FOLDER)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" ||
            ext == ".tif" || ext == ".tiff")
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) { std::cerr << "未找到图片: " << IMAGE_FOLDER << std::endl; return -1; }
    std::cout << "共 " << files.size() << " 张图片，开始推理..." << std::endl;

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> in_shape{1, 3, INPUT_H, INPUT_W};


    std::cout << "共 " << files.size() << " 张图片，";
    size_t start = std::max(0, START_INDEX);
    if (start >= files.size()) {
        std::cerr << "起始索引 " << START_INDEX << " 超出图片总数 " << files.size() << "，将从第 1 张开始。" << std::endl;
        start = 0;
    }
    bool shape_printed = false;
    for (size_t i = start; i < files.size(); ++i) {
        std::cout << "\n[" << (i + 1) << "/" << files.size() << "] " << files[i] << std::endl;
        cv::Mat img = cv::imread(files[i]);
        if (img.empty()) { std::cerr << "  读取失败，跳过" << std::endl; continue; }

        try {
            // 预处理
            cv::Mat lb;
            LBInfo lbi = letterbox(img, lb, INPUT_H, INPUT_W);
            std::vector<float> input = preprocess(lb, INPUT_H, INPUT_W);

            // 推理
            Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
                mem_info, input.data(), input.size(), in_shape.data(), in_shape.size());
            auto outputs = session.Run(Ort::RunOptions{nullptr},
                                       in_names.data(), &in_tensor, 1,
                                       out_names.data(), 1);

            auto& out = outputs[0];
            auto out_type_info = out.GetTensorTypeAndShapeInfo();
            auto out_dims = out_type_info.GetShape();
            const float* out_data = out.GetTensorMutableData<float>();
            if (!shape_printed) {
                std::cout << "  实际输出形状: [";
                for (size_t k = 0; k < out_dims.size(); ++k)
                    std::cout << out_dims[k] << (k + 1 < out_dims.size() ? "," : "");
                std::cout << "]  (格式 [1, 4+nc, anchors]，anchor 数随输入尺寸变化)" << std::endl;
                shape_printed = true;
            }

            // 后处理
            auto dets = postprocess(out_data, out_dims, lbi.ratio, lbi.pad_left, lbi.pad_top,
                                    img.cols, img.rows, CONF_THRES, IOU_THRES);

            // 画框
            cv::Mat annotated = img.clone();
            draw(annotated, dets);
            for (const auto& d : dets)
                std::cout << "  框: (" << (int)d.x1 << "," << (int)d.y1 << ","
                          << (int)d.x2 << "," << (int)d.y2 << ") conf=" << d.score << std::endl;
            if (dets.empty()) std::cout << "  未检测到目标" << std::endl;

            cv::imshow("ONNX Detection - ESC to exit", annotated);
            int key = cv::waitKey(WAIT_KEY);
            if (key == 27) { std::cout << "用户中断，退出..." << std::endl; break; }

            if (!OUTPUT_FOLDER.empty()) {
                std::string save = OUTPUT_FOLDER + "/" +
                                   std::filesystem::path(files[i]).filename().string();
                cv::imwrite(save, annotated);
                std::cout << "  已保存: " << save << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "  本张处理出错，跳过: " << e.what() << std::endl;
            continue;
        }
    }
    cv::destroyAllWindows();
    std::cout << "推理完成！" << std::endl;
    return 0;
}
