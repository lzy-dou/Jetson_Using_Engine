#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <dirent.h>
#include <cstring>
#include <sys/stat.h>
#include <algorithm>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>


// ==================== 直接在这里修改参数 ====================
const std::string ENGINE_PATH = "/home/nvidia/code-main/chapter4/YoloLearn/runs/detect/car_det_run/weights/best.engine";
const std::string INPUT_DIR = "/home/nvidia/code-main/data/camera";
const int INPUT_SIZE = 640;
const bool SAVE_RESULTS = false;
const std::string SAVE_DIR = "/home/nvidia/code-main/chapter4/YoloLearn/runs/infer_results";
const int DELAY_MS = 1;

// COCO class names (adjust for your dataset)
const std::vector<std::string> CLASS_NAMES = {"car"};

// Detection confidence threshold
const float CONF_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.45f;
// =========================================================


class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};


std::vector<char> loadEngineFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << path << std::endl;
        exit(1);
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);
    file.close();
    return data;
}


std::vector<std::string> listImages(const std::string& dir) {
    std::vector<std::string> files;
    DIR* d = opendir(dir.c_str());
    if (!d) {
        std::cerr << "Cannot open directory: " << dir << std::endl;
        return files;
    }
    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name = entry->d_name;
        std::string lower = name;
        for (auto& c : lower) c = tolower(c);
        if (lower.size() > 4 &&
            (lower.substr(lower.size()-4) == ".jpg" ||
             lower.substr(lower.size()-5) == ".jpeg" ||
             lower.substr(lower.size()-4) == ".png" ||
             lower.substr(lower.size()-4) == ".bmp")) {
            files.push_back(dir + "/" + name);
        }
    }
    closedir(d);
    std::sort(files.begin(), files.end());
    return files;
}


// Letterbox resize: keep aspect ratio, pad with 114
cv::Mat letterbox(const cv::Mat& img, int target_size, float& scale, int& pad_w, int& pad_h) {
    int h = img.rows;
    int w = img.cols;
    scale = std::min((float)target_size / w, (float)target_size / h);
    int new_w = std::round(w * scale);
    int new_h = std::round(h * scale);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    pad_w = (target_size - new_w) / 2;
    pad_h = (target_size - new_h) / 2;
    cv::Mat padded = cv::Mat::ones(target_size, target_size, img.type()) * 114;
    resized.copyTo(padded(cv::Rect(pad_w, pad_h, new_w, new_h)));
    return padded;
}


void preprocess(const cv::Mat& img, float* input_data, int input_size, float& scale, int& pad_w, int& pad_h) {
    cv::Mat lb = letterbox(img, input_size, scale, pad_w, pad_h);
    cv::Mat rgb;
    cv::cvtColor(lb, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    // HWC -> CHW
    int area = input_size * input_size;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < area; i++) {
            input_data[c * area + i] = rgb.at<cv::Vec3f>(i)[c];
        }
    }
}


struct Detection {
    float x1, y1, x2, y2;
    float conf;
    int class_id;
};


float iou(const Detection& a, const Detection& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}


void nms(std::vector<Detection>& dets, float threshold) {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.conf > b.conf;
    });
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            if (iou(dets[i], dets[j]) > threshold) {
                suppressed[j] = true;
            }
        }
    }
    std::vector<Detection> result;
    for (size_t i = 0; i < dets.size(); i++) {
        if (!suppressed[i]) result.push_back(dets[i]);
    }
    dets = result;
}


// YOLO11 detect output: [1, 4+nc, 8400] or [1, 6, 8400] for nc=1
// YOLOv8 detect output: [1, 4+nc, 8400] or [1, 6, 8400] for nc=1
// Format: [cx, cy, w, h, class_scores...]  transposed as [6, 8400]
std::vector<Detection> postprocess(const float* output_data, const nvinfer1::Dims& output_dims,
                                    int input_size, float scale, int pad_w, int pad_h,
                                    int num_classes) {
    std::vector<Detection> dets;

    // YOLO output shape: [1, 4+nc, num_boxes] or [1, num_boxes, 4+nc]
    int num_dims = output_dims.nbDims;
    int num_boxes = 0;
    int data_per_box = 4 + num_classes;
    bool transposed = false;

    if (num_dims == 3) {
        // [1, 4+nc, num_boxes] - transposed format (common in YOLOv8/v11)
        if (output_dims.d[1] == data_per_box) {
            num_boxes = output_dims.d[2];
            transposed = true;
        }
        // [1, num_boxes, 4+nc]
        else if (output_dims.d[2] == data_per_box) {
            num_boxes = output_dims.d[1];
            transposed = false;
        }
    }

    if (num_boxes == 0) {
        std::cerr << "Unexpected output shape: [";
        for (int i = 0; i < num_dims; i++) std::cerr << output_dims.d[i] << " ";
        std::cerr << "]" << std::endl;
        return dets;
    }

    for (int i = 0; i < num_boxes; i++) {
        float best_score = 0.0f;
        int best_class = -1;

        for (int c = 0; c < num_classes; c++) {
            float score;
            if (transposed) {
                // [4+nc, num_boxes] -> index = (4+c) * num_boxes + i
                score = output_data[(4 + c) * num_boxes + i];
            } else {
                // [num_boxes, 4+nc] -> index = i * data_per_box + 4 + c
                score = output_data[i * data_per_box + 4 + c];
            }
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score < CONF_THRESHOLD) continue;

        float cx, cy, w, h;
        if (transposed) {
            cx = output_data[0 * num_boxes + i];
            cy = output_data[1 * num_boxes + i];
            w  = output_data[2 * num_boxes + i];
            h  = output_data[3 * num_boxes + i];
        } else {
            cx = output_data[i * data_per_box + 0];
            cy = output_data[i * data_per_box + 1];
            w  = output_data[i * data_per_box + 2];
            h  = output_data[i * data_per_box + 3];
        }

        // Convert from center to corner
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        // Remove letterbox padding and scale back
        x1 = (x1 - pad_w) / scale;
        y1 = (y1 - pad_h) / scale;
        x2 = (x2 - pad_w) / scale;
        y2 = (y2 - pad_h) / scale;

        dets.push_back({x1, y1, x2, y2, best_score, best_class});
    }

    nms(dets, NMS_THRESHOLD);
    return dets;
}


int main() {
    Logger logger;

    // 1. Load engine
    std::cout << "Loading engine: " << ENGINE_PATH << std::endl;
    auto engine_data = loadEngineFile(ENGINE_PATH);
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (!engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return 1;
    }
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    std::cout << "Engine loaded." << std::endl;

    // 2. Get input/output info
    int num_io = engine->getNbIOTensors();
    std::string input_name, output_name;
    nvinfer1::Dims input_dims, output_dims;
    size_t input_size = 0, output_size = 0;

    for (int i = 0; i < num_io; i++) {
        const char* name = engine->getIOTensorName(i);
        auto dims = engine->getTensorShape(name);
        size_t sz = 1;
        for (int j = 0; j < dims.nbDims; j++) sz *= dims.d[j];

        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_name = name;
            input_dims = dims;
            input_size = sz;
        } else {
            output_name = name;
            output_dims = dims;
            output_size = sz;
        }
    }

    std::cout << "Input: " << input_name << " shape: [";
    for (int i = 0; i < input_dims.nbDims; i++) std::cout << input_dims.d[i] << (i < input_dims.nbDims-1 ? ", " : "");
    std::cout << "]" << std::endl;

    std::cout << "Output: " << output_name << " shape: [";
    for (int i = 0; i < output_dims.nbDims; i++) std::cout << output_dims.d[i] << (i < output_dims.nbDims-1 ? ", " : "");
    std::cout << "]" << std::endl;

    // 3. Set dynamic input shape and allocate buffers
    // Override dynamic dims with actual INPUT_SIZE
    nvinfer1::Dims actual_input_dims = input_dims;
    actual_input_dims.d[0] = 1;
    actual_input_dims.d[1] = 3;
    actual_input_dims.d[2] = INPUT_SIZE;
    actual_input_dims.d[3] = INPUT_SIZE;
    context->setInputShape(input_name.c_str(), actual_input_dims);

    // Get actual output shape after setting input
    output_dims = context->getTensorShape(output_name.c_str());

    // Recalculate sizes with actual dims
    input_size = 1;
    for (int j = 0; j < actual_input_dims.nbDims; j++) input_size *= actual_input_dims.d[j];
    output_size = 1;
    for (int j = 0; j < output_dims.nbDims; j++) output_size *= output_dims.d[j];

    std::cout << "Actual input shape: [";
    for (int i = 0; i < actual_input_dims.nbDims; i++) std::cout << actual_input_dims.d[i] << (i < actual_input_dims.nbDims-1 ? ", " : "");
    std::cout << "]" << std::endl;
    std::cout << "Actual output shape: [";
    for (int i = 0; i < output_dims.nbDims; i++) std::cout << output_dims.d[i] << (i < output_dims.nbDims-1 ? ", " : "");
    std::cout << "]" << std::endl;

    // Allocate buffers based on actual sizes
    void* d_input = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    std::vector<float> h_input(input_size);
    std::vector<float> h_output(output_size);

    context->setTensorAddress(input_name.c_str(), d_input);
    context->setTensorAddress(output_name.c_str(), d_output);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 4. Collect images
    std::vector<std::string> img_files;
    struct stat st;
    if (stat(INPUT_DIR.c_str(), &st) == 0) {
        if (S_ISDIR(st.st_mode)) {
            img_files = listImages(INPUT_DIR);
        } else {
            img_files.push_back(INPUT_DIR);
        }
    }

    if (img_files.empty()) {
        std::cerr << "No images found in: " << INPUT_DIR << std::endl;
        return 1;
    }
    std::cout << "Found " << img_files.size() << " images to infer" << std::endl;

    if (SAVE_RESULTS) {
        std::string cmd = "mkdir -p " + SAVE_DIR;
        system(cmd.c_str());
    }

    // Determine num_classes from output shape
    // Output: [1, 4+nc, num_boxes] or [1, num_boxes, 4+nc]
    int num_classes = CLASS_NAMES.size();
    if (output_dims.nbDims == 3) {
        if (output_dims.d[1] < output_dims.d[2]) {
            // [1, 4+nc, num_boxes]
            num_classes = output_dims.d[1] - 4;
        } else {
            // [1, num_boxes, 4+nc]
            num_classes = output_dims.d[2] - 4;
        }
    }
    std::cout << "Num classes: " << num_classes << std::endl;

    // 5. Inference loop
    for (size_t idx = 0; idx < img_files.size(); idx++) {
        cv::Mat img = cv::imread(img_files[idx]);
        if (img.empty()) {
            std::cerr << "[" << idx+1 << "/" << img_files.size() << "] Cannot read: " << img_files[idx] << std::endl;
            continue;
        }

        // Preprocess
        float scale;
        int pad_w, pad_h;
        preprocess(img, h_input.data(), INPUT_SIZE, scale, pad_w, pad_h);

        // Copy to device
        cudaMemcpyAsync(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice, stream);

        // Infer
        auto t0 = std::chrono::high_resolution_clock::now();
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
        auto t1 = std::chrono::high_resolution_clock::now();

        // Copy output
        cudaMemcpyAsync(h_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // Postprocess
        auto detections = postprocess(h_output.data(), output_dims, INPUT_SIZE, scale, pad_w, pad_h, num_classes);

        // Print
        std::string basename = img_files[idx].substr(img_files[idx].find_last_of('/') + 1);
        printf("[%zu/%zu] %s -> %zu detections (%.1fms)\n",
               idx+1, img_files.size(), basename.c_str(), detections.size(), infer_ms);

        // Draw
        cv::Mat display = img.clone();
        for (const auto& det : detections) {
            int x1 = std::max(0, (int)det.x1);
            int y1 = std::max(0, (int)det.y1);
            int x2 = std::min(display.cols - 1, (int)det.x2);
            int y2 = std::min(display.rows - 1, (int)det.y2);

            cv::Scalar color(0, 255, 0);
            cv::rectangle(display, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

            std::string label;
            if (det.class_id < (int)CLASS_NAMES.size()) {
                label = CLASS_NAMES[det.class_id] + " " + cv::format("%.2f", det.conf);
            } else {
                label = cv::format("cls%d %.2f", det.class_id, det.conf);
            }

            int baseline;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
            cv::rectangle(display, cv::Point(x1, y1 - text_size.height - 5),
                          cv::Point(x1 + text_size.width, y1), color, -1);
            cv::putText(display, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        }

        // Resize for display if too large
        if (display.cols > 1280) {
            double scale = 1280.0 / display.cols;
            cv::resize(display, display, cv::Size(), scale, scale);
        }

        // Save
        if (SAVE_RESULTS) {
            std::string save_path = SAVE_DIR + "/pred_" + basename;
            cv::imwrite(save_path, display);
        }

        // Show
        cv::imshow("TensorRT Detection", display);
        if (DELAY_MS == 0) {
            int key = cv::waitKey(0);
            if (key == 27) {
                std::cout << "Interrupted by user" << std::endl;
                break;
            }
        } else {
            cv::waitKey(DELAY_MS);
        }
    }

    // Cleanup
    cv::destroyAllWindows();
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
    delete context;
    delete engine;
    delete runtime;

    std::cout << "Inference complete!" << std::endl;
    return 0;
}


