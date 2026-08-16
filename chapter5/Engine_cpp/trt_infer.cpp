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
const std::string ENGINE_PATH = "/home/nvidia/code-main/chapter5/YoloLearn/runs/segment/car_seg_run/weights/best.engine";
const std::string INPUT_DIR = "/home/nvidia/code-main/data/camera";
const int INPUT_SIZE = 640;
const bool SAVE_RESULTS = false;
const std::string SAVE_DIR = "/home/nvidia/code-main/chapter5/YoloLearn/runs/infer_results";
const int DELAY_MS = 1;

const std::vector<std::string> CLASS_NAMES = {
    "class0", "class1", "class2", "class3", "class4",
    "class5", "class6", "class7", "class8", "class9",
    "class10", "class11", "class12", "class13", "class14",
    "class15", "class16", "class17", "class18"
};

const float CONF_THRESHOLD = 0.5f;
const float NMS_THRESHOLD = 0.45f;
const int MASK_THRESHOLD = 50;  // 0-255
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
    std::vector<float> mask_coeffs;  // 32 coefficients
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


// Fixed color per class: BGR format
// Modify these to match your class names and desired colors
cv::Scalar getColorByClass(int class_id) {
    // Index must match CLASS_NAMES order above
    static cv::Scalar colors[] = {
        {0, 0, 255},       // class0 - red
        {0, 255, 0},       // class1 - green
        {0, 128, 0},       // class2 - half green
        {0, 255, 255},     // class3 - yellow
        {0, 100, 100},     // class4 - brown-ish
        {255, 0, 0},       // class5 - blue
        {255, 0, 255},     // class6 - magenta
        {203, 192, 255},   // class7 - pink
        {255, 255, 0},     // class8 - cyan
        {0, 0, 128},       // class9 - dark red
        {0, 128, 128},     // class10 - olive
        {128, 0, 0},       // class11 - navy
        {0, 69, 255},      // class12 - orange
        {128, 128, 0},     // class13 - teal
        {128, 0, 128},     // class14 - purple
        {0, 128, 128},     // class15 - olive
        {100, 100, 100},   // class16 - gray
        {50, 50, 50},      // class17 - dark gray
        {200, 200, 200}    // class18 - light gray
    };
    if (class_id >= 0 && class_id < 19) {
        return colors[class_id];
    }
    return cv::Scalar(255, 255, 255);  // default white
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
    std::string input_name;
    std::string output0_name, output1_name;
    nvinfer1::Dims input_dims, output0_dims, output1_dims;

    for (int i = 0; i < num_io; i++) {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        auto dims = engine->getTensorShape(name);

        std::cout << "Tensor[" << i << "]: " << name << " mode: ";
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name = name;
            input_dims = dims;
            std::cout << "INPUT";
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            if (output0_name.empty()) {
                output0_name = name;
                output0_dims = dims;
            } else {
                output1_name = name;
                output1_dims = dims;
            }
            std::cout << "OUTPUT";
        } else {
            std::cout << "OTHER";
        }

        std::cout << " shape: [";
        for (int j = 0; j < dims.nbDims; j++) std::cout << dims.d[j] << (j < dims.nbDims-1 ? ", " : "");
        std::cout << "]" << std::endl;
    }

    // 3. Set input shape
    nvinfer1::Dims actual_input_dims = input_dims;
    actual_input_dims.d[0] = 1;
    if (input_dims.d[2] == -1) actual_input_dims.d[2] = INPUT_SIZE;
    if (input_dims.d[3] == -1) actual_input_dims.d[3] = INPUT_SIZE;
    context->setInputShape(input_name.c_str(), actual_input_dims);

    // Get actual output shapes
    output0_dims = context->getTensorShape(output0_name.c_str());
    output1_dims = context->getTensorShape(output1_name.c_str());

    // Print actual shapes
    std::cout << "Input: [";
    for (int i = 0; i < actual_input_dims.nbDims; i++) std::cout << actual_input_dims.d[i] << (i < actual_input_dims.nbDims-1 ? ", " : "");
    std::cout << "]" << std::endl;

    std::cout << "Output0 (" << output0_name << "): [";
    for (int i = 0; i < output0_dims.nbDims; i++) std::cout << output0_dims.d[i] << (i < output0_dims.nbDims-1 ? ", " : "");
    std::cout << "]" << std::endl;

    std::cout << "Output1 (" << output1_name << "): [";
    for (int i = 0; i < output1_dims.nbDims; i++) std::cout << output1_dims.d[i] << (i < output1_dims.nbDims-1 ? ", " : "");
    std::cout << "]" << std::endl;

    // Calculate sizes
    size_t input_count = 1;
    for (int i = 0; i < actual_input_dims.nbDims; i++) input_count *= actual_input_dims.d[i];

    size_t output0_count = 1;
    for (int i = 0; i < output0_dims.nbDims; i++) output0_count *= output0_dims.d[i];

    size_t output1_count = 1;
    for (int i = 0; i < output1_dims.nbDims; i++) output1_count *= output1_dims.d[i];

    // Determine num_classes from output0 shape
    // output0: [1, 4+nc+32, num_boxes] or [1, num_boxes, 4+nc+32]
    int num_classes = 0;
    int num_masks = 32;
    int num_boxes = 0;
    int data_per_box = 0;
    bool transposed = false;

    if (output0_dims.nbDims == 3) {
        if (output0_dims.d[1] < output0_dims.d[2]) {
            // [1, 4+nc+32, num_boxes]
            data_per_box = output0_dims.d[1];
            num_boxes = output0_dims.d[2];
            num_classes = data_per_box - 4 - 32;
            transposed = true;
        } else {
            // [1, num_boxes, 4+nc+32]
            data_per_box = output0_dims.d[2];
            num_boxes = output0_dims.d[1];
            num_classes = data_per_box - 4 - 32;
            transposed = false;
        }
    }

    std::cout << "Num classes: " << num_classes << ", Num masks: " << num_masks
              << ", Num boxes: " << num_boxes << ", Transposed: " << transposed << std::endl;

    // Mask proto shape: [1, 32, mask_h, mask_w]
    int mask_c = output1_dims.d[1];
    int mask_h = output1_dims.d[2];
    int mask_w = output1_dims.d[3];
    std::cout << "Mask proto: " << mask_c << "x" << mask_h << "x" << mask_w << std::endl;

    // Allocate buffers
    void* d_input = nullptr;
    void* d_output0 = nullptr;
    void* d_output1 = nullptr;
    cudaMalloc(&d_input, input_count * sizeof(float));
    cudaMalloc(&d_output0, output0_count * sizeof(float));
    cudaMalloc(&d_output1, output1_count * sizeof(float));

    std::vector<float> h_input(input_count);
    std::vector<float> h_output0(output0_count);
    std::vector<float> h_output1(output1_count);

    context->setTensorAddress(input_name.c_str(), d_input);
    context->setTensorAddress(output0_name.c_str(), d_output0);
    context->setTensorAddress(output1_name.c_str(), d_output1);

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
        cudaMemcpyAsync(d_input, h_input.data(), input_count * sizeof(float), cudaMemcpyHostToDevice, stream);

        // Infer
        auto t0 = std::chrono::high_resolution_clock::now();
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
        auto t1 = std::chrono::high_resolution_clock::now();

        // Copy outputs
        cudaMemcpyAsync(h_output0.data(), d_output0, output0_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_output1.data(), d_output1, output1_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // Parse detections from output0
        std::vector<Detection> detections;

        for (int i = 0; i < num_boxes; i++) {
            float best_score = 0.0f;
            int best_class = -1;

            for (int c = 0; c < num_classes; c++) {
                float score;
                if (transposed) {
                    score = h_output0[(4 + c) * num_boxes + i];
                } else {
                    score = h_output0[i * data_per_box + 4 + c];
                }
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }

            if (best_score < CONF_THRESHOLD) continue;

            float cx, cy, w, h;
            if (transposed) {
                cx = h_output0[0 * num_boxes + i];
                cy = h_output0[1 * num_boxes + i];
                w  = h_output0[2 * num_boxes + i];
                h  = h_output0[3 * num_boxes + i];
            } else {
                cx = h_output0[i * data_per_box + 0];
                cy = h_output0[i * data_per_box + 1];
                w  = h_output0[i * data_per_box + 2];
                h  = h_output0[i * data_per_box + 3];
            }

            float x1 = cx - w / 2.0f;
            float y1 = cy - h / 2.0f;
            float x2 = cx + w / 2.0f;
            float y2 = cy + h / 2.0f;

            x1 = (x1 - pad_w) / scale;
            y1 = (y1 - pad_h) / scale;
            x2 = (x2 - pad_w) / scale;
            y2 = (y2 - pad_h) / scale;

            // Extract mask coefficients
            std::vector<float> coeffs(num_masks);
            for (int m = 0; m < num_masks; m++) {
                if (transposed) {
                    coeffs[m] = h_output0[(4 + num_classes + m) * num_boxes + i];
                } else {
                    coeffs[m] = h_output0[i * data_per_box + 4 + num_classes + m];
                }
            }

            detections.push_back({x1, y1, x2, y2, best_score, best_class, coeffs});
        }

        // NMS
        nms(detections, NMS_THRESHOLD);

        // Print
        std::string basename = img_files[idx].substr(img_files[idx].find_last_of('/') + 1);
        printf("[%zu/%zu] %s -> %zu instances (%.1fms)\n",
               idx+1, img_files.size(), basename.c_str(), detections.size(), infer_ms);

        // Create black background (only show masks)
        cv::Mat mask_display = cv::Mat::zeros(img.size(), CV_8UC3);

        // Process each detection: generate mask and draw on black background
        for (size_t d = 0; d < detections.size(); d++) {
            const auto& det = detections[d];

            // Generate mask: mask_coeffs(1x32) x mask_proto(32x160x160) = mask(160x160)
            cv::Mat single_mask(mask_h, mask_w, CV_32F);
            for (int mh = 0; mh < mask_h; mh++) {
                for (int mw = 0; mw < mask_w; mw++) {
                    float val = 0.0f;
                    for (int c = 0; c < mask_c; c++) {
                        float proto = h_output1[c * mask_h * mask_w + mh * mask_w + mw];
                        float coeff = det.mask_coeffs[c];
                        val += proto * coeff;
                    }
                    single_mask.at<float>(mh, mw) = val;
                }
            }

            // Apply sigmoid
            cv::Mat sigmoid_mask(mask_h, mask_w, CV_32F);
            for (int mh = 0; mh < mask_h; mh++) {
                for (int mw = 0; mw < mask_w; mw++) {
                    float v = single_mask.at<float>(mh, mw);
                    sigmoid_mask.at<float>(mh, mw) = 1.0f / (1.0f + expf(-v));
                }
            }

            // Convert to 8-bit
            cv::Mat mask_8u;
            sigmoid_mask.convertTo(mask_8u, CV_8U, 255.0);

            // Resize mask to input_size
            cv::Mat mask_resized;
            cv::resize(mask_8u, mask_resized, cv::Size(INPUT_SIZE, INPUT_SIZE));

            // Remove letterbox padding
            int img_new_w = std::round(img.cols * scale);
            int img_new_h = std::round(img.rows * scale);
            cv::Mat mask_unpadded = cv::Mat::zeros(img.rows, img.cols, CV_8U);
            cv::Rect roi(pad_w, pad_h, std::min(img_new_w, INPUT_SIZE - 2*pad_w),
                         std::min(img_new_h, INPUT_SIZE - 2*pad_h));
            if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= INPUT_SIZE && roi.y + roi.height <= INPUT_SIZE) {
                mask_resized(roi).copyTo(mask_unpadded);
            }

            // Crop mask to detection box
            int x1 = std::max(0, (int)det.x1);
            int y1 = std::max(0, (int)det.y1);
            int x2 = std::min(img.cols - 1, (int)det.x2);
            int y2 = std::min(img.rows - 1, (int)det.y2);

            // Get color by class (same class = same color)
            cv::Scalar color = getColorByClass(det.class_id);

            // Draw mask on black background (only where mask > threshold AND inside detection box)
            for (int y = y1; y <= y2; y++) {
                for (int x = x1; x <= x2; x++) {
                    if (mask_unpadded.at<uchar>(y, x) > MASK_THRESHOLD) {
                        mask_display.at<cv::Vec3b>(y, x) = cv::Vec3b(color[0], color[1], color[2]);
                    }
                }
            }
        }

        // Resize for display if too large
        cv::Mat display = mask_display;
        if (display.cols > 1280) {
            double disp_scale = 1280.0 / display.cols;
            cv::resize(display, display, cv::Size(), disp_scale, disp_scale);
        }

        // Save
        if (SAVE_RESULTS) {
            std::string save_path = SAVE_DIR + "/mask_" + basename;
            cv::imwrite(save_path, mask_display);
        }

        // Show
        cv::imshow("Mask Only", display);
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
    cudaFree(d_output0);
    cudaFree(d_output1);
    delete context;
    delete engine;
    delete runtime;

    std::cout << "Inference complete!" << std::endl;
    return 0;
}
