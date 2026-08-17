#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <dirent.h>
#include <cstring>
#include <sys/stat.h>
#include <algorithm>
#include <cmath>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>


// ==================== 直接在这里修改参数 ====================
const std::string ENGINE_PATH = "/home/nvidia/code-main/chapter6/YoloLearn/runs/depth/train/weights/best.engine";
const std::string INPUT_DIR = "/home/nvidia/code-main/data/camera";
const int INPUT_SIZE = 640;
const bool SAVE_RESULTS = false;
const std::string SAVE_DIR = "/home/nvidia/code-main/chapter6/YoloLearn/runs/infer_results";
const int DELAY_MS = 1;
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
            output_name = name;
            output_dims = dims;
            std::cout << "OUTPUT";
        }

        std::cout << " shape: [";
        for (int j = 0; j < dims.nbDims; j++) std::cout << dims.d[j] << (j < dims.nbDims-1 ? ", " : "");
        std::cout << "]" << std::endl;
    }

    // 3. Set input shape
    nvinfer1::Dims4 actual_input_dims;
    actual_input_dims.d[0] = 1;
    actual_input_dims.d[1] = 3;
    actual_input_dims.d[2] = INPUT_SIZE;
    actual_input_dims.d[3] = INPUT_SIZE;
    context->setInputShape(input_name.c_str(), actual_input_dims);

    // Get actual output shape
    output_dims = context->getTensorShape(output_name.c_str());

    std::cout << "Actual input: [1, 3, " << INPUT_SIZE << ", " << INPUT_SIZE << "]" << std::endl;
    std::cout << "Actual output: [";
    for (int i = 0; i < output_dims.nbDims; i++) std::cout << output_dims.d[i] << (i < output_dims.nbDims-1 ? ", " : "");
    std::cout << "]" << std::endl;

    // Output: [1, 1, H, W]
    int out_h = output_dims.d[2];
    int out_w = output_dims.d[3];

    size_t input_count = 1 * 3 * INPUT_SIZE * INPUT_SIZE;
    size_t output_count = 1 * 1 * out_h * out_w;

    // Allocate buffers
    void* d_input = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input, input_count * sizeof(float));
    cudaMalloc(&d_output, output_count * sizeof(float));

    std::vector<float> h_input(input_count, 0.0f);
    std::vector<float> h_output(output_count, 0.0f);

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

    // 5. Inference loop
    for (size_t idx = 0; idx < img_files.size(); idx++) {
        cv::Mat img = cv::imread(img_files[idx]);
        if (img.empty()) {
            std::cerr << "[" << idx+1 << "/" << img_files.size() << "] Cannot read: " << img_files[idx] << std::endl;
            continue;
        }

        // === Preprocess: resize to INPUT_SIZE x INPUT_SIZE, normalize ===
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(INPUT_SIZE, INPUT_SIZE));
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

        // HWC -> CHW
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < INPUT_SIZE; y++) {
                for (int x = 0; x < INPUT_SIZE; x++) {
                    h_input[c * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] =
                        rgb.at<cv::Vec3f>(y, x)[c];
                }
            }
        }

        // Copy to device
        cudaMemcpyAsync(d_input, h_input.data(), input_count * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        // Infer
        auto t0 = std::chrono::high_resolution_clock::now();
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
        auto t1 = std::chrono::high_resolution_clock::now();

        // Copy output
        cudaMemcpyAsync(h_output.data(), d_output, output_count * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // === Debug: print first 10 output values ===
        if (idx == 0) {
            std::cout << "First 10 output values: ";
            for (int i = 0; i < 10 && i < (int)output_count; i++) {
                std::cout << h_output[i] << " ";
            }
            std::cout << std::endl;
        }

        // === Parse depth output: [1, 1, out_h, out_w] ===
        cv::Mat depth_map(out_h, out_w, CV_32F, h_output.data());

        // Find min/max
        double min_val, max_val;
        cv::minMaxLoc(depth_map, &min_val, &max_val);

        // Debug output
        std::string basename = img_files[idx].substr(img_files[idx].find_last_of('/') + 1);
        printf("[%zu/%zu] %s -> depth: [%.4f, %.4f] (%.1fms)\n",
               idx+1, img_files.size(), basename.c_str(), min_val, max_val, infer_ms);

        // === If output is all zeros, skip visualization ===
        if (max_val < 1e-6 && min_val > -1e-6) {
            std::cerr << "  WARNING: Output is all zeros!" << std::endl;
        }

        // Normalize to 0-255
        cv::Mat depth_norm;
        if (max_val - min_val > 1e-6) {
            // Clamp to valid range first
            cv::Mat clamped;
            cv::max(depth_map, min_val, clamped);
            cv::min(clamped, max_val, clamped);
            clamped.convertTo(depth_norm, CV_32F, 1.0 / (max_val - min_val),
                              -min_val / (max_val - min_val));
        } else {
            depth_map.convertTo(depth_norm, CV_32F, 0.0);  // all zero
        }
        depth_norm.convertTo(depth_norm, CV_8U, 255.0);

        // Apply JET colormap
        cv::Mat depth_color;
        cv::applyColorMap(depth_norm, depth_color, cv::COLORMAP_JET);

        // Resize depth to original image size for side-by-side
        cv::Mat depth_full;
        cv::resize(depth_color, depth_full, img.size());

        // Side by side: original | depth
        cv::Mat display;
        cv::hconcat(img, depth_full, display);

        // Resize for display if too large
        if (display.cols > 1280) {
            double disp_scale = 1280.0 / display.cols;
            cv::resize(display, display, cv::Size(), disp_scale, disp_scale);
        }

        // Save
        if (SAVE_RESULTS) {
            std::string save_path = SAVE_DIR + "/depth_" + basename;
            cv::imwrite(save_path, depth_full);
        }

        // Show
        cv::imshow("Depth Estimation (left: original, right: depth)", display);
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

