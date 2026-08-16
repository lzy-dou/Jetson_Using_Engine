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
const std::string ENGINE_PATH = "/home/nvidia/code-main/chapter2/YoloLearn/runs/classify/car_cls_run/weights/best.engine";
const std::string INPUT_DIR = "/home/nvidia/code-main/chapter2/YoloLearn/car_classification_split/val/car";
const int INPUT_SIZE = 64;
const bool SAVE_RESULTS = false;
const std::string SAVE_DIR = "/home/nvidia/code-main/chapter2/YoloLearn/runs/infer_results";
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


void preprocess(const cv::Mat& img, float* input_data, int input_size) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_size, input_size));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    // HWC -> CHW
    int area = input_size * input_size;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < area; i++) {
            input_data[c * area + i] = rgb.at<cv::Vec3f>(i)[c];
        }
    }
}


void softmax(float* data, int n) {
    float max_val = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    for (int i = 0; i < n; i++) {
        data[i] /= sum;
    }
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
    std::vector<std::string> tensor_names(num_io);
    std::vector<nvinfer1::DataType> tensor_dtypes(num_io);
    std::vector<nvinfer1::Dims> tensor_dims(num_io);
    std::vector<size_t> tensor_sizes(num_io);

    std::string input_name, output_name;
    nvinfer1::Dims input_dims, output_dims;
    size_t input_size = 0, output_size = 0;

    for (int i = 0; i < num_io; i++) {
        tensor_names[i] = engine->getIOTensorName(i);
        tensor_dtypes[i] = engine->getTensorDataType(tensor_names[i].c_str());
        tensor_dims[i] = engine->getTensorShape(tensor_names[i].c_str());
        tensor_sizes[i] = 1;
        for (int j = 0; j < tensor_dims[i].nbDims; j++) {
            tensor_sizes[i] *= tensor_dims[i].d[j];
        }
        if (engine->getTensorIOMode(tensor_names[i].c_str()) == nvinfer1::TensorIOMode::kINPUT) {
            input_name = tensor_names[i];
            input_dims = tensor_dims[i];
            input_size = tensor_sizes[i];
        } else {
            output_name = tensor_names[i];
            output_dims = tensor_dims[i];
            output_size = tensor_sizes[i];
        }
    }

    std::cout << "Input: " << input_name << " shape: [";
    for (int i = 0; i < input_dims.nbDims; i++) std::cout << input_dims.d[i] << (i < input_dims.nbDims-1 ? ", " : "");
    std::cout << "] size=" << input_size << std::endl;

    std::cout << "Output: " << output_name << " shape: [";
    for (int i = 0; i < output_dims.nbDims; i++) std::cout << output_dims.d[i] << (i < output_dims.nbDims-1 ? ", " : "");
    std::cout << "] size=" << output_size << std::endl;

    // 3. Allocate buffers
    void* d_input = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    std::vector<float> h_input(input_size);
    std::vector<float> h_output(output_size);

    // Set tensor addresses (TensorRT 10.x)
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

    // Create save dir
    if (SAVE_RESULTS) {
        std::string cmd = "mkdir -p " + SAVE_DIR;
        system(cmd.c_str());
    }

    // 5. Inference loop
    int num_classes = output_dims.d[output_dims.nbDims - 1];

    for (size_t idx = 0; idx < img_files.size(); idx++) {
        cv::Mat img = cv::imread(img_files[idx]);
        if (img.empty()) {
            std::cerr << "[" << idx+1 << "/" << img_files.size() << "] Cannot read: " << img_files[idx] << std::endl;
            continue;
        }

        // Preprocess
        preprocess(img, h_input.data(), INPUT_SIZE);

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

        // Postprocess: softmax + argmax
        softmax(h_output.data(), num_classes);
        int pred_class = 0;
        float confidence = h_output[0];
        for (int i = 1; i < num_classes; i++) {
            if (h_output[i] > confidence) {
                confidence = h_output[i];
                pred_class = i;
            }
        }

        // Print
        std::string basename = img_files[idx].substr(img_files[idx].find_last_of('/') + 1);
        printf("[%zu/%zu] %s -> Class: %d (%.4f) (%.1fms)\n",
               idx+1, img_files.size(), basename.c_str(), pred_class, confidence, infer_ms);

        // Draw
        cv::Mat display = img.clone();
        char label[128];
        snprintf(label, sizeof(label), "Class: %d (%.4f)", pred_class, confidence);
        cv::putText(display, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(display, basename, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 1);

        // Resize for display if too small
        if (display.cols < 400) {
            double scale = 400.0 / display.cols;
            cv::resize(display, display, cv::Size(), scale, scale);
        }

        // Save
        if (SAVE_RESULTS) {
            std::string save_path = SAVE_DIR + "/pred_" + basename;
            cv::imwrite(save_path, display);
        }

        // Show
        cv::imshow("TensorRT Inference", display);
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


