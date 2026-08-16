#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <sys/stat.h>
#include <algorithm>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>


// ==================== 直接在这里修改参数 ====================
const std::string ONNX_PATH  = "/home/nvidia/code-main/chapter6/YoloLearn/runs/depth/train/weights/best.onnx";
const std::string ENGINE_PATH = "/home/nvidia/code-main/chapter6/YoloLearn/runs/depth/train/weights/best.engine";
const std::string PRECISION  = "fp16";  // "fp32" or "fp16"
const int MIN_SIZE   = 320;
const int OPT_SIZE   = 640;
const int MAX_SIZE   = 960;
// =========================================================


class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};


int main() {
    Logger logger;

    std::cout << "Building engine from: " << ONNX_PATH << std::endl;
    std::cout << "Precision: " << PRECISION << std::endl;

    // Create builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

    // Create network with explicit batch
    const auto explicitBatch = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    // Create ONNX parser
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if (!parser) {
        std::cerr << "Failed to create ONNX parser" << std::endl;
        return 1;
    }

    // Parse ONNX file
    std::ifstream file(ONNX_PATH, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open ONNX file: " << ONNX_PATH << std::endl;
        return 1;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);
    file.close();

    if (!parser->parse(data.data(), size)) {
        std::cerr << "Failed to parse ONNX file" << std::endl;
        for (int i = 0; i < parser->getNbErrors(); i++) {
            std::cerr << "  " << parser->getError(i)->desc() << std::endl;
        }
        return 1;
    }
    std::cout << "ONNX parsed successfully" << std::endl;

    // Print input/output info
    for (int i = 0; i < network->getNbInputs(); i++) {
        auto* input = network->getInput(i);
        auto dims = input->getDimensions();
        std::cout << "Input[" << i << "]: " << input->getName() << " shape: [";
        for (int j = 0; j < dims.nbDims; j++) std::cout << dims.d[j] << (j < dims.nbDims-1 ? ", " : "");
        std::cout << "]" << std::endl;
    }
    for (int i = 0; i < network->getNbOutputs(); i++) {
        auto* output = network->getOutput(i);
        auto dims = output->getDimensions();
        std::cout << "Output[" << i << "]: " << output->getName() << " shape: [";
        for (int j = 0; j < dims.nbDims; j++) std::cout << dims.d[j] << (j < dims.nbDims-1 ? ", " : "");
        std::cout << "]" << std::endl;
    }

    // Create builder config
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 4ULL << 30);  // 4GB

    if (PRECISION == "fp16" && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 enabled" << std::endl;
    }

    // Check if input has dynamic shape
    bool has_dynamic = false;
    for (int i = 0; i < network->getNbInputs(); i++) {
        auto* input = network->getInput(i);
        auto dims = input->getDimensions();
        for (int j = 0; j < dims.nbDims; j++) {
            if (dims.d[j] == -1) {
                has_dynamic = true;
                break;
            }
        }
    }

    if (has_dynamic) {
        std::cout << "Dynamic input detected, adding optimization profile" << std::endl;
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();

        for (int i = 0; i < network->getNbInputs(); i++) {
            auto* input = network->getInput(i);
            auto dims = input->getDimensions();
            std::string name = input->getName();

            if (dims.nbDims == 4) {
                nvinfer1::Dims d_min(dims), d_opt(dims), d_max(dims);
                d_min.d[0] = 1; d_min.d[1] = dims.d[1]; d_min.d[2] = MIN_SIZE; d_min.d[3] = MIN_SIZE;
                d_opt.d[0] = 1; d_opt.d[1] = dims.d[1]; d_opt.d[2] = OPT_SIZE; d_opt.d[3] = OPT_SIZE;
                d_max.d[0] = 1; d_max.d[1] = dims.d[1]; d_max.d[2] = MAX_SIZE; d_max.d[3] = MAX_SIZE;
                profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN, d_min);
                profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT, d_opt);
                profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX, d_max);
                std::cout << "  Profile for " << name << ": min=" << MIN_SIZE << " opt=" << OPT_SIZE << " max=" << MAX_SIZE << std::endl;
            }
        }
        config->addOptimizationProfile(profile);
    }

    // Build serialized engine
    std::cout << "Building engine (this may take 1-5 minutes)..." << std::endl;
    nvinfer1::IHostMemory* serialized = builder->buildSerializedNetwork(*network, *config);
    if (!serialized) {
        std::cerr << "Failed to build engine" << std::endl;
        return 1;
    }

    // Save to file
    std::ofstream out(ENGINE_PATH, std::ios::binary);
    out.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
    out.close();

    std::cout << "Engine saved to: " << ENGINE_PATH << " (" << serialized->size() << " bytes)" << std::endl;

    // Cleanup
    delete serialized;
    delete config;
    delete network;
    delete parser;
    delete builder;

    std::cout << "Done!" << std::endl;
    return 0;
}

