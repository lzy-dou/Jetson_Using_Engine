#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <NvInfer.h>
#include <NvOnnxParser.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    std::string onnx_path = "/home/nvidia/code-main/chapter2/YoloLearn/runs/classify/car_cls_run/weights/best.onnx";
    std::string engine_path = "/home/nvidia/code-main/chapter2/YoloLearn/runs/classify/car_cls_run/weights/best.engine";

    if (argc >= 2) onnx_path = argv[1];
    if (argc >= 3) engine_path = argv[2];

    Logger logger;

    std::cout << "Building engine from: " << onnx_path << std::endl;

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

    std::ifstream file(onnx_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Cannot open ONNX file: " << onnx_path << std::endl;
        return 1;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);
    file.close();

    if (!parser->parse(data.data(), size)) {
        std::cerr << "Failed to parse ONNX" << std::endl;
        for (int i = 0; i < parser->getNbErrors(); i++) {
            std::cerr << parser->getError(i)->desc() << std::endl;
        }
        return 1;
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    // Enable FP16
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 enabled" << std::endl;
    }

    std::cout << "Building engine..." << std::endl;
    nvinfer1::IHostMemory* serialized = builder->buildSerializedNetwork(*network, *config);
    if (!serialized) {
        std::cerr << "Failed to build engine" << std::endl;
        return 1;
    }

    std::ofstream out(engine_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
    out.close();

    std::cout << "Engine saved to: " << engine_path << " (" << serialized->size() << " bytes)" << std::endl;

    delete serialized;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return 0;
}


