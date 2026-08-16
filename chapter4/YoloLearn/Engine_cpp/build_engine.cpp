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
    std::string onnx_path = "best.onnx";
    std::string engine_path = "best.engine";
    bool use_fp16 = true;

    if (argc >= 2) onnx_path = argv[1];
    if (argc >= 3) engine_path = argv[2];
    if (argc >= 4) use_fp16 = (std::string(argv[3]) == "fp16");

    Logger logger;

    std::cout << "Building engine from: " << onnx_path << std::endl;
    std::cout << "Precision: " << (use_fp16 ? "FP16" : "FP32") << std::endl;

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
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 2ULL << 30);

    if (use_fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 enabled" << std::endl;
    }

    // Print input/output info and create optimization profile for dynamic shapes
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    for (int i = 0; i < network->getNbInputs(); i++) {
        auto* input = network->getInput(i);
        auto dims = input->getDimensions();
        std::cout << "Input[" << i << "]: " << input->getName() << " shape: [";
        for (int j = 0; j < dims.nbDims; j++) std::cout << dims.d[j] << (j < dims.nbDims-1 ? ", " : "");
        std::cout << "]" << std::endl;

        // If dynamic shape (-1 values), set optimization profile
        bool is_dynamic = false;
        for (int j = 0; j < dims.nbDims; j++) {
            if (dims.d[j] == -1) { is_dynamic = true; break; }
        }

        if (is_dynamic) {
            nvinfer1::Dims d_min = dims, d_opt = dims, d_max = dims;
            // batch=1, H/W: 320 to 640 (must be multiple of 32)
            d_min.d[0] = 1; d_opt.d[0] = 1; d_max.d[0] = 1;
            d_min.d[2] = 320; d_min.d[3] = 320;
            d_opt.d[2] = 640; d_opt.d[3] = 640;
            d_max.d[2] = 960; d_max.d[3] = 960;

            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, d_min);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, d_opt);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, d_max);
            std::cout << "  -> Set dynamic profile: min [1,3,320,320] opt [1,3,640,640] max [1,3,960,960]" << std::endl;
        }
    }
    config->addOptimizationProfile(profile);

    for (int i = 0; i < network->getNbOutputs(); i++) {
        auto* output = network->getOutput(i);
        auto dims = output->getDimensions();
        std::cout << "Output[" << i << "]: " << output->getName() << " shape: [";
        for (int j = 0; j < dims.nbDims; j++) std::cout << dims.d[j] << (j < dims.nbDims-1 ? ", " : "");
        std::cout << "]" << std::endl;
    }

    std::cout << "Building engine (this may take 1-5 minutes)..." << std::endl;
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


