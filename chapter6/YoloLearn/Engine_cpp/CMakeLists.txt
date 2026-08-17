cmake_minimum_required(VERSION 3.10)
project(depth_trt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)

# CUDA - auto detect path
if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    if(EXISTS "/usr/local/cuda")
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
    elseif(EXISTS "/usr/local/cuda-12.6")
        set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6")
    else()
        file(GLOB CUDA_DIRS "/usr/local/cuda-*")
        if(CUDA_DIRS)
            list(GET CUDA_DIRS 0 CUDA_TOOLKIT_ROOT_DIR)
        endif()
    endif()
    message(STATUS "Auto-detected CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
endif()

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES_DIR})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib)

# TensorRT
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

# Build engine tool: ONNX -> Engine
add_executable(build_engine build_engine.cpp)
target_link_libraries(build_engine
    nvinfer
    nvonnxparser
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

# Inference executable
add_executable(trt_infer trt_infer.cpp)
target_link_libraries(trt_infer
    ${OpenCV_LIBS}
    nvinfer
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

