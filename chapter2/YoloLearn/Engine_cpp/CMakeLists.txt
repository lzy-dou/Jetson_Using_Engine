cmake_minimum_required(VERSION 3.10)
project(trt_infer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES_DIR})
link_directories(/usr/local/cuda-12.6/targets/aarch64-linux/lib)

# TensorRT
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

add_executable(trt_infer trt_infer.cpp)

target_link_libraries(trt_infer
    ${OpenCV_LIBS}
    nvinfer
    nvonnxparser
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

add_executable(build_engine build_engine.cpp)

target_link_libraries(build_engine
    nvinfer
    nvonnxparser
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)


