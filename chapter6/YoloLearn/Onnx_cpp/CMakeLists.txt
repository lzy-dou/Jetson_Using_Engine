# ===========================================================================
# 深度估计 ONNX C++ 推理构建脚本
# 用法：
#   export ONNXRUNTIME_ROOTDIR=/path/to/onnxruntime-linux-x64-1.x.x
#   mkdir build && cd build
#   cmake .. && make -j
#   export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOTDIR/lib:$LD_LIBRARY_PATH
#   ./depth_onnx_inf
# ===========================================================================
cmake_minimum_required(VERSION 3.12)
project(depth_onnx_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)

# 默认路径已硬编码，直接 cmake .. 即可；如需覆盖可用 -DONNXRUNTIME_ROOTDIR=... 或环境变量
if(NOT DEFINED ONNXRUNTIME_ROOTDIR)
    if(DEFINED ENV{ONNXRUNTIME_ROOTDIR})
        set(ONNXRUNTIME_ROOTDIR $ENV{ONNXRUNTIME_ROOTDIR})
    else()
        set(ONNXRUNTIME_ROOTDIR "/home/nvidia/libs/onnxruntime-linux-aarch64-1.20.0")
    endif()
endif()
message(STATUS "ONNXRUNTIME_ROOTDIR = ${ONNXRUNTIME_ROOTDIR}")

set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOTDIR}/include")
set(ORT_LIB_DIR     "${ONNXRUNTIME_ROOTDIR}/lib")
find_library(ORT_LIB NAMES onnxruntime PATHS ${ORT_LIB_DIR} NO_DEFAULT_PATH)
if(NOT ORT_LIB)
    message(FATAL_ERROR "未在 ${ORT_LIB_DIR} 找到 libonnxruntime，请检查路径")
endif()

add_executable(depth_onnx_inf onnx_depth_inf.cpp)

target_include_directories(depth_onnx_inf PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_INCLUDE_DIR}
)

target_link_libraries(depth_onnx_inf PRIVATE
    ${OpenCV_LIBS}
    ${ORT_LIB}
)

set_target_properties(depth_onnx_inf PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR}"
    INSTALL_RPATH "${ORT_LIB_DIR}"
)