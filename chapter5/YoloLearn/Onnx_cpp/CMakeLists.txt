cmake_minimum_required(VERSION 3.14)
project(infer_onnx_seg)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 通过 -DONNXRUNTIME_ROOT=/path/to/onnxruntime 指定
# 该目录下应包含 include/onnxruntime_cxx_api.h 和 lib/libonnxruntime.so
if(NOT DEFINED ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime")
endif()

find_package(OpenCV REQUIRED)

add_executable(infer_onnx_seg infer_onnx_seg.cpp)

target_include_directories(infer_onnx_seg PRIVATE
    ${ONNXRUNTIME_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(infer_onnx_seg PRIVATE
    ${ONNXRUNTIME_ROOT}/lib
)

target_link_libraries(infer_onnx_seg PRIVATE
    onnxruntime
    ${OpenCV_LIBS}
)

# 运行时需要能找到 libonnxruntime.so，可用以下方式之一：
#   export LD_LIBRARY_PATH=${ONNXRUNTIME_ROOT}/lib:$LD_LIBRARY_PATH
#   或在本CMakeLists中设置 RPATH（跨机器部署时不推荐硬编码路径）
