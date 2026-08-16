# ===========================================================================
# YOLO11 ONNX C++ 推理构建脚本
# 用法：
#   1) 安装 OpenCV (apt install libopencv-dev 或自行编译)
#   2) 下载 ONNX Runtime C/C++ 压缩包并解压，设置环境变量指向其根目录：
#        export ONNXRUNTIME_ROOTDIR=/path/to/onnxruntime-linux-x64-1.x.x
#   3) 构建：
#        mkdir build && cd build
#        cmake .. && make -j
#   4) 运行前确保 libonnxruntime.so 在动态库搜索路径：
#        export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOTDIR/lib:$LD_LIBRARY_PATH
#        ./yolo_onnx_inf
# ===========================================================================
cmake_minimum_required(VERSION 3.12)
project(yolo_onnx_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# ---------- OpenCV ----------
find_package(OpenCV REQUIRED)

# ---------- ONNX Runtime ----------
if(NOT DEFINED ONNXRUNTIME_ROOTDIR AND DEFINED ENV{ONNXRUNTIME_ROOTDIR})
    set(ONNXRUNTIME_ROOTDIR $ENV{ONNXRUNTIME_ROOTDIR})
endif()
if(NOT ONNXRUNTIME_ROOTDIR)
    message(FATAL_ERROR "请设置 ONNXRUNTIME_ROOTDIR 指向 onnxruntime 解压根目录 "
                        "(含 include/ 与 lib/)，例如: "
                        "cmake .. -DONNXRUNTIME_ROOTDIR=/path/to/onnxruntime")
endif()

set(ORT_INCLUDE_DIR "${ONNXRUNTIME_ROOTDIR}/include")
set(ORT_LIB_DIR     "${ONNXRUNTIME_ROOTDIR}/lib")
find_library(ORT_LIB NAMES onnxruntime PATHS ${ORT_LIB_DIR} NO_DEFAULT_PATH)
if(NOT ORT_LIB)
    message(FATAL_ERROR "未在 ${ORT_LIB_DIR} 找到 libonnxruntime，请检查路径")
endif()

add_executable(yolo_onnx_inf onnx_inf.cpp)

target_include_directories(yolo_onnx_inf PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_INCLUDE_DIR}
)

target_link_libraries(yolo_onnx_inf PRIVATE
    ${OpenCV_LIBS}
    ${ORT_LIB}
)

# 运行时自动把 ORT 的 lib 目录加入 rpath，免手动设 LD_LIBRARY_PATH
set_target_properties(yolo_onnx_inf PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR}"
    INSTALL_RPATH "${ORT_LIB_DIR}"
)
