cmake_minimum_required(VERSION 3.10)
project(rk3588_yolo_detect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==================== 第三方库根路径 ====================
set(3RDPARTY_DIR ${CMAKE_SOURCE_DIR}/3rdparty)

# ==================== RKNN ====================
set(RKNN_DIR ${3RDPARTY_DIR}/rknpu2)
set(RKNN_INCLUDE_DIR ${RKNN_DIR}/include)
file(GLOB RKNN_LIB
    ${RKNN_DIR}/lib/librknnrt.so
    ${RKNN_DIR}/lib64/librknnrt.so
    ${RKNN_DIR}/lib/librknn_api.so
)

# ==================== librga (RK3588 2D 硬件加速) ====================
set(RGA_DIR ${3RDPARTY_DIR}/librga CACHE PATH "librga root dir")

find_path(RGA_INCLUDE_DIR im2d.hpp
    HINTS ${RGA_DIR}/include ${RGA_DIR}/include/im2d_api
    PATHS /usr/include /usr/local/include
)

find_library(RGA_LIB rga
    HINTS ${RGA_DIR}/lib ${RGA_DIR}/lib64
    PATHS /usr/lib /usr/local/lib /usr/lib/aarch64-linux-gnu
)

if(NOT RGA_INCLUDE_DIR OR NOT RGA_LIB)
  message(FATAL_ERROR
    "librga not found!\n"
    "  Searched include: ${RGA_DIR}/include, /usr/include\n"
    "  Searched lib:     ${RGA_DIR}/lib, /usr/lib\n"
    "  Please set -DRGA_DIR=/path/to/librga or install librga to system"
  )
endif()

# ==================== OpenCV ====================
set(OpenCV_DIR ${3RDPARTY_DIR}/opencv/share/opencv4
    CACHE PATH "OpenCV cmake config dir")
find_package(OpenCV QUIET)

if(NOT OpenCV_FOUND)
  message(STATUS "OpenCV find_package failed, using manual paths")
  set(OpenCV_INCLUDE_DIRS ${3RDPARTY_DIR}/opencv/include)
  file(GLOB OpenCV_LIBS
      ${3RDPARTY_DIR}/opencv/lib/libopencv_core.so
      ${3RDPARTY_DIR}/opencv/lib/libopencv_imgproc.so
      ${3RDPARTY_DIR}/opencv/lib/libopencv_imgcodecs.so
      ${3RDPARTY_DIR}/opencv/lib/libopencv_highgui.so
      ${3RDPARTY_DIR}/opencv/lib/libopencv_videoio.so
  )
endif()

# ==================== 源文件 ====================
# 新流水线架构: pipeline.cpp 替换 rknn_pool.cpp
# 不再需要 threadpool.h (用 bounded_queue.h 替代)
set(SOURCES
    src/main.cpp
    src/pipeline.cpp
    src/videofile.cpp
    src/image_process.cpp
    src/yolov8.cpp
    src/postprocess.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${RKNN_INCLUDE_DIR}
    ${RGA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${RKNN_LIB}
    ${RGA_LIB}
    ${OpenCV_LIBS}
    pthread
)

# ==================== 多源推流可执行文件 ====================
set(MULTI_STREAM_SOURCES
    src/multi_stream_main.cpp
    src/channel_pipeline.cpp
    src/videofile.cpp
    src/image_process.cpp
    src/yolov8.cpp
    src/postprocess.cpp
)

add_executable(multi_stream ${MULTI_STREAM_SOURCES})

target_include_directories(multi_stream PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${RKNN_INCLUDE_DIR}
    ${RGA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(multi_stream PRIVATE
    ${RKNN_LIB}
    ${RGA_LIB}
    ${OpenCV_LIBS}
    pthread
)

# ==================== 打印配置 ====================
message(STATUS "RKNN include: ${RKNN_INCLUDE_DIR}")
message(STATUS "RKNN lib:     ${RKNN_LIB}")
message(STATUS "RGA include:  ${RGA_INCLUDE_DIR}")
message(STATUS "RGA lib:      ${RGA_LIB}")
message(STATUS "OpenCV inc:   ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs:  ${OpenCV_LIBS}")
