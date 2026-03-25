# RK3588 YOLO 视频推理并使用FFMPEG推流

基于 RK3588 NPU 的多线程 YOLO 目标检测。

## 项目结构

```
rk3588_yolo/
├── CMakeLists.txt
├── include/
│   ├── utils.h           # 替代 kaylordut（日志/计时/run_once_with_delay）
│   ├── threadpool.h      # 线程池
│   ├── postprocess.h     # 后处理结构体定义
│   ├── videofile.h
│   ├── image_process.h
│   ├── yolov8.h
│   └── rknn_pool.h
└── src/
    ├── main.cpp           # 主程序入口
    ├── videofile.cpp
    ├── image_process.cpp
    ├── yolov8.cpp
    ├── rknn_pool.cpp
    └── postprocess.cpp    # 后处理
```

## 依赖

| 依赖                | 说明                                                             |
| ----------------- | -------------------------------------------------------------- |
| RKNN SDK (rknpu2) | 包含 `rknn_api.h`、`rknn_matmul_api.h`、`Float16.h`、`librknnrt.so` |
| OpenCV 4.x        | 图像处理与显示                                                        |
| LIBRGA            | rk3588专用加速图像处理                                                 |

## 编译（在 RK3588 板子上原生编译）

```bash
mkdir build && cd build
cmake ..
make -j4
```

## 交叉编译

```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/aarch64-toolchain.cmake \
  -DRKNN_SDK_PATH=/path/to/rknpu2/runtime/Linux/librknn_api/aarch64 \
  -DOpenCV_DIR=/path/to/opencv/aarch64/lib/cmake/opencv4
make -j$(nproc)
```

## 运行

```bash
#先进入miediamtx文件夹，运行客户端
./mediamtx
./rk3588_yolo_detect \
  -m ../model/yolov8n.rknn \
  -l ../model/coco_80_labels.txt \
  -i ../data/test.mp4 \
  -t 3 \
  -f 30 \
  --proto rtmp
```

| 参数             | 说明                        | 默认值                          |
|:-------------- |:-------------------------:|:---------------------------- |
| `-m`           | RKNN 模型路径                 | 必填                           |
| `-l`           | 标签文件路径                    | 必填                           |
| `-i`           | 输入视频路径                    | 必填                           |
| `-t`           | 推理线程数（建议 3，对应 3 个 NPU 核心） | 3                            |
| `-f`           | 播放帧率                      | 30                           |
| `-proto`       | 选择推流协议                    | rtsp                         |
| `-rtsp`        | rtsp地址                    | rtsp://127.0.0.1:8554/stream |
| `-rtmp`        | rtmp地址                    | rtmp://127.0.0.1:1935/stream |
| `--bitrate`    | 码率                        | 4M                           |
| `--no-display` | 不显示本地窗口                   |                              |

## 支持的模型类型

程序会自动识别模型类型：

- **YOLOv8 Detection** — 标准目标检测
- **YOLOv8 Segment** — 实例分割（输出 13 个 tensor）
- **YOLOv8 OBB** — 旋转框检测
- **YOLOv8 Pose** — 姿态估计
- **YOLOv10 Detection** — v10 检测

## 如需日志输出

`include/utils.h` 中的日志宏默认为空操作。如需实际输出，推荐引入 spdlog：

```cpp
// 在 utils.h 中替换为:
#include "spdlog/spdlog.h"
#define KAYLORDUT_LOG_INFO(...)  spdlog::info(__VA_ARGS__)
#define KAYLORDUT_LOG_ERROR(...) spdlog::error(__VA_ARGS__)
#define KAYLORDUT_LOG_DEBUG(...) spdlog::debug(__VA_ARGS__)
```
