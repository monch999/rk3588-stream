# RK3588 YOLO 视频推理与多源推流

基于 RK3588 NPU 的多线程 YOLO 目标检测与多源视频推流系统。

## 项目结构

```
rk3588-stream/
├── CMakeLists.txt
├── include/
│   ├── utils.h                # 日志/计时/run_once_with_delay
│   ├── bounded_queue.h        # 线程安全有界队列
│   ├── postprocess.h          # 后处理结构体定义
│   ├── videofile.h            # 视频文件读取
│   ├── image_process.h        # RGA 硬件加速图像处理 (直接 resize，无 letterbox)
│   ├── yolov8.h               # RKNN 模型封装
│   ├── pipeline.h             # 单路 YOLO 推理流水线
│   ├── shared_clock.h         # 多通道统一时间基准 (monotonic clock)
│   ├── ffmpeg_streamer.h      # FFmpeg 管道推流器
│   ├── algorithm_interface.h  # 算法处理器接口 (YOLO/NDVI/火点检测)
│   └── channel_pipeline.h     # 单通道推流流水线
└── src/
    ├── main.cpp               # 单路 YOLO 推流入口
    ├── multi_stream_main.cpp  # 三路多源推流入口
    ├── pipeline.cpp           # 单路流水线实现
    ├── channel_pipeline.cpp   # 多源通道流水线实现
    ├── videofile.cpp
    ├── image_process.cpp
    ├── yolov8.cpp
    └── postprocess.cpp
```

## 依赖

| 依赖                | 说明                                                             |
| ----------------- | -------------------------------------------------------------- |
| RKNN SDK (rknpu2) | 包含 `rknn_api.h`、`rknn_matmul_api.h`、`Float16.h`、`librknnrt.so` |
| OpenCV 4.x        | 图像处理与显示                                                        |
| LIBRGA            | RK3588 专用 2D 硬件加速                                              |
| FFmpeg            | 视频编码与 RTSP/RTMP 推流（需支持 `h264_rkmpp` 硬件编码器）                     |

## 编译

### 在 RK3588 板子上原生编译

```bash
mkdir build && cd build
cmake ..
make -j4
```

编译产物：

- `rk3588_yolo_detect` — 单路 YOLO 推理推流
- `multi_stream` — 三路多源推流

### 交叉编译

```bash
mkdir build && cd build
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/aarch64-toolchain.cmake \
  -DRKNN_SDK_PATH=/path/to/rknpu2/runtime/Linux/librknn_api/aarch64 \
  -DOpenCV_DIR=/path/to/opencv/aarch64/lib/cmake/opencv4
make -j$(nproc)
```

---

## 一、单路 YOLO 推流 (`rk3588_yolo_detect`)

原有功能，单路视频 YOLO 推理 + RTSP/RTMP 推流。

### 运行

```bash
# 先启动 mediamtx RTSP 服务器
./mediamtx

./rk3588_yolo_detect \
  -m ../model/yolov8n.rknn \
  -l ../model/coco_80_labels_list.txt \
  -i ../data/test.mp4 \
  -t 3 -f 30 --proto rtsp
```

### 参数

| 参数             | 说明                        | 默认值                          |
|:-------------- |:-------------------------:|:---------------------------- |
| `-m`           | RKNN 模型路径                 | 必填                           |
| `-l`           | 标签文件路径                    | 必填                           |
| `-i`           | 输入视频路径                    | 必填                           |
| `-t`           | 推理线程数（建议 3，对应 3 个 NPU 核心） | 3                            |
| `-f`           | 播放帧率                      | 30                           |
| `-n`           | 每 N 帧推理一次                 | 3                            |
| `--proto`      | 推流协议 (rtsp/rtmp/both)     | rtsp                         |
| `--rtsp`       | RTSP 地址                   | rtsp://127.0.0.1:8554/stream |
| `--rtmp`       | RTMP 地址                   | rtmp://127.0.0.1:1935/stream |
| `--bitrate`    | 码率                        | 4M                           |
| `--no-display` | 不显示本地窗口                   |                              |
| `--loop`       | 视频循环播放                    |                              |

---

## 二、三路多源推流 (`multi_stream`)

三路独立视频通道（RGB、多光谱、热红外），每路独立采集、处理、推流。

### 架构

```
SharedClock (monotonic epoch) ─────────────────────────────────
     │                         │                         │
 ChannelPipeline            ChannelPipeline           ChannelPipeline
 [RGB]                      [多光谱]                  [热红外]
     │                         │                         │
 Reader(30fps)              Reader(60fps)             Reader(30fps)
   ├→ RawWriter               ├→ RawWriter               ├→ RawWriter
   │  → /rgb_raw              │  → /ms_raw               │  → /thermal_raw
   │                          │                          │
   └→ YoloProcessor(×3NPU)   └→ NdviProcessor(stub)     └→ FireDetector(stub)
      → /rgb_yolo (15fps)       → /ms_ndvi (30fps)         → /thermal_fire (30fps)
```

### 关键特性

- **PTS 时间对齐**：所有通道共享 `steady_clock` epoch，PTS = epoch + seq × (1e9/fps)，不做帧级同步
- **独立帧率**：RGB 原始 30fps / 推理 15fps，多光谱原始 60fps / 处理 30fps，热红外 30fps / 30fps
- **多核 NPU 推理**：RGB YOLO 使用 3 个 NPU 核心并行推理，ProcessWorker 多线程 + 重排序缓冲
- **推流模式控制**：`--stream-mode` 参数支持 `raw`（仅原始流）、`processed`（仅处理流）、`both`（全部推流）
- **无 Letterbox 预处理**：YOLO 输入直接 resize 至 640×384（W×H），使用独立 scale_w / scale_h 坐标映射
- **算法接口预留**：`NdviProcessor`（植被指数）和 `FireDetector`（火点检测）为 stub 实现，接口已定义

### 运行

```bash
# 先启动 mediamtx RTSP 服务器
./mediamtx

# 三路全推（原始 + 处理后，默认模式）
./multi_stream \
  --rgb-input ../data/rgb_4k.mp4 \
  --ms-input ../data/multispectral.mp4 \
  --thermal-input ../data/thermal_640x512.mp4 \
  -m ../model/yolov8n.rknn \
  -l ../model/coco_80_labels_list.txt \
  --npu-cores 3 \
  --loop

# 仅推原始流（不加载模型，不进行任何算法处理）
./multi_stream \
  --rgb-input ../data/rgb_4k.mp4 \
  --ms-input ../data/multispectral.mp4 \
  --thermal-input ../data/thermal_640x512.mp4 \
  --stream-mode raw \
  --loop

# 仅推处理后的流
./multi_stream \
  --rgb-input ../data/rgb_4k.mp4 \
  --ms-input ../data/multispectral.mp4 \
  --thermal-input ../data/thermal_640x512.mp4 \
  -m ../model/yolov8n.rknn \
  -l ../model/coco_80_labels_list.txt \
  --stream-mode processed \
  --loop
```

### 参数

| 参数                | 说明                   | 默认值                   |
|:----------------- |:--------------------:|:--------------------- |
| `--rgb-input`     | RGB 视频文件 (4K, 30fps) | 可选                    |
| `--ms-input`      | 多光谱视频文件 (60fps)      | 可选                    |
| `--thermal-input` | 热红外视频文件 (640×512)    | 可选                    |
| `-m, --model`     | RKNN 模型文件            | 推处理流时 RGB 通道必填        |
| `-l, --labels`    | 标签文件                 | 推处理流时 RGB 通道必填        |
| `--npu-cores`     | NPU 核心数              | 3                     |
| `--rtsp-base`     | RTSP 基地址             | rtsp://127.0.0.1:8554 |
| `--rtmp-base`     | RTMP 基地址             | rtmp://127.0.0.1:1935 |
| `--bitrate`       | 码率                   | 4M                    |
| `--stream-mode`   | 推流模式                 | both                  |
| `--loop`          | 视频循环播放               |                       |
| `--no-display`    | 不显示本地窗口              | (暂未添加)                |

### 推流模式 (`--stream-mode`)

| 模式          | 推原始流 | 推处理流 | 是否加载模型/算法 | 说明               |
|:----------- |:----:|:----:|:---------:|:---------------- |
| `raw`       | yes  | no   | no        | 仅转发原始视频，零处理开销    |
| `processed` | no   | yes  | yes       | 仅推算法处理后的流        |
| `both`      | yes  | yes  | yes       | 同时推原始流和处理后的流（默认） |

### 推流地址

| 流名称          | URL                                  | 帧率    | 内容        | 模式               |
|:------------ |:------------------------------------ |:----- |:--------- |:---------------- |
| RGB 原始流      | `rtsp://127.0.0.1:8554/rgb_raw`      | 30fps | 原始 4K 视频  | raw / both       |
| RGB YOLO 推理流 | `rtsp://127.0.0.1:8554/rgb_yolo`     | 15fps | YOLO 检测结果 | processed / both |
| 多光谱原始流       | `rtsp://127.0.0.1:8554/ms_raw`       | 60fps | 原始多光谱视频   | raw / both       |
| 多光谱植被指数流     | `rtsp://127.0.0.1:8554/ms_ndvi`      | 30fps | 植被指数结果    | processed / both |
| 热红外原始流       | `rtsp://127.0.0.1:8554/thermal_raw`  | 30fps | 原始热红外视频   | raw / both       |
| 热红外火点检测流     | `rtsp://127.0.0.1:8554/thermal_fire` | 30fps | 火点检测结果    | processed / both |

### 算法接口说明

三个处理器均实现 `IFrameProcessor` 接口：

```cpp
class IFrameProcessor {
public:
  virtual bool Init() = 0;
  virtual bool Process(int worker_id, const cv::Mat &input, cv::Mat &output) = 0;
  virtual std::string Name() const = 0;
  virtual int NumWorkers() const { return 1; }
};
```

- **YoloProcessor**：已实现，封装 Yolov8 + ImageProcess + 多 NPU 核心
- **NdviProcessor**：stub，预留 6 通道多光谱植被指数计算接口
- **FireDetector**：stub，预留热红外火点检测接口

---

## 图像预处理

YOLO 模型输入为 640(W) × 384(H)，**不使用 letterbox**，直接 resize：

```
原始帧 (如 3840×2160) ──RGA resize──→ 640×384 (BGR) ──RGA cvtcolor──→ 640×384 (RGB) ──→ RKNN NPU
```

后处理坐标映射使用独立的宽高缩放因子：

- `scale_w = 640 / src_width`
- `scale_h = 384 / src_height`
- 检测框坐标：`x_原始 = x_模型 / scale_w`，`y_原始 = y_模型 / scale_h`

---

## 支持的模型类型

程序自动识别模型类型：

- **YOLOv8 Detection** — 标准目标检测
- **YOLOv8 Segment** — 实例分割
- **YOLOv8 OBB** — 旋转框检测
- **YOLOv8 Pose** — 姿态估计
- **YOLOv10 Detection** — YOLOv10 检测

## 如需日志输出

`include/utils.h` 中的日志宏默认为空操作。如需实际输出，推荐引入 spdlog：

```cpp
// 在 utils.h 中替换为:
#include "spdlog/spdlog.h"
#define KAYLORDUT_LOG_INFO(...)  spdlog::info(__VA_ARGS__)
#define KAYLORDUT_LOG_ERROR(...) spdlog::error(__VA_ARGS__)
#define KAYLORDUT_LOG_DEBUG(...) spdlog::debug(__VA_ARGS__)
```
