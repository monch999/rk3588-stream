# RK3588 YOLO 视频推理、多源推流与云台目标追踪

基于 RK3588 NPU 的多线程 YOLO 目标检测、ByteTrack 目标追踪与多源视频推流系统。

支持：

- **全链路 Zero-Copy**
  （Decode → RGA → NPU → CPU 画框 → Encode 全程共享 DMA buffer）
- 多路 RTSP/RTMP 推流
- ByteTrack 多目标追踪
- 云台 PTZ 控制
- 自动变焦
- 多源视频（RGB / 多光谱 / 热红外）
- RK3588 NPU 多核并行推理

---

# 项目结构

```text
rk3588-stream/
├── CMakeLists.txt
├── include/
│   ├── utils.h                # 日志/计时/run_once_with_delay
│   ├── bounded_queue.h        # 线程安全有界队列
│   ├── postprocess.h          # 后处理结构体定义
│   ├── videofile.h            # 视频文件读取 (MPP 硬解 -> DrmFrame)
│   ├── drm_allocator.h        # DRM buffer 池 (单例, MPP_BUFFER_TYPE_DRM)
│   ├── drm_frame.h            # 统一 DMA 帧封装 + Y 平面绘制工具
│   ├── image_process.h        # RGA 硬件加速 (NV12->RGB+scale, Y 平面画框)
│   ├── yolov8.h               # RKNN 模型封装
│   ├── shared_clock.h         # 多通道统一时间基准
│   ├── ffmpeg_streamer.h      # FFmpeg 管道推流器
│   ├── mpp_encoder.h          # MPP 硬件编码
│   ├── rtsp_muxer.h           # RTSP/RTMP 输出复用器
│   ├── algorithm_interface.h  # 算法接口
│   ├── channel_pipeline.h     # 单通道推流流水线
│   ├── lens_model.h           # 镜头模型与 FOV 计算
│   ├── zoom_controller.h      # 自动变焦控制
│   ├── gimbal_state.h         # 云台状态管理
│   ├── ByteTrack/             # ByteTrack 追踪算法
│   └── angle_sink.h           # 云台角度输出接口
│
└── src/
    ├── multi_stream_main.cpp
    ├── channel_pipeline.cpp
    ├── videofile.cpp
    ├── image_process.cpp
    ├── mpp_encoder.cpp
    ├── rtsp_muxer.cpp
    ├── yolov8.cpp
    └── postprocess.cpp
```

---

# 依赖

| 依赖                | 说明             |
| ----------------- | -------------- |
| RKNN SDK (rknpu2) | RK3588 NPU SDK |
| OpenCV 4.x        | 图像处理与显示        |
| LIBRGA            | RGA 硬件加速       |
| Rockchip MPP      | 硬件编解码          |
| FFmpeg            | RTSP/RTMP 推流   |
| ByteTrack         | 多目标追踪          |

---

# 编译

## RK3588 板端原生编译

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

---

# 系统架构

支持：

- RGB
- 多光谱
- 热红外

三路独立视频流。

---

# Zero-Copy 数据流

```text
Decode -> RGA -> NPU -> Draw -> Encode
```

统一基于 DMA Buffer：

- 不进行 CPU memcpy
- 不进行 CPU cvtColor
- 不进行 BGR 中转

---

# 核心特性

## 全链路 Zero-Copy

统一使用：

```cpp
DrmFrame
```

贯穿：

```text
Decode → RGA → NPU → Draw → Encode
```

---

## NV12 Y 平面直接画框

直接在：

```text
NV12 Y Plane
```

绘制：

- 白色框
- 黑边白字

无需 NV12 ↔ BGR 转换。

---

## 多核 NPU 推理

普通检测：

```text
YoloProcessor
```

支持：

- 多 NPU 核
- 多线程
- 无序处理
- Writer reorder

---

## ByteTrack 目标追踪

追踪模式：

```text
YoloTrackProcessor
```

特点：

- 单线程
- 单 NPU
- 严格时序
- ByteTrack 关联

---

## PTZ 云台控制

支持：

- yaw
- pitch
- 自动跟踪
- 自动变焦

---

# 运行

## 启动 RTSP 服务

```bash
./mediamtx
```

---

## processed 模式

```bash
./multi_stream \
  --rgb-input ../data/rgb_4k.mp4 \
  --ms-input ../data/multispectral.mp4 \
  --thermal-input ../data/thermal_640x512.mp4 \
  -m ../model/yolov8n.rknn \
  -l ../model/coco_80_labels_list.txt \
  --npu-cores 3 \
  --stream-mode processed \
  --loop
```

---

## raw 模式

```bash
./multi_stream \
  --rgb-input ../data/rgb_4k.mp4 \
  --ms-input ../data/multispectral.mp4 \
  --thermal-input ../data/thermal_640x512.mp4 \
  --stream-mode raw \
  --loop
```

---

## track 模式

```bash
./multi_stream \
  --rgb-input /dev/video0 \
  -m ../model/yolov8n.rknn \
  -l ../model/coco_80_labels_list.txt \
  --stream-mode track \
  --lens-config ../config/lens_params.json \
  --tracking-config ../config/tracking_params.json \
  --gimbal-state ../config/gimbal_init.json \
  --track-target 1 \
  --display
```

---

# 推流模式

| 模式        | 内容   | 算法                     |
| --------- | ---- | ---------------------- |
| raw       | 原始视频 | 无                      |
| processed | 检测结果 | YOLO                   |
| track     | 目标追踪 | YOLO + ByteTrack + PTZ |

---

# RTSP 地址

| URL             | 内容      |
| --------------- | ------- |
| `/rgb_raw`      | RGB 原始流 |
| `/rgb_yolo`     | RGB 检测流 |
| `/rgb_track`    | RGB 追踪流 |
| `/ms_raw`       | 多光谱原始流  |
| `/ms_ndvi`      | NDVI 结果 |
| `/thermal_raw`  | 热红外原始流  |
| `/thermal_fire` | 火点检测    |

---

# 算法接口

```cpp
class IFrameProcessor {
public:
  virtual bool Init() = 0;

  virtual bool Process(
      int worker_id,
      const std::shared_ptr<DrmFrame>& frame) = 0;

  virtual std::string Name() const = 0;

  virtual int NumWorkers() const {
      return 1;
  }
};
```

---

# 已实现算法

| 算法                 | 状态   |
| ------------------ | ---- |
| YoloProcessor      | 已实现  |
| YoloTrackProcessor | 已实现  |
| ByteTrack          | 已集成  |
| NdviProcessor      | Stub |
| FireDetector       | Stub |

---

# 模型支持

当前支持：

```text
YOLOv8 Detection
```

未来可扩展：

- OBB
- Pose
- Segmentation

---

# 日志输出

推荐：

```cpp
#include "spdlog/spdlog.h"

#define KAYLORDUT_LOG_INFO(...) \
    spdlog::info(__VA_ARGS__)

#define KAYLORDUT_LOG_ERROR(...) \
    spdlog::error(__VA_ARGS__)

#define KAYLORDUT_LOG_DEBUG(...) \
    spdlog::debug(__VA_ARGS__)
```
