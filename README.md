# RK3588 YOLO 视频推理与多源推流

基于 RK3588 NPU 的多线程 YOLO 目标检测与多源视频推流系统。
**全链路 Zero-Copy** 架构:Decode → RGA → NPU → CPU 画框 → Encode 全程共享 DMA buffer。

## 项目结构

```
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
│   ├── shared_clock.h         # 多通道统一时间基准 (monotonic clock)
│   ├── ffmpeg_streamer.h      # FFmpeg 管道推流器 (raw 模式)
│   ├── mpp_encoder.h          # MPP 硬件编码 (支持 DRM fd zero-copy 输入)
│   ├── rtsp_muxer.h           # RTSP/RTMP 输出复用器
│   ├── algorithm_interface.h  # 算法处理器接口 (in-place on DrmFrame)
│   └── channel_pipeline.h     # 单通道推流流水线 (zero-copy)
└── src/
    ├── multi_stream_main.cpp  # 三路多源推流入口
    ├── channel_pipeline.cpp
    ├── videofile.cpp
    ├── image_process.cpp
    ├── mpp_encoder.cpp
    ├── rtsp_muxer.cpp
    ├── yolov8.cpp
    └── postprocess.cpp
```

## 依赖

| 依赖                | 说明                                                             |
| ----------------- | -------------------------------------------------------------- |
| RKNN SDK (rknpu2) | 包含 `rknn_api.h`、`rknn_matmul_api.h`、`Float16.h`、`librknnrt.so` |
| OpenCV 4.x        | 图像处理与显示                                                        |
| LIBRGA            | RK3588 专用 2D 硬件加速 (≥ 1.10, 需 `improcess` API)                  |
| Rockchip MPP      | 硬件编解码,支持 DRM buffer import (`mpp_buffer_import`)                |
| FFmpeg            | 视频解封装与 RTSP/RTMP 推流                                           |

## 编译

### 在 RK3588 板子上原生编译

```bash
mkdir build && cd build
cmake ..
make -j4
```

编译产物:
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


## 一、三路多源推流 (`multi_stream`)

三路独立视频通道(RGB、多光谱、热红外),每路独立采集、处理、推流。

### 架构

```
SharedClock (monotonic epoch) ─────────────────────────────────
     │                         │                         │
 ChannelPipeline            ChannelPipeline           ChannelPipeline
 [RGB]                      [多光谱]                  [热红外]
     │                         │                         │
 Reader (MPP HW decode -> DrmFrame -> RGA copy -> 自管 NV12 DRM buffer)
   ├→ RawWriter               ├→ RawWriter               ├→ RawWriter
   │  → /rgb_raw              │  → /ms_raw               │  → /thermal_raw
   │                          │                          │
   └→ YoloProcessor(×3NPU)   └→ NdviProcessor(stub)     └→ FireDetector(stub)
      → /rgb_yolo               → /ms_ndvi                  → /thermal_fire
```

### Zero-Copy 数据流(processed 模式)

```
 MPP Decoder (NV12 fd, decoder pool)
        │
        │  [RGA imcopy fd→fd]   (decoder buffer 立即归还)
        ▼
 自管 NV12 DRM buffer (DrmAllocator 池)
        │
        │  [RGA improcess: NV12 → RGB + resize, fd→fd]
        ▼
 RGB DRM buffer (mmap 后取 vaddr)
        │
        │  [NPU rknn_inputs_set, CPU 指针, 务实零拷贝]
        ▼
 检测结果 (object_detect_result_list)
        │
        │  [CPU 在原 NV12 的 Y 平面画灰度框 + 黑边白字]
        │  [SyncEnd: cache flush]
        ▼
 MPP Encoder.EncodeFd(fd, h_stride, v_stride, pts)  ← 同一块 buffer
        │
        ▼
 RTSP/RTMP Muxer
```

整条管线只有一次 NV12 在自管 buffer 内的 RGA copy(为了让 decoder buffer 立即归还,避免下游卡顿导致 decoder 池死锁),
NPU 输入做一次硬件加速的 NV12→RGB+scale。**不再有 CPU 端的 BGR cvtColor / memcpy。**

### 关键特性

- **全链路 Zero-Copy**:`DrmFrame` 统一封装 DMA buffer,贯穿 Decode → RGA → NPU → 画框 → Encode
- **DRM Buffer 池**(`DrmAllocator` 单例):管理自管 NV12/RGB DMA buffer,与 decoder pool 解耦,避免下游卡顿引发 decoder 死锁
- **NV12 Y 平面直接绘制**:检测框/文字直接画在 Y 平面(白色边框 + 黑边白字),省去 NV12↔BGR 双向转换。文字基于 `cv::rectangle/putText` 在 Y 平面单通道 Mat 上操作,稳定可靠
- **PTS 双轨设计**:
  - `pts_ns`(steady_clock 纳秒)用于 `SharedClock::WaitUntilPts` 帧率节流和多通道时间对齐
  - 送 encoder/muxer 的 PTS = `seq`(帧序号),与 muxer time_base = `{1, fps}` 对齐
- **多核 NPU 推理**:RGB YOLO 使用 3 个 NPU 核心并行,Process worker 多线程
- **轻量 Reorder Buffer**:Writer 处用 `std::map` 做按 seq 排序(容量 = `worker × 2 + 2`),保证 encoder 输入 PTS 严格单调,避免 RTSP muxer 拒包
- **推流模式控制**:`--stream-mode` 参数支持 `raw`(仅原始流)、`processed`(仅推流处理后流)
- **算法接口预留**:`NdviProcessor`(植被指数)和 `FireDetector`(火点检测)为 stub 实现,接口已定义

### 运行

```bash
# 先启动 mediamtx RTSP 服务器
./mediamtx

# 三路全推 processed
./multi_stream \
  --rgb-input ../data/rgb_4k.mp4 \
  --ms-input ../data/multispectral.mp4 \
  --thermal-input ../data/thermal_640x512.mp4 \
  -m ../model/yolov8n.rknn \
  -l ../model/coco_80_labels_list.txt \
  --npu-cores 3 \
  --stream-mode processed \
  --loop

# 仅推原始流(不加载模型,不进行任何算法处理)
./multi_stream \
  --rgb-input ../data/rgb_4k.mp4 \
  --ms-input ../data/multispectral.mp4 \
  --thermal-input ../data/thermal_640x512.mp4 \
  --stream-mode raw \
  --loop
```

### 参数

| 参数                | 说明                    | 默认值                      |
|:------------------ |:---------------------:|:-------------------------- |
| `--rgb-input`      | RGB 视频文件             | 可选                        |
| `--ms-input`       | 多光谱视频文件             | 可选                        |
| `--thermal-input`  | 热红外视频文件             | 可选                        |
| `-m, --model`      | RKNN 模型文件             | 推处理流时 RGB 通道必填         |
| `-l, --labels`     | 标签文件                  | 推处理流时 RGB 通道必填         |
| `--npu-cores`      | NPU 核心数               | 3                          |
| `--rtsp-base`      | RTSP 基地址              | rtsp://127.0.0.1:8554      |
| `--bitrate`        | 码率                    | 4M                         |
| `--stream-mode`    | 推流模式 (raw/processed)  | raw                        |
| `--loop`           | 视频循环播放                |                            |
| `--display`        | 显示本地窗口(慢路径,转 BGR) |                            |

### 推流模式 (`--stream-mode`)

| 模式         | 推原始流 | 推处理流 | 是否加载模型/算法 | 说明                     |
|:----------- |:------:|:------:|:-------------:|:----------------------- |
| `raw`       | yes    | no     | no            | 仅转发原始视频,零处理开销    |
| `processed` | no     | yes    | yes           | 仅推算法处理后的流          |

### 推流地址

| 流名称            | URL                                   | 内容         | 模式        |
|:---------------- |:------------------------------------- |:----------- |:---------- |
| RGB 原始流        | `rtsp://127.0.0.1:8554/rgb_raw`      | 原始视频     | raw        |
| RGB YOLO 推理流   | `rtsp://127.0.0.1:8554/rgb_yolo`     | YOLO 检测结果 | processed  |
| 多光谱原始流       | `rtsp://127.0.0.1:8554/ms_raw`       | 原始多光谱视频 | raw        |
| 多光谱植被指数流    | `rtsp://127.0.0.1:8554/ms_ndvi`      | 植被指数结果   | processed  |
| 热红外原始流       | `rtsp://127.0.0.1:8554/thermal_raw`  | 原始热红外视频 | raw        |
| 热红外火点检测流    | `rtsp://127.0.0.1:8554/thermal_fire` | 火点检测结果   | processed  |

### 算法接口说明

所有处理器实现 `IFrameProcessor` 接口,**直接 in-place 修改 `DrmFrame` 内容**(无需返回新 buffer):

```cpp
class IFrameProcessor {
public:
  virtual bool Init() = 0;
  // frame 既是输入也是输出: 算法在 NV12 buffer 上 in-place 绘制结果
  // 返回 true: 该帧继续推流; false: 丢弃
  virtual bool Process(int worker_id,
                       const std::shared_ptr<DrmFrame>& frame) = 0;
  virtual std::string Name() const = 0;
  virtual int NumWorkers() const { return 1; }
};
```

- **YoloProcessor**:已实现。流程为 RGA(NV12→RGB+scale) → NPU 推理 → CPU 在原 NV12 Y 平面画框 → cache flush
- **NdviProcessor**:stub,预留多光谱植被指数计算接口
- **FireDetector**:stub,预留热红外火点检测接口

---

## 图像预处理

YOLO 模型输入为 640(W) × 384(H),**不使用 letterbox**,直接 resize:

```
DrmFrame (NV12, 任意分辨率) ──RGA improcess──→ DrmFrame (RGB, 640×384) ──→ RKNN NPU
```

后处理坐标映射使用独立的宽高缩放因子:
- `scale_w = 640 / src_width`
- `scale_h = 384 / src_height`
- 检测框坐标:`x_原始 = x_模型 / scale_w`,`y_原始 = y_模型 / scale_h`

---

## 检测框可视化

为最大化 zero-copy 收益,**直接在 NV12 的 Y 平面上绘制灰度框**,不做 NV12↔BGR 转换。

- **白色边框**:Y = 255,在彩色背景上显示为亮白色
- **文字**:黑色描边(Y = 0)+ 白色填充(Y = 255),通过 `cv::rectangle/putText` 在 Y 平面单通道 cv::Mat 上绘制
- **UV 平面不动**:背景颜色完全保留,只在边框/文字区域改亮度

边框粗细和字体大小按分辨率自适应(1080p → 2px 边框 / 字号 1.4,4K → 4px 边框 / 字号 2.8)。

---

## DRM Buffer 与 Cache 管理

`drm_allocator.h` 提供单例 `DrmAllocator`,基于 `mpp_buffer_group_get_internal(MPP_BUFFER_TYPE_DRM)` 申请 DMA buffer。

```cpp
auto buf = DrmAllocator::Instance().Acquire(DrmAllocator::NV12, 1920, 1080);
int fd = buf->fd;
void* vaddr = buf->vaddr;
buf->SyncEnd();    // CPU 写完, 设备读前 (encoder/RGA)
buf->SyncBegin();  // 设备写完, CPU 读前
// shared_ptr 析构时自动 mpp_buffer_put 归还到池子
```

`DrmFrame::FromAllocator` / `DrmFrame::FromMppFrame` 提供两类来源的统一封装。

退出前建议调用 `DrmAllocator::Instance().Shutdown()` 主动释放 buffer group。

---

## 支持的模型类型

程序自动识别模型类型:

- **YOLOv8 Detection** — 标准目标检测(当前 zero-copy 路径仅启用 Detection 的画框逻辑;OBB / Pose / Segmentation 暂未在 NV12 路径上实现)

## 如需日志输出

`include/utils.h` 中的日志宏默认为空操作。如需实际输出,推荐引入 spdlog:

```cpp
// 在 utils.h 中替换为:
#include "spdlog/spdlog.h"
#define KAYLORDUT_LOG_INFO(...)  spdlog::info(__VA_ARGS__)
#define KAYLORDUT_LOG_ERROR(...) spdlog::error(__VA_ARGS__)
#define KAYLORDUT_LOG_DEBUG(...) spdlog::debug(__VA_ARGS__)
```
