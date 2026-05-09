#pragma once
// V4l2Source: 通过 V4L2 M-Plane API 从 HDMI-RX 等设备采集 NV12 帧
// - 启动时 VIDIOC_QUERY_DV_TIMINGS 读取实际分辨率/帧率
// - VIDIOC_S_FMT 协商 NV12
// - mmap + VIDIOC_EXPBUF 拿 DMABUF fd (零拷贝)
// - 订阅 V4L2_EVENT_SOURCE_CHANGE 处理热插拔/分辨率变化
//
// 注意: 输出的 DrmFrame 引用 V4L2 内部缓冲, 析构时会自动 QBUF 归还
//       (V4L2 buffer 数量有限, 下游应尽快消耗或 RGA copy 到自管 buffer)
#include "video_source.h"
#include <atomic>
#include <linux/videodev2.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class V4l2Source : public IVideoSource {
public:
  explicit V4l2Source(const std::string& dev_path, int buffer_count = 6);
  ~V4l2Source() override;

  V4l2Source(const V4l2Source&) = delete;
  V4l2Source& operator=(const V4l2Source&) = delete;

  std::shared_ptr<DrmFrame> GetNextDrmFrame() override;
  int    get_frame_width()  const override { return width_; }
  int    get_frame_height() const override { return height_; }
  double get_fps()          const override { return fps_; }

private:
  // 完整初始化 (open + 协商 + REQBUFS + STREAMON)
  bool Open();
  // 仅协商格式 + buffer 准备 + STREAMON (假设 fd 已开)
  bool NegotiateAndStart();
  // 仅停止 + 释放 buffer (不关 fd, 用于热插拔重新协商)
  void StopAndReleaseBuffers();

  bool QueryDvTimings();           // 读 HDMI 实际分辨率/帧率
  bool SetFormat();                // VIDIOC_S_FMT NV12 M-Plane
  bool RequestBuffers();           // REQBUFS + mmap + EXPBUF + 全部 QBUF
  bool SubscribeSourceChange();    // 订阅 SOURCE_CHANGE 事件

  // 处理 epoll 事件中的 V4L2_EVENT_SOURCE_CHANGE -> 重协商
  bool HandleSourceChange();

  // 把 V4L2 buffer 包成 DrmFrame, 析构时 QBUF
  std::shared_ptr<DrmFrame> WrapBuffer(int buf_index, int64_t pts_us);
  void ReturnBuffer(int buf_index);

  struct V4l2Buffer {
    int    dmabuf_fd = -1;     // VIDIOC_EXPBUF 返回, 由本类持有
    void*  mmap_addr = nullptr;
    size_t mmap_size = 0;
    bool   queued    = false;
  };

  std::string dev_path_;
  int         buffer_count_ = 6;
  int         fd_ = -1;

  int    width_  = 0;
  int    height_ = 0;
  int    h_stride_ = 0;
  int    v_stride_ = 0;
  double fps_    = 25.0;
  uint32_t pixel_format_ = V4L2_PIX_FMT_NV12;  // 实际协商到的格式

  std::vector<V4l2Buffer> buffers_;
  std::mutex              return_mtx_;     // 保护 ReturnBuffer (DrmFrame 析构在任意线程)

  std::atomic<bool>       streaming_{false};
  std::atomic<bool>       resolution_changed_{false};
};
