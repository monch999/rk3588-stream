#include "v4l2_source.h"
#include "drm_allocator.h"
#include "im2d.hpp"
#include "RgaUtils.h"
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

// 工具: 对所有 EINTR 自动重试的 ioctl
static int xioctl(int fd, unsigned long req, void* arg) {
  int r;
  do { r = ::ioctl(fd, req, arg); }
  while (r == -1 && errno == EINTR);
  return r;
}

// ==================== 构造 / 析构 ====================
V4l2Source::V4l2Source(const std::string& dev_path, int buffer_count)
    : dev_path_(dev_path), buffer_count_(buffer_count) {
  if (!Open()) {
    fprintf(stderr, "[V4L2 ] open/init failed: %s\n", dev_path.c_str());
  }
}

V4l2Source::~V4l2Source() {
  StopAndReleaseBuffers();
  if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

// ==================== Open: 完整初始化 ====================
bool V4l2Source::Open() {
  fd_ = ::open(dev_path_.c_str(), O_RDWR | O_NONBLOCK);
  if (fd_ < 0) {
    fprintf(stderr, "[V4L2 ] open(%s) failed: %s\n",
            dev_path_.c_str(), strerror(errno));
    return false;
  }

  // 验证 capability: 必须是 M-Plane Capture
  v4l2_capability cap{};
  if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
    fprintf(stderr, "[V4L2 ] VIDIOC_QUERYCAP failed: %s\n", strerror(errno));
    return false;
  }
  if (!(cap.device_caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE)
      || !(cap.device_caps & V4L2_CAP_STREAMING)) {
    fprintf(stderr, "[V4L2 ] device caps insufficient: 0x%x\n", cap.device_caps);
    return false;
  }
  printf("[V4L2 ] device: %s (driver=%s)\n", cap.card, cap.driver);

  if (!SubscribeSourceChange())  // 订阅热插拔事件
    fprintf(stderr, "[V4L2 ] (warn) subscribe SOURCE_CHANGE failed\n");

  return NegotiateAndStart();
}

// ==================== 协商 + STREAMON ====================
bool V4l2Source::NegotiateAndStart() {
  if (!QueryDvTimings()) {
    // DV Timings 不可用 (USB 摄像头等), 用 G_FMT 读取当前格式
    printf("[V4L2 ] DV timings unavailable, querying current format...\n");
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (xioctl(fd_, VIDIOC_G_FMT, &fmt) == 0) {
      if (fmt.fmt.pix_mp.width > 0 && fmt.fmt.pix_mp.height > 0) {
        width_  = fmt.fmt.pix_mp.width;
        height_ = fmt.fmt.pix_mp.height;
        printf("[V4L2 ] G_FMT: %dx%d\n", width_, height_);
      }
    }
    if (width_ <= 0 || height_ <= 0) {
      fprintf(stderr, "[V4L2 ] cannot determine resolution\n");
      return false;
    }
  }
  if (!SetFormat())      return false;
  if (!RequestBuffers()) return false;

  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
    fprintf(stderr, "[V4L2 ] STREAMON failed: %s\n", strerror(errno));
    return false;
  }
  streaming_ = true;
  printf("[V4L2 ] streaming ON: %dx%d@%.2f stride=%d v=%d, %d buffers\n",
         width_, height_, fps_, h_stride_, v_stride_, (int)buffers_.size());
  return true;
}

void V4l2Source::StopAndReleaseBuffers() {
  if (streaming_.exchange(false)) {
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    xioctl(fd_, VIDIOC_STREAMOFF, &type);
  }

  std::lock_guard<std::mutex> lk(return_mtx_);
  for (auto& b : buffers_) {
    if (b.mmap_addr && b.mmap_size > 0) ::munmap(b.mmap_addr, b.mmap_size);
    if (b.dmabuf_fd >= 0)               ::close(b.dmabuf_fd);
    b = V4l2Buffer{};
  }
  buffers_.clear();

  // REQBUFS(0) 释放驱动里的 buffer
  if (fd_ >= 0) {
    v4l2_requestbuffers req{};
    req.count  = 0;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_MMAP;
    xioctl(fd_, VIDIOC_REQBUFS, &req);
  }
}

// ==================== DV Timings: 读 HDMI 实际分辨率/帧率 ====================
bool V4l2Source::QueryDvTimings() {
  v4l2_dv_timings t{};
  if (xioctl(fd_, VIDIOC_QUERY_DV_TIMINGS, &t) < 0) {
    fprintf(stderr, "[V4L2 ] QUERY_DV_TIMINGS failed: %s "
                    "(HDMI 信号未就绪? 检查线缆与对端)\n", strerror(errno));
    return false;
  }

  // RK HDMI-RX 驱动要求 type 字段必须正确
  t.type = V4L2_DV_BT_656_1120;

  if (xioctl(fd_, VIDIOC_S_DV_TIMINGS, &t) < 0) {
    fprintf(stderr, "[V4L2 ] S_DV_TIMINGS failed: %s (尝试不设 timings 继续)\n",
            strerror(errno));
    // 虽然 S_DV_TIMINGS 失败, 但 QUERY 成功了, 至少拿到分辨率和帧率
    // 某些 RK 驱动版本中 timings 已经在 QUERY 时隐式生效
  }

  width_  = t.bt.width;
  height_ = t.bt.height;

  // fps 计算: pixelclock / (htotal * vtotal)
  uint64_t htotal = (uint64_t)t.bt.width  + t.bt.hfrontporch + t.bt.hsync + t.bt.hbackporch;
  uint64_t vtotal = (uint64_t)t.bt.height + t.bt.vfrontporch + t.bt.vsync + t.bt.vbackporch;
  if (t.bt.pixelclock > 0 && htotal > 0 && vtotal > 0) {
    fps_ = (double)t.bt.pixelclock / (double)(htotal * vtotal);
  }
  printf("[V4L2 ] DV timings: %dx%d @ %.2f fps (pclk=%llu)\n",
         width_, height_, fps_, (unsigned long long)t.bt.pixelclock);
  return true;
}

// ==================== Set Format ====================
// 策略: 优先尝试设置 NV12 (最适合后续 pipeline)
//        全部失败时, 用 G_FMT 接受驱动当前格式 (如 BGR3)
//        非 NV12 格式在 WrapBuffer 中通过 RGA 转换
bool V4l2Source::SetFormat() {
  // 按优先级尝试
  struct FmtCandidate {
    uint32_t fourcc;
    const char* name;
  };
  FmtCandidate candidates[] = {
    {V4L2_PIX_FMT_NV12, "NV12"},
    {V4L2_PIX_FMT_NV16, "NV16"},
    {V4L2_PIX_FMT_NV24, "NV24"},
    {V4L2_PIX_FMT_BGR24, "BGR3"},
    {V4L2_PIX_FMT_UYVY, "UYVY"},
  };

  v4l2_format fmt{};
  bool found = false;

  for (auto& cand : candidates) {
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width       = width_;
    fmt.fmt.pix_mp.height      = height_;
    fmt.fmt.pix_mp.pixelformat = cand.fourcc;
    fmt.fmt.pix_mp.field       = V4L2_FIELD_NONE;
    fmt.fmt.pix_mp.num_planes  = 1;

    int ret = xioctl(fd_, VIDIOC_S_FMT, &fmt);
    if (ret < 0) {
      // 静默跳过, 不刷屏
      continue;
    }
    if (fmt.fmt.pix_mp.pixelformat != cand.fourcc) {
      continue;
    }
    printf("[V4L2 ] S_FMT %s OK\n", cand.name);
    found = true;
    break;
  }

  // 所有 S_FMT 失败 → 用 G_FMT 接受驱动当前格式
  if (!found) {
    printf("[V4L2 ] S_FMT all failed, falling back to G_FMT (accept current format)\n");
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (xioctl(fd_, VIDIOC_G_FMT, &fmt) < 0) {
      fprintf(stderr, "[V4L2 ] G_FMT also failed: %s\n", strerror(errno));
      return false;
    }
    // 接受驱动给的任何格式
    found = true;
  }

  pixel_format_ = fmt.fmt.pix_mp.pixelformat;
  width_    = fmt.fmt.pix_mp.width;
  height_   = fmt.fmt.pix_mp.height;
  h_stride_ = fmt.fmt.pix_mp.plane_fmt[0].bytesperline;

  // 计算 v_stride
  int sizeimage = fmt.fmt.pix_mp.plane_fmt[0].sizeimage;
  if (pixel_format_ == V4L2_PIX_FMT_NV12 && h_stride_ > 0) {
    v_stride_ = (sizeimage * 2) / (h_stride_ * 3);
  } else if (pixel_format_ == V4L2_PIX_FMT_NV16 && h_stride_ > 0) {
    v_stride_ = sizeimage / (h_stride_ * 2);
  } else if (pixel_format_ == V4L2_PIX_FMT_NV24 && h_stride_ > 0) {
    v_stride_ = sizeimage / (h_stride_ * 3);
  } else if ((pixel_format_ == V4L2_PIX_FMT_BGR24 ||
              pixel_format_ == V4L2_PIX_FMT_RGB24) && h_stride_ > 0) {
    // BGR3/RGB3: bytesperline = width * 3, sizeimage = bpl * height
    v_stride_ = sizeimage / h_stride_;
  } else if (pixel_format_ == V4L2_PIX_FMT_UYVY && h_stride_ > 0) {
    v_stride_ = sizeimage / h_stride_;
  } else {
    v_stride_ = height_;
  }
  if (v_stride_ < height_) v_stride_ = height_;

  // 格式名称
  char fourcc_str[5] = {};
  fourcc_str[0] = (pixel_format_)       & 0xFF;
  fourcc_str[1] = (pixel_format_ >> 8)  & 0xFF;
  fourcc_str[2] = (pixel_format_ >> 16) & 0xFF;
  fourcc_str[3] = (pixel_format_ >> 24) & 0xFF;

  printf("[V4L2 ] format: %s (0x%x) %dx%d h_stride=%d v_stride=%d sizeimage=%d\n",
         fourcc_str, pixel_format_, width_, height_, h_stride_, v_stride_, sizeimage);
  return true;
}

// ==================== Request Buffers: mmap + EXPBUF + QBUF ====================
bool V4l2Source::RequestBuffers() {
  v4l2_requestbuffers req{};
  req.count  = buffer_count_;
  req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  req.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) {
    fprintf(stderr, "[V4L2 ] REQBUFS failed or too few: count=%d\n", req.count);
    return false;
  }
  buffers_.assign(req.count, V4l2Buffer{});

  for (size_t i = 0; i < buffers_.size(); i++) {
    v4l2_plane planes[VIDEO_MAX_PLANES]{};
    v4l2_buffer buf{};
    buf.type     = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory   = V4L2_MEMORY_MMAP;
    buf.index    = i;
    buf.m.planes = planes;
    buf.length   = 1;
    if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
      fprintf(stderr, "[V4L2 ] QUERYBUF[%zu] failed: %s\n", i, strerror(errno));
      return false;
    }
    void* p = ::mmap(nullptr, planes[0].length, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd_, planes[0].m.mem_offset);
    if (p == MAP_FAILED) {
      fprintf(stderr, "[V4L2 ] mmap[%zu] failed: %s\n", i, strerror(errno));
      return false;
    }
    buffers_[i].mmap_addr = p;
    buffers_[i].mmap_size = planes[0].length;

    // 导出 DMABUF fd (供 RGA / MPP 零拷贝使用)
    v4l2_exportbuffer eb{};
    eb.type  = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    eb.index = i;
    eb.plane = 0;
    eb.flags = O_RDWR;
    if (xioctl(fd_, VIDIOC_EXPBUF, &eb) < 0) {
      fprintf(stderr, "[V4L2 ] EXPBUF[%zu] failed: %s\n", i, strerror(errno));
      return false;
    }
    buffers_[i].dmabuf_fd = eb.fd;

    // 入队
    v4l2_plane qplanes[VIDEO_MAX_PLANES]{};
    v4l2_buffer qbuf{};
    qbuf.type     = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    qbuf.memory   = V4L2_MEMORY_MMAP;
    qbuf.index    = i;
    qbuf.m.planes = qplanes;
    qbuf.length   = 1;
    if (xioctl(fd_, VIDIOC_QBUF, &qbuf) < 0) {
      fprintf(stderr, "[V4L2 ] QBUF[%zu] failed: %s\n", i, strerror(errno));
      return false;
    }
    buffers_[i].queued = true;
  }
  return true;
}

// ==================== SubscribeSourceChange ====================
bool V4l2Source::SubscribeSourceChange() {
  v4l2_event_subscription sub{};
  sub.type = V4L2_EVENT_SOURCE_CHANGE;
  return xioctl(fd_, VIDIOC_SUBSCRIBE_EVENT, &sub) == 0;
}

bool V4l2Source::HandleSourceChange() {
  // 拉空所有事件
  while (true) {
    v4l2_event ev{};
    if (xioctl(fd_, VIDIOC_DQEVENT, &ev) < 0) break;
    if (ev.type == V4L2_EVENT_SOURCE_CHANGE) {
      printf("[V4L2 ] SOURCE_CHANGE event\n");
      resolution_changed_ = true;
    }
  }
  if (!resolution_changed_.exchange(false)) return true;

  printf("[V4L2 ] re-negotiating after source change...\n");
  StopAndReleaseBuffers();

  // HDMI 信号可能还没稳定, 重试
  for (int retry = 0; retry < 50; retry++) {
    if (NegotiateAndStart()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  fprintf(stderr, "[V4L2 ] re-negotiation gave up after 5s\n");
  return false;
}

// ==================== 取一帧 ====================
std::shared_ptr<DrmFrame> V4l2Source::GetNextDrmFrame() {
  if (fd_ < 0) return nullptr;

  // 等待 V4L2 就绪 (含 EXCEPT 事件: SOURCE_CHANGE)
  pollfd pfd{};
  pfd.fd = fd_;
  pfd.events = POLLIN | POLLPRI;
  int pr = ::poll(&pfd, 1, 1000);   // 1s 超时
  if (pr <= 0) {
    if (pr == 0) fprintf(stderr, "[V4L2 ] poll timeout (无信号?)\n");
    else         fprintf(stderr, "[V4L2 ] poll error: %s\n", strerror(errno));
    return nullptr;
  }

  // 优先处理事件 (热插拔)
  if (pfd.revents & POLLPRI) {
    if (!HandleSourceChange()) return nullptr;
    // 重新协商后此次返回 nullptr, 让上层下次再来
    return nullptr;
  }

  if (!streaming_) return nullptr;

  // DQBUF
  v4l2_plane planes[VIDEO_MAX_PLANES]{};
  v4l2_buffer buf{};
  buf.type     = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  buf.memory   = V4L2_MEMORY_MMAP;
  buf.m.planes = planes;
  buf.length   = 1;
  if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
    if (errno != EAGAIN)
      fprintf(stderr, "[V4L2 ] DQBUF failed: %s\n", strerror(errno));
    return nullptr;
  }
  if (buf.index >= buffers_.size()) {
    fprintf(stderr, "[V4L2 ] DQBUF bad index %u\n", buf.index);
    return nullptr;
  }
  buffers_[buf.index].queued = false;

  int64_t pts_us = (int64_t)buf.timestamp.tv_sec * 1000000 + buf.timestamp.tv_usec;
  return WrapBuffer(buf.index, pts_us);
}

// ==================== Wrap V4L2 Buffer -> DrmFrame ====================
// NV12: 零拷贝直接包装
// NV16/NV24/UYVY: RGA 转 NV12 到自管 buffer, 立即归还 V4L2 buffer
std::shared_ptr<DrmFrame> V4l2Source::WrapBuffer(int idx, int64_t pts_us) {
  auto* self = this;

  if (pixel_format_ != V4L2_PIX_FMT_NV12) {
    // 非 NV12 → RGA 转换到自管 NV12 DRM buffer
    auto nv12_buf = DrmAllocator::Instance().Acquire(
        DrmAllocator::NV12, width_, height_);
    if (!nv12_buf) {
      fprintf(stderr, "[V4L2 ] alloc NV12 convert buffer failed\n");
      ReturnBuffer(idx);
      return nullptr;
    }

    // 确定源 RGA 格式
    uint32_t src_rga_fmt = 0;
    if      (pixel_format_ == V4L2_PIX_FMT_UYVY)  src_rga_fmt = RK_FORMAT_UYVY_422;
    else if (pixel_format_ == V4L2_PIX_FMT_NV16)  src_rga_fmt = RK_FORMAT_YCbCr_422_SP;
    else if (pixel_format_ == V4L2_PIX_FMT_NV24)  src_rga_fmt = RK_FORMAT_YCbCr_444_SP;
    else if (pixel_format_ == V4L2_PIX_FMT_BGR24) src_rga_fmt = RK_FORMAT_BGR_888;
    else if (pixel_format_ == V4L2_PIX_FMT_RGB24) src_rga_fmt = RK_FORMAT_RGB_888;
    else {
      fprintf(stderr, "[V4L2 ] unsupported pixel format 0x%x for RGA convert\n", pixel_format_);
      ReturnBuffer(idx);
      return nullptr;
    }

    // 源: V4L2 mmap buffer → importbuffer_virtualaddr 获取 handle
    // (RGA 要求 src/dst 统一使用 handle 或统一不用)
    im_handle_param_t src_param = {(uint32_t)width_,
                                    (uint32_t)v_stride_,
                                    src_rga_fmt};
    rga_buffer_handle_t src_h = importbuffer_virtualaddr(
        buffers_[idx].mmap_addr, &src_param);
    if (src_h == 0) {
      fprintf(stderr, "[V4L2 ] importbuffer_virtualaddr for src failed\n");
      ReturnBuffer(idx);
      return nullptr;
    }

    rga_buffer_t src_rga = wrapbuffer_handle(src_h,
        width_, height_, src_rga_fmt);
    src_rga.wstride = width_;
    src_rga.hstride = v_stride_;

    // 目标: 自管 NV12 DRM buffer (通过 fd → handle)
    im_handle_param_t dst_param = {(uint32_t)nv12_buf->h_stride,
                                    (uint32_t)nv12_buf->v_stride,
                                    RK_FORMAT_YCbCr_420_SP};
    rga_buffer_handle_t dst_h = importbuffer_fd(nv12_buf->fd, &dst_param);
    if (dst_h == 0) {
      fprintf(stderr, "[V4L2 ] importbuffer_fd for dst failed\n");
      releasebuffer_handle(src_h);
      ReturnBuffer(idx);
      return nullptr;
    }

    rga_buffer_t dst_rga = wrapbuffer_handle(dst_h,
        width_, height_, RK_FORMAT_YCbCr_420_SP);
    dst_rga.wstride = nv12_buf->h_stride;
    dst_rga.hstride = nv12_buf->v_stride;

    IM_STATUS st = imcvtcolor(src_rga, dst_rga, src_rga_fmt, RK_FORMAT_YCbCr_420_SP);
    releasebuffer_handle(src_h);
    releasebuffer_handle(dst_h);

    // 立即归还 V4L2 buffer
    ReturnBuffer(idx);

    if (st != IM_STATUS_SUCCESS) {
      static int err_cnt = 0;
      if (++err_cnt <= 5)
        fprintf(stderr, "[V4L2 ] RGA convert->NV12 failed: %s\n", imStrError(st));
      return nullptr;
    }

    nv12_buf->SyncBegin();
    auto frame = DrmFrame::FromAllocator(nv12_buf, DrmFrame::NV12);
    frame->pts = pts_us * 1000;
    return frame;
  }

  // NV12 路径: 零拷贝包装, 析构时归还
  auto frame = DrmFrame::FromExternalFd(
      buffers_[idx].dmabuf_fd,
      width_, height_, h_stride_, v_stride_,
      DrmFrame::NV12,
      [self, idx]() { self->ReturnBuffer(idx); }
  );
  if (!frame) {
    ReturnBuffer(idx);
    return nullptr;
  }
  frame->vaddr = buffers_[idx].mmap_addr;
  frame->pts = pts_us * 1000;
  return frame;
}

void V4l2Source::ReturnBuffer(int idx) {
  std::lock_guard<std::mutex> lk(return_mtx_);
  if (fd_ < 0 || !streaming_) return;       // 已经在停流, 不再 QBUF
  if (idx < 0 || (size_t)idx >= buffers_.size()) return;
  if (buffers_[idx].queued) return;         // 防双重入队

  v4l2_plane planes[VIDEO_MAX_PLANES]{};
  v4l2_buffer buf{};
  buf.type     = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  buf.memory   = V4L2_MEMORY_MMAP;
  buf.index    = idx;
  buf.m.planes = planes;
  buf.length   = 1;
  if (xioctl(fd_, VIDIOC_QBUF, &buf) == 0) {
    buffers_[idx].queued = true;
  }
}
