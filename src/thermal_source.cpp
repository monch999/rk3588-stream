#include "thermal_source.h"
#include "drm_allocator.h"
#include "im2d.hpp"
#include "RgaUtils.h"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

// 辅助 ioctl 函数
static int xioctl(int fd, unsigned long req, void* arg) {
    int r;
    do { r = ::ioctl(fd, req, arg); }
    while (r == -1 && errno == EINTR);
    return r;
}

ThermalSource::ThermalSource(const std::string& dev_path, int buffer_count)
    : dev_path_(dev_path), buffer_count_(buffer_count) {
    if (!Open()) {
        fprintf(stderr, "[THERMAL] open/init failed: %s\n", dev_path.c_str());
    }
}

ThermalSource::~ThermalSource() {
    StopAndReleaseBuffers();
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

bool ThermalSource::Open() {
    fd_ = ::open(dev_path_.c_str(), O_RDWR | O_NONBLOCK);
    if (fd_ < 0) return false;

    v4l2_capability cap{};
    if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) return false;
    
    return NegotiateAndStart();
}

bool ThermalSource::NegotiateAndStart() {
    if (!SetFormat()) return false;
    if (!RequestBuffers()) return false;

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) return false;
    
    streaming_ = true;
    printf("[THERMAL] Streaming ON: Raw=%dx%d, Output=640x512\n", raw_width_, raw_height_);
    return true;
}

void ThermalSource::StopAndReleaseBuffers() {
    if (streaming_.exchange(false)) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(fd_, VIDIOC_STREAMOFF, &type);
    }
    std::lock_guard<std::mutex> lk(return_mtx_);
    for (auto& b : buffers_) {
        if (b.mmap_addr && b.mmap_size > 0) ::munmap(b.mmap_addr, b.mmap_size);
        if (b.dmabuf_fd >= 0)               ::close(b.dmabuf_fd);
        b = V4l2Buffer{};
    }
    buffers_.clear();
}

bool ThermalSource::SetFormat() {
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; // 单平面
    fmt.fmt.pix_mp.width       = raw_width_;
    fmt.fmt.pix_mp.height      = raw_height_;
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_UYVY; // 机芯硬性要求 UYVY
    fmt.fmt.pix_mp.field       = V4L2_FIELD_NONE;
    fmt.fmt.pix_mp.num_planes  = 1;

    if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        fprintf(stderr, "[THERMAL] Failed to set UYVY 1280x520\n");
        return false;
    }
    return true;
}

bool ThermalSource::RequestBuffers() {
    v4l2_requestbuffers req{};
    req.count  = buffer_count_;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) return false;

    buffers_.assign(req.count, V4l2Buffer{});
    for (size_t i = 0; i < buffers_.size(); i++) {
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        xioctl(fd_, VIDIOC_QUERYBUF, &buf);

        buffers_[i].mmap_addr = ::mmap(nullptr, buf.length,
            PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
        buffers_[i].mmap_size = buf.length;

        v4l2_exportbuffer eb{};
        eb.type  = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        eb.index = i;
        eb.flags = O_RDWR;
        xioctl(fd_, VIDIOC_EXPBUF, &eb);
        buffers_[i].dmabuf_fd = eb.fd;

        xioctl(fd_, VIDIOC_QBUF, &buf);
        buffers_[i].queued = true;
    }
    return true;
}

std::shared_ptr<DrmFrame> ThermalSource::GetNextDrmFrame() {
    if (fd_ < 0 || !streaming_) return nullptr;

    pollfd pfd{};
    pfd.fd = fd_;
    pfd.events = POLLIN;
    if (::poll(&pfd, 1, 1000) <= 0) return nullptr;

    v4l2_buffer buf{};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) return nullptr;

    buffers_[buf.index].queued = false;
    int64_t pts_us = (int64_t)buf.timestamp.tv_sec * 1000000 + buf.timestamp.tv_usec;
    return ProcessAndWrapBuffer(buf.index, pts_us);
}

// ==================== 核心解析 ====================
void ThermalSource::ExtractThermalData(uint8_t* raw_ptr, int64_t pts_us) {
    // 1280宽 * 2字节(UYVY) = 2560 字节/行
    // 跳过前 512 行画面，直接定位到底部 8 行冗余数据的起点
    const size_t redund_offset = 512 * (1280 * 2);
    uint8_t* buf = raw_ptr + redund_offset;

    ThermalData data{};
    data.pts_us = pts_us;

    // 根据数据协议(小端序，有效位被0xfe隔开)
    data.hot_y  = (buf[3] << 8) | buf[1];
    data.hot_x  = (buf[7] << 8) | buf[5];
    data.cold_y = (buf[11] << 8) | buf[9];
    data.cold_x = (buf[15] << 8) | buf[13];

    // 解析温度 (按 32 位整型拼装后转 float)
    int32_t hot_raw = (buf[23] << 24) | (buf[21] << 16) | (buf[19] << 8) | buf[17];
    data.max_temp = static_cast<float>(hot_raw) / 200.0f;

    int32_t cold_raw = (buf[31] << 24) | (buf[29] << 16) | (buf[27] << 8) | buf[25];
    data.min_temp = static_cast<float>(cold_raw) / 200.0f;

    // 缓存到内部，供外部按需查询
    {
        std::lock_guard<std::mutex> lk(thermal_mtx_);
        latest_thermal_ = data;
    }
}

// ==================== 核心裁剪与转换 ====================
std::shared_ptr<DrmFrame> ThermalSource::ProcessAndWrapBuffer(int idx, int64_t pts_us) {
    auto* self = this;
    
    // 1. 拦截并提取测温数据 (在 RGA 处理前，防止数据被破坏)
    ExtractThermalData((uint8_t*)buffers_[idx].mmap_addr, pts_us);

    // 2. 申请用于输出纯净画面的 NV12 DrmFrame (640x512)
    auto out_buf = DrmAllocator::Instance().Acquire(DrmAllocator::NV12, 640, 512);
    if (!out_buf) {
        ReturnBuffer(idx);
        return nullptr;
    }

    // 3. 配置源 RGA 属性 (1280x520 UYVY)
    im_handle_param_t src_param = {(uint32_t)raw_width_, (uint32_t)raw_height_, RK_FORMAT_UYVY_422};
    rga_buffer_handle_t src_h = importbuffer_virtualaddr(buffers_[idx].mmap_addr, &src_param);
    
    rga_buffer_t src_rga = wrapbuffer_handle(src_h, raw_width_, raw_height_, RK_FORMAT_UYVY_422);
    // UYVY 是 packed 格式，wstride 在 RGA 中就是像素宽度 1280
    src_rga.wstride = raw_width_;
    src_rga.hstride = raw_height_;

    // 4. 配置目标 RGA 属性 (640x512 NV12)
    im_handle_param_t dst_param = {(uint32_t)out_buf->h_stride, (uint32_t)out_buf->v_stride, RK_FORMAT_YCbCr_420_SP};
    rga_buffer_handle_t dst_h = importbuffer_fd(out_buf->fd, &dst_param);
    
    rga_buffer_t dst_rga = wrapbuffer_handle(dst_h, 640, 512, RK_FORMAT_YCbCr_420_SP);
    dst_rga.wstride = out_buf->h_stride;
    dst_rga.hstride = out_buf->v_stride;

    // 5. ★ 见证魔法的时刻：一键 Crop + Format Convert
    // 源矩形：起于 x=640 (跳过左边 RAW)，y=0，宽高 640x512 (舍弃底部8行)
    im_rect src_rect = {640, 0, 640, 512};
    im_rect dst_rect = {0, 0, 640, 512};
    
    IM_STATUS st = improcess(src_rga, dst_rga, {}, src_rect, dst_rect, {}, IM_SYNC);

    releasebuffer_handle(src_h);
    releasebuffer_handle(dst_h);
    
    // 立即归还 V4L2 原始 buffer
    ReturnBuffer(idx);

    if (st != IM_STATUS_SUCCESS) {
        fprintf(stderr, "[THERMAL] RGA crop/convert failed: %s\n", imStrError(st));
        return nullptr;
    }

    out_buf->SyncBegin();
    auto frame = DrmFrame::FromAllocator(out_buf, DrmFrame::NV12);
    frame->pts = pts_us * 1000;
    return frame;
}

void ThermalSource::ReturnBuffer(int idx) {
    std::lock_guard<std::mutex> lk(return_mtx_);
    if (fd_ < 0 || !streaming_) return;
    if (idx < 0 || (size_t)idx >= buffers_.size()) return;
    if (buffers_[idx].queued) return;

    v4l2_plane planes[VIDEO_MAX_PLANES]{};
    v4l2_buffer buf{};
    buf.type     = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory   = V4L2_MEMORY_MMAP;
    buf.index    = idx;
    buf.m.planes = planes;
    buf.length   = 1;
    if (xioctl(fd_, VIDIOC_QBUF, &buf) == 0) {
        buffers_[idx].queued = true;
    }
}