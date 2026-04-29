#pragma once
#include <cstdint>
#include <cstring>
#include <memory>
#include <functional>
#include <opencv2/opencv.hpp>
#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
#include "drm_allocator.h"

// ==================== DRM 帧封装 ====================
// 统一的硬件 buffer 封装, 支持 zero-copy 在管线各阶段间传递
//
// 两种来源:
//   1. FromMppFrame(): 来自 decoder 输出, 持有 MppFrame, 析构时调 mpp_frame_deinit
//   2. FromAllocator(): 来自自管 DRM 池 (DrmAllocator), shared_ptr 析构时自动归还
//
struct DrmFrame {
  int      width    = 0;
  int      height   = 0;
  int      h_stride = 0;
  int      v_stride = 0;
  int      fd       = -1;
  void    *vaddr    = nullptr;
  int64_t  pts      = 0;

  MppBuffer mpp_buf = nullptr;
  MppFrame  mpp_frm = nullptr;  // 仅 decoder 来源时持有

  enum Format { NV12, BGR24, RGB24 } format = NV12;

  // 自管 buffer 引用 (FromAllocator 时持有, 析构时自动归还)
  std::shared_ptr<DrmAllocator::Buffer> owned_;

  // ---- 来自 MPP decoder 输出 ----
  static std::shared_ptr<DrmFrame> FromMppFrame(MppFrame frame) {
    if (!frame) return nullptr;
    auto df       = std::make_shared<DrmFrame>();
    df->mpp_frm   = frame;
    df->width     = mpp_frame_get_width(frame);
    df->height    = mpp_frame_get_height(frame);
    df->h_stride  = mpp_frame_get_hor_stride(frame);
    df->v_stride  = mpp_frame_get_ver_stride(frame);
    df->pts       = mpp_frame_get_pts(frame);
    df->format    = NV12;

    MppBuffer buf = mpp_frame_get_buffer(frame);
    if (buf) {
      df->mpp_buf = buf;
      df->fd      = mpp_buffer_get_fd(buf);
      df->vaddr   = mpp_buffer_get_ptr(buf);
    }
    return df;
  }

  // ---- 来自自管 DRM 池 ----
  static std::shared_ptr<DrmFrame> FromAllocator(
      std::shared_ptr<DrmAllocator::Buffer> buf, Format fmt = NV12) {
    if (!buf) return nullptr;
    auto df       = std::make_shared<DrmFrame>();
    df->owned_    = buf;
    df->mpp_buf   = buf->mpp_buf;
    df->fd        = buf->fd;
    df->vaddr     = buf->vaddr;
    df->width     = buf->width;
    df->height    = buf->height;
    df->h_stride  = buf->h_stride;
    df->v_stride  = buf->v_stride;
    df->format    = fmt;
    return df;
  }

  // ---- cv::Mat 包装 (零拷贝, 仅供 CPU 访问) ----
  cv::Mat GetNV12Mat() const {
    if (!vaddr) return {};
    return cv::Mat(v_stride * 3 / 2, h_stride, CV_8UC1, vaddr);
  }

  // ---- 兼容路径: NV12 → BGR (CPU 转换, 仅用于 display) ----
  std::unique_ptr<cv::Mat> ToBgrMat() const {
    if (!vaddr) return nullptr;
    cv::Mat nv12(v_stride * 3 / 2, h_stride, CV_8UC1, vaddr);
    auto bgr = std::make_unique<cv::Mat>();
    cv::cvtColor(nv12, *bgr, cv::COLOR_YUV2BGR_NV12);
    if (bgr->cols != width || bgr->rows != height)
      *bgr = (*bgr)(cv::Rect(0, 0, width, height)).clone();
    return bgr;
  }

  // ---- Cache 同步 ----
  void SyncEnd()   { if (owned_) owned_->SyncEnd(); }
  void SyncBegin() { if (owned_) owned_->SyncBegin(); }

  ~DrmFrame() {
    if (mpp_frm) {
      mpp_frame_deinit(&mpp_frm);
      mpp_frm = nullptr;
    }
  }

  DrmFrame() = default;
  DrmFrame(const DrmFrame&) = delete;
  DrmFrame& operator=(const DrmFrame&) = delete;
};

// ==================== Y 平面绘制工具 (灰度框 + 黑边白字) ====================
namespace nv12_draw {

// 在 Y 平面画矩形边框 (用 cv::rectangle 单通道, 最可靠)
inline void DrawRect(uint8_t* y_plane, int h_stride, int v_stride,
                     int img_w, int img_h,
                     int x1, int y1, int x2, int y2,
                     uint8_t luma, int thickness) {
  if (x1 > x2) std::swap(x1, x2);
  if (y1 > y2) std::swap(y1, y2);
  // clamp 到有效区域 (保留 thickness 余量, 避免 cv::rectangle 写到 stride padding)
  x1 = std::max(0, std::min(x1, img_w - 1));
  x2 = std::max(0, std::min(x2, img_w - 1));
  y1 = std::max(0, std::min(y1, img_h - 1));
  y2 = std::max(0, std::min(y2, img_h - 1));

  cv::Mat y_mat(v_stride, h_stride, CV_8UC1, y_plane);
  cv::Mat roi = y_mat(cv::Rect(0, 0, img_w, img_h));
  cv::rectangle(roi, cv::Point(x1, y1), cv::Point(x2, y2),
                cv::Scalar(luma), thickness);
}

// 在 Y 平面绘制文字: 黑色描边 + 白色填充
inline void DrawText(uint8_t* y_plane, int h_stride, int v_stride,
                     int img_w, int img_h,
                     const std::string& text, int x, int y,
                     double font_scale = 0.6, int thickness = 1) {
  cv::Mat y_mat(v_stride, h_stride, CV_8UC1, y_plane);
  cv::Mat roi = y_mat(cv::Rect(0, 0, img_w, img_h));

  // 黑色描边 (Y=0)
  cv::putText(roi, text, cv::Point(x, y),
              cv::FONT_HERSHEY_SIMPLEX, font_scale,
              cv::Scalar(0), thickness + 2, cv::LINE_AA);
  // 白色文字 (Y=255)
  cv::putText(roi, text, cv::Point(x, y),
              cv::FONT_HERSHEY_SIMPLEX, font_scale,
              cv::Scalar(255), thickness, cv::LINE_AA);
}

} // namespace nv12_draw
