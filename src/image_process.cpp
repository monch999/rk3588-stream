#include "image_process.h"
#include "im2d.hpp"
#include "RgaUtils.h"
#include <cstdio>
#include <cstring>

ImageProcess::ImageProcess(int src_w, int src_h, int target_w, int target_h)
    : src_w_(src_w), src_h_(src_h), target_w_(target_w), target_h_(target_h) {
  letterbox_.scale_w = static_cast<double>(target_w) / src_w;
  letterbox_.scale_h = static_cast<double>(target_h) / src_h;
  letterbox_.x_pad   = 0;
  letterbox_.y_pad   = 0;
}

// ==================== RGA: NV12 → RGB888 + resize ====================
std::shared_ptr<DrmFrame> ImageProcess::ConvertToRgb(
    const std::shared_ptr<DrmFrame>& src) {
  if (!src || src->fd < 0) {
    fprintf(stderr, "[IMG_PROC] invalid src DrmFrame\n");
    return nullptr;
  }

  // 申请目标 RGB DRM buffer (来自池子)
  auto rgb_buf = DrmAllocator::Instance().Acquire(
      DrmAllocator::RGB888, target_w_, target_h_);
  if (!rgb_buf) {
    fprintf(stderr, "[IMG_PROC] alloc RGB buffer failed\n");
    return nullptr;
  }

  // 用 importbuffer_fd 走 IOMMU 路径
  // 注意: im_handle_param_t 应传 stride 而不是 width/height
  im_handle_param_t src_param = {(uint32_t)src->h_stride,
                                  (uint32_t)src->v_stride,
                                  RK_FORMAT_YCbCr_420_SP};
  im_handle_param_t dst_param = {(uint32_t)rgb_buf->h_stride,
                                  (uint32_t)rgb_buf->v_stride,
                                  RK_FORMAT_RGB_888};

  rga_buffer_handle_t src_h = importbuffer_fd(src->fd, &src_param);
  rga_buffer_handle_t dst_h = importbuffer_fd(rgb_buf->fd, &dst_param);

  if (src_h == 0 || dst_h == 0) {
    fprintf(stderr, "[IMG_PROC] importbuffer_fd failed\n");
    if (src_h) releasebuffer_handle(src_h);
    if (dst_h) releasebuffer_handle(dst_h);
    return nullptr;
  }

  rga_buffer_t src_rga = wrapbuffer_handle(src_h,
      src->width, src->height, RK_FORMAT_YCbCr_420_SP);
  src_rga.wstride = src->h_stride;
  src_rga.hstride = src->v_stride;

  rga_buffer_t dst_rga = wrapbuffer_handle(dst_h,
      target_w_, target_h_, RK_FORMAT_RGB_888);
  dst_rga.wstride = rgb_buf->h_stride;
  dst_rga.hstride = rgb_buf->v_stride;

  im_rect src_rect = {0, 0, src->width, src->height};
  im_rect dst_rect = {0, 0, target_w_, target_h_};
  IM_STATUS status = improcess(src_rga, dst_rga, {}, src_rect, dst_rect, {}, IM_SYNC);

  releasebuffer_handle(src_h);
  releasebuffer_handle(dst_h);

  if (status != IM_STATUS_SUCCESS) {
    fprintf(stderr, "[IMG_PROC] RGA NV12->RGB+scale failed: %s\n", imStrError(status));
    return nullptr;
  }

  // RGA 写完后, CPU 读前需要 sync
  rgb_buf->SyncBegin();

  return DrmFrame::FromAllocator(rgb_buf, DrmFrame::RGB24);
}

// ==================== 在 Y 平面画检测框 ====================
void ImageProcess::DrawDetections(
    const std::shared_ptr<DrmFrame>& frame,
    const object_detect_result_list& od_results) {
  if (!frame || !frame->vaddr) return;
  if (frame->format != DrmFrame::NV12) {
    fprintf(stderr, "[IMG_PROC] DrawDetections only supports NV12\n");
    return;
  }

  uint8_t* y_plane = static_cast<uint8_t*>(frame->vaddr);
  const int h_stride = frame->h_stride;
  const int v_stride = frame->v_stride;
  const int W = frame->width;
  const int H = frame->height;

  // 边框粗细按分辨率自适应 (4K -> 4px, 1080p -> 2px)
  const int thickness = std::max(2, W / 960);
  // 文字放大一倍: 1080p -> 1.4, 4K -> 2.8
  const double font_scale = std::max(1.0, W / 1920.0 * 1.4);
  const int font_thick = std::max(2, W / 960);

  for (int i = 0; i < od_results.count; ++i) {
    const auto& r = od_results.results[i];

    // 白色边框 (Y=255)
    nv12_draw::DrawRect(y_plane, h_stride, v_stride, W, H,
                        r.box.left, r.box.top,
                        r.box.right, r.box.bottom,
                        255, thickness);

    // 类别标签
    char text[128];
    snprintf(text, sizeof(text), "%s %.0f%%",
             coco_cls_to_name(r.cls_id), r.prop * 100);

    // 文字位置: 框上方; 太靠顶时放框内
    // 字体放大后, 偏移量也跟着放大 (避免文字贴到框线)
    const int text_offset = static_cast<int>(font_scale * 25);
    int text_y = r.box.top - 8;
    if (text_y < text_offset) text_y = r.box.top + text_offset;

    nv12_draw::DrawText(y_plane, h_stride, v_stride, W, H,
                        text, r.box.left, text_y,
                        font_scale, font_thick);
  }
}
