#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include "drm_frame.h"
#include "postprocess.h"

// ==================== 图像处理 (RGA 硬件加速) ====================
// zero-copy 链路:
//   1. ConvertToRgb(src_nv12) -> RGB DrmFrame (NPU 输入)
//   2. NPU 推理后, DrawDetections() 直接在原 NV12 的 Y 平面画灰度框
//
class ImageProcess {
public:
  // src_w/h: 原始视频分辨率, target_w/h: 模型输入分辨率
  ImageProcess(int src_w, int src_h, int target_w, int target_h);

  // RGA: NV12 (h_stride×v_stride) -> RGB888 (target_w×target_h)
  // 一步完成 NV12→RGB + resize, 输出来自 DrmAllocator 的 buffer
  std::shared_ptr<DrmFrame> ConvertToRgb(const std::shared_ptr<DrmFrame>& src);

  const letterbox_t& get_letter_box() const { return letterbox_; }

  // 在 NV12 帧的 Y 平面上画检测框 (白色边框 + 黑边白字)
  // od_results 中的坐标已经是原始图像坐标系 (postprocess 已反算)
  void DrawDetections(const std::shared_ptr<DrmFrame>& nv12_frame,
                      const object_detect_result_list& od_results);

private:
  int src_w_, src_h_;
  int target_w_, target_h_;
  letterbox_t letterbox_;
};
