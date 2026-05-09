#pragma once
// 镜头模型: 基于物理公式 f * tan(fov/2) = const
// 已知广角端 (focal_min, hfov, vfov), 推算任意焦距下的 FOV 与角度
#include <string>

class LensModel {
public:
  bool LoadFromFile(const std::string& path);

  // 由焦距(mm)算 FOV(度)
  float ComputeHFovDeg(float focal_mm) const;
  float ComputeVFovDeg(float focal_mm) const;

  // 由目标在画面中的中心点(像素)算偏离画面中心的增量角度(度)
  // 输出: yaw_inc (右为正), pitch_inc_image (下为正, 物理符号转换由 GimbalState 负责)
  void ComputeAngles(float cx, float cy,
                     int frame_w, int frame_h,
                     float focal_mm,
                     float* yaw_inc_deg, float* pitch_inc_image_deg) const;

  float focal_min() const { return focal_min_mm_; }
  float focal_max() const { return focal_max_mm_; }

private:
  float focal_min_mm_  = 0;
  float focal_max_mm_  = 0;
  float k_h_           = 0;  // K_h = focal_min * tan(hfov_min/2)
  float k_v_           = 0;
};
