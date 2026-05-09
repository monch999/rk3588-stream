#include "lens_model.h"
#include "json_helper.h"
#include <cmath>
#include <cstdio>

static constexpr float kDeg2Rad = 3.14159265358979323846f / 180.0f;
static constexpr float kRad2Deg = 180.0f / 3.14159265358979323846f;

bool LensModel::LoadFromFile(const std::string& path) {
  JsonFlat j;
  if (!j.LoadFile(path)) return false;

  focal_min_mm_ = static_cast<float>(j.GetDouble("focal_min_mm", 0));
  focal_max_mm_ = static_cast<float>(j.GetDouble("focal_max_mm", 0));
  float hfov_min = static_cast<float>(j.GetDouble("hfov_at_focal_min_deg", 0));
  float vfov_min = static_cast<float>(j.GetDouble("vfov_at_focal_min_deg", 0));

  if (focal_min_mm_ <= 0 || focal_max_mm_ <= focal_min_mm_
      || hfov_min <= 0 || vfov_min <= 0) {
    fprintf(stderr, "[LENS ] invalid lens.json: focal_min=%.2f max=%.2f hfov=%.2f vfov=%.2f\n",
            focal_min_mm_, focal_max_mm_, hfov_min, vfov_min);
    return false;
  }

  k_h_ = focal_min_mm_ * std::tan(hfov_min * 0.5f * kDeg2Rad);
  k_v_ = focal_min_mm_ * std::tan(vfov_min * 0.5f * kDeg2Rad);

  printf("[LENS ] loaded: f=[%.1f,%.1f]mm  K_h=%.3f K_v=%.3f\n",
         focal_min_mm_, focal_max_mm_, k_h_, k_v_);
  return true;
}

float LensModel::ComputeHFovDeg(float focal_mm) const {
  if (focal_mm <= 0) return 0;
  return 2.0f * std::atan(k_h_ / focal_mm) * kRad2Deg;
}

float LensModel::ComputeVFovDeg(float focal_mm) const {
  if (focal_mm <= 0) return 0;
  return 2.0f * std::atan(k_v_ / focal_mm) * kRad2Deg;
}

void LensModel::ComputeAngles(float cx, float cy,
                              int frame_w, int frame_h,
                              float focal_mm,
                              float* yaw_inc_deg,
                              float* pitch_inc_image_deg) const {
  // 针孔模型: tan(angle) = (offset_pixel / half_size) * tan(fov/2)
  float hfov = ComputeHFovDeg(focal_mm);
  float vfov = ComputeVFovDeg(focal_mm);
  float dx = cx - frame_w * 0.5f;
  float dy = cy - frame_h * 0.5f;
  float tan_h = std::tan(hfov * 0.5f * kDeg2Rad);
  float tan_v = std::tan(vfov * 0.5f * kDeg2Rad);
  *yaw_inc_deg          = std::atan(dx / (frame_w * 0.5f) * tan_h) * kRad2Deg;
  *pitch_inc_image_deg  = std::atan(dy / (frame_h * 0.5f) * tan_v) * kRad2Deg;
}
