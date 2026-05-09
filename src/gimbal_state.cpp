#include "gimbal_state.h"
#include "json_helper.h"
#include <cmath>
#include <cstdio>

bool GimbalState::LoadInitialFromFile(const std::string& path) {
  JsonFlat j;
  if (!j.LoadFile(path)) return false;

  yaw_abs_   = static_cast<float>(j.GetDouble("yaw_abs_deg",   0.0));
  pitch_abs_ = static_cast<float>(j.GetDouble("pitch_abs_deg", 90.0));

  // 校验
  if (pitch_abs_ < 0 || pitch_abs_ > 180) {
    fprintf(stderr, "[GIMBL] invalid initial pitch_abs=%.2f (must be 0-180)\n", pitch_abs_);
    return false;
  }
  // yaw 折叠到 [0, 360)
  yaw_abs_ = std::fmod(std::fmod(yaw_abs_, 360.0f) + 360.0f, 360.0f);

  printf("[GIMBL] initial pose: yaw_abs=%.2f pitch_abs=%.2f\n", yaw_abs_, pitch_abs_);
  return true;
}

void GimbalState::ApplyIncrement(float yaw_inc_in, float pitch_inc_image_in,
                                 float* yaw_inc_out, float* pitch_inc_out,
                                 bool* clamped_out) {
  // ---- yaw: 折叠到 [-180, 180], 永远不 clamped ----
  float yaw_inc = std::fmod(yaw_inc_in + 540.0f, 360.0f) - 180.0f;
  yaw_abs_ = std::fmod(std::fmod(yaw_abs_ + yaw_inc, 360.0f) + 360.0f, 360.0f);
  *yaw_inc_out = yaw_inc;

  // ---- pitch: 符号反转后做绝对位置限位 ----
  float pitch_inc = pitch_sign_ * pitch_inc_image_in;
  float new_pitch = pitch_abs_ + pitch_inc;
  bool clamped = false;
  if (new_pitch < 0.0f || new_pitch > 180.0f) {
    pitch_inc = 0.0f;
    clamped = true;
  } else {
    pitch_abs_ = new_pitch;
  }
  *pitch_inc_out = pitch_inc;
  *clamped_out   = clamped;
}
