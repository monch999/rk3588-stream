#pragma once
// 云台状态: 启动时读 gimbal_state.json 一次, 之后开环积分维护
// pitch: 0=正下, 90=水平, 180=正上 (天底-天顶编码)
// yaw:   0-360, 单次增量限制 ±180 (取近路)
#include <string>

class GimbalState {
public:
  bool LoadInitialFromFile(const std::string& path);

  // 输入: 来自镜头模型的画面增量角度 (yaw 右为正, pitch_image 下为正)
  // 输出: 限位后的实际增量(传给伺服) + 是否被限位
  // 内部: 更新绝对位置
  void ApplyIncrement(float yaw_inc_in, float pitch_inc_image_in,
                      float* yaw_inc_out, float* pitch_inc_out,
                      bool* clamped_out);

  float GetYawAbs() const   { return yaw_abs_; }
  float GetPitchAbs() const { return pitch_abs_; }

  // 配置: 画面 pitch 增量 → 云台 pitch 增量的符号 (默认 -1)
  void SetPitchSign(int sign) { pitch_sign_ = sign >= 0 ? 1 : -1; }

private:
  float yaw_abs_   = 0.0f;   // [0, 360)
  float pitch_abs_ = 90.0f;  // [0, 180]
  int   pitch_sign_ = -1;    // 画面 y 向下 vs 云台 pitch 向上 → 反号
};
