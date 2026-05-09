#pragma once
// 变焦状态机
// IDLE → 检测到目标过小/过大 + 连续稳定 N 帧 → 写 zoom_command.json → ZOOMING(eta)
// ZOOMING → 超时到 → IDLE(focal=new_focal), 进入 cooldown
// cooldown 期间不再触发 (防止变焦后尺度突变又触发)
#include <atomic>
#include <chrono>
#include <string>

class ZoomController {
public:
  bool LoadFromFile(const std::string& path);

  // 每帧调用; 输入: 目标 bbox 占画面比例, 是否找到目标
  // 内部: 状态机更新 + 必要时写 zoom_command.json
  void Update(float bbox_ratio, bool target_visible);

  bool   IsZooming() const       { return state_ == State::ZOOMING; }
  float  GetCurrentFocal() const { return current_focal_; }

  // 给 LensModel 用
  void SetLensFocalRange(float fmin, float fmax) {
    focal_min_ = fmin; focal_max_ = fmax;
  }

private:
  enum class State { IDLE, ZOOMING, COOLDOWN };

  void WriteZoomCommand(float focal, const char* reason);

  // 配置
  float       initial_focal_      = 5.1f;
  float       zoom_in_thr_        = 0.01f;
  float       zoom_out_thr_       = 0.25f;
  float       zoom_step_          = 1.5f;
  int         zoom_duration_ms_   = 5500;
  int         stable_frames_      = 15;
  int         cooldown_frames_    = 30;
  std::string zoom_command_path_  = "/tmp/zoom_command.json";

  // 镜头物理范围 (由外部 SetLensFocalRange 注入)
  float focal_min_ = 5.1f;
  float focal_max_ = 153.0f;

  // 运行时状态
  State  state_                   = State::IDLE;
  float  current_focal_           = 5.1f;
  float  pending_focal_           = 5.1f;
  std::chrono::steady_clock::time_point zoom_start_;
  int    consecutive_trigger_     = 0;
  int    cooldown_counter_        = 0;
};
