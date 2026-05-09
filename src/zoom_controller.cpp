#include "zoom_controller.h"
#include "json_helper.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>

bool ZoomController::LoadFromFile(const std::string& path) {
  JsonFlat j;
  if (!j.LoadFile(path)) return false;

  initial_focal_     = static_cast<float>(j.GetDouble("initial_focal_mm",          5.1));
  zoom_in_thr_       = static_cast<float>(j.GetDouble("zoom_in_threshold_ratio",   0.01));
  zoom_out_thr_      = static_cast<float>(j.GetDouble("zoom_out_threshold_ratio",  0.25));
  zoom_step_         = static_cast<float>(j.GetDouble("zoom_step",                 1.5));
  zoom_duration_ms_  = j.GetInt("zoom_duration_ms",          5500);
  stable_frames_     = j.GetInt("stable_frames_before_zoom", 15);
  cooldown_frames_   = j.GetInt("post_zoom_cooldown_frames", 30);
  zoom_command_path_ = j.GetString("zoom_command_path", "/tmp/zoom_command.json");

  current_focal_ = initial_focal_;
  pending_focal_ = initial_focal_;

  printf("[ZOOM ] loaded: initial_f=%.1f thr=[%.3f,%.3f] step=%.2f dur=%dms "
         "stable=%d cooldown=%d cmd=%s\n",
         initial_focal_, zoom_in_thr_, zoom_out_thr_, zoom_step_,
         zoom_duration_ms_, stable_frames_, cooldown_frames_,
         zoom_command_path_.c_str());
  return true;
}

void ZoomController::Update(float bbox_ratio, bool target_visible) {
  auto now = std::chrono::steady_clock::now();

  // 1. ZOOMING -> COOLDOWN(超时到达)
  if (state_ == State::ZOOMING) {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - zoom_start_).count();
    if (elapsed >= zoom_duration_ms_) {
      current_focal_ = pending_focal_;
      state_ = State::COOLDOWN;
      cooldown_counter_ = cooldown_frames_;
      consecutive_trigger_ = 0;
      printf("[ZOOM ] ZOOMING -> COOLDOWN (focal=%.2f)\n", current_focal_);
    }
    return;
  }

  // 2. COOLDOWN -> IDLE(帧数倒数完)
  if (state_ == State::COOLDOWN) {
    if (--cooldown_counter_ <= 0) {
      state_ = State::IDLE;
      printf("[ZOOM ] COOLDOWN -> IDLE\n");
    }
    return;
  }

  // 3. IDLE: 评估是否触发变焦
  if (!target_visible) {
    consecutive_trigger_ = 0;
    return;
  }

  bool need_zoom_in  = bbox_ratio < zoom_in_thr_;
  bool need_zoom_out = bbox_ratio > zoom_out_thr_;

  if (!need_zoom_in && !need_zoom_out) {
    consecutive_trigger_ = 0;
    return;
  }

  consecutive_trigger_++;
  if (consecutive_trigger_ < stable_frames_) return;

  // 触发变焦
  float new_focal = need_zoom_in
      ? std::min(current_focal_ * zoom_step_, focal_max_)
      : std::max(current_focal_ / zoom_step_, focal_min_);

  if (std::fabs(new_focal - current_focal_) < 0.01f) {
    // 已在物理边界, 无法继续
    consecutive_trigger_ = 0;
    return;
  }

  WriteZoomCommand(new_focal, need_zoom_in ? "zoom_in" : "zoom_out");
  pending_focal_ = new_focal;
  zoom_start_ = now;
  state_ = State::ZOOMING;
  consecutive_trigger_ = 0;
}

void ZoomController::WriteZoomCommand(float focal, const char* reason) {
  // 原子写: 先写 .tmp 再 rename
  std::string tmp_path = zoom_command_path_ + ".tmp";
  auto ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();

  FILE* fp = std::fopen(tmp_path.c_str(), "w");
  if (!fp) {
    fprintf(stderr, "[ZOOM ] open tmp failed: %s\n", tmp_path.c_str());
    return;
  }
  std::fprintf(fp,
      "{\n"
      "  \"focal_mm\": %.3f,\n"
      "  \"timestamp_ns\": %lld,\n"
      "  \"reason\": \"%s\"\n"
      "}\n", focal, (long long)ts, reason);
  std::fclose(fp);

  if (std::rename(tmp_path.c_str(), zoom_command_path_.c_str()) != 0) {
    fprintf(stderr, "[ZOOM ] rename failed: %s -> %s\n",
            tmp_path.c_str(), zoom_command_path_.c_str());
    return;
  }
  printf("[ZOOM ] CMD: focal=%.2f reason=%s\n", focal, reason);
}
