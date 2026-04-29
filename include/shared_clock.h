#pragma once
#include <chrono>
#include <cstdint>
#include <thread>

// ==================== 统一时间基准 ====================
// 所有通道共享同一个 epoch，基于 steady_clock (monotonic)
// PTS = nanoseconds since epoch，确保三路流时间对齐
class SharedClock {
public:
  using clock_type  = std::chrono::steady_clock;
  using time_point  = clock_type::time_point;

  void Start() { epoch_ = clock_type::now(); }

  time_point GetEpoch() const { return epoch_; }

  // 当前时刻距 epoch 的纳秒数
  int64_t NowNs() const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock_type::now() - epoch_)
        .count();
  }

  // 根据帧序号和帧率计算理论 PTS (纳秒)
  static int64_t FramePtsNs(int64_t seq, double fps) {
    return static_cast<int64_t>(seq * 1000000000.0 / fps);
  }

  // 阻塞等待直到 PTS 时刻到达 (用于帧率节流)
  void WaitUntilPts(int64_t pts_ns) const {
    auto target = epoch_ + std::chrono::nanoseconds(pts_ns);
    auto now    = clock_type::now();
    if (target > now)
      std::this_thread::sleep_until(target);
  }

private:
  time_point epoch_;
};
