#pragma once
#include <chrono>
#include <cstdio>
#include <thread>

// ==================== 日志宏（替代 kaylordut/log） ====================
// 原项目使用 fmt 库的 {} 占位符，这里简化为空操作
// 如需日志输出，建议引入 spdlog: #include "spdlog/spdlog.h"
// 然后替换为 spdlog::info(...) / spdlog::error(...)
template<typename... Args>
inline void _log_noop(const char*, Args&&...) {}

#define KAYLORDUT_LOG_INFO(fmt, ...)  _log_noop(fmt, ##__VA_ARGS__)
#define KAYLORDUT_LOG_ERROR(fmt, ...) _log_noop(fmt, ##__VA_ARGS__)
#define KAYLORDUT_LOG_DEBUG(fmt, ...) _log_noop(fmt, ##__VA_ARGS__)

// 计时宏（替代 KAYLORDUT_TIME_COST_*），直接执行语句
#define KAYLORDUT_TIME_COST_DEBUG(name, stmt) do { stmt; } while(0)
#define KAYLORDUT_TIME_COST_INFO(name, stmt)  do { stmt; } while(0)

// ==================== TimeDuration（替代 kaylordut/time） ====================
class TimeDuration {
public:
  TimeDuration() : start_(std::chrono::steady_clock::now()) {}

  std::chrono::steady_clock::duration DurationSinceLastTime() {
    auto now = std::chrono::steady_clock::now();
    auto d = now - start_;
    start_ = now;
    return d;
  }

private:
  std::chrono::steady_clock::time_point start_;
};

// ==================== run_once_with_delay ====================
template <typename Func, typename Duration>
void run_once_with_delay(Func&& func, Duration delay) {
  auto start = std::chrono::steady_clock::now();
  func();
  auto elapsed = std::chrono::steady_clock::now() - start;
  if (elapsed < delay) {
    std::this_thread::sleep_for(delay - elapsed);
  }
}
