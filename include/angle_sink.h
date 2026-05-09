#pragma once
// 角度输出接口: stdout 实现
// 后续接伺服只需实现 UartAngleSink / UdpAngleSink 等
#include <cstdio>

struct AngleMsg {
  int     track_id;
  float   yaw_inc_deg;
  float   pitch_inc_deg;
  float   yaw_abs_deg;
  float   pitch_abs_deg;
  bool    lost;
  bool    zooming;
  bool    clamped;
  float   focal_mm;
  int64_t pts_ns;
};

class IAngleSink {
public:
  virtual ~IAngleSink() = default;
  virtual void Push(const AngleMsg& m) = 0;
};

class StdoutAngleSink : public IAngleSink {
public:
  void Push(const AngleMsg& m) override {
    std::printf(
        "[ANGLE] id=%d yaw_inc=%+.2f pitch_inc=%+.2f "
        "yaw_abs=%.2f pitch_abs=%.2f lost=%d zooming=%d clamped=%d "
        "focal=%.2f pts=%lld\n",
        m.track_id, m.yaw_inc_deg, m.pitch_inc_deg,
        m.yaw_abs_deg, m.pitch_abs_deg,
        m.lost ? 1 : 0, m.zooming ? 1 : 0, m.clamped ? 1 : 0,
        m.focal_mm, (long long)m.pts_ns);
    std::fflush(stdout);  // 保证下游 grep / 管道能实时收到
  }
};
