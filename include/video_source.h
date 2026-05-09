#pragma once
// 视频源抽象接口
// 实现:
//   - VideoFile  (文件 / RTSP URL, 已存在)
//   - V4l2Source (HDMI-RX, /dev/videoN, 新增)
#include <memory>
#include "drm_frame.h"

class IVideoSource {
public:
  virtual ~IVideoSource() = default;

  virtual std::shared_ptr<DrmFrame> GetNextDrmFrame() = 0;
  virtual int    get_frame_width()  const = 0;
  virtual int    get_frame_height() const = 0;
  virtual double get_fps()          const = 0;
};
