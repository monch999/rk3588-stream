#pragma once
#include <cstdio>
#include <string>
#include <opencv2/opencv.hpp>

// ==================== 推流配置 ====================
struct StreamConfig {
  std::string rtsp_url;
  std::string rtmp_url;
  std::string codec   = "h264";
  std::string bitrate  = "4M";
  int         gop      = 0;   // 0 = auto (2 × fps)
};

// ==================== FFmpeg 管道推流 ====================
class FFmpegStreamer {
public:
  FFmpegStreamer() = default;
  ~FFmpegStreamer() { Close(); }

  // 禁止拷贝
  FFmpegStreamer(const FFmpegStreamer &)            = delete;
  FFmpegStreamer &operator=(const FFmpegStreamer &)  = delete;

  // 允许移动
  FFmpegStreamer(FFmpegStreamer &&o) noexcept : pipe_(o.pipe_) { o.pipe_ = nullptr; }
  FFmpegStreamer &operator=(FFmpegStreamer &&o) noexcept {
    Close();
    pipe_ = o.pipe_;
    o.pipe_ = nullptr;
    return *this;
  }

  bool Open(int width, int height, double fps, const StreamConfig &cfg) {
    std::string encoder = cfg.codec + "_rkmpp";
    int gop = cfg.gop > 0 ? cfg.gop : static_cast<int>(fps * 2);

    char cmd[2048];
    if (!cfg.rtsp_url.empty() && !cfg.rtmp_url.empty()) {
      snprintf(cmd, sizeof(cmd),
        "ffmpeg -hide_banner -loglevel warning "
        "-f rawvideo -pixel_format bgr24 -video_size %dx%d -framerate %.1f "
        "-i pipe:0 "
        "-c:v %s -rc_mode CBR -b:v %s -g %d -bf 0 "
        "-flags +low_delay -fflags +genpts "
        "-map 0:v "
        "-f tee '"
        "[f=rtsp:rtsp_transport=tcp]%s"
        "|[f=flv]%s"
        "'",
        width, height, fps,
        encoder.c_str(), cfg.bitrate.c_str(), gop,
        cfg.rtsp_url.c_str(), cfg.rtmp_url.c_str());
    } else if (!cfg.rtsp_url.empty()) {
      snprintf(cmd, sizeof(cmd),
        "ffmpeg -hide_banner -loglevel warning "
        "-f rawvideo -pixel_format bgr24 -video_size %dx%d -framerate %.1f "
        "-i pipe:0 "
        "-c:v %s -rc_mode CBR -b:v %s -g %d -bf 0 "
        "-flags +low_delay -fflags +genpts "
        "-rtsp_transport tcp -f rtsp %s",
        width, height, fps,
        encoder.c_str(), cfg.bitrate.c_str(), gop,
        cfg.rtsp_url.c_str());
    } else if (!cfg.rtmp_url.empty()) {
      snprintf(cmd, sizeof(cmd),
        "ffmpeg -hide_banner -loglevel warning "
        "-f rawvideo -pixel_format bgr24 -video_size %dx%d -framerate %.1f "
        "-i pipe:0 "
        "-c:v %s -rc_mode CBR -b:v %s -g %d -bf 0 "
        "-flags +low_delay -fflags +genpts "
        "-f flv %s",
        width, height, fps,
        encoder.c_str(), cfg.bitrate.c_str(), gop,
        cfg.rtmp_url.c_str());
    } else {
      fprintf(stderr, "[STREAM] No output URL configured\n");
      return false;
    }

    printf("[STREAM] CMD: %s\n", cmd);
    pipe_ = popen(cmd, "w");
    if (!pipe_) {
      fprintf(stderr, "[STREAM] Failed to open ffmpeg pipe\n");
      return false;
    }
    return true;
  }

  bool Write(const cv::Mat &frame) {
    if (!pipe_) return false;
    cv::Mat cont = frame.isContinuous() ? frame : frame.clone();
    size_t sz = cont.total() * cont.elemSize();
    return fwrite(cont.data, 1, sz, pipe_) == sz;
  }

  void Close() {
    if (pipe_) {
      pclose(pipe_);
      pipe_ = nullptr;
    }
  }

  bool IsOpen() const { return pipe_ != nullptr; }

private:
  FILE *pipe_ = nullptr;
};
