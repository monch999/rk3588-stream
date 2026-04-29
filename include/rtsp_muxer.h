#pragma once
#include <cstdint>
#include <string>
#include <mutex>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

class RTSPMuxer {
public:
  struct Config {
    std::string rtsp_url;
    std::string rtmp_url;
    std::string codec   = "h264";
    int         width   = 1920;
    int         height  = 1080;
    double      fps     = 30.0;
    std::string bitrate = "4M";
  };

  RTSPMuxer();
  ~RTSPMuxer();

  RTSPMuxer(const RTSPMuxer &) = delete;
  RTSPMuxer &operator=(const RTSPMuxer &) = delete;

  bool Open(const Config &cfg);
  void Close();

  // 从 MppEncoder callback 调用, 线程安全
  void WritePacket(const uint8_t *data, size_t size,
                   int64_t pts, bool is_key);

private:
  struct OutputCtx {
    AVFormatContext *fmt_ctx = nullptr;
    int              stream_idx = -1;
    std::string      url;
    std::string      format;
    bool             header_written = false;
  };

  bool InitOutput(OutputCtx &out, const std::string &url,
                  const std::string &format, const Config &cfg);
  bool LazyInit(const uint8_t *data, size_t size);
  void CloseOutput(OutputCtx &out);
  void WriteToOutput(OutputCtx &out, const uint8_t *data, size_t size,
                     int64_t pts, bool is_key);

  OutputCtx rtsp_;
  OutputCtx rtmp_;
  Config    cfg_;

  AVCodecParameters *codec_par_ = nullptr;
  int64_t frame_count_ = 0;
  AVRational time_base_ = {1, 30};

  std::mutex write_mutex_;
  bool opened_      = false;
  bool initialized_ = false;   // 延迟初始化: 等第一个 IDR
};
