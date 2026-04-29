#pragma once
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_buffer.h>
#include "drm_frame.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
}

class VideoFile {
public:
  explicit VideoFile(const std::string &path);
  ~VideoFile();

  VideoFile(const VideoFile &) = delete;
  VideoFile &operator=(const VideoFile &) = delete;

  std::shared_ptr<DrmFrame> GetNextDrmFrame();
  std::unique_ptr<cv::Mat> GetNextFrame();

  int get_frame_width()  const { return width_; }
  int get_frame_height() const { return height_; }
  double get_fps()       const { return fps_; }

private:
  bool InitDemuxer(const std::string &path);
  bool InitMppDecoder();
  bool SendPacket(AVPacket *pkt);
  std::shared_ptr<DrmFrame> ReceiveFrame();

  AVFormatContext *fmt_ctx_    = nullptr;
  AVBSFContext    *bsf_ctx_    = nullptr;
  int              video_idx_  = -1;
  MppCodingType    codec_type_ = MPP_VIDEO_CodingAVC;

  MppCtx  mpp_ctx_ = nullptr;
  MppApi *mpp_mpi_ = nullptr;

  int    width_  = 0;
  int    height_ = 0;
  double fps_    = 30.0;
  bool   eof_    = false;
};
