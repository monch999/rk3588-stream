#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_buffer.h>

using EncodedPacketCallback =
    std::function<void(const uint8_t* data, size_t size,
                       int64_t pts, bool is_key)>;

class MppEncoder {
public:
  struct Config {
    int         width   = 1920;
    int         height  = 1080;
    double      fps     = 30.0;
    std::string codec   = "h264";
    std::string bitrate = "4M";
    int         gop     = 60;
    int         profile = 100;
  };

  MppEncoder();
  ~MppEncoder();

  MppEncoder(const MppEncoder&) = delete;
  MppEncoder& operator=(const MppEncoder&) = delete;

  bool Open(const Config& cfg, EncodedPacketCallback cb);
  void Close();

  // BGR cv::Mat -> RGA NV12 -> MPP encode (兼容路径)
  bool Encode(const cv::Mat& bgr, int64_t pts);

  // ★ Zero-copy: 直接吃外部 NV12 DRM fd
  // h_stride / v_stride 必须与 encoder 配置一致 (MPP_ALIGN(width, 16) 等)
  bool EncodeFd(int drm_fd, int h_stride, int v_stride, int64_t pts);

private:
  bool ConfigEncoder();
  bool PrepareBuffers();
  bool EncodeFrame(MppBuffer buf, int64_t pts);
  void BgrToNv12(const cv::Mat& bgr, uint8_t* nv12);
  uint64_t ParseBitrate(const std::string& s);

  MppCtx           mpp_ctx_   = nullptr;
  MppApi*          mpp_mpi_   = nullptr;
  MppBufferGroup   buf_grp_   = nullptr;
  MppBuffer        frame_buf_ = nullptr;

  Config           cfg_;
  MppCodingType    codec_type_ = MPP_VIDEO_CodingAVC;
  int              hor_stride_ = 0;
  int              ver_stride_ = 0;
  size_t           frame_size_ = 0;

  EncodedPacketCallback callback_;
  bool opened_ = false;
};
