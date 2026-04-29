#include "mpp_encoder.h"
#include "im2d.hpp"
#include "RgaUtils.h"
#include <cstdio>
#include <cstring>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
#include <rockchip/rk_venc_cfg.h>
#include <rockchip/rk_venc_cmd.h>

#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

MppEncoder::MppEncoder() = default;
MppEncoder::~MppEncoder() { Close(); }

bool MppEncoder::Open(const Config& cfg, EncodedPacketCallback cb) {
  if (opened_) Close();
  cfg_      = cfg;
  callback_ = std::move(cb);

  codec_type_ = (cfg_.codec == "h265" || cfg_.codec == "hevc")
                  ? MPP_VIDEO_CodingHEVC : MPP_VIDEO_CodingAVC;

  hor_stride_ = MPP_ALIGN(cfg_.width, 16);
  ver_stride_ = MPP_ALIGN(cfg_.height, 16);
  frame_size_ = (size_t)hor_stride_ * ver_stride_ * 3 / 2;

  if (mpp_create(&mpp_ctx_, &mpp_mpi_) != MPP_OK) {
    fprintf(stderr, "[ENC   ] mpp_create failed\n");
    return false;
  }
  if (mpp_init(mpp_ctx_, MPP_CTX_ENC, codec_type_) != MPP_OK) {
    fprintf(stderr, "[ENC   ] mpp_init failed\n");
    return false;
  }
  if (!ConfigEncoder()) return false;
  if (!PrepareBuffers()) return false;

  opened_ = true;
  printf("[ENC   ] Open OK: %dx%d (stride %dx%d), %s, bitrate=%s, gop=%d\n",
         cfg_.width, cfg_.height, hor_stride_, ver_stride_,
         cfg_.codec.c_str(), cfg_.bitrate.c_str(), cfg_.gop);
  return true;
}

void MppEncoder::Close() {
  if (!opened_) return;
  opened_ = false;

  if (frame_buf_) { mpp_buffer_put(frame_buf_); frame_buf_ = nullptr; }
  if (buf_grp_)   { mpp_buffer_group_put(buf_grp_); buf_grp_ = nullptr; }
  if (mpp_mpi_)   { mpp_mpi_->reset(mpp_ctx_); }
  if (mpp_ctx_)   {
    mpp_destroy(mpp_ctx_);
    mpp_ctx_ = nullptr;
    mpp_mpi_ = nullptr;
  }
  printf("[ENC   ] Closed\n");
}

bool MppEncoder::ConfigEncoder() {
  MppEncCfg enc_cfg = nullptr;
  mpp_enc_cfg_init(&enc_cfg);
  mpp_mpi_->control(mpp_ctx_, MPP_ENC_GET_CFG, enc_cfg);

  uint64_t bps = ParseBitrate(cfg_.bitrate);
  int fps_num  = static_cast<int>(cfg_.fps);
  int gop      = cfg_.gop > 0 ? cfg_.gop : fps_num * 2;

  mpp_enc_cfg_set_s32(enc_cfg, "prep:width",      cfg_.width);
  mpp_enc_cfg_set_s32(enc_cfg, "prep:height",     cfg_.height);
  mpp_enc_cfg_set_s32(enc_cfg, "prep:hor_stride", hor_stride_);
  mpp_enc_cfg_set_s32(enc_cfg, "prep:ver_stride", ver_stride_);
  mpp_enc_cfg_set_s32(enc_cfg, "prep:format",     MPP_FMT_YUV420SP);

  mpp_enc_cfg_set_s32(enc_cfg, "rc:mode",          MPP_ENC_RC_MODE_CBR);
  mpp_enc_cfg_set_s32(enc_cfg, "rc:fps_in_flex",   0);
  mpp_enc_cfg_set_s32(enc_cfg, "rc:fps_in_num",    fps_num);
  mpp_enc_cfg_set_s32(enc_cfg, "rc:fps_in_denorm", 1);
  mpp_enc_cfg_set_s32(enc_cfg, "rc:fps_out_flex",  0);
  mpp_enc_cfg_set_s32(enc_cfg, "rc:fps_out_num",   fps_num);
  mpp_enc_cfg_set_s32(enc_cfg, "rc:fps_out_denorm",1);
  mpp_enc_cfg_set_s32(enc_cfg, "rc:gop",           gop);
  mpp_enc_cfg_set_s32(enc_cfg, "rc:bps_target",    (int)bps);
  mpp_enc_cfg_set_s32(enc_cfg, "rc:bps_max",       (int)(bps * 3 / 2));
  mpp_enc_cfg_set_s32(enc_cfg, "rc:bps_min",       (int)(bps / 2));

  if (codec_type_ == MPP_VIDEO_CodingAVC) {
    mpp_enc_cfg_set_s32(enc_cfg, "codec:id",       MPP_VIDEO_CodingAVC);
    mpp_enc_cfg_set_s32(enc_cfg, "h264:profile",   cfg_.profile);
    mpp_enc_cfg_set_s32(enc_cfg, "h264:level",     41);
    mpp_enc_cfg_set_s32(enc_cfg, "h264:cabac_en",  1);
    mpp_enc_cfg_set_s32(enc_cfg, "h264:cabac_idc", 0);
  } else {
    mpp_enc_cfg_set_s32(enc_cfg, "codec:id", MPP_VIDEO_CodingHEVC);
  }

  MPP_RET ret = mpp_mpi_->control(mpp_ctx_, MPP_ENC_SET_CFG, enc_cfg);
  mpp_enc_cfg_deinit(enc_cfg);
  if (ret != MPP_OK) {
    fprintf(stderr, "[ENC   ] SET_CFG failed: %d\n", ret);
    return false;
  }

  MppEncHeaderMode hm = MPP_ENC_HEADER_MODE_EACH_IDR;
  mpp_mpi_->control(mpp_ctx_, MPP_ENC_SET_HEADER_MODE, &hm);
  return true;
}

bool MppEncoder::PrepareBuffers() {
  if (mpp_buffer_group_get_internal(&buf_grp_, MPP_BUFFER_TYPE_DRM) != MPP_OK) {
    fprintf(stderr, "[ENC   ] buffer group create failed\n");
    return false;
  }
  if (mpp_buffer_get(buf_grp_, &frame_buf_, frame_size_) != MPP_OK) {
    fprintf(stderr, "[ENC   ] buffer alloc failed (size=%zu)\n", frame_size_);
    return false;
  }
  return true;
}

void MppEncoder::BgrToNv12(const cv::Mat& bgr, uint8_t* nv12) {
  rga_buffer_t src = wrapbuffer_virtualaddr(
      bgr.data, bgr.cols, bgr.rows, RK_FORMAT_BGR_888);
  rga_buffer_t dst = wrapbuffer_virtualaddr(
      nv12, cfg_.width, cfg_.height, RK_FORMAT_YCbCr_420_SP);
  dst.wstride = hor_stride_;
  dst.hstride = ver_stride_;

  IM_STATUS st = imcvtcolor(src, dst, RK_FORMAT_BGR_888, RK_FORMAT_YCbCr_420_SP);
  if (st != IM_STATUS_SUCCESS) {
    fprintf(stderr, "[ENC   ] RGA BGR->NV12 failed: %s\n", imStrError(st));
  }
}

bool MppEncoder::EncodeFrame(MppBuffer buf, int64_t pts) {
  MppFrame frame = nullptr;
  mpp_frame_init(&frame);
  mpp_frame_set_width(frame, cfg_.width);
  mpp_frame_set_height(frame, cfg_.height);
  mpp_frame_set_hor_stride(frame, hor_stride_);
  mpp_frame_set_ver_stride(frame, ver_stride_);
  mpp_frame_set_fmt(frame, MPP_FMT_YUV420SP);
  mpp_frame_set_buffer(frame, buf);
  mpp_frame_set_pts(frame, pts);

  if (mpp_mpi_->encode_put_frame(mpp_ctx_, frame) != MPP_OK) {
    mpp_frame_deinit(&frame);
    return false;
  }

  MppPacket pkt = nullptr;
  if (mpp_mpi_->encode_get_packet(mpp_ctx_, &pkt) != MPP_OK || !pkt) {
    mpp_frame_deinit(&frame);
    return false;
  }

  auto* data = static_cast<const uint8_t*>(mpp_packet_get_data(pkt));
  size_t len = mpp_packet_get_length(pkt);

  bool is_key = false;
  MppMeta meta = mpp_packet_get_meta(pkt);
  if (meta) {
    RK_S32 v = 0;
    mpp_meta_get_s32(meta, KEY_OUTPUT_INTRA, &v);
    is_key = (v != 0);
  }

  if (callback_ && len > 0)
    callback_(data, len, pts, is_key);

  mpp_packet_deinit(&pkt);
  mpp_frame_deinit(&frame);
  return true;
}

// ==================== 兼容路径: BGR cv::Mat ====================
bool MppEncoder::Encode(const cv::Mat& bgr, int64_t pts) {
  if (!opened_ || bgr.empty()) return false;
  auto* p = static_cast<uint8_t*>(mpp_buffer_get_ptr(frame_buf_));
  BgrToNv12(bgr, p);
  return EncodeFrame(frame_buf_, pts);
}

// ==================== Zero-copy: 直接吃外部 fd ====================
bool MppEncoder::EncodeFd(int drm_fd, int h_stride, int v_stride, int64_t pts) {
  if (!opened_ || drm_fd < 0) return false;

  // stride 必须与 encoder 配置一致, 否则编码出图会错位
  if (h_stride != hor_stride_ || v_stride != ver_stride_) {
    static int warn = 0;
    if (++warn <= 3) {
      fprintf(stderr, "[ENC   ] stride mismatch: in=%dx%d, cfg=%dx%d\n",
              h_stride, v_stride, hor_stride_, ver_stride_);
    }
    return false;
  }

  MppBufferInfo info;
  memset(&info, 0, sizeof(info));
  info.type  = MPP_BUFFER_TYPE_DRM;
  info.fd    = drm_fd;
  info.size  = (size_t)h_stride * v_stride * 3 / 2;

  MppBuffer buf = nullptr;
  if (mpp_buffer_import(&buf, &info) != MPP_OK) {
    fprintf(stderr, "[ENC   ] mpp_buffer_import fd=%d failed\n", drm_fd);
    return false;
  }

  bool ok = EncodeFrame(buf, pts);
  mpp_buffer_put(buf);  // 仅释放 import 引用, 原 buffer 由调用方管理
  return ok;
}

uint64_t MppEncoder::ParseBitrate(const std::string& s) {
  uint64_t val = 0;
  char unit = 0;
  sscanf(s.c_str(), "%lu%c", &val, &unit);
  if (unit == 'M' || unit == 'm') return val * 1000000;
  if (unit == 'K' || unit == 'k') return val * 1000;
  return val;
}
