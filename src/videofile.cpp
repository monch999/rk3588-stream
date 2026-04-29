#include "videofile.h"
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>

// ==================== 构造 / 析构 ====================
VideoFile::VideoFile(const std::string &path) {
  if (!InitDemuxer(path)) {
    fprintf(stderr, "[VIDEO ] Failed to init demuxer: %s\n", path.c_str());
    return;
  }
  if (!InitMppDecoder()) {
    fprintf(stderr, "[VIDEO ] Failed to init MPP decoder\n");
    return;
  }
  printf("[VIDEO ] MPP HW decode OK: %dx%d @ %.1ffps, codec=%s\n",
         width_, height_, fps_,
         codec_type_ == MPP_VIDEO_CodingHEVC ? "H265" : "H264");
}

VideoFile::~VideoFile() {
  if (bsf_ctx_) {
    av_bsf_free(&bsf_ctx_);
    bsf_ctx_ = nullptr;
  }
  if (mpp_mpi_) {
    mpp_mpi_->reset(mpp_ctx_);
  }
  if (mpp_ctx_) {
    mpp_destroy(mpp_ctx_);
    mpp_ctx_ = nullptr;
    mpp_mpi_ = nullptr;
  }
  if (fmt_ctx_) {
    avformat_close_input(&fmt_ctx_);
  }
}

// ==================== FFmpeg 解封装初始化 ====================
bool VideoFile::InitDemuxer(const std::string &path) {
  int ret = avformat_open_input(&fmt_ctx_, path.c_str(), nullptr, nullptr);
  if (ret < 0) {
    char errbuf[128];
    av_strerror(ret, errbuf, sizeof(errbuf));
    fprintf(stderr, "[VIDEO ] avformat_open_input failed: %s\n", errbuf);
    return false;
  }

  ret = avformat_find_stream_info(fmt_ctx_, nullptr);
  if (ret < 0) return false;

  AVCodecParameters *par = nullptr;
  for (unsigned i = 0; i < fmt_ctx_->nb_streams; i++) {
    par = fmt_ctx_->streams[i]->codecpar;
    if (par->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_idx_ = i;
      width_     = par->width;
      height_    = par->height;

      AVRational rate = fmt_ctx_->streams[i]->avg_frame_rate;
      if (rate.num > 0 && rate.den > 0)
        fps_ = av_q2d(rate);

      switch (par->codec_id) {
        case AV_CODEC_ID_H264:  codec_type_ = MPP_VIDEO_CodingAVC;  break;
        case AV_CODEC_ID_HEVC:  codec_type_ = MPP_VIDEO_CodingHEVC; break;
        case AV_CODEC_ID_VP9:   codec_type_ = MPP_VIDEO_CodingVP9;  break;
        case AV_CODEC_ID_AV1:   codec_type_ = MPP_VIDEO_CodingAV1;  break;
        default:
          fprintf(stderr, "[VIDEO ] Unsupported codec: %d\n", par->codec_id);
          return false;
      }
      break;
    }
  }

  if (video_idx_ < 0) {
    fprintf(stderr, "[VIDEO ] No video stream found\n");
    return false;
  }

  // MP4 H264/HEVC: AVCC → Annex-B (MPP 需要 start code 格式)
  if (codec_type_ == MPP_VIDEO_CodingAVC || codec_type_ == MPP_VIDEO_CodingHEVC) {
    const char *filter = (codec_type_ == MPP_VIDEO_CodingAVC)
                          ? "h264_mp4toannexb" : "hevc_mp4toannexb";
    const AVBitStreamFilter *bsf = av_bsf_get_by_name(filter);
    if (bsf) {
      av_bsf_alloc(bsf, &bsf_ctx_);
      avcodec_parameters_copy(bsf_ctx_->par_in, par);
      av_bsf_init(bsf_ctx_);
      printf("[VIDEO ] BSF %s enabled\n", filter);
    } else {
      fprintf(stderr, "[VIDEO ] BSF %s not found!\n", filter);
    }
  }

  return true;
}

// ==================== MPP 解码器初始化 ====================
bool VideoFile::InitMppDecoder() {
  MPP_RET ret = mpp_create(&mpp_ctx_, &mpp_mpi_);
  if (ret != MPP_OK) {
    fprintf(stderr, "[VIDEO ] mpp_create failed: %d\n", ret);
    return false;
  }

  RK_U32 split = 1;
  ret = mpp_mpi_->control(mpp_ctx_, MPP_DEC_SET_PARSER_SPLIT_MODE, &split);
  if (ret != MPP_OK) {
    fprintf(stderr, "[VIDEO ] Set split mode failed: %d\n", ret);
    return false;
  }

  ret = mpp_init(mpp_ctx_, MPP_CTX_DEC, codec_type_);
  if (ret != MPP_OK) {
    fprintf(stderr, "[VIDEO ] mpp_init DEC failed: %d\n", ret);
    return false;
  }

  return true;
}

// ==================== 发送压缩包给解码器 ====================
bool VideoFile::SendPacket(AVPacket *pkt) {
  if (bsf_ctx_) {
    AVPacket *filtered = av_packet_clone(pkt);
    if (av_bsf_send_packet(bsf_ctx_, filtered) < 0) {
      av_packet_free(&filtered);
      return false;
    }
    while (av_bsf_receive_packet(bsf_ctx_, filtered) == 0) {
      MppPacket mpp_pkt = nullptr;
      mpp_packet_init(&mpp_pkt, filtered->data, filtered->size);
      mpp_packet_set_pts(mpp_pkt, filtered->pts);
      MPP_RET ret;
      do {
        ret = mpp_mpi_->decode_put_packet(mpp_ctx_, mpp_pkt);
        if (ret == MPP_ERR_BUFFER_FULL) {
          // 排空解码器，处理 info_change 等
          MppFrame frm = nullptr;
          mpp_mpi_->decode_get_frame(mpp_ctx_, &frm);
          if (frm) {
            if (mpp_frame_get_info_change(frm)) {
              width_  = mpp_frame_get_width(frm);
              height_ = mpp_frame_get_height(frm);
              printf("[VIDEO ] Info change in SendPacket: %dx%d\n", width_, height_);
              mpp_mpi_->control(mpp_ctx_, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
            }
            mpp_frame_deinit(&frm);
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      } while (ret == MPP_ERR_BUFFER_FULL);
      mpp_packet_deinit(&mpp_pkt);
      av_packet_unref(filtered);
    }
    av_packet_free(&filtered);
    return true;
  }

  // 原始路径
  MppPacket mpp_pkt = nullptr;
  mpp_packet_init(&mpp_pkt, pkt->data, pkt->size);
  mpp_packet_set_pts(mpp_pkt, pkt->pts);
  MPP_RET ret;
  do {
    ret = mpp_mpi_->decode_put_packet(mpp_ctx_, mpp_pkt);
    if (ret == MPP_ERR_BUFFER_FULL) {
      MppFrame frm = nullptr;
      mpp_mpi_->decode_get_frame(mpp_ctx_, &frm);
      if (frm) {
        if (mpp_frame_get_info_change(frm)) {
          mpp_mpi_->control(mpp_ctx_, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
        }
        mpp_frame_deinit(&frm);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } while (ret == MPP_ERR_BUFFER_FULL);
  mpp_packet_deinit(&mpp_pkt);
  return ret == MPP_OK;
}

// ==================== 从解码器取一帧 ====================
std::shared_ptr<DrmFrame> VideoFile::ReceiveFrame() {
  MppFrame mpp_frm = nullptr;
  MPP_RET ret = mpp_mpi_->decode_get_frame(mpp_ctx_, &mpp_frm);

  if (ret != MPP_OK || !mpp_frm)
    return nullptr;

  if (mpp_frame_get_eos(mpp_frm)) {
    eof_ = true;
    mpp_frame_deinit(&mpp_frm);
    return nullptr;
  }

  if (mpp_frame_get_info_change(mpp_frm)) {
    width_  = mpp_frame_get_width(mpp_frm);
    height_ = mpp_frame_get_height(mpp_frm);
    printf("[VIDEO ] Info change: %dx%d\n", width_, height_);
    mpp_mpi_->control(mpp_ctx_, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
    mpp_frame_deinit(&mpp_frm);
    return nullptr;
  }

  if (mpp_frame_get_errinfo(mpp_frm) || mpp_frame_get_discard(mpp_frm)) {
    mpp_frame_deinit(&mpp_frm);
    return nullptr;
  }

  return DrmFrame::FromMppFrame(mpp_frm);
}

// ==================== 获取下一帧 (DRM 零拷贝) ====================
std::shared_ptr<DrmFrame> VideoFile::GetNextDrmFrame() {
  if (eof_) return nullptr;

  AVPacket *pkt = av_packet_alloc();
  if (!pkt) return nullptr;

  while (!eof_) {
    // 轮询取帧
    for (int i = 0; i < 10; i++) {
      auto frame = ReceiveFrame();
      if (frame) {
        av_packet_free(&pkt);
        return frame;
      }
    }

    int ret = av_read_frame(fmt_ctx_, pkt);
    if (ret < 0) {
      // flush
      MppPacket flush_pkt = nullptr;
      mpp_packet_init(&flush_pkt, nullptr, 0);
      mpp_packet_set_eos(flush_pkt);
      mpp_mpi_->decode_put_packet(mpp_ctx_, flush_pkt);
      mpp_packet_deinit(&flush_pkt);

      for (int i = 0; i < 1000; i++) {
        auto f = ReceiveFrame();
        if (f) {
          av_packet_free(&pkt);
          return f;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      eof_ = true;
      av_packet_free(&pkt);
      return nullptr;
    }

    if (pkt->stream_index == video_idx_) {
      SendPacket(pkt);
    }
    av_packet_unref(pkt);
  }

  av_packet_free(&pkt);
  return nullptr;
}

// ==================== 兼容接口: cv::Mat BGR ====================
std::unique_ptr<cv::Mat> VideoFile::GetNextFrame() {
  auto drm = GetNextDrmFrame();
  if (!drm) return nullptr;
  return drm->ToBgrMat();
}
