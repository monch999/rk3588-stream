#include "rtsp_muxer.h"
#include <cstdio>
#include <cstring>

// ==================== 构造 / 析构 ====================
RTSPMuxer::RTSPMuxer() = default;
RTSPMuxer::~RTSPMuxer() { Close(); }

static uint64_t ParseBitrate(const std::string &s) {
  uint64_t val = 0;
  char unit = 0;
  sscanf(s.c_str(), "%lu%c", &val, &unit);
  if (unit == 'M' || unit == 'm') return val * 1000000;
  if (unit == 'K' || unit == 'k') return val * 1000;
  return val;
}

// ==================== 从 Annex-B 流中提取 SPS/PPS ====================
static bool ExtractSpsPps(const uint8_t *data, size_t size,
                          const uint8_t **extra, size_t *extra_size) {
  size_t sps_start = 0;
  bool found_sps = false;

  for (size_t i = 0; i + 4 < size; i++) {
    bool is_start4 = (data[i] == 0 && data[i+1] == 0 &&
                      data[i+2] == 0 && data[i+3] == 1);
    bool is_start3 = (!is_start4 && data[i] == 0 &&
                      data[i+1] == 0 && data[i+2] == 1);
    if (!is_start4 && !is_start3) continue;

    int nal_off = is_start4 ? 4 : 3;
    if (i + nal_off >= size) break;
    uint8_t nal_type = data[i + nal_off] & 0x1F;

    if (nal_type == 7) {  // SPS
      if (!found_sps) { sps_start = i; found_sps = true; }
    } else if (nal_type == 8) {  // PPS
      // continue collecting
    } else if (found_sps) {
      // 非 SPS/PPS NAL → extradata 到此为止
      *extra = data + sps_start;
      *extra_size = i - sps_start;
      return true;
    }
  }

  if (found_sps) {
    *extra = data + sps_start;
    *extra_size = size - sps_start;
    return true;
  }
  return false;
}

// ==================== 初始化单路输出 ====================
bool RTSPMuxer::InitOutput(OutputCtx &out, const std::string &url,
                           const std::string &format, const Config &cfg) {
  out.url = url;
  out.format = format;

  int ret = avformat_alloc_output_context2(&out.fmt_ctx, nullptr,
                                           format.c_str(), url.c_str());
  if (ret < 0 || !out.fmt_ctx) {
    char err[128];
    av_strerror(ret, err, sizeof(err));
    fprintf(stderr, "[MUXER ] alloc output failed [%s]: %s\n", url.c_str(), err);
    return false;
  }

  AVStream *st = avformat_new_stream(out.fmt_ctx, nullptr);
  if (!st) {
    fprintf(stderr, "[MUXER ] new_stream failed [%s]\n", url.c_str());
    return false;
  }
  out.stream_idx = st->index;

  avcodec_parameters_copy(st->codecpar, codec_par_);
  st->time_base = time_base_;

  if (!(out.fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
    if (format != "rtsp") {
      ret = avio_open(&out.fmt_ctx->pb, url.c_str(), AVIO_FLAG_WRITE);
      if (ret < 0) {
        char err[128];
        av_strerror(ret, err, sizeof(err));
        fprintf(stderr, "[MUXER ] avio_open failed [%s]: %s\n", url.c_str(), err);
        return false;
      }
    }
  }

  AVDictionary *opts = nullptr;
  if (format == "rtsp")
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);

  ret = avformat_write_header(out.fmt_ctx, &opts);
  av_dict_free(&opts);

  if (ret < 0) {
    char err[128];
    av_strerror(ret, err, sizeof(err));
    fprintf(stderr, "[MUXER ] write_header failed [%s]: %s\n", url.c_str(), err);
    return false;
  }

  out.header_written = true;
  printf("[MUXER ] Output opened: %s\n", url.c_str());
  return true;
}

// ==================== 打开 (仅保存配置, 延迟初始化) ====================
bool RTSPMuxer::Open(const Config &cfg) {
  if (opened_) Close();

  cfg_ = cfg;
  time_base_ = {1, static_cast<int>(cfg.fps)};

  codec_par_ = avcodec_parameters_alloc();
  codec_par_->codec_type = AVMEDIA_TYPE_VIDEO;
  codec_par_->codec_id   = (cfg.codec == "h265" || cfg.codec == "hevc")
                            ? AV_CODEC_ID_HEVC : AV_CODEC_ID_H264;
  codec_par_->width      = cfg.width;
  codec_par_->height     = cfg.height;
  codec_par_->bit_rate   = ParseBitrate(cfg.bitrate);

  opened_ = true;
  initialized_ = false;
  printf("[MUXER ] Configured (waiting for first IDR): RTSP=%s, RTMP=%s\n",
         cfg.rtsp_url.empty() ? "(off)" : cfg.rtsp_url.c_str(),
         cfg.rtmp_url.empty() ? "(off)" : cfg.rtmp_url.c_str());
  return true;
}

// ==================== 延迟初始化: 从第一个 IDR 提取 extradata ====================
bool RTSPMuxer::LazyInit(const uint8_t *data, size_t size) {
  const uint8_t *extra = nullptr;
  size_t extra_size = 0;
  if (ExtractSpsPps(data, size, &extra, &extra_size)) {
    codec_par_->extradata = (uint8_t *)av_mallocz(extra_size + AV_INPUT_BUFFER_PADDING_SIZE);
    memcpy(codec_par_->extradata, extra, extra_size);
    codec_par_->extradata_size = static_cast<int>(extra_size);
    printf("[MUXER ] Extracted SPS/PPS: %zu bytes\n", extra_size);
  } else {
    fprintf(stderr, "[MUXER ] Warning: no SPS/PPS found in first IDR\n");
  }

  bool any_ok = false;
  if (!cfg_.rtsp_url.empty()) {
    if (InitOutput(rtsp_, cfg_.rtsp_url, "rtsp", cfg_))
      any_ok = true;
  }
  if (!cfg_.rtmp_url.empty()) {
    if (InitOutput(rtmp_, cfg_.rtmp_url, "flv", cfg_))
      any_ok = true;
  }

  if (!any_ok) {
    fprintf(stderr, "[MUXER ] No output initialized\n");
    return false;
  }

  initialized_ = true;
  return true;
}

// ==================== 写一路 ====================
void RTSPMuxer::WriteToOutput(OutputCtx &out, const uint8_t *data, size_t size,
                              int64_t pts, bool is_key) {
  if (!out.fmt_ctx || !out.header_written) return;

  AVPacket *pkt = av_packet_alloc();
  if (!pkt) return;

  av_new_packet(pkt, static_cast<int>(size));
  memcpy(pkt->data, data, size);

  pkt->stream_index = out.stream_idx;
  pkt->pts = pts;
  pkt->dts = pts;

  AVStream *st = out.fmt_ctx->streams[out.stream_idx];
  av_packet_rescale_ts(pkt, time_base_, st->time_base);

  if (is_key)
    pkt->flags |= AV_PKT_FLAG_KEY;

  int ret = av_interleaved_write_frame(out.fmt_ctx, pkt);
  if (ret < 0) {
    static int err_count = 0;
    if (++err_count <= 5) {
      char err[128];
      av_strerror(ret, err, sizeof(err));
      fprintf(stderr, "[MUXER ] write_frame [%s] failed: %s\n",
              out.url.c_str(), err);
    }
  }

  av_packet_free(&pkt);
}

// ==================== 写入 (线程安全) ====================
void RTSPMuxer::WritePacket(const uint8_t *data, size_t size,
                            int64_t pts, bool is_key) {
  if (!opened_) return;
  std::lock_guard<std::mutex> lock(write_mutex_);

  // 延迟初始化: 等第一个 keyframe 到来
  if (!initialized_) {
    if (!is_key) return;  // 丢弃 IDR 之前的帧
    if (!LazyInit(data, size)) {
      fprintf(stderr, "[MUXER ] LazyInit failed, disabling muxer\n");
      opened_ = false;
      return;
    }
  }

  WriteToOutput(rtsp_, data, size, pts, is_key);
  WriteToOutput(rtmp_, data, size, pts, is_key);
  frame_count_++;
}

// ==================== 关闭单路 ====================
void RTSPMuxer::CloseOutput(OutputCtx &out) {
  if (!out.fmt_ctx) return;

  if (out.header_written) {
    av_write_trailer(out.fmt_ctx);
    out.header_written = false;
  }

  if (out.fmt_ctx->pb &&
      !(out.fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
    avio_closep(&out.fmt_ctx->pb);
  }

  avformat_free_context(out.fmt_ctx);
  out.fmt_ctx = nullptr;
  out.stream_idx = -1;
  printf("[MUXER ] Closed: %s\n", out.url.c_str());
}

// ==================== 关闭 ====================
void RTSPMuxer::Close() {
  if (!opened_) return;
  opened_ = false;

  CloseOutput(rtsp_);
  CloseOutput(rtmp_);

  if (codec_par_) {
    avcodec_parameters_free(&codec_par_);
    codec_par_ = nullptr;
  }
  printf("[MUXER ] All closed, %ld packets written\n", (long)frame_count_);
}
