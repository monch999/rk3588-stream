#include "channel_pipeline.h"
#include "algorithm_interface.h"
#include "im2d.hpp"
#include "RgaUtils.h"
#include <cstdio>
#include <chrono>
#include <map>

// ==================== 构造 / 析构 ====================
ChannelPipeline::ChannelPipeline(const Config& cfg) : cfg_(cfg) {
  need_process_ = cfg_.enable_processed && cfg_.processor != nullptr;

  if (need_process_) {
    video_ = std::make_unique<VideoFile>(cfg_.input_file);
    frame_w_ = video_->get_frame_width();
    frame_h_ = video_->get_frame_height();
    skip_ratio_ = std::max(1, static_cast<int>(
        std::round(cfg_.raw_fps / cfg_.processed_fps)));
  }

  printf("[%-6s] Init OK: mode=%s\n", cfg_.name.c_str(),
         need_process_ ? "Processed(zero-copy)" : "Raw-Only");
}

ChannelPipeline::~ChannelPipeline() { Stop(); }

// ==================== Decoder NV12 -> 自管 NV12 ====================
// RGA 硬拷贝, decoder buffer 立即归还给 decoder pool
std::shared_ptr<DrmFrame> ChannelPipeline::CopyToOwnedNv12(
    const std::shared_ptr<DrmFrame>& src) {
  if (!src || src->fd < 0) return nullptr;

  auto buf = DrmAllocator::Instance().Acquire(
      DrmAllocator::NV12, src->width, src->height);
  if (!buf) {
    fprintf(stderr, "[%-6s] alloc owned NV12 failed\n", cfg_.name.c_str());
    return nullptr;
  }

  // 用 importbuffer_fd 走 IOMMU 路径 (RK3588 RGA3 不支持 >4GB 物理地址)
  // 注意: im_handle_param_t 描述的是 buffer 物理尺寸, 应传 stride 而不是 width/height
  im_handle_param_t src_param = {(uint32_t)src->h_stride,
                                  (uint32_t)src->v_stride,
                                  RK_FORMAT_YCbCr_420_SP};
  im_handle_param_t dst_param = {(uint32_t)buf->h_stride,
                                  (uint32_t)buf->v_stride,
                                  RK_FORMAT_YCbCr_420_SP};

  rga_buffer_handle_t src_h = importbuffer_fd(src->fd, &src_param);
  rga_buffer_handle_t dst_h = importbuffer_fd(buf->fd, &dst_param);

  if (src_h == 0 || dst_h == 0) {
    fprintf(stderr, "[%-6s] importbuffer_fd failed: src_h=%lu dst_h=%lu\n",
            cfg_.name.c_str(), (unsigned long)src_h, (unsigned long)dst_h);
    if (src_h) releasebuffer_handle(src_h);
    if (dst_h) releasebuffer_handle(dst_h);
    return nullptr;
  }

  rga_buffer_t src_rga = wrapbuffer_handle(src_h,
      src->width, src->height, RK_FORMAT_YCbCr_420_SP);
  src_rga.wstride = src->h_stride;
  src_rga.hstride = src->v_stride;

  rga_buffer_t dst_rga = wrapbuffer_handle(dst_h,
      buf->width, buf->height, RK_FORMAT_YCbCr_420_SP);
  dst_rga.wstride = buf->h_stride;
  dst_rga.hstride = buf->v_stride;

  // 诊断: 首次调用时打印 RGA 参数
  static thread_local int dbg_count = 0;
  if (dbg_count < 2) {
    fprintf(stderr,
        "[%-6s][RGA-DBG] src: fd=%d handle=%lu w=%d h=%d ws=%d hs=%d\n"
        "[%-6s][RGA-DBG] dst: fd=%d handle=%lu w=%d h=%d ws=%d hs=%d\n",
        cfg_.name.c_str(), src->fd, (unsigned long)src_h,
        src->width, src->height, src_rga.wstride, src_rga.hstride,
        cfg_.name.c_str(), buf->fd, (unsigned long)dst_h,
        buf->width, buf->height, dst_rga.wstride, dst_rga.hstride);
    dbg_count++;
  }

  im_rect src_rect = {0, 0, src->width, src->height};
  im_rect dst_rect = {0, 0, buf->width, buf->height};
  IM_STATUS st = improcess(src_rga, dst_rga, {}, src_rect, dst_rect, {}, IM_SYNC);

  releasebuffer_handle(src_h);
  releasebuffer_handle(dst_h);

  if (st != IM_STATUS_SUCCESS) {
    fprintf(stderr, "[%-6s] RGA NV12 copy failed: %s\n",
            cfg_.name.c_str(), imStrError(st));
    return nullptr;
  }

  buf->SyncBegin();  // RGA 写完, CPU 后续要读/写

  auto out = DrmFrame::FromAllocator(buf, DrmFrame::NV12);
  out->pts = src->pts;
  return out;
}

// ==================== 启动 ====================
void ChannelPipeline::Start() {
  if (cfg_.enable_raw) {
    StreamConfig sc;
    sc.rtsp_url = cfg_.raw_rtsp_url;
    sc.bitrate  = cfg_.bitrate;
    if (!raw_streamer_.OpenDirect(cfg_.input_file, cfg_.raw_fps, sc, cfg_.loop_video)) {
      fprintf(stderr, "[%-6s] Failed to open direct raw streamer\n", cfg_.name.c_str());
    } else {
      printf("[%-6s] Raw stream (Direct) started: %s\n",
             cfg_.name.c_str(), cfg_.raw_rtsp_url.c_str());
    }
    return;
  }

  if (!need_process_) {
    printf("[%-6s] No processor, nothing to start\n", cfg_.name.c_str());
    return;
  }

  // ---- Muxer ----
  muxer_ = std::make_unique<RTSPMuxer>();
  RTSPMuxer::Config mcfg;
  mcfg.rtsp_url = cfg_.processed_rtsp_url;
  mcfg.codec    = cfg_.codec;
  mcfg.width    = frame_w_;
  mcfg.height   = frame_h_;
  mcfg.fps      = cfg_.processed_fps;
  mcfg.bitrate  = cfg_.bitrate;
  if (!muxer_->Open(mcfg)) {
    fprintf(stderr, "[%-6s] Failed to open RTSPMuxer\n", cfg_.name.c_str());
    return;
  }

  // ---- Encoder ----
  encoder_ = std::make_unique<MppEncoder>();
  MppEncoder::Config ecfg;
  ecfg.width   = frame_w_;
  ecfg.height  = frame_h_;
  ecfg.fps     = cfg_.processed_fps;
  ecfg.codec   = cfg_.codec;
  ecfg.bitrate = cfg_.bitrate;
  ecfg.gop     = static_cast<int>(cfg_.processed_fps);

  auto* muxer_ptr = muxer_.get();
  EncodedPacketCallback enc_cb = [muxer_ptr](const uint8_t* data, size_t size,
                                              int64_t pts, bool is_key) {
    muxer_ptr->WritePacket(data, size, pts, is_key);
  };

  if (!encoder_->Open(ecfg, std::move(enc_cb))) {
    fprintf(stderr, "[%-6s] Failed to open MppEncoder\n", cfg_.name.c_str());
    muxer_->Close();
    return;
  }

  running_ = true;

  reader_thread_ = std::thread(&ChannelPipeline::ReaderLoop, this);
  int nw = cfg_.processor->NumWorkers();
  for (int i = 0; i < nw; i++)
    process_threads_.emplace_back(&ChannelPipeline::ProcessWorker, this, i);
  processed_writer_thread_ = std::thread(&ChannelPipeline::ProcessedWriterLoop, this);

  printf("[%-6s] AI Pipeline (zero-copy) Started: reader(1) + process(%d) + writer(1)\n",
         cfg_.name.c_str(), nw);
}

// ==================== 停止 ====================
void ChannelPipeline::Stop() {
  raw_streamer_.Close();

  if (!running_.exchange(false)) return;

  process_input_queue_.shutdown();
  processed_queue_.shutdown();

  if (reader_thread_.joinable()) reader_thread_.join();
  for (auto& t : process_threads_) if (t.joinable()) t.join();
  if (processed_writer_thread_.joinable()) processed_writer_thread_.join();

  if (encoder_) encoder_->Close();
  if (muxer_)   muxer_->Close();
}

// ==================== Reader: 解码 + RGA copy + 入队 ====================
void ChannelPipeline::ReaderLoop() {
  int64_t file_seq = 0;
  auto start_time = std::chrono::steady_clock::now();
  auto interval = std::chrono::nanoseconds(static_cast<int64_t>(1e9 / cfg_.raw_fps));

  while (running_) {
    auto src_drm = video_->GetNextDrmFrame();
    if (!src_drm) {
      if (cfg_.loop_video) {
        video_.reset();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        video_ = std::make_unique<VideoFile>(cfg_.input_file);
        file_seq = 0;
        start_time = std::chrono::steady_clock::now();
        printf("[%-6s] Loop restart\n", cfg_.name.c_str());
        continue;
      }
      break;
    }

    // 跳帧 (按 raw_fps -> processed_fps 比例)
    if (file_seq % skip_ratio_ != 0) {
      file_seq++;
      // src_drm 离开作用域, decoder buffer 立即归还
      continue;
    }

    // 帧率节流
    auto target = start_time + interval * file_seq;
    auto now = std::chrono::steady_clock::now();
    if (now > target + std::chrono::milliseconds(100)) {
      start_time = now - interval * file_seq;
      target = now;
    }
    std::this_thread::sleep_until(target);

    // ★ 关键: RGA copy 到自管 NV12 buffer, decoder buffer 立即释放
    auto owned_nv12 = CopyToOwnedNv12(src_drm);
    src_drm.reset();  // 显式释放 decoder frame (回归 decoder pool)

    if (!owned_nv12) {
      file_seq++;
      continue;
    }

    int64_t pts = SharedClock::FramePtsNs(total_proc_seq_++, cfg_.processed_fps);
    owned_nv12->pts = pts;

    TimestampedFrame tf{next_seq_, pts, std::move(owned_nv12)};
    if (!process_input_queue_.push(std::move(tf))) break;
    next_seq_++;
    file_seq++;
  }
}

// ==================== ProcessWorker: 算法处理 (in-place) ====================
void ChannelPipeline::ProcessWorker(int worker_id) {
  while (true) {
    TimestampedFrame tf;
    if (!process_input_queue_.pop(tf)) break;

    if (tf.frame) {
      // in-place 处理: 算法直接修改 frame 的 Y 平面
      bool ok = cfg_.processor->Process(worker_id, tf.frame);
      if (!ok) {
        // Process 返回 false 表示丢弃
        tf.frame.reset();
      }
    }

    // 不论是否处理成功都入队 (空帧用于保持 seq 连续, writer 会跳过)
    if (!processed_queue_.push(std::move(tf))) break;
  }
}

// ==================== ProcessedWriter: 轻量 reorder + encoder ====================
void ChannelPipeline::ProcessedWriterLoop() {
  // 轻量 reorder buffer: 容量 = NPU worker 数 × 2 已足够
  // 不会无限缓存 (process_input_queue/processed_queue 容量已限制总在途帧数)
  std::map<int64_t, TimestampedFrame> reorder_buf;
  int64_t expected_seq = 0;
  const int nw = cfg_.processor ? cfg_.processor->NumWorkers() : 1;
  const size_t MAX_REORDER = static_cast<size_t>(nw * 2 + 2);

  auto emit = [&](TimestampedFrame& f) {
    if (!f.frame) {
      processed_frames_++;
      return;
    }
    if (cfg_.clock)
      cfg_.clock->WaitUntilPts(f.pts_ns);

    // ★ encoder/muxer PTS 必须与 muxer time_base 对齐 (time_base = 1/fps)
    //   即: PTS 单位是"帧数", 等同于 seq (从 0 起递增)
    int64_t enc_pts = f.seq;

    if (encoder_) {
      encoder_->EncodeFd(f.frame->fd,
                         f.frame->h_stride, f.frame->v_stride,
                         enc_pts);
    }
    if (cfg_.enable_display) {
      auto bgr = f.frame->ToBgrMat();
      if (bgr) {
        std::lock_guard<std::mutex> lk(display_mtx_);
        display_frame_ = std::move(*bgr);
      }
    }
    processed_frames_++;
  };

  while (true) {
    TimestampedFrame tf;
    if (!processed_queue_.pop(tf)) break;

    reorder_buf.emplace(tf.seq, std::move(tf));

    // 水位线: 缓冲过多时强行跳过缺失 seq (避免无限等待)
    if (reorder_buf.size() > MAX_REORDER) {
      auto it = reorder_buf.begin();
      if (it->first > expected_seq) {
        fprintf(stderr, "[%-6s] reorder skip missing seq %ld -> %ld\n",
                cfg_.name.c_str(), expected_seq, it->first);
        expected_seq = it->first;
      }
    }

    // 按 seq 升序输出连续段
    auto it = reorder_buf.find(expected_seq);
    while (it != reorder_buf.end()) {
      emit(it->second);
      reorder_buf.erase(it);
      expected_seq++;
      it = reorder_buf.find(expected_seq);
    }
  }

  // 退出时刷出剩余帧 (按 seq 顺序)
  for (auto& kv : reorder_buf) emit(kv.second);
  reorder_buf.clear();
}
