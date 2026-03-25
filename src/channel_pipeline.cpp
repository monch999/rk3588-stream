#include "channel_pipeline.h"
#include "algorithm_interface.h"
#include <map>
#include <cstdio>

// ==================== 构造 / 析构 ====================
ChannelPipeline::ChannelPipeline(const Config &cfg) : cfg_(cfg) {
  video_ = std::make_unique<VideoFile>(cfg_.input_file);
  frame_w_ = video_->get_frame_width();
  frame_h_ = video_->get_frame_height();

  // 计算帧抽取比: 每 skip_ratio_ 帧送一帧给处理管线
  skip_ratio_ = std::max(1, static_cast<int>(
      std::round(cfg_.raw_fps / cfg_.processed_fps)));

  printf("[%-6s] Init OK: %dx%d, raw=%.0ffps, proc=%.0ffps, skip=%d\n",
         cfg_.name.c_str(), frame_w_, frame_h_,
         cfg_.raw_fps, cfg_.processed_fps, skip_ratio_);
}

ChannelPipeline::~ChannelPipeline() { Stop(); }

// ==================== 启动 ====================
void ChannelPipeline::Start() {
  // 打开原始推流
  {
    StreamConfig sc;
    sc.rtsp_url = cfg_.raw_rtsp_url;
    sc.rtmp_url = cfg_.raw_rtmp_url;
    sc.bitrate  = cfg_.bitrate;
    if (!sc.rtsp_url.empty() || !sc.rtmp_url.empty()) {
      if (!raw_streamer_.Open(frame_w_, frame_h_, cfg_.raw_fps, sc)) {
        fprintf(stderr, "[%-6s] Failed to open raw streamer\n",
                cfg_.name.c_str());
      } else {
        printf("[%-6s] Raw stream -> %s\n",
               cfg_.name.c_str(), cfg_.raw_rtsp_url.c_str());
      }
    }
  }

  // 打开处理后推流
  if (cfg_.processor) {
    StreamConfig sc;
    sc.rtsp_url = cfg_.processed_rtsp_url;
    sc.rtmp_url = cfg_.processed_rtmp_url;
    sc.bitrate  = cfg_.bitrate;
    if (!sc.rtsp_url.empty() || !sc.rtmp_url.empty()) {
      if (!processed_streamer_.Open(frame_w_, frame_h_,
                                     cfg_.processed_fps, sc)) {
        fprintf(stderr, "[%-6s] Failed to open processed streamer\n",
                cfg_.name.c_str());
      } else {
        printf("[%-6s] Processed stream -> %s\n",
               cfg_.name.c_str(), cfg_.processed_rtsp_url.c_str());
      }
    }
  }

  running_ = true;

  // 启动线程
  reader_thread_     = std::thread(&ChannelPipeline::ReaderLoop, this);
  raw_writer_thread_ = std::thread(&ChannelPipeline::RawWriterLoop, this);

  if (cfg_.processor) {
    int nw = cfg_.processor->NumWorkers();
    for (int i = 0; i < nw; i++)
      process_threads_.emplace_back(
          &ChannelPipeline::ProcessWorker, this, i);
    processed_writer_thread_ =
        std::thread(&ChannelPipeline::ProcessedWriterLoop, this);

    printf("[%-6s] Started: reader(1) + raw_writer(1) + "
           "process(%d) + proc_writer(1)\n",
           cfg_.name.c_str(), nw);
  } else {
    printf("[%-6s] Started: reader(1) + raw_writer(1) (no processor)\n",
           cfg_.name.c_str());
  }
}

// ==================== 停止 ====================
void ChannelPipeline::Stop() {
  if (!running_.exchange(false)) return;

  raw_queue_.shutdown();
  process_input_queue_.shutdown();
  processed_queue_.shutdown();

  if (reader_thread_.joinable())     reader_thread_.join();
  if (raw_writer_thread_.joinable()) raw_writer_thread_.join();
  for (auto &t : process_threads_)
    if (t.joinable()) t.join();
  if (processed_writer_thread_.joinable())
    processed_writer_thread_.join();

  raw_streamer_.Close();
  processed_streamer_.Close();

  printf("[%-6s] Stopped. raw=%d, processed=%d\n",
         cfg_.name.c_str(), raw_frames_.load(), processed_frames_.load());
}

// ==================== Reader: 读帧 + 分发 ====================
// 读取视频帧，分配 PTS，分发到 raw 队列和 process 队列
void ChannelPipeline::ReaderLoop() {
  int64_t seq      = 0;
  int64_t proc_seq = 0;

  while (running_) {
    auto frame = video_->GetNextFrame();
    if (!frame) {
      if (cfg_.loop_video) {
        video_ = std::make_unique<VideoFile>(cfg_.input_file);
        printf("[%-6s] Loop restart\n", cfg_.name.c_str());
        continue;
      }
      break;
    }

    // PTS = 基于帧序号和原始帧率的理论时刻
    int64_t pts = SharedClock::FramePtsNs(seq, cfg_.raw_fps);
    auto shared_frame = std::shared_ptr<cv::Mat>(std::move(frame));

    // 送入 raw 推流队列
    TimestampedFrame raw_tf{seq, pts, shared_frame};
    if (!raw_queue_.push(std::move(raw_tf))) break;

    // 每 skip_ratio_ 帧送一帧给处理管线
    if (cfg_.processor && (seq % skip_ratio_ == 0)) {
      TimestampedFrame proc_tf{proc_seq, pts, shared_frame};
      if (!process_input_queue_.push(std::move(proc_tf))) break;
      proc_seq++;
    }

    seq++;
  }
}

// ==================== RawWriter: 原始帧推流 ====================
// 按 raw_fps 帧率节流，基于 SharedClock PTS 对齐
void ChannelPipeline::RawWriterLoop() {
  while (true) {
    TimestampedFrame tf;
    if (!raw_queue_.pop(tf)) break;

    // 基于共享时钟的 PTS 进行帧率节流
    if (cfg_.clock)
      cfg_.clock->WaitUntilPts(tf.pts_ns);

    if (raw_streamer_.IsOpen())
      raw_streamer_.Write(*tf.image);

    raw_frames_++;
  }
}

// ==================== ProcessWorker: 算法处理 ====================
// 多个 worker 可并行（如 RGB YOLO 使用多 NPU 核心）
void ChannelPipeline::ProcessWorker(int worker_id) {
  while (true) {
    TimestampedFrame tf;
    if (!process_input_queue_.pop(tf)) break;

    cv::Mat output;
    if (cfg_.processor->Process(worker_id, *tf.image, output)) {
      TimestampedFrame out_tf{tf.seq, tf.pts_ns,
                              std::make_shared<cv::Mat>(std::move(output))};
      if (!processed_queue_.push(std::move(out_tf))) break;
    }
  }
}

// ==================== ProcessedWriter: 处理结果推流 ====================
// 带重排序缓冲（多 worker 可能乱序完成），按 PTS 节流输出
void ChannelPipeline::ProcessedWriterLoop() {
  std::map<int64_t, TimestampedFrame> reorder_buf;
  int64_t next_seq = 0;

  while (true) {
    TimestampedFrame tf;
    if (!processed_queue_.pop(tf)) break;

    reorder_buf[tf.seq] = std::move(tf);

    // 按序号连续输出
    while (reorder_buf.count(next_seq)) {
      auto &frame = reorder_buf[next_seq];

      // 基于共享时钟的 PTS 进行帧率节流
      if (cfg_.clock)
        cfg_.clock->WaitUntilPts(frame.pts_ns);

      if (processed_streamer_.IsOpen())
        processed_streamer_.Write(*frame.image);

      processed_frames_++;
      reorder_buf.erase(next_seq);
      next_seq++;
    }
  }

  // 刷出剩余有序帧
  while (reorder_buf.count(next_seq)) {
    if (processed_streamer_.IsOpen())
      processed_streamer_.Write(*reorder_buf[next_seq].image);
    processed_frames_++;
    reorder_buf.erase(next_seq);
    next_seq++;
  }
}
