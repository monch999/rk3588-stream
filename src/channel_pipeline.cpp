#include "channel_pipeline.h"
#include "algorithm_interface.h"
#include <map>
#include <cstdio>

// ==================== 构造 / 析构 ====================
ChannelPipeline::ChannelPipeline(const Config &cfg) : cfg_(cfg) {
  need_process_ = cfg_.enable_processed && cfg_.processor != nullptr;

  if (need_process_) {
    video_ = std::make_unique<VideoFile>(cfg_.input_file);
    frame_w_ = video_->get_frame_width();
    frame_h_ = video_->get_frame_height();
    skip_ratio_ = std::max(1, static_cast<int>(std::round(cfg_.raw_fps / cfg_.processed_fps)));
  } else {
    // raw 模式下，我们不需要 video_ 对象，也不需要 frame_w_/h_
    frame_w_ = 0;
    frame_h_ = 0;
  }

  printf("[%-6s] Init OK: mode=%s\n", cfg_.name.c_str(), need_process_ ? "AI+Raw" : "Raw-Only");
}

ChannelPipeline::~ChannelPipeline() { Stop(); }

// ==================== 启动 ====================
void ChannelPipeline::Start() {
  // ---- 1. 原始流：全硬件 Direct 模式 ----
  if (cfg_.enable_raw) {
    StreamConfig sc;
    sc.rtsp_url = cfg_.raw_rtsp_url;
    sc.rtmp_url = cfg_.raw_rtmp_url;
    sc.bitrate  = cfg_.bitrate;
    if (!raw_streamer_.OpenDirect(cfg_.input_file, cfg_.raw_fps, sc, cfg_.loop_video)) {
      fprintf(stderr, "[%-6s] Failed to open direct raw streamer\n", cfg_.name.c_str());
    } else {
      printf("[%-6s] Raw stream (Direct) started: %s\n", cfg_.name.c_str(), cfg_.raw_rtsp_url.c_str());
    }
  }

  // ---- 2. 处理流：只有需要 AI 处理时才启动 C++ 管线线程 ----
  if (need_process_) {
    StreamConfig sc;
    sc.rtsp_url = cfg_.processed_rtsp_url;
    sc.rtmp_url = cfg_.processed_rtmp_url;
    sc.bitrate  = cfg_.bitrate;
    if (!processed_streamer_.Open(frame_w_, frame_h_, cfg_.processed_fps, sc)) {
      fprintf(stderr, "[%-6s] Failed to open processed streamer\n", cfg_.name.c_str());
    }

    running_ = true;

    // 只有在此处才启动线程！
    reader_thread_ = std::thread(&ChannelPipeline::ReaderLoop, this);
    
    int nw = cfg_.processor->NumWorkers();
    for (int i = 0; i < nw; i++) {
      process_threads_.emplace_back(&ChannelPipeline::ProcessWorker, this, i);
    }
    
    processed_writer_thread_ = std::thread(&ChannelPipeline::ProcessedWriterLoop, this);

    printf("[%-6s] AI Pipeline Started: reader(1) + process(%d) + writer(1)\n",
           cfg_.name.c_str(), nw);
  } else {
    // 如果不需要处理，就不标记 running_ 为 true，也不启动任何 reader/writer 线程
    printf("[%-6s] Running in Direct-Raw mode only. No C++ threads started.\n", cfg_.name.c_str());
  }
}

// ==================== 停止 ====================
void ChannelPipeline::Stop() {
  if (!running_.exchange(false)) {
    // 如果没有运行 C++ 管线，只需关闭 FFmpeg 进程
    raw_streamer_.Close();
    return;
  }

  raw_queue_.shutdown();
  process_input_queue_.shutdown();
  processed_queue_.shutdown();

  if (reader_thread_.joinable())     reader_thread_.join();
  // ... 其他线程的 joinable 检查 ...

  raw_streamer_.Close();
  processed_streamer_.Close();
}

// ==================== Reader: 读帧 + 分发 ====================
void ChannelPipeline::ReaderLoop() {
  int64_t seq      = 0;
  int64_t proc_seq = 0;
  auto start = std::chrono::steady_clock::now();          // +
  auto interval = std::chrono::nanoseconds(               // +
      static_cast<int64_t>(1e9 / cfg_.raw_fps));

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
    // ---- 帧率节流: 控制读取速度 ----
    auto target = start + interval * seq;                  
    std::this_thread::sleep_until(target);
    int64_t pts = SharedClock::FramePtsNs(seq, cfg_.raw_fps);
    auto shared_frame = std::shared_ptr<cv::Mat>(std::move(frame));

    // 每 skip_ratio_ 帧送一帧给处理管线 (仅当需要处理时)
    if (need_process_ && (seq % skip_ratio_ == 0)) {
      TimestampedFrame proc_tf{proc_seq, pts, shared_frame};
      if (!process_input_queue_.push(std::move(proc_tf))) break;
      proc_seq++;
    }

    seq++;
  }
}

// ==================== RawWriter: 原始帧推流 ====================
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
