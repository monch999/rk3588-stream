#pragma once
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include "bounded_queue.h"
#include "ffmpeg_streamer.h"
#include "shared_clock.h"
#include "videofile.h"

class IFrameProcessor;  // 前向声明

// ==================== 带时间戳的帧 ====================
struct TimestampedFrame {
  int64_t                  seq;      // 帧序号
  int64_t                  pts_ns;   // PTS (纳秒, 相对于 SharedClock epoch)
  std::shared_ptr<cv::Mat> image;
};

// ==================== 单通道推流流水线 ====================
//
// 每个通道（RGB / 多光谱 / 热红外）独立运行一个 ChannelPipeline：
//
//   Reader ──→ raw_queue ──→ RawWriter ──→ FFmpeg (原始推流)
//     │
//     └──→ process_input_queue ──→ ProcessWorker(×N) ──→ processed_queue
//                                                            │
//                                    ProcessedWriter ←───────┘
//                                         │
//                                    FFmpeg (处理后推流)
//
// PTS 对齐: 所有通道共享同一个 SharedClock epoch
//           帧的 PTS = epoch + seq × (1e9 / raw_fps)
//           Writer 基于 PTS 进行帧率节流
//
class ChannelPipeline {
public:
  struct Config {
    std::string name;                 // 通道名: "rgb", "ms", "thermal"
    std::string input_file;           // 输入视频文件

    double raw_fps        = 30.0;     // 原始推流帧率
    double processed_fps  = 30.0;     // 处理后推流帧率

    // 推流 URL (RTSP)
    std::string raw_rtsp_url;
    std::string processed_rtsp_url;
    // 推流 URL (RTMP, 可选)
    std::string raw_rtmp_url;
    std::string processed_rtmp_url;

    std::string bitrate = "4M";
    bool loop_video     = false;

    // 推流模式: 控制是否推原始流和/或处理后的流
    bool enable_raw       = true;   // 推原始流
    bool enable_processed = true;   // 推处理后的流

    SharedClock      *clock     = nullptr;  // 共享时钟 (外部持有)
    IFrameProcessor  *processor = nullptr;  // 帧处理器 (外部持有, 可为 null)
  };

  explicit ChannelPipeline(const Config &cfg);
  ~ChannelPipeline();

  void Start();
  void Stop();

  int GetFrameWidth()       const { return frame_w_; }
  int GetFrameHeight()      const { return frame_h_; }
  int GetRawFrames()        const { return raw_frames_.load(); }
  int GetProcessedFrames()  const { return processed_frames_.load(); }
  const std::string &GetName() const { return cfg_.name; }

private:
  void ReaderLoop();
  void RawWriterLoop();
  void ProcessWorker(int worker_id);
  void ProcessedWriterLoop();

  Config cfg_;
  int frame_w_ = 0, frame_h_ = 0;
  int skip_ratio_ = 1;               // raw_fps / processed_fps (帧抽取比)
  bool need_process_ = false;         // 是否需要处理管线
  std::atomic<bool> running_{false};

  // 视频读取
  std::unique_ptr<VideoFile> video_;

  // 阶段间队列
  BoundedQueue<TimestampedFrame> raw_queue_{10};
  BoundedQueue<TimestampedFrame> process_input_queue_{8};
  BoundedQueue<TimestampedFrame> processed_queue_{8};

  // FFmpeg 推流器
  FFmpegStreamer raw_streamer_;
  FFmpegStreamer processed_streamer_;

  // 线程
  std::thread              reader_thread_;
  std::vector<std::thread> process_threads_;
  std::thread              processed_writer_thread_;

  // 统计
  std::atomic<int> raw_frames_{0};
  std::atomic<int> processed_frames_{0};
};
