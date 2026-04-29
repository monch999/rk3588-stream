#pragma once
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include "bounded_queue.h"
#include "drm_frame.h"
#include "ffmpeg_streamer.h"
#include "mpp_encoder.h"
#include "rtsp_muxer.h"
#include "shared_clock.h"
#include "videofile.h"

class IFrameProcessor;

// ==================== 带时间戳的帧 (zero-copy) ====================
struct TimestampedFrame {
  int64_t seq    = 0;
  int64_t pts_ns = 0;
  std::shared_ptr<DrmFrame> frame;  // NV12, 自管 buffer
};

// ==================== 单通道推流流水线 (全链路 zero-copy) ====================
class ChannelPipeline {
public:
  struct Config {
    std::string name;
    std::string input_file;

    double raw_fps        = 30.0;
    double processed_fps  = 30.0;

    std::string raw_rtsp_url;
    std::string processed_rtsp_url;

    std::string bitrate = "4M";
    std::string codec   = "h264";
    bool loop_video     = false;

    bool enable_raw       = true;
    bool enable_processed = false;
    bool enable_display   = false;

    SharedClock      *clock     = nullptr;
    IFrameProcessor  *processor = nullptr;
  };

  explicit ChannelPipeline(const Config& cfg);
  ~ChannelPipeline();

  void Start();
  void Stop();

  int GetFrameWidth()      const { return frame_w_; }
  int GetFrameHeight()     const { return frame_h_; }
  int GetProcessedFrames() const { return processed_frames_.load(); }
  const std::string& GetName() const { return cfg_.name; }

  bool GetDisplayFrame(cv::Mat& out) {
    std::lock_guard<std::mutex> lk(display_mtx_);
    if (display_frame_.empty()) return false;
    display_frame_.copyTo(out);
    return true;
  }

private:
  void ReaderLoop();
  void ProcessWorker(int worker_id);
  void ProcessedWriterLoop();

  // decoder NV12 (来自 video_) -> 自管 NV12 DRM buffer (decoder buffer 立即释放)
  std::shared_ptr<DrmFrame> CopyToOwnedNv12(const std::shared_ptr<DrmFrame>& src);

  Config cfg_;
  int frame_w_ = 0, frame_h_ = 0;
  int skip_ratio_ = 1;
  bool need_process_ = false;
  std::atomic<bool> running_{false};

  int64_t total_proc_seq_ = 0;
  int64_t next_seq_ = 0;

  std::unique_ptr<VideoFile> video_;

  // 队列容量小: 防止 buffer 池占用过多
  BoundedQueue<TimestampedFrame> process_input_queue_{2};
  BoundedQueue<TimestampedFrame> processed_queue_{2};

  FFmpegStreamer raw_streamer_;
  std::unique_ptr<MppEncoder> encoder_;
  std::unique_ptr<RTSPMuxer>  muxer_;

  std::thread              reader_thread_;
  std::vector<std::thread> process_threads_;
  std::thread              processed_writer_thread_;

  std::atomic<int> processed_frames_{0};

  std::mutex display_mtx_;
  cv::Mat    display_frame_;
};
