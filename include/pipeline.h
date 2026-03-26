#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include "bounded_queue.h"
#include "image_process.h"
#include "postprocess.h"
#include "videofile.h"
#include "yolov8.h"

// ==================== 阶段间数据结构 ====================
struct FrameData {
  int64_t seq;
  std::shared_ptr<cv::Mat> image;
  bool need_infer;
};

struct PreprocessedData {
  int64_t seq;
  std::shared_ptr<cv::Mat> original;
  std::unique_ptr<cv::Mat> rgb;  // RGA转换后的RGB图
};

struct RenderFrame {
  int64_t seq;
  std::shared_ptr<cv::Mat> image;  // 画完框的最终帧
};

// 写帧回调类型
using WriteFrameFunc = std::function<bool(const cv::Mat &)>;

// ==================== 多阶段流水线 ====================
//
//  Reader(1) → frame_queue → Preprocess(1) → infer_queue → InferWorker(N)
//                                  ↓                            ↓
//                            (non-infer帧                (infer帧推理+画框)
//                             画缓存框)                        ↓
//                                  ↓                            ↓
//                                  └──── render_queue ──────────┘
//                                              ↓
//                                       Writer(1) [重排序缓冲]
//                                              ↓
//                                    fwrite → FFmpeg pipe → RTSP/RTMP
//
class Pipeline {
public:
  struct Config {
    std::string model_path;
    std::string label_path;
    std::string input_file;
    int    infer_threads    = 3;
    int    infer_interval   = 15;    // 每N帧推理一次
    int    model_input_w    = 640;   // 模型输入宽
    int    model_input_h    = 384;   // 模型输入高
    double framerate        = 30.0;
    bool   loop_video       = false;
  };

  explicit Pipeline(const Config &cfg);
  ~Pipeline();

  void Start(WriteFrameFunc write_func);
  void Stop();

  // 主线程调用: 非阻塞获取一帧用于显示
  bool GetDisplayFrame(std::shared_ptr<cv::Mat> &frame);

  int GetFrameWidth()    const { return frame_w_; }
  int GetFrameHeight()   const { return frame_h_; }
  int GetTotalFrames()   const { return total_frames_.load(); }
  int GetInferredFrames() const { return inferred_frames_.load(); }

private:
  // 各阶段线程函数
  void ReaderLoop();
  void PreprocessLoop();
  void InferWorker(int id);
  void WriterLoop();

  Config cfg_;
  int frame_w_ = 0, frame_h_ = 0;
  std::atomic<bool> running_{false};

  // 视频 & 模型
  std::unique_ptr<VideoFile>              video_;
  std::unique_ptr<ImageProcess>           image_process_;
  std::vector<std::shared_ptr<Yolov8>>    models_;

  // 阶段间队列
  BoundedQueue<FrameData>                 frame_queue_{8};
  BoundedQueue<PreprocessedData>          infer_queue_{4};
  BoundedQueue<RenderFrame>               render_queue_{8};
  BoundedQueue<std::shared_ptr<cv::Mat>>  display_queue_{2};

  // 缓存检测结果 (preprocess线程读, infer线程写)
  std::mutex                    cache_mutex_;
  object_detect_result_list     cached_od_{};
  int64_t                       cached_seq_ = -1;

  // 线程
  std::thread              reader_thread_;
  std::thread              preprocess_thread_;
  std::vector<std::thread> infer_threads_;
  std::thread              writer_thread_;

  // 统计
  std::atomic<int> total_frames_{0};
  std::atomic<int> inferred_frames_{0};

  // 推流写入回调
  WriteFrameFunc write_func_;
};
