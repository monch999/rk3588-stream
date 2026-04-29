#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>
#include "bounded_queue.h"
#include "drm_frame.h"
#include "mpp_encoder.h"
#include "videofile.h"
#include "yolov8.h"
#include "postprocess.h"

class ImageProcess;


// ==================== 写帧回调 (兼容旧 FFmpegStreamer pipe) ====================
using WriteFrameFunc = std::function<void(const cv::Mat &)>;

// ==================== 阶段间数据结构 ====================
struct FrameData {
  int64_t                  seq;
  std::shared_ptr<DrmFrame> drm;    // MPP 解码输出 (NV12, DRM buffer)
  std::shared_ptr<cv::Mat>  image;  // BGR cv::Mat (从 DRM 转换, 用于画框)
  bool                     need_infer;
};

struct PreprocessedData {
  int64_t                  seq;
  std::shared_ptr<cv::Mat> original;  // BGR 原图 (画框用)
  std::unique_ptr<cv::Mat> rgb;       // RGA resize+cvtcolor 后的 RGB
};

struct RenderFrame {
  int64_t                  seq;
  std::shared_ptr<cv::Mat> image;     // 画完框的 BGR
  std::shared_ptr<DrmFrame> drm;      // 保留 DRM 引用 (未来零拷贝编码用)
};

// ==================== 推理流水线 ====================
//
//   Reader ──→ frame_queue ──→ Preprocess ──┬─(推理帧)──→ infer_queue ──→ InferWorker(×N)──┐
//                                           │                                               │
//                                           └─(非推理帧, 复用缓存)──────────────────────────┤
//                                                                                           ↓
//                               WriterLoop ←── render_queue ←───────────────────────────────┘
//                                  │
//                        ┌─────────┴──────────┐
//                  MppEncoder            write_func_
//                (零拷贝 H264)         (兼容 pipe 模式)
//
class Pipeline {
public:
  struct Config {
    std::string input_file;
    std::string model_path;
    std::string label_path;
    int    model_input_w  = 640;
    int    model_input_h  = 384;
    int    infer_threads  = 3;
    int    infer_interval = 1;      // 每 N 帧推理 1 帧
    double framerate      = 30.0;
    bool   loop_video     = false;

    // ---- MPP 编码器配置 (可选, 设置后替代 write_func_) ----
    bool        use_mpp_encoder = true;
    std::string enc_codec       = "h264";
    std::string enc_bitrate     = "4M";
    EncodedPacketCallback enc_callback = nullptr;  // 编码输出回调
  };

  explicit Pipeline(const Config &cfg);
  ~Pipeline();

  // write_func: 兼容旧的 FFmpegStreamer pipe 模式
  // 如果 cfg.use_mpp_encoder=true, 则 write_func 可传 nullptr
  void Start(WriteFrameFunc write_func = nullptr);
  void Stop();

  bool GetDisplayFrame(std::shared_ptr<cv::Mat> &frame);

  int GetFrameWidth()      const { return frame_w_; }
  int GetFrameHeight()     const { return frame_h_; }
  int GetTotalFrames()     const { return total_frames_.load(); }
  int GetInferredFrames()  const { return inferred_frames_.load(); }

  std::atomic<bool> finished_{false};
  bool IsFinished() const { return finished_; }

private:
  void ReaderLoop();
  void PreprocessLoop();
  void InferWorker(int id);
  void WriterLoop();

  Config cfg_;
  int frame_w_ = 0, frame_h_ = 0;
  std::atomic<bool> running_{false};

  // 视频读取 (MPP 硬件解码)
  std::unique_ptr<VideoFile> video_;

  // 图像处理
  std::unique_ptr<ImageProcess> image_process_;

  // RKNN 模型
  std::vector<std::shared_ptr<Yolov8>> models_;

  // 推理结果缓存
  std::mutex                 cache_mutex_;
  object_detect_result_list  cached_od_;
  int64_t                    cached_seq_ = -1;

  // 阶段间队列
  BoundedQueue<FrameData>        frame_queue_{16};
  BoundedQueue<PreprocessedData> infer_queue_{4};
  BoundedQueue<RenderFrame>      render_queue_{4};
  BoundedQueue<std::shared_ptr<cv::Mat>> display_queue_{2};

  // 输出: 二选一
  WriteFrameFunc write_func_;           // 兼容模式: pipe rawvideo
  std::unique_ptr<MppEncoder> encoder_; // MPP 硬件编码

  // 线程
  std::thread              reader_thread_;
  std::thread              preprocess_thread_;
  std::vector<std::thread> infer_threads_;
  std::thread              writer_thread_;

  // 统计
  std::atomic<int> total_frames_{0};
  std::atomic<int> inferred_frames_{0};
};
