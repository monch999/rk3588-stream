#include "pipeline.h"
#include "postprocess.h"
#include <map>
#include <chrono>
#include <thread>
#include <cstdio>

// ==================== 构造 / 析构 ====================
Pipeline::Pipeline(const Config &cfg) : cfg_(cfg) {
  init_post_process(const_cast<std::string &>(cfg_.label_path));

  video_ = std::make_unique<VideoFile>(cfg_.input_file);
  frame_w_ = video_->get_frame_width();
  frame_h_ = video_->get_frame_height();

  image_process_ = std::make_unique<ImageProcess>(
      frame_w_, frame_h_, cfg_.model_input_w, cfg_.model_input_h);

  // 初始化 RKNN 模型 (与原 RknnPool 逻辑相同)
  for (int i = 0; i < cfg_.infer_threads; i++)
    models_.push_back(std::make_shared<Yolov8>(std::string(cfg_.model_path)));
  for (int i = 0; i < cfg_.infer_threads; i++) {
    if (models_[i]->Init(models_[0]->get_rknn_context(), i != 0) != 0) {
      fprintf(stderr, "[ERROR] Init RKNN model %d failed\n", i);
      exit(EXIT_FAILURE);
    }
  }

  cached_od_.count = 0;
  printf("[PIPE  ] Init OK: %dx%d, %d infer threads, interval=%d\n",
         frame_w_, frame_h_, cfg_.infer_threads, cfg_.infer_interval);
}

Pipeline::~Pipeline() {
  Stop();
  deinit_post_process();
}

// ==================== 生命周期 ====================
void Pipeline::Start(WriteFrameFunc write_func) {
  write_func_ = std::move(write_func);
  running_ = true;

  reader_thread_     = std::thread(&Pipeline::ReaderLoop, this);
  preprocess_thread_ = std::thread(&Pipeline::PreprocessLoop, this);
  for (int i = 0; i < cfg_.infer_threads; i++)
    infer_threads_.emplace_back(&Pipeline::InferWorker, this, i);
  writer_thread_ = std::thread(&Pipeline::WriterLoop, this);

  printf("[PIPE  ] Started: reader(1) + preprocess(1) + infer(%d) + writer(1)\n",
         cfg_.infer_threads);
}

void Pipeline::Stop() {
  if (!running_.exchange(false)) return;

  // 关闭所有队列，唤醒阻塞线程
  frame_queue_.shutdown();
  infer_queue_.shutdown();
  render_queue_.shutdown();
  display_queue_.shutdown();

  if (reader_thread_.joinable())     reader_thread_.join();
  if (preprocess_thread_.joinable()) preprocess_thread_.join();
  for (auto &t : infer_threads_)
    if (t.joinable()) t.join();
  if (writer_thread_.joinable()) writer_thread_.join();

  printf("[PIPE  ] Stopped. frames=%d, inferred=%d\n",
         total_frames_.load(), inferred_frames_.load());
}

bool Pipeline::GetDisplayFrame(std::shared_ptr<cv::Mat> &frame) {
  return display_queue_.try_pop(frame);
}

// ==================== Stage A: 读帧 ====================
// 独立线程，解除磁盘I/O对主流水线的阻塞
void Pipeline::ReaderLoop() {
  int64_t seq = 0;
  while (running_) {
    auto frame = video_->GetNextFrame();
    if (!frame) {
      if (cfg_.loop_video) {
        video_ = std::make_unique<VideoFile>(cfg_.input_file);
        printf("[READER] Loop restart\n");
        continue;
      }
      break;  // 不循环则结束
    }

    FrameData fd;
    fd.seq        = seq;
    fd.image      = std::shared_ptr<cv::Mat>(std::move(frame));
    fd.need_infer = (seq % cfg_.infer_interval == 0);
    seq++;

    if (!frame_queue_.push(std::move(fd))) break;
  }
}

// ==================== Stage B: 预处理 ====================
// 独立线程，RGA硬件resize+色彩转换，不占NPU时间
void Pipeline::PreprocessLoop() {
  while (true) {
    FrameData fd;
    if (!frame_queue_.pop(fd)) break;

    if (fd.need_infer) {
      // --- 推理帧: RGA 硬件加速 resize + BGR→RGB ---
      auto rgb = image_process_->Convert(*fd.image);
      if (!rgb) {
        // 转换失败，原帧直接送渲染 (避免序号空洞导致Writer死等)
        RenderFrame rf{fd.seq, fd.image};
        render_queue_.push(std::move(rf));
        continue;
      }

      PreprocessedData pd;
      pd.seq      = fd.seq;
      pd.original = fd.image;
      pd.rgb      = std::move(rgb);
      if (!infer_queue_.push(std::move(pd))) break;

    } else {
      // --- 非推理帧: 复用缓存检测结果画框 ---
      {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (cached_od_.count > 0) {
          object_detect_result_list od_copy = cached_od_;
          image_process_->ImagePostProcess(*fd.image, od_copy);
        }
      }

      RenderFrame rf{fd.seq, fd.image};
      if (!render_queue_.push(std::move(rf))) break;
    }
  }
}

// ==================== Stage C: 推理 (×N) ====================
// 每个Worker绑定一个RKNN模型实例，只做NPU推理+画框
void Pipeline::InferWorker(int id) {
  while (true) {
    PreprocessedData pd;
    if (!infer_queue_.pop(pd)) break;

    // NPU推理
    object_detect_result_list od_results;
    models_[id]->Inference(pd.rgb->ptr(), &od_results,
                           image_process_->get_letter_box());

    // 在原图上画框
    image_process_->ImagePostProcess(*pd.original, od_results);

    // 清理seg_mask (已被ImagePostProcess释放)
    for (int i = 0; i < od_results.count; i++)
      od_results.results_seg[i].seg_mask = nullptr;

    // 更新缓存 (仅当此帧比缓存更新)
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      if (pd.seq > cached_seq_) {
        cached_od_ = od_results;
        cached_seq_ = pd.seq;
      }
    }

    inferred_frames_++;

    RenderFrame rf{pd.seq, pd.original};
    if (!render_queue_.push(std::move(rf))) break;
  }
}

// ==================== Stage D: 写入推流 ====================
// 独立线程，重排序缓冲保证帧顺序，按实际帧率节流 (等效于 ffmpeg -re)
void Pipeline::WriterLoop() {
  std::map<int64_t, std::shared_ptr<cv::Mat>> reorder_buf;
  int64_t next_seq = 0;

  // 帧率节流: 每帧间隔
  const auto frame_interval = std::chrono::microseconds(
      static_cast<int64_t>(1000000.0 / cfg_.framerate));
  auto next_send_time = std::chrono::steady_clock::now();

  while (true) {
    RenderFrame rf;
    if (!render_queue_.pop(rf)) break;

    reorder_buf[rf.seq] = rf.image;

    // 按序号顺序输出连续帧
    while (reorder_buf.count(next_seq)) {
      // 等待到目标发送时刻 (帧率节流)
      auto now = std::chrono::steady_clock::now();
      if (next_send_time > now)
        std::this_thread::sleep_until(next_send_time);
      next_send_time += frame_interval;

      auto &img = reorder_buf[next_seq];
      write_func_(*img);
      display_queue_.try_push(img);
      total_frames_++;
      reorder_buf.erase(next_seq);
      next_seq++;
    }
  }

  // 刷出剩余有序帧
  while (reorder_buf.count(next_seq)) {
    write_func_(*reorder_buf[next_seq]);
    total_frames_++;
    reorder_buf.erase(next_seq);
    next_seq++;
  }
}
