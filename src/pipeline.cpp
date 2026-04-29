#include "pipeline.h"
#include "image_process.h"
#include "postprocess.h"
#include <map>
#include <chrono>
#include <thread>
#include <cstdio>

// ==================== 构造 / 析构 ====================
Pipeline::Pipeline(const Config &cfg) : cfg_(cfg) {
  init_post_process(const_cast<std::string &>(cfg_.label_path));

  // VideoFile 现在使用 MPP 硬件解码
  video_ = std::make_unique<VideoFile>(cfg_.input_file);
  frame_w_ = video_->get_frame_width();
  frame_h_ = video_->get_frame_height();

  image_process_ = std::make_unique<ImageProcess>(
      frame_w_, frame_h_, cfg_.model_input_w, cfg_.model_input_h);

  // 初始化 RKNN 模型
  for (int i = 0; i < cfg_.infer_threads; i++)
    models_.push_back(std::make_shared<Yolov8>(std::string(cfg_.model_path)));
  for (int i = 0; i < cfg_.infer_threads; i++) {
    if (models_[i]->Init(models_[0]->get_rknn_context(), i != 0) != 0) {
      fprintf(stderr, "[ERROR] Init RKNN model %d failed\n", i);
      exit(EXIT_FAILURE);
    }
  }

  cached_od_.count = 0;
  printf("[PIPE  ] Init OK: %dx%d, %d infer threads, interval=%d, decode=MPP\n",
         frame_w_, frame_h_, cfg_.infer_threads, cfg_.infer_interval);
}

Pipeline::~Pipeline() {
  Stop();
  deinit_post_process();
}

// ==================== 生命周期 ====================
void Pipeline::Start(WriteFrameFunc write_func) {
  // ---- 初始化输出: MppEncoder 或兼容 pipe ----
  if (cfg_.use_mpp_encoder && cfg_.enc_callback) {
    encoder_ = std::make_unique<MppEncoder>();
    MppEncoder::Config ecfg;
    ecfg.width   = frame_w_;
    ecfg.height  = frame_h_;
    ecfg.fps     = cfg_.framerate;
    ecfg.codec   = cfg_.enc_codec;
    ecfg.bitrate = cfg_.enc_bitrate;
    if (!encoder_->Open(ecfg, cfg_.enc_callback)) {
      fprintf(stderr, "[PIPE  ] MppEncoder open failed, fallback to write_func\n");
      encoder_.reset();
      write_func_ = std::move(write_func);
    } else {
      printf("[PIPE  ] Using MPP hardware encoder\n");
    }
  } else {
    write_func_ = std::move(write_func);
    printf("[PIPE  ] Using pipe encoder\n");

  }

  running_ = true;

  reader_thread_     = std::thread(&Pipeline::ReaderLoop, this);
  preprocess_thread_ = std::thread(&Pipeline::PreprocessLoop, this);
  for (int i = 0; i < cfg_.infer_threads; i++)
    infer_threads_.emplace_back(&Pipeline::InferWorker, this, i);
  writer_thread_ = std::thread(&Pipeline::WriterLoop, this);

  printf("[PIPE  ] Started: reader(1) + preprocess(1) + infer(%d) + writer(1)\n", cfg_.infer_threads);
}

void Pipeline::Stop() {
  if (!running_.exchange(false)) return;

  frame_queue_.shutdown();
  infer_queue_.shutdown();
  render_queue_.shutdown();
  display_queue_.shutdown();

  if (reader_thread_.joinable())     reader_thread_.join();
  if (preprocess_thread_.joinable()) preprocess_thread_.join();
  for (auto &t : infer_threads_)
    if (t.joinable()) t.join();
  if (writer_thread_.joinable()) writer_thread_.join();

  if (encoder_) encoder_->Close();

  printf("[PIPE  ] Stopped. frames=%d, inferred=%d\n",
         total_frames_.load(), inferred_frames_.load());
}

bool Pipeline::GetDisplayFrame(std::shared_ptr<cv::Mat> &frame) {
  return display_queue_.try_pop(frame);
}

// ==================== Stage A: 读帧 (MPP 硬件解码) ====================
void Pipeline::ReaderLoop() {
  int64_t seq = 0;
  printf("[READER] Thread started\n");
  while (running_) {
    auto drm = video_->GetNextDrmFrame();
    if (!drm) {
      printf("[READER] GetNextDrmFrame returned null at seq=%ld\n", (long)seq);
      if (cfg_.loop_video) {
        video_ = std::make_unique<VideoFile>(cfg_.input_file);
        printf("[READER] Loop restart\n");
        continue;
      }
      break;
    }
    //printf("[READER] DRM frame seq=%ld: %dx%d, vaddr=%p, fd=%d\n",
           //(long)seq, drm->width, drm->height, drm->vaddr, drm->fd);

    auto bgr = drm->ToBgrMat();
    if (!bgr) {
      printf("[READER] ToBgrMat FAILED seq=%ld, skipping!\n", (long)seq);
      continue;
    }
    //printf("[READER] BGR OK seq=%ld: %dx%d\n", (long)seq, bgr->cols, bgr->rows);

    FrameData fd;
    fd.seq        = seq;
    fd.drm        = std::move(drm);
    fd.image      = std::shared_ptr<cv::Mat>(std::move(bgr));
    fd.need_infer = (seq % cfg_.infer_interval == 0);
    seq++;

    if (!frame_queue_.push(std::move(fd))) {
      printf("[READER] frame_queue_.push failed, exiting\n");
      break;
    }
  }
  printf("[READER] Thread exiting, seq=%ld\n", (long)seq);
  frame_queue_.shutdown();
  finished_ = true;
  printf("[READER] Video finished\n");
}

// ==================== Stage B: 预处理 ====================
void Pipeline::PreprocessLoop() {
  printf("[PREPROC] Thread started\n");
  while (true) {
    FrameData fd;
    if (!frame_queue_.pop(fd)) {
      printf("[PREPROC] frame_queue_ pop failed (shutdown), exiting\n");
      break;
    }
 
    // printf("[PREPROC] Got frame seq=%ld, need_infer=%d, image=%p (%dx%d)\n",
    //        fd.seq, fd.need_infer,
    //        fd.image ? fd.image->data : nullptr,
    //        fd.image ? fd.image->cols : 0,
    //        fd.image ? fd.image->rows : 0);
 
    if (fd.need_infer) {
      // RGA 硬件 resize + BGR→RGB
      auto rgb = image_process_->Convert(*fd.image);
      if (!rgb) {
        printf("[PREPROC] seq=%ld Convert failed, skip to render\n", fd.seq);
        RenderFrame rf{fd.seq, fd.image, fd.drm};
        render_queue_.push(std::move(rf));
        continue;
      }
 
      //printf("[PREPROC] seq=%ld Convert OK, pushing to infer_queue\n", fd.seq);
      PreprocessedData pd;
      pd.seq      = fd.seq;
      pd.original = fd.image;
      pd.rgb      = std::move(rgb);
      if (!infer_queue_.push(std::move(pd))) {
        printf("[PREPROC] infer_queue_ push failed (shutdown), exiting\n");
        break;
      }
 
    } else {
      // 非推理帧: 复用缓存检测结果画框
      {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (cached_od_.count > 0) {
          object_detect_result_list od_copy = cached_od_;
          image_process_->ImagePostProcess(*fd.image, od_copy);
        }
      }
 
      //printf("[PREPROC] seq=%ld direct frame, pushing to render_queue\n", fd.seq);
      RenderFrame rf{fd.seq, fd.image, fd.drm};
      if (!render_queue_.push(std::move(rf))) {
        printf("[PREPROC] render_queue_ push failed (shutdown), exiting\n");
        break;
      }
    }
  }
  printf("[PREPROC] Thread exited\n");
}

// ==================== Stage C: 推理 (×N) ====================
void Pipeline::InferWorker(int id) {
  printf("[INFER%d] Thread started\n", id);
  while (true) {
    PreprocessedData pd;
    if (!infer_queue_.pop(pd)) {
      printf("[INFER%d] infer_queue_ pop failed (shutdown), exiting\n", id);
      break;
    }
 
    // printf("[INFER%d] Got frame seq=%ld, rgb=%p, original=%p\n",
    //        id, pd.seq, pd.rgb ? pd.rgb->data : nullptr,
    //        pd.original ? pd.original->data : nullptr);
 
    object_detect_result_list od_results;
    models_[id]->Inference(pd.rgb->ptr(), &od_results,
                           image_process_->get_letter_box());
 
    // printf("[INFER%d] seq=%ld Inference done, detected %d objects\n",
    //        id, pd.seq, od_results.count);
 
    image_process_->ImagePostProcess(*pd.original, od_results);
 
    for (int i = 0; i < od_results.count; i++)
      od_results.results_seg[i].seg_mask = nullptr;
 
    if (pd.seq > cached_seq_) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (pd.seq > cached_seq_) {  // 双重检查
            cached_od_ = od_results;
            cached_seq_ = pd.seq;
        }
      }
 
    inferred_frames_++;
 
    // 注意: 推理路径不保留 DRM 引用 (已画框在 cv::Mat 上)
    RenderFrame rf{pd.seq, pd.original, nullptr};
    if (!render_queue_.push(std::move(rf))) {
      printf("[INFER%d] render_queue_ push failed (shutdown), exiting\n", id);
      break;
    }
    //printf("[INFER%d] seq=%ld pushed to render_queue\n", id, pd.seq);
  }
  printf("[INFER%d] Thread exited\n", id);
}

// ==================== Stage D: 写入推流 ====================
void Pipeline::WriterLoop() {
  std::map<int64_t, RenderFrame> reorder_buf;
  int64_t next_seq = 0;

  const auto frame_interval = std::chrono::microseconds(
      static_cast<int64_t>(1000000.0 / cfg_.framerate));
  auto next_send_time = std::chrono::steady_clock::now();

  while (true) {
    RenderFrame rf;
    if (!render_queue_.pop(rf)) break;

    reorder_buf[rf.seq] = std::move(rf);

    while (reorder_buf.count(next_seq)) {
      auto now = std::chrono::steady_clock::now();
      if (next_send_time > now)
        std::this_thread::sleep_until(next_send_time);
      next_send_time += frame_interval;

      auto &frame = reorder_buf[next_seq];

      // ---- 编码输出: MppEncoder 或兼容 pipe ----
      if (encoder_) {
        // MPP 硬件编码 (cv::Mat BGR → NV12 → H264)
        // TODO: 当画框也迁移到 DRM 后, 可用 Encode(drm) 零拷贝
        encoder_->Encode(*frame.image, next_seq);
      } else if (write_func_) {
        write_func_(*frame.image);
      }

      display_queue_.try_push(frame.image);
      total_frames_++;
      reorder_buf.erase(next_seq);
      next_seq++;
    }
  }

  // 刷出剩余有序帧
  while (reorder_buf.count(next_seq)) {
    auto &frame = reorder_buf[next_seq];
    if (encoder_)
      encoder_->Encode(*frame.image, next_seq);
    else if (write_func_)
      write_func_(*frame.image);
    total_frames_++;
    reorder_buf.erase(next_seq);
    next_seq++;
  }
}
