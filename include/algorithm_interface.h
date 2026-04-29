#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cstdio>
#include "drm_frame.h"
#include "image_process.h"
#include "postprocess.h"
#include "yolov8.h"

// ==================== 帧处理器接口 (zero-copy 版) ====================
// 算法直接消费 DrmFrame, 在 Y 平面/原 buffer 上修改 (in-place)
// Process() 必须线程安全 (NumWorkers() > 1 时多线程调用)
class IFrameProcessor {
public:
  virtual ~IFrameProcessor() = default;

  virtual bool Init() = 0;

  // 处理一帧 (in-place: 直接修改 frame 内容)
  // 返回 true 表示该帧应继续推流; false 表示丢弃
  virtual bool Process(int worker_id,
                       const std::shared_ptr<DrmFrame>& frame) = 0;

  virtual std::string Name() const = 0;
  virtual int NumWorkers() const { return 1; }
};

// ==================== RGB YOLO 推理处理器 ====================
class YoloProcessor : public IFrameProcessor {
public:
  struct Config {
    std::string model_path;
    std::string label_path;
    int  num_cores      = 3;
    int  frame_w        = 0;
    int  frame_h        = 0;
    int  model_input_w  = 640;
    int  model_input_h  = 384;
  };

  explicit YoloProcessor(const Config& cfg) : cfg_(cfg) {}

  bool Init() override {
    if (cfg_.model_path.empty() || cfg_.label_path.empty()) {
      fprintf(stderr, "[YOLO] model/label path empty\n");
      return false;
    }
    if (cfg_.frame_w <= 0 || cfg_.frame_h <= 0 || cfg_.num_cores <= 0) {
      fprintf(stderr, "[YOLO] invalid cfg\n");
      return false;
    }

    init_post_process(const_cast<std::string&>(cfg_.label_path));

    models_.reserve(cfg_.num_cores);
    image_processors_.reserve(cfg_.num_cores);

    for (int i = 0; i < cfg_.num_cores; i++) {
      models_.emplace_back(std::make_unique<Yolov8>(std::string(cfg_.model_path)));
      image_processors_.emplace_back(std::make_unique<ImageProcess>(
          cfg_.frame_w, cfg_.frame_h,
          cfg_.model_input_w, cfg_.model_input_h));
    }

    for (int i = 0; i < cfg_.num_cores; i++) {
      rknn_context* ctx = (i == 0) ? nullptr : models_[0]->get_rknn_context();
      if (models_[i]->Init(ctx, i != 0) != 0) {
        fprintf(stderr, "[YOLO] Init RKNN model %d failed\n", i);
        return false;
      }
    }

    printf("[YOLO] Init OK: %d NPU cores, %dx%d -> %dx%d\n",
           cfg_.num_cores, cfg_.frame_w, cfg_.frame_h,
           cfg_.model_input_w, cfg_.model_input_h);
    return true;
  }

  bool Process(int worker_id,
               const std::shared_ptr<DrmFrame>& frame) override {
    if (!frame || frame->format != DrmFrame::NV12) return false;

    auto& ip    = image_processors_[worker_id];
    auto& model = models_[worker_id];

    // 1. RGA: NV12 -> RGB (zero-copy via fd)
    auto rgb_frame = ip->ConvertToRgb(frame);
    if (!rgb_frame) return true;  // 转换失败仍推原帧

    // 2. NPU 推理 (CPU 指针路径, 务实零拷贝)
    thread_local object_detect_result_list od_results;
    od_results.count = 0;
    model->Inference(rgb_frame->vaddr, &od_results, ip->get_letter_box());

    // 3. 在原 NV12 帧的 Y 平面画检测框
    if (od_results.count > 0) {
      ip->DrawDetections(frame, od_results);
    }

    // 4. 清理分割掩码指针 (本次只画框, 但保险释放)
    for (int i = 0; i < od_results.count; i++) {
      if (od_results.results_seg[i].seg_mask) {
        free(od_results.results_seg[i].seg_mask);
        od_results.results_seg[i].seg_mask = nullptr;
      }
    }

    // 5. CPU 写完, encoder 读前必须 cache flush
    frame->SyncEnd();
    return true;
  }

  std::string Name() const override { return "YoloProcessor"; }
  int NumWorkers() const override { return cfg_.num_cores; }

private:
  Config cfg_;
  std::vector<std::unique_ptr<Yolov8>>       models_;
  std::vector<std::unique_ptr<ImageProcess>> image_processors_;
};

// ==================== 多光谱植被指数处理器 (stub) ====================
class NdviProcessor : public IFrameProcessor {
public:
  bool Init() override {
    printf("[NDVI ] Stub processor initialized\n");
    return true;
  }
  bool Process(int /*worker_id*/,
               const std::shared_ptr<DrmFrame>& frame) override {
    // TODO: 实现植被指数计算 (在 Y 平面上叠加伪彩色信息)
    if (frame) frame->SyncEnd();
    return true;
  }
  std::string Name() const override { return "NdviProcessor"; }
};

// ==================== 热红外火点检测处理器 (stub) ====================
class FireDetector : public IFrameProcessor {
public:
  bool Init() override {
    printf("[FIRE ] Stub processor initialized\n");
    return true;
  }
  bool Process(int /*worker_id*/,
               const std::shared_ptr<DrmFrame>& frame) override {
    // TODO: 实现火点检测
    if (frame) frame->SyncEnd();
    return true;
  }
  std::string Name() const override { return "FireDetector"; }
};
