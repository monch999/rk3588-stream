#pragma once
#include <memory>
#include <string>
#include <vector>
#include <cstdio>
#include <atomic>

#include "drm_frame.h"
#include "image_process.h"
#include "postprocess.h"
#include "yolov8.h"
#include "ByteTrack/BYTETracker.h" // 引入 ByteTrack
#include "ByteTrack/Object.h"
#include "lens_model.h"
#include "zoom_controller.h"
#include "gimbal_state.h"
#include "angle_sink.h"


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

    init_post_process(cfg_.label_path);

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

    // 2. NPU 推理
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

// ==================== 单线程目标追踪处理器 ====================
// 专为强时序依赖的 Tracker 设计，仅使用 1 个 Worker 保证帧顺序
class YoloTrackProcessor : public IFrameProcessor {
public:
  struct Config {
    // 复用 YoloProcessor 的字段
    std::string model_path;
    std::string label_path;
    int  num_cores      = 1;       // 单线程, 强制 1
    int  frame_w        = 0;
    int  frame_h        = 0;
    int  model_input_w  = 640;
    int  model_input_h  = 384;

    // 注入的模块 (由 main 创建并保证生命周期)
    LensModel*       lens     = nullptr;
    ZoomController*  zoom     = nullptr;
    GimbalState*     gimbal   = nullptr;
    IAngleSink*      sink     = nullptr;

    // 初始追踪目标 ID; -1 表示尚未指定
    int initial_target_id = -1;
  };

  explicit YoloTrackProcessor(const Config& cfg)
      : cfg_(cfg), target_id_(cfg.initial_target_id) {}

  bool Init() override {
    if (cfg_.model_path.empty() || cfg_.label_path.empty()) {
      fprintf(stderr, "[TRACK] model/label path empty\n"); return false;
    }
    if (!cfg_.lens || !cfg_.zoom || !cfg_.gimbal || !cfg_.sink) {
      fprintf(stderr, "[TRACK] lens/zoom/gimbal/sink must be injected\n"); return false;
    }

    init_post_process(cfg_.label_path);

    model_ = std::make_unique<Yolov8>(std::string(cfg_.model_path));
    if (model_->Init(nullptr, false) != 0) {
      fprintf(stderr, "[TRACK] Init RKNN model failed\n"); return false;
    }
    image_processor_ = std::make_unique<ImageProcess>(
        cfg_.frame_w, cfg_.frame_h, cfg_.model_input_w, cfg_.model_input_h);

    tracker_ = std::make_unique<byte_track::BYTETracker>(30, 30, 0.1f, 0.2f, 0.4f);

    // 让 ZoomController 知道镜头物理范围
    cfg_.zoom->SetLensFocalRange(cfg_.lens->focal_min(), cfg_.lens->focal_max());

    printf("[TRACK] Init OK: target_id=%d frame=%dx%d\n",
           target_id_.load(), cfg_.frame_w, cfg_.frame_h);
    return true;
  }

  bool Process(int /*worker_id*/,
               const std::shared_ptr<DrmFrame>& frame) override {
    if (!frame || frame->format != DrmFrame::NV12) return false;

    // 1. RGA: NV12 -> RGB
    auto rgb_frame = image_processor_->ConvertToRgb(frame);
    if (!rgb_frame) return true;

    // 2. YOLO 推理
    object_detect_result_list od_results;
    od_results.count = 0;
    model_->Inference(rgb_frame->vaddr, &od_results, image_processor_->get_letter_box());

    static int dbg_cnt = 0;
    ++dbg_cnt;
    if (dbg_cnt % 30 == 0) {
      fprintf(stderr, "[TRACK-DBG] frame#%d detected %d objects\n",
              dbg_cnt, od_results.count);
      for (int i = 0; i < od_results.count; i++) {
        fprintf(stderr, "[TRACK-DBG]   det[%d] cls=%d(%s) score=%.3f "
                "box=[%d,%d,%d,%d]\n",
                i, od_results.results[i].cls_id,
                coco_cls_to_name(od_results.results[i].cls_id),
                od_results.results[i].prop,
                od_results.results[i].box.left, od_results.results[i].box.top,
                od_results.results[i].box.right, od_results.results[i].box.bottom);
      }
    }

    // 3. 喂 ByteTrack
    std::vector<byte_track::Object> bt_objects;
    bt_objects.reserve(od_results.count);
    for (int i = 0; i < od_results.count; i++) {
      float x1 = od_results.results[i].box.left;
      float y1 = od_results.results[i].box.top;
      float w  = od_results.results[i].box.right  - x1;
      float h  = od_results.results[i].box.bottom - y1;
      bt_objects.push_back({byte_track::Rect<float>(x1, y1, w, h),
                             od_results.results[i].cls_id,
                             od_results.results[i].prop});
    }
    auto tracked = tracker_->update(bt_objects);

    if (dbg_cnt % 30 == 0) {
      fprintf(stderr, "[TRACK-DBG] tracked=%zu, target_id=%d ever_seen=%d\n",
              tracked.size(), target_id_.load(), ever_seen_.load());
      for (auto& tr : tracked) {
        fprintf(stderr, "[TRACK-DBG]   id=%zu label=%d score=%.2f "
                "rect=[%.0f,%.0f,%.0f,%.0f]\n",
                tr->getTrackId(), tr->getLabel(), tr->getScore(),
                tr->getRect().x(), tr->getRect().y(),
                tr->getRect().width(), tr->getRect().height());
      }
    }

    // 4. 找目标 ID 对应的 track
    int   target = target_id_.load();
    bool  target_found = false;
    float t_cx = 0, t_cy = 0, t_w = 0, t_h = 0;
    if (target >= 0) {
      for (const auto& tr : tracked) {
        if (static_cast<int>(tr->getTrackId()) == target) {
          const auto& r = tr->getRect();
          t_w  = r.width();
          t_h  = r.height();
          t_cx = r.x() + t_w * 0.5f;
          t_cy = r.y() + t_h * 0.5f;
          target_found = true;
          break;
        }
      }
    }

    // 5. ZoomController 状态更新
    float bbox_ratio = 0.0f;
    if (target_found && cfg_.frame_w > 0 && cfg_.frame_h > 0) {
      bbox_ratio = (t_w * t_h) / static_cast<float>(cfg_.frame_w * cfg_.frame_h);
    }
    cfg_.zoom->Update(bbox_ratio, target_found);
    bool zooming = cfg_.zoom->IsZooming();
    float focal  = cfg_.zoom->GetCurrentFocal();

    // 6. 角度输出决策
    AngleMsg msg;
    msg.track_id      = target;
    msg.focal_mm      = focal;
    msg.zooming       = zooming;
    msg.pts_ns        = frame->pts;
    msg.yaw_abs_deg   = cfg_.gimbal->GetYawAbs();
    msg.pitch_abs_deg = cfg_.gimbal->GetPitchAbs();

    if (zooming) {
      // 变焦中: 输出上一帧增量, lost=1
      if (ever_seen_) {
        msg.yaw_inc_deg   = last_yaw_inc_;
        msg.pitch_inc_deg = last_pitch_inc_;
        msg.lost = true;
        msg.clamped = false;
        cfg_.sink->Push(msg);
      }
    } else if (target_found) {
      // 实时角度
      float yaw_in, pitch_in_image;
      cfg_.lens->ComputeAngles(t_cx, t_cy, cfg_.frame_w, cfg_.frame_h,
                                focal, &yaw_in, &pitch_in_image);
      float yaw_out, pitch_out;
      bool clamped;
      cfg_.gimbal->ApplyIncrement(yaw_in, pitch_in_image,
                                   &yaw_out, &pitch_out, &clamped);
      msg.yaw_inc_deg   = yaw_out;
      msg.pitch_inc_deg = pitch_out;
      msg.yaw_abs_deg   = cfg_.gimbal->GetYawAbs();
      msg.pitch_abs_deg = cfg_.gimbal->GetPitchAbs();
      msg.lost          = false;
      msg.clamped       = clamped;
      cfg_.sink->Push(msg);

      last_yaw_inc_   = yaw_out;
      last_pitch_inc_ = pitch_out;
      ever_seen_      = true;
    } else if (ever_seen_) {
      // 跟丢: 输出上一帧
      msg.yaw_inc_deg   = last_yaw_inc_;
      msg.pitch_inc_deg = last_pitch_inc_;
      msg.lost          = true;
      msg.clamped       = false;
      cfg_.sink->Push(msg);
    }
    // (else: 从未见过, 不输出)

    // 7. 把 tracked 结果回写到 od_results 用于绘制 (保留原 cls_id, 修复原 bug)
    od_results.count = 0;
    for (const auto& tr : tracked) {
      if (od_results.count >= OBJ_NUMB_MAX_SIZE) break;
      const auto& r = tr->getRect();
      auto& dst = od_results.results[od_results.count];
      dst.box.left   = r.x();
      dst.box.top    = r.y();
      dst.box.right  = r.x() + r.width();
      dst.box.bottom = r.y() + r.height();
      dst.prop       = tr->getScore();
      dst.cls_id     = tr->getLabel();   // ★ 保留原始类别 (修复 bug)
      // 如需画 ID, 可借 cls_id 的扩展通道; 这里暂保留检测原貌
      od_results.count++;
    }

    if (dbg_cnt % 30 == 0) {
      fprintf(stderr, "[TRACK-DBG] final od_results.count=%d before DrawDetections\n", od_results.count);
    }

    if (od_results.count > 0) {
      image_processor_->DrawDetections(frame, od_results);
    }
    for (int i = 0; i < od_results.count; i++) {
      if (od_results.results_seg[i].seg_mask) {
        free(od_results.results_seg[i].seg_mask);
        od_results.results_seg[i].seg_mask = nullptr;
      }
    }

    frame->SyncEnd();
    return true;
  }

  std::string Name() const override { return "YoloTrackProcessor"; }
  int NumWorkers() const override { return 1; }

  // 留给后续控制接口使用 (stdin/UDP/...)
  void SetTargetId(int id) { target_id_.store(id); ever_seen_ = false; }

private:
  Config cfg_;
  std::unique_ptr<Yolov8>                   model_;
  std::unique_ptr<ImageProcess>             image_processor_;
  std::unique_ptr<byte_track::BYTETracker>  tracker_;

  std::atomic<int> target_id_;
  std::atomic<bool> ever_seen_{false};
  float            last_yaw_inc_   = 0.0f;
  float            last_pitch_inc_ = 0.0f;
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
