#pragma once
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "image_process.h"
#include "postprocess.h"
#include "yolov8.h"

// ==================== 帧处理器接口 ====================
// 所有算法（YOLO、植被指数、火点检测）统一实现此接口
// Process() 必须线程安全（当 NumWorkers() > 1 时多线程调用）
class IFrameProcessor {
public:
  virtual ~IFrameProcessor() = default;

  // 初始化资源（模型加载等）
  virtual bool Init() = 0;

  // 处理一帧
  // worker_id: 调用者的工作线程 ID [0, NumWorkers())
  // input:     原始帧 (只读，不可修改)
  // output:    处理结果帧 (由实现者分配)
  // 返回 true 表示处理成功
  virtual bool Process(int worker_id, const cv::Mat &input, cv::Mat &output) = 0;

  // 算法名称（用于日志）
  virtual std::string Name() const = 0;

  // 推荐的并行工作线程数（默认1）
  // 对于 YOLO，等于 NPU 核心数
  virtual int NumWorkers() const { return 1; }
};

// ==================== RGB YOLO 推理处理器 ====================
// 封装现有 Yolov8 + ImageProcess，支持多 NPU 核心并行推理
class YoloProcessor : public IFrameProcessor {
public:
  struct Config {
    std::string model_path;
    std::string label_path;
    int  num_cores        = 3;    // NPU 核心数
    int  frame_w          = 0;    // 视频帧宽（Init 前必须设置）
    int  frame_h          = 0;    // 视频帧高
    int  model_input_w    = 640;  // 模型输入宽
    int  model_input_h    = 384;  // 模型输入高
  };

  explicit YoloProcessor(const Config &cfg) : cfg_(cfg) {}

  bool Init() override {
    // 初始化后处理（标签加载，全局状态）
    init_post_process(const_cast<std::string &>(cfg_.label_path));

    // 为每个 NPU 核心创建独立的模型实例和图像处理器
    for (int i = 0; i < cfg_.num_cores; i++) {
      models_.push_back(
          std::make_shared<Yolov8>(std::string(cfg_.model_path)));
      image_processors_.push_back(std::make_unique<ImageProcess>(
          cfg_.frame_w, cfg_.frame_h, cfg_.model_input_w, cfg_.model_input_h));
    }

    // 初始化 RKNN 模型（第一个完整加载，后续共享权重）
    for (int i = 0; i < cfg_.num_cores; i++) {
      if (models_[i]->Init(models_[0]->get_rknn_context(), i != 0) != 0) {
        fprintf(stderr, "[YOLO  ] Init RKNN model %d failed\n", i);
        return false;
      }
    }

    printf("[YOLO  ] Init OK: %d NPU cores, input %dx%d -> %dx%d\n",
           cfg_.num_cores, cfg_.frame_w, cfg_.frame_h,
           cfg_.model_input_w, cfg_.model_input_h);
    return true;
  }

  bool Process(int worker_id, const cv::Mat &input, cv::Mat &output) override {
    // 克隆输入帧（原始帧可能同时被 raw 推流使用）
    output = input.clone();

    auto &ip    = image_processors_[worker_id];
    auto &model = models_[worker_id];

    // RGA 硬件加速: resize + BGR→RGB
    auto rgb = ip->Convert(output);
    if (!rgb) return false;

    // NPU 推理
    object_detect_result_list od_results;
    model->Inference(rgb->ptr(), &od_results, ip->get_letter_box());

    // 在克隆帧上绘制检测结果
    ip->ImagePostProcess(output, od_results);

    // 清理分割掩码指针（已被 ImagePostProcess 释放）
    for (int i = 0; i < od_results.count; i++)
      od_results.results_seg[i].seg_mask = nullptr;

    return true;
  }

  std::string Name() const override { return "YoloProcessor"; }
  int NumWorkers() const override { return cfg_.num_cores; }

private:
  Config cfg_;
  std::vector<std::shared_ptr<Yolov8>>           models_;
  std::vector<std::unique_ptr<ImageProcess>>      image_processors_;
};

// ==================== 多光谱植被指数处理器 (接口预留) ====================
// 当前为 stub 实现，后续接入实际算法
//
// 预期输入: 多光谱图像（测试阶段为 BGR，正式阶段为 6 通道原始数据）
// 预期输出: 植被指数伪彩色可视化图（BGR）
//
// 常用植被指数:
//   NDVI = (NIR - Red) / (NIR + Red)
//   GNDVI = (NIR - Green) / (NIR + Green)
//   EVI, SAVI 等
//
// TODO: 实现时需要:
//   1. 定义 6 通道波段映射关系
//   2. 提取 NIR 和可见光波段
//   3. 计算植被指数
//   4. 生成伪彩色可视化输出
class NdviProcessor : public IFrameProcessor {
public:
  bool Init() override {
    printf("[NDVI  ] Stub processor initialized (algorithm pending)\n");
    return true;
  }

  bool Process(int /*worker_id*/, const cv::Mat &input,
               cv::Mat &output) override {
    // ---- 算法接口预留 ----
    // 在此处实现植被指数计算逻辑
    // input: 当前为 BGR 测试帧，正式接入后为 6 通道多光谱帧
    // output: 处理后的可视化结果帧（BGR）
    output = input.clone();
    return true;
  }

  std::string Name() const override { return "NdviProcessor"; }
};

// ==================== 热红外火点检测处理器 (接口预留) ====================
// 当前为 stub 实现，后续接入实际算法
//
// 预期输入: 热红外图像 640×512（测试阶段为 BGR，正式阶段可能为 16-bit 温度数据）
// 预期输出: 标注火点位置的可视化图（BGR）
//
// TODO: 实现时需要:
//   1. 温度阈值分割（识别高温区域）
//   2. 连通域分析（提取火点候选区域）
//   3. 形态学滤波（去除噪点）
//   4. 在输出帧上标注火点位置和温度信息
class FireDetector : public IFrameProcessor {
public:
  bool Init() override {
    printf("[FIRE  ] Stub processor initialized (algorithm pending)\n");
    return true;
  }

  bool Process(int /*worker_id*/, const cv::Mat &input,
               cv::Mat &output) override {
    // ---- 算法接口预留 ----
    // 在此处实现火点检测逻辑
    // input: 当前为 BGR 测试帧，正式接入后为热红外原始帧
    // output: 标注火点位置的可视化帧（BGR）
    output = input.clone();
    return true;
  }

  std::string Name() const override { return "FireDetector"; }
};
