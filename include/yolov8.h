#pragma once
#include <memory>
#include <mutex>
#include <string>
#include "rknn_api.h"
#include "postprocess.h"

class Yolov8 {
public:
  explicit Yolov8(std::string &&model_path);
  ~Yolov8();

  int Init(rknn_context *ctx_in, bool copy_weight);
  int Inference(void *image_buf, object_detect_result_list *od_results,
                letterbox_t letter_box);
  int DeInit();

  rknn_context *get_rknn_context();
  int get_model_width();
  int get_model_height();

private:
  std::string model_path_;
  rknn_context ctx_ = 0;
  rknn_app_context_t app_ctx_ = {};
  ModelType model_type_ = ModelType::DETECTION;
  std::unique_ptr<rknn_input[]> inputs_;
  std::unique_ptr<rknn_output[]> outputs_;
  std::mutex outputs_lock_;
};
