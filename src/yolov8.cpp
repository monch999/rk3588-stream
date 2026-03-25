#include "yolov8.h"
#include "utils.h"
#include "postprocess.h"
#include <cstdio>
#include <cstring>

static const int RK3588_NPU_CORES = 3;

static int get_core_num() {
  static int core_num = 0;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  int temp = core_num % RK3588_NPU_CORES;
  core_num++;
  return temp;
}

static int read_data_from_file(const char *path, char **out_data) {
  FILE *fp = fopen(path, "rb");
  if (!fp) return -1;
  fseek(fp, 0, SEEK_END);
  int file_size = ftell(fp);
  char *data = (char *)malloc(file_size + 1);
  data[file_size] = 0;
  fseek(fp, 0, SEEK_SET);
  if (file_size != (int)fread(data, 1, file_size, fp)) {
    free(data);
    fclose(fp);
    return -1;
  }
  fclose(fp);
  *out_data = data;
  return file_size;
}

Yolov8::Yolov8(std::string &&model_path) : model_path_(std::move(model_path)) {}

int Yolov8::Init(rknn_context *ctx_in, bool copy_weight) {
  char *model = nullptr;
  int model_len = read_data_from_file(model_path_.c_str(), &model);
  if (model == nullptr) {
    fprintf(stderr, "[ERROR] Load model failed: %s\n", model_path_.c_str());
    return -1;
  }

  int ret = 0;
  if (copy_weight) {
    ret = rknn_dup_context(ctx_in, &ctx_);
    if (ret != RKNN_SUCC) {
      fprintf(stderr, "[ERROR] rknn_dup_context failed: %d\n", ret);
      free(model);
      return -1;
    }
  } else {
    ret = rknn_init(&ctx_, model, model_len, 0, NULL);
    free(model);
    if (ret != RKNN_SUCC) {
      fprintf(stderr, "[ERROR] rknn_init failed: %d\n", ret);
      return -1;
    }
  }

  // 绑定 NPU 核心
  rknn_core_mask core_mask;
  switch (get_core_num()) {
    case 0: core_mask = RKNN_NPU_CORE_0; break;
    case 1: core_mask = RKNN_NPU_CORE_1; break;
    default: core_mask = RKNN_NPU_CORE_2; break;
  }
  rknn_set_core_mask(ctx_, core_mask);

  // 查询输入输出信息
  rknn_input_output_num io_num;
  ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) return -1;

  // 获取输入属性
  rknn_tensor_attr *input_attrs =
      (rknn_tensor_attr *)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
  for (uint32_t i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs[i],
               sizeof(rknn_tensor_attr));
  }

  // 获取输出属性并判断模型类型
  rknn_tensor_attr *output_attrs =
      (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));

  model_type_ = (io_num.n_output == 13) ? ModelType::SEGMENT
                                        : ModelType::DETECTION;
  for (uint32_t i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i],
               sizeof(rknn_tensor_attr));
    if (i == 2) {
      if (strstr(output_attrs[i].name, "angle"))
        model_type_ = ModelType::OBB;
      if (strstr(output_attrs[i].name, "kpt"))
        model_type_ = ModelType::POSE;
      if (strstr(output_attrs[i].name, "yolov10"))
        model_type_ = ModelType::V10_DETECTION;
    }
  }

  // 填充 app context
  app_ctx_.rknn_ctx = ctx_;
  app_ctx_.is_quant = (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
                       output_attrs[0].type == RKNN_TENSOR_INT8);
  app_ctx_.io_num = io_num;
  app_ctx_.input_attrs =
      (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
  memcpy(app_ctx_.input_attrs, input_attrs,
         io_num.n_input * sizeof(rknn_tensor_attr));
  app_ctx_.output_attrs =
      (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
  memcpy(app_ctx_.output_attrs, output_attrs,
         io_num.n_output * sizeof(rknn_tensor_attr));

  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
    app_ctx_.model_channel = input_attrs[0].dims[1];
    app_ctx_.model_height = input_attrs[0].dims[2];
    app_ctx_.model_width = input_attrs[0].dims[3];
  } else {
    app_ctx_.model_height = input_attrs[0].dims[1];
    app_ctx_.model_width = input_attrs[0].dims[2];
    app_ctx_.model_channel = input_attrs[0].dims[3];
  }

  free(input_attrs);
  free(output_attrs);

  // 初始化输入输出参数
  inputs_ = std::make_unique<rknn_input[]>(io_num.n_input);
  outputs_ = std::make_unique<rknn_output[]>(io_num.n_output);
  inputs_[0].index = 0;
  inputs_[0].type = RKNN_TENSOR_UINT8;
  inputs_[0].fmt = RKNN_TENSOR_NHWC;
  inputs_[0].size = app_ctx_.model_width * app_ctx_.model_height *
                    app_ctx_.model_channel;
  inputs_[0].buf = nullptr;

  printf("[INFO ] Model loaded: %dx%d, type=%d, quant=%d\n",
         app_ctx_.model_width, app_ctx_.model_height,
         (int)model_type_, app_ctx_.is_quant);
  return 0;
}

Yolov8::~Yolov8() { DeInit(); }

int Yolov8::DeInit() {
  if (app_ctx_.rknn_ctx != 0) {
    rknn_destroy(app_ctx_.rknn_ctx);
    app_ctx_.rknn_ctx = 0;
  }
  free(app_ctx_.input_attrs);
  app_ctx_.input_attrs = nullptr;
  free(app_ctx_.output_attrs);
  app_ctx_.output_attrs = nullptr;
  return 0;
}

rknn_context *Yolov8::get_rknn_context() { return &ctx_; }

int Yolov8::Inference(void *image_buf, object_detect_result_list *od_results,
                      letterbox_t letter_box) {
  inputs_[0].buf = image_buf;
  int ret = rknn_inputs_set(app_ctx_.rknn_ctx, app_ctx_.io_num.n_input,
                            inputs_.get());
  if (ret < 0) return -1;

  ret = rknn_run(app_ctx_.rknn_ctx, nullptr);
  if (ret != RKNN_SUCC) return -1;

  for (uint32_t i = 0; i < app_ctx_.io_num.n_output; ++i) {
    outputs_[i].index = i;
    outputs_[i].want_float = (!app_ctx_.is_quant);
  }

  outputs_lock_.lock();
  ret = rknn_outputs_get(app_ctx_.rknn_ctx, app_ctx_.io_num.n_output,
                         outputs_.get(), nullptr);
  if (ret != RKNN_SUCC) {
    outputs_lock_.unlock();
    return -1;
  }

  memset(od_results, 0, sizeof(object_detect_result_list));
  od_results->model_type = model_type_;

  if (model_type_ == ModelType::SEGMENT) {
    post_process_seg(&app_ctx_, outputs_.get(), &letter_box, BOX_THRESH,
                     NMS_THRESH, od_results);
  } else if (model_type_ == ModelType::DETECTION ||
             model_type_ == ModelType::V10_DETECTION) {
    post_process(&app_ctx_, outputs_.get(), &letter_box, BOX_THRESH,
                 NMS_THRESH, od_results);
  } else if (model_type_ == ModelType::OBB) {
    post_process_obb(&app_ctx_, outputs_.get(), &letter_box, BOX_THRESH,
                     NMS_THRESH, od_results);
  } else if (model_type_ == ModelType::POSE) {
    post_process_pose(&app_ctx_, outputs_.get(), &letter_box, BOX_THRESH,
                      NMS_THRESH, od_results);
  }
  od_results->model_type = model_type_;

  rknn_outputs_release(app_ctx_.rknn_ctx, app_ctx_.io_num.n_output,
                       outputs_.get());
  outputs_lock_.unlock();
  return 0;
}

int Yolov8::get_model_width() { return app_ctx_.model_width; }
int Yolov8::get_model_height() { return app_ctx_.model_height; }
