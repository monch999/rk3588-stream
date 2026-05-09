#pragma once
#include <stdint.h>
#include <string>
#include "rknn_api.h"

// ==================== 常量定义 ====================
#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define NMS_THRESH        0.45
#define BOX_THRESH        0.25

// ==================== 坐标映射参数 ====================
typedef struct {
  double scale_w;   // 宽度缩放因子: target_w / src_w
  double scale_h;   // 高度缩放因子: target_h / src_h
  int x_pad;        // x 方向填充 (无 letterbox 时为 0)
  int y_pad;        // y 方向填充 (无 letterbox 时为 0)
} letterbox_t;

// ==================== 检测框 ====================
typedef struct {
  int left;
  int top;
  int right;
  int bottom;
} image_rect_t;

// ==================== 检测结果 ====================
typedef struct {
  image_rect_t box;
  float prop;
  int cls_id;
} object_detect_result;

// ==================== 分割结果 (保留结构体用于兼容) ====================
typedef struct {
  uint8_t *seg_mask;
} object_segment_result;

// ==================== 结果列表 ====================
typedef struct {
  int count;
  object_detect_result results[OBJ_NUMB_MAX_SIZE];
  object_segment_result results_seg[OBJ_NUMB_MAX_SIZE];  // 保留兼容, 实际未使用
} object_detect_result_list;

// ==================== RKNN App 上下文 ====================
typedef struct {
  rknn_context rknn_ctx;
  rknn_input_output_num io_num;
  rknn_tensor_attr *input_attrs;
  rknn_tensor_attr *output_attrs;
  int model_channel;
  int model_width;
  int model_height;
  bool is_quant;
} rknn_app_context_t;

// ==================== 函数声明 ====================
int init_post_process(const std::string &label_path);
void deinit_post_process();
const char *coco_cls_to_name(int cls_id);
int get_num_labels();

int clamp(float val, int min, int max);
void compute_dfl(float *tensor, int dfl_len, float *box);

int post_process(rknn_app_context_t *app_ctx, rknn_output *outputs,
                 letterbox_t *letter_box, float conf_threshold,
                 float nms_threshold, object_detect_result_list *od_results);
