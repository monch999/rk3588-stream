#pragma once
#include <stdint.h>
#include <string>
#include "rknn_api.h"

// ==================== 常量定义 ====================
#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define OBJ_CLASS_NUM     80
#define NMS_THRESH        0.45
#define BOX_THRESH        0.25
#define PROTO_CHANNEL     32
#define PROTO_HEIGHT      160
#define PROTO_WEIGHT      160

// ==================== 模型类型 ====================
enum class ModelType {
  DETECTION,
  SEGMENT,
  OBB,
  POSE,
  V10_DETECTION,
};

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

// ==================== OBB 旋转框 ====================
typedef struct {
  float x;
  float y;
  float w;
  float h;
  float theta;
} image_obb_rect_t;

// ==================== 检测结果 ====================
typedef struct {
  image_rect_t box;
  float prop;
  int cls_id;
} object_detect_result;

// ==================== OBB 检测结果 ====================
typedef struct {
  image_obb_rect_t box;
  float prop;
  int cls_id;
} object_detect_obb_result;

// ==================== 分割结果 ====================
typedef struct {
  uint8_t *seg_mask;
} object_segment_result;

// ==================== 姿态结果 ====================
typedef struct {
  float kpt[34];         // 17 keypoints * 2 (x, y)
  float visibility[17];
} object_pose_result;

// ==================== 结果列表 ====================
typedef struct {
  int count;
  ModelType model_type;
  object_detect_result results[OBJ_NUMB_MAX_SIZE];
  object_detect_obb_result results_obb[OBJ_NUMB_MAX_SIZE];
  object_segment_result results_seg[OBJ_NUMB_MAX_SIZE];
  object_pose_result results_pose[OBJ_NUMB_MAX_SIZE];
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
int init_post_process(std::string &label_path);
void deinit_post_process();
const char *coco_cls_to_name(int cls_id);
int clamp(float val, int min, int max);
void compute_dfl(float *tensor, int dfl_len, float *box);

int post_process(rknn_app_context_t *app_ctx, rknn_output *outputs,
                 letterbox_t *letter_box, float conf_threshold,
                 float nms_threshold, object_detect_result_list *od_results);

int post_process_seg(rknn_app_context_t *app_ctx, rknn_output *outputs,
                     letterbox_t *letter_box, float conf_threshold,
                     float nms_threshold, object_detect_result_list *od_results);

int post_process_obb(rknn_app_context_t *app_ctx, rknn_output *outputs,
                     letterbox_t *letter_box, float conf_threshold,
                     float nms_threshold, object_detect_result_list *od_results);

int post_process_pose(rknn_app_context_t *app_ctx, rknn_output *outputs,
                      letterbox_t *letter_box, float conf_threshold,
                      float nms_threshold, object_detect_result_list *od_results);

int post_process_v10_detection(rknn_app_context_t *app_ctx,
                               rknn_output *outputs, letterbox_t *letter_box,
                               float conf_threshold,
                               object_detect_result_list *od_results);
