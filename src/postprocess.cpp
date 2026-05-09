#include "postprocess.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <set>
#include <vector>
#include <algorithm>

// ==================== 全局标签表 ====================
static constexpr int MAX_LABELS = 256;
static char *labels[MAX_LABELS];
static int num_labels = 0;

// ==================== 工具函数 ====================
int clamp(float val, int min, int max) {
  return val > min ? (val < max ? val : max) : min;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
  return ((float)qnt - (float)zp) * scale;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
  float dst_val = (f32 / scale) + zp;
  if (dst_val < -128) dst_val = -128;
  if (dst_val >  127) dst_val = 127;
  return (int8_t)dst_val;
}

// ==================== 标签文件加载 ====================
static char *readLine(FILE *fp) {
  size_t cap = 64;
  char *buf = (char *)malloc(cap);
  if (!buf) return nullptr;

  int ch, i = 0;
  while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
    if ((size_t)i + 1 >= cap) {
      cap *= 2;
      char *tmp = (char *)realloc(buf, cap);
      if (!tmp) { free(buf); return nullptr; }
      buf = tmp;
    }
    buf[i++] = (char)ch;
  }
  // 去除行尾 \r (Windows 换行)
  if (i > 0 && buf[i - 1] == '\r') i--;
  buf[i] = '\0';

  if (ch == EOF && i == 0) { free(buf); return nullptr; }
  return buf;
}

int init_post_process(const std::string &label_path) {
  // 先释放旧标签 (允许重复调用)
  deinit_post_process();

  FILE *fp = fopen(label_path.c_str(), "r");
  if (!fp) {
    fprintf(stderr, "[POST] Failed to open label file: %s\n", label_path.c_str());
    return -1;
  }

  char *line;
  while ((line = readLine(fp)) != nullptr && num_labels < MAX_LABELS) {
    if (line[0] == '\0') { free(line); continue; }  // 跳过空行
    labels[num_labels++] = line;
  }
  fclose(fp);

  fprintf(stderr, "[POST] Loaded %d labels from %s\n", num_labels, label_path.c_str());
  return 0;
}

void deinit_post_process() {
  for (int i = 0; i < num_labels; i++) {
    free(labels[i]);
    labels[i] = nullptr;
  }
  num_labels = 0;
}

const char *coco_cls_to_name(int cls_id) {
  if (cls_id < 0 || cls_id >= num_labels) return "?";
  return labels[cls_id] ? labels[cls_id] : "?";
}

int get_num_labels() { return num_labels; }

// ==================== DFL 解码 ====================
void compute_dfl(float *tensor, int dfl_len, float *box) {
  for (int b = 0; b < 4; b++) {
    float exp_sum = 0, acc_sum = 0;
    for (int i = 0; i < dfl_len; i++) {
      float e = expf(tensor[i + b * dfl_len]);
      exp_sum += e;
      // 延迟归一化
      acc_sum += e * i;
    }
    box[b] = acc_sum / exp_sum;
  }
}

// ==================== NMS ====================
static float calc_iou(float x0, float y0, float w0, float h0,
                       float x1, float y1, float w1, float h1) {
  float ax2 = x0 + w0, ay2 = y0 + h0;
  float bx2 = x1 + w1, by2 = y1 + h1;
  float inter_w = fmaxf(0, fminf(ax2, bx2) - fmaxf(x0, x1));
  float inter_h = fmaxf(0, fminf(ay2, by2) - fmaxf(y0, y1));
  float inter = inter_w * inter_h;
  float uni = w0 * h0 + w1 * h1 - inter;
  return uni > 0 ? inter / uni : 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left,
                                      int right, std::vector<int> &indices) {
  if (left >= right) return 0;
  float key = input[left];
  int key_index = indices[left];
  int low = left, high = right;
  while (low < high) {
    while (low < high && input[high] <= key) high--;
    input[low] = input[high];
    indices[low] = indices[high];
    while (low < high && input[low] >= key) low++;
    input[high] = input[low];
    indices[high] = indices[low];
  }
  input[low] = key;
  indices[low] = key_index;
  quick_sort_indice_inverse(input, left, low - 1, indices);
  quick_sort_indice_inverse(input, low + 1, right, indices);
  return 0;
}

static void nms(int validCount, std::vector<float> &boxes,
                std::vector<int> &classIds, std::vector<int> &order,
                int filterId, float threshold) {
  for (int i = 0; i < validCount; ++i) {
    if (order[i] == -1 || classIds[order[i]] != filterId) continue;
    int n = order[i];
    for (int j = i + 1; j < validCount; ++j) {
      if (order[j] == -1 || classIds[order[j]] != filterId) continue;
      int m = order[j];
      float iou = calc_iou(
          boxes[n*4], boxes[n*4+1], boxes[n*4+2], boxes[n*4+3],
          boxes[m*4], boxes[m*4+1], boxes[m*4+2], boxes[m*4+3]);
      if (iou > threshold) order[j] = -1;
    }
  }
}

// ==================== INT8 单分支解码 ====================
static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp,
                      float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      int n_classes,
                      std::vector<float> &boxes,
                      std::vector<float> &objProbs,
                      std::vector<int> &classId,
                      float threshold) {
  int validCount = 0;
  int grid_len = grid_h * grid_w;
  int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
  int8_t score_sum_thres_i8 =
      qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

  for (int i = 0; i < grid_h; i++) {
    for (int j = 0; j < grid_w; j++) {
      int offset = i * grid_w + j;

      // score_sum 快速过滤
      if (score_sum_tensor != nullptr) {
        if (score_sum_tensor[offset] < score_sum_thres_i8) continue;
      }

      // 遍历所有类别找最大 score
      int8_t max_score = -128;
      int max_class_id = -1;
      int class_offset = offset;
      for (int c = 0; c < n_classes; c++) {
        if (score_tensor[class_offset] > max_score) {
          max_score = score_tensor[class_offset];
          max_class_id = c;
        }
        class_offset += grid_len;
      }

      if (max_score <= score_thres_i8) continue;

      // 解码 box
      offset = i * grid_w + j;
      float box[4];
      float before_dfl[dfl_len * 4];
      for (int k = 0; k < dfl_len * 4; k++) {
        before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
        offset += grid_len;
      }
      compute_dfl(before_dfl, dfl_len, box);

      float x1 = (-box[0] + j + 0.5f) * stride;
      float y1 = (-box[1] + i + 0.5f) * stride;
      float x2 = ( box[2] + j + 0.5f) * stride;
      float y2 = ( box[3] + i + 0.5f) * stride;
      boxes.push_back(x1);
      boxes.push_back(y1);
      boxes.push_back(x2 - x1);  // w
      boxes.push_back(y2 - y1);  // h

      objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
      classId.push_back(max_class_id);
      validCount++;
    }
  }
  return validCount;
}

// ==================== post_process: YOLOv8 Detection ====================
int post_process(rknn_app_context_t *app_ctx, rknn_output *outputs,
                 letterbox_t *letter_box, float conf_threshold,
                 float nms_threshold, object_detect_result_list *od_results) {
  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;
  int validCount = 0;

  int model_in_w = app_ctx->model_width;
  int model_in_h = app_ctx->model_height;
  int dfl_len = app_ctx->output_attrs[0].dims[1] / 4;
  int output_per_branch = app_ctx->io_num.n_output / 3;

  // 从模型输出张量推断类别数 (而非依赖全局 num_labels)
  int n_classes = app_ctx->output_attrs[1].dims[1];

  for (int i = 0; i < 3; i++) {
    int box_idx   = i * output_per_branch;
    int score_idx = i * output_per_branch + 1;

    int grid_h = app_ctx->output_attrs[box_idx].dims[2];
    int grid_w = app_ctx->output_attrs[box_idx].dims[3];
    int stride = model_in_h / grid_h;

    void *score_sum = nullptr;
    int32_t score_sum_zp = 0;
    float score_sum_scale = 1.0f;
    if (output_per_branch == 3) {
      score_sum      = outputs[i * output_per_branch + 2].buf;
      score_sum_zp   = app_ctx->output_attrs[i * output_per_branch + 2].zp;
      score_sum_scale = app_ctx->output_attrs[i * output_per_branch + 2].scale;
    }

    validCount += process_i8(
        (int8_t *)outputs[box_idx].buf,
        app_ctx->output_attrs[box_idx].zp,
        app_ctx->output_attrs[box_idx].scale,
        (int8_t *)outputs[score_idx].buf,
        app_ctx->output_attrs[score_idx].zp,
        app_ctx->output_attrs[score_idx].scale,
        (int8_t *)score_sum, score_sum_zp, score_sum_scale,
        grid_h, grid_w, stride, dfl_len, n_classes,
        filterBoxes, objProbs, classId, conf_threshold);
  }

  if (validCount <= 0) {
    od_results->count = 0;
    return 0;
  }

  // NMS
  std::vector<int> indexArray(validCount);
  for (int i = 0; i < validCount; i++) indexArray[i] = i;
  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(classId.begin(), classId.end());
  for (auto c : class_set) {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  // 输出结果, 坐标映射回原图
  int last_count = 0;
  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) continue;
    int n = indexArray[i];

    float x1 = filterBoxes[n * 4 + 0] - letter_box->x_pad;
    float y1 = filterBoxes[n * 4 + 1] - letter_box->y_pad;
    float x2 = x1 + filterBoxes[n * 4 + 2];
    float y2 = y1 + filterBoxes[n * 4 + 3];

    od_results->results[last_count].box.left =
        (int)(clamp(x1, 0, model_in_w) / letter_box->scale_w);
    od_results->results[last_count].box.top =
        (int)(clamp(y1, 0, model_in_h) / letter_box->scale_h);
    od_results->results[last_count].box.right =
        (int)(clamp(x2, 0, model_in_w) / letter_box->scale_w);
    od_results->results[last_count].box.bottom =
        (int)(clamp(y2, 0, model_in_h) / letter_box->scale_h);
    // ★ 修复: 使用 objProbs[n] 而不是 objProbs[i],
    //   排序后 i 是排序位置, n 才是原始检测索引
    od_results->results[last_count].prop = objProbs[n];
    od_results->results[last_count].cls_id = classId[n];
    last_count++;
  }
  od_results->count = last_count;
  return 0;
}
