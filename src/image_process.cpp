#include "image_process.h"
#include "im2d.hpp"
#include "RgaUtils.h"
#include <cstring>

#define N_CLASS_COLORS 20
static unsigned char class_colors[][3] = {
    {255, 56, 56},   {255, 157, 151}, {255, 112, 31},  {255, 178, 29},
    {207, 210, 49},  {72, 249, 10},   {146, 204, 23},  {61, 219, 134},
    {26, 147, 52},   {0, 212, 187},   {44, 153, 168},  {0, 194, 255},
    {52, 69, 147},   {100, 115, 255}, {0, 24, 236},    {132, 56, 255},
    {82, 0, 133},    {203, 56, 255},  {255, 149, 200}, {255, 55, 199}};

ImageProcess::ImageProcess(int src_w, int src_h, int target_w, int target_h)
    : target_w_(target_w), target_h_(target_h) {
  // 直接 resize，不做 letterbox，使用独立的宽高缩放因子
  letterbox_.scale_w = static_cast<double>(target_w) / src_w;
  letterbox_.scale_h = static_cast<double>(target_h) / src_h;
  letterbox_.x_pad   = 0;
  letterbox_.y_pad   = 0;
}

// RGA 硬件加速: resize(target_w × target_h) + BGR→RGB
std::unique_ptr<cv::Mat> ImageProcess::Convert(const cv::Mat &src) {
  if (src.empty()) return nullptr;

  // 1. RGA resize: src (BGR) → resized_bgr (BGR, target_w_ × target_h_)
  cv::Mat resized_bgr(target_h_, target_w_, CV_8UC3);
  rga_buffer_t src_buf = wrapbuffer_virtualaddr(
      src.data, src.cols, src.rows, RK_FORMAT_BGR_888);
  rga_buffer_t resize_buf = wrapbuffer_virtualaddr(
      resized_bgr.data, target_w_, target_h_, RK_FORMAT_BGR_888);

  IM_STATUS status = imresize(src_buf, resize_buf);
  if (status != IM_STATUS_SUCCESS) {
    fprintf(stderr, "[ERROR] RGA imresize failed: %s\n", imStrError(status));
    return nullptr;
  }

  // 2. RGA cvtcolor: resized_bgr (BGR) → resized_rgb (RGB)
  auto rgb_img = std::make_unique<cv::Mat>(target_h_, target_w_, CV_8UC3);
  rga_buffer_t rgb_buf = wrapbuffer_virtualaddr(
      rgb_img->data, target_w_, target_h_, RK_FORMAT_RGB_888);

  status = imcvtcolor(resize_buf, rgb_buf, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888);
  if (status != IM_STATUS_SUCCESS) {
    fprintf(stderr, "[ERROR] RGA imcvtcolor failed: %s\n", imStrError(status));
    return nullptr;
  }

  return rgb_img;
}

const letterbox_t &ImageProcess::get_letter_box() { return letterbox_; }

void ImageProcess::ImagePostProcess(cv::Mat &image,
                                    object_detect_result_list &od_results) {
  // 分割 mask 叠加
  if (od_results.count >= 1) {
    int width = image.rows;
    int height = image.cols;
    auto *ori_img = image.ptr();
    uint8_t *seg_mask = od_results.results_seg[0].seg_mask;
    float alpha = 0.5f;
    if (seg_mask != nullptr) {
      for (int j = 0; j < height; j++) {
        for (int k = 0; k < width; k++) {
          int offset = 3 * (j * width + k);
          if (seg_mask[j * width + k] != 0) {
            int ci = seg_mask[j * width + k] % N_CLASS_COLORS;
            ori_img[offset + 0] = (uint8_t)clamp(
                class_colors[ci][0] * (1 - alpha) + ori_img[offset + 0] * alpha, 0, 255);
            ori_img[offset + 1] = (uint8_t)clamp(
                class_colors[ci][1] * (1 - alpha) + ori_img[offset + 1] * alpha, 0, 255);
            ori_img[offset + 2] = (uint8_t)clamp(
                class_colors[ci][2] * (1 - alpha) + ori_img[offset + 2] * alpha, 0, 255);
          }
        }
      }
      free(seg_mask);
    }
  }

  if (od_results.model_type == ModelType::DETECTION ||
      od_results.model_type == ModelType::V10_DETECTION) {
    ProcessDetectionImage(image, od_results);
  } else if (od_results.model_type == ModelType::OBB) {
    ProcessOBBImage(image, od_results);
  } else if (od_results.model_type == ModelType::POSE) {
    ProcessPoseImage(image, od_results);
  }
}

// ==================== Detection ====================
void ImageProcess::ProcessDetectionImage(
    cv::Mat &image, object_detect_result_list &od_results) const {
  for (int i = 0; i < od_results.count; ++i) {
    auto *r = &od_results.results[i];
    cv::rectangle(image, cv::Point(r->box.left, r->box.top),
                  cv::Point(r->box.right, r->box.bottom),
                  cv::Scalar(0, 0, 255), 2);
    char text[256];
    sprintf(text, "%s %.1f%%", coco_cls_to_name(r->cls_id), r->prop * 100);
    cv::putText(image, text, cv::Point(r->box.left, r->box.top + 20),
                cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0, 0), 2);
  }
}

// ==================== OBB ====================
static void DrawRotatedRect(cv::Mat &image, float x, float y, float w, float h,
                            float theta, const cv::Scalar &color, int thickness) {
  cv::RotatedRect rr(cv::Point2f(x, y), cv::Size2f(w, h), theta);
  cv::Point2f vertices[4];
  rr.points(vertices);
  for (int i = 0; i < 4; i++)
    cv::line(image, vertices[i], vertices[(i + 1) % 4], color, thickness);
}

void ImageProcess::ProcessOBBImage(
    cv::Mat &image, const object_detect_result_list &od_results) const {
  for (int i = 0; i < od_results.count; ++i) {
    auto &r = od_results.results_obb[i];
    DrawRotatedRect(image, r.box.x, r.box.y, r.box.w, r.box.h,
                    r.box.theta * 180.0f / CV_PI, cv::Scalar(0, 255, 0), 2);
  }
}

// ==================== Pose ====================
static void drawSkeleton(cv::Mat &img, const std::vector<cv::Point> &points,
                          const std::vector<int> &pairs,
                          const cv::Scalar &color, int thickness) {
  for (size_t i = 0; i < pairs.size(); i += 2) {
    int a = pairs[i], b = pairs[i + 1];
    if (points[a].x != -1 && points[b].x != -1)
      cv::line(img, points[a], points[b], color, thickness);
  }
}

void ImageProcess::ProcessPoseImage(
    cv::Mat &image, object_detect_result_list &od_results) const {
  for (int i = 0; i < od_results.count; ++i) {
    auto *r = &od_results.results[i];
    cv::rectangle(image, cv::Point(r->box.left, r->box.top),
                  cv::Point(r->box.right, r->box.bottom),
                  cv::Scalar(0, 0, 255), 2);
    std::vector<cv::Point> pts(17);
    for (int j = 0; j < 17; ++j) {
      if (od_results.results_pose[i].visibility[j] <= 0.6) {
        pts[j] = cv::Point(-1, -1);
        continue;
      }
      pts[j] = cv::Point((int)od_results.results_pose[i].kpt[j * 2],
                          (int)od_results.results_pose[i].kpt[j * 2 + 1]);
      cv::circle(image, pts[j], 10, cv::Scalar(0, 0, 255), cv::FILLED);
    }
    std::vector<int> pairs = {0,1, 1,3, 0,2, 2,4, 0,5, 5,7, 7,9,
                              0,6, 6,8, 8,10, 5,6, 11,12, 11,5,
                              12,6, 11,13, 12,14, 13,15, 14,16};
    drawSkeleton(image, pts, pairs, cv::Scalar(255, 0, 0), 2);
  }
}
