#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include "postprocess.h"

class ImageProcess {
public:
  // target_w / target_h: 模型输入宽高 (如 640×384)，直接 resize 不做 letterbox
  ImageProcess(int src_w, int src_h, int target_w, int target_h);

  // RGA 硬件加速 resize + BGR→RGB，输出 target_w × target_h RGB 图
  std::unique_ptr<cv::Mat> Convert(const cv::Mat &src);

  const letterbox_t &get_letter_box();
  void ImagePostProcess(cv::Mat &image, object_detect_result_list &od_results);

private:
  void ProcessDetectionImage(cv::Mat &image,
                             object_detect_result_list &od_results) const;
  void ProcessOBBImage(cv::Mat &image,
                       const object_detect_result_list &od_results) const;
  void ProcessPoseImage(cv::Mat &image,
                        object_detect_result_list &od_results) const;

  int target_w_;
  int target_h_;
  letterbox_t letterbox_;
};
