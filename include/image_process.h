#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include "postprocess.h"

class ImageProcess {
public:
  ImageProcess(int width, int height, int target_size, float framerate);

  // Letterbox resize + BGR→RGB，输出直接是 RGB 格式，可直接送入 RKNN
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

  double scale_;
  int padding_x_;
  int padding_y_;
  int new_width_;
  int new_height_;
  int target_size_;
  letterbox_t letterbox_;
};
