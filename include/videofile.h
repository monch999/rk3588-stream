#pragma once
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

class VideoFile {
public:
  explicit VideoFile(const std::string &filename);
  ~VideoFile();

  std::unique_ptr<cv::Mat> GetNextFrame();
  int get_frame_width();
  int get_frame_height();

private:
  std::string filename_;
  std::unique_ptr<cv::VideoCapture> capture_;
};
