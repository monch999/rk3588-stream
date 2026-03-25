#include "videofile.h"
#include <cstdio>

VideoFile::VideoFile(const std::string &filename) : filename_(filename) {
  capture_ = std::make_unique<cv::VideoCapture>(filename_);
  if (!capture_->isOpened()) {
    fprintf(stderr, "[ERROR] Failed to open video: %s\n", filename.c_str());
    exit(EXIT_FAILURE);
  }
}

VideoFile::~VideoFile() {
  if (capture_) capture_->release();
}

std::unique_ptr<cv::Mat> VideoFile::GetNextFrame() {
  auto frame = std::make_unique<cv::Mat>();
  *capture_ >> *frame;
  if (frame->empty()) return nullptr;
  return frame;
}

int VideoFile::get_frame_width() {
  return static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_WIDTH));
}

int VideoFile::get_frame_height() {
  return static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_HEIGHT));
}
