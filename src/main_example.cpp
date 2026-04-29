// ====== main.cpp 关键改动示例 (替换 FFmpegStreamer pipe 模式) ======
// 只展示核心改动部分, 参数解析等保持不变

#include "pipeline.h"
#include "rtsp_muxer.h"
// ... 其余 include 同原 main.cpp

int main(int argc, char *argv[]) {
  // ... 参数解析同原来 ...

  // ====== 创建 RTSPMuxer ======
  RTSPMuxer muxer;
  RTSPMuxer::Config mcfg;
  mcfg.rtsp_url = stream_cfg.rtsp_url;  // 来自命令行
  mcfg.rtmp_url = stream_cfg.rtmp_url;
  mcfg.codec    = "h264";
  mcfg.width    = 0;   // 后面从 pipeline 获取
  mcfg.height   = 0;
  mcfg.fps      = framerate;
  mcfg.bitrate  = stream_cfg.bitrate;

  // ====== 创建流水线 (启用 MPP 编码) ======
  Pipeline::Config pcfg;
  pcfg.model_path       = model_path;
  pcfg.label_path       = label_path;
  pcfg.input_file       = input_file;
  pcfg.infer_threads    = thread_count;
  pcfg.infer_interval   = infer_interval;
  pcfg.framerate        = framerate;
  pcfg.loop_video       = loop_video;

  // 关键: 启用 MPP 硬件编码
  pcfg.use_mpp_encoder  = true;
  pcfg.enc_codec        = "h264";
  pcfg.enc_bitrate      = stream_cfg.bitrate;
  pcfg.enc_callback     = [&muxer](const uint8_t *data, size_t size,
                                    int64_t pts, bool is_key) {
    muxer.WritePacket(data, size, pts, is_key);
  };

  Pipeline pipeline(pcfg);

  // 用 pipeline 的实际分辨率打开 muxer
  mcfg.width  = pipeline.GetFrameWidth();
  mcfg.height = pipeline.GetFrameHeight();
  if (!muxer.Open(mcfg)) {
    fprintf(stderr, "[ERROR] Failed to open muxer\n");
    return 1;
  }

  // ====== 启动 (不再需要 write_func) ======
  pipeline.Start();  // write_func = nullptr, 走 MppEncoder 路径

  printf("\n===== Pipeline streaming (MPP encode) =====\n");
  printf("  Threads: reader(1) + preprocess(1) + infer(%d) + writer(1)\n", thread_count);

  // ====== 主线程显示 / 等待 ======
  if (stream_cfg.enable_display) {
    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
    while (g_running && !pipeline.IsFinished()) {
      std::shared_ptr<cv::Mat> frame;
      if (pipeline.GetDisplayFrame(frame))
        cv::imshow("Video", *frame);
      if (cv::waitKey(1) == 27) break;
    }
    cv::destroyAllWindows();
  } else {
    while (g_running && !pipeline.IsFinished())
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // ====== 停止 ======
  pipeline.Stop();
  muxer.Close();

  return 0;
}
