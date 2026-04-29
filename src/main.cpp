#include <cstdio>
#include <csignal>
#include <chrono>
#include <thread>
#include <atomic>
#include <string>
#include <opencv2/opencv.hpp>
#include "pipeline.h"
#include "rtsp_muxer.h"
#include "utils.h"

static std::atomic<bool> g_running{true};
void signal_handler(int) { g_running = false; }

// ==================== 推流配置 ====================
struct StreamConfig {
  std::string rtsp_url  = "rtsp://127.0.0.1:8554/stream";
  std::string rtmp_url  = "rtmp://127.0.0.1:1935/stream";
  std::string codec     = "h264";
  std::string bitrate   = "4M";
  int         gop       = 60;
  bool        enable_display = true;
};

// ==================== 主程序 ====================
void print_usage(const char *prog) {
  printf("Usage: %s -m <model.rknn> -l <labels.txt> -i <video>\n"
         "  -t <threads>               推理线程数 (默认: 3)\n"
         "  -f <fps>                   帧率 (默认: 30)\n"
         "  -n <interval>              每N帧推理一次 (默认: 3)\n"
         "  --proto <rtsp|rtmp|both>   推流协议 (默认: rtsp)\n"
         "  --rtsp <URL>               RTSP地址\n"
         "  --rtmp <URL>               RTMP地址\n"
         "  --bitrate <rate>           码率 (默认: 4M)\n"
         "  --no-display               不显示本地窗口\n"
         "  --loop                     视频loop\n", prog);
}

int main(int argc, char *argv[]) {
  std::string model_path, label_path, input_file;
  int thread_count   = 3;
  int infer_interval = 3;
  double framerate   = 30.0;
  bool loop_video    = false;
  StreamConfig stream_cfg;
  std::string proto  = "rtsp";

  // 参数解析
  for (int i = 1; i < argc; i++) {
    std::string opt = argv[i];
    if (opt == "-h" || opt == "--help")    { print_usage(argv[0]); return 0; }
    if (opt == "--no-display") { stream_cfg.enable_display = true; continue; }
    if (opt == "--loop")       { loop_video = true; continue; }
    if (i + 1 >= argc) { print_usage(argv[0]); return 1; }
    std::string val = argv[++i];
    if      (opt == "-m")        model_path = val;
    else if (opt == "-l")        label_path = val;
    else if (opt == "-i")        input_file = val;
    else if (opt == "-t")        thread_count = std::stoi(val);
    else if (opt == "-f")        { framerate = std::stod(val); stream_cfg.gop = 2 * static_cast<int>(framerate); }
    else if (opt == "-n")        infer_interval = std::stoi(val);
    else if (opt == "--proto")   proto = val;
    else if (opt == "--rtsp")    stream_cfg.rtsp_url = val;
    else if (opt == "--rtmp")    stream_cfg.rtmp_url = val;
    else if (opt == "--bitrate") stream_cfg.bitrate = val;
    else { print_usage(argv[0]); return 1; }
  }

  if (proto == "rtsp")      stream_cfg.rtmp_url = "";
  else if (proto == "rtmp") stream_cfg.rtsp_url = "";
  else if (proto != "both") {
    fprintf(stderr, "错误: --proto 必须是 rtsp, rtmp 或 both\n");
    return 1;
  }

  if (model_path.empty() || label_path.empty() || input_file.empty()) {
    print_usage(argv[0]);
    return 1;
  }

  signal(SIGINT,  signal_handler);
  signal(SIGTERM, signal_handler);

  // ====== 创建 RTSPMuxer ======
  RTSPMuxer muxer;
  RTSPMuxer::Config mcfg;
  mcfg.rtsp_url = stream_cfg.rtsp_url;
  mcfg.rtmp_url = stream_cfg.rtmp_url;
  mcfg.codec    = stream_cfg.codec;
  mcfg.fps      = framerate;
  mcfg.bitrate  = stream_cfg.bitrate;

  // ====== 创建流水线 (MPP 硬件编码) ======
  Pipeline::Config pcfg;
  pcfg.model_path      = model_path;
  pcfg.label_path      = label_path;
  pcfg.input_file      = input_file;
  pcfg.infer_threads   = thread_count;
  pcfg.infer_interval  = infer_interval;
  pcfg.framerate       = framerate;
  pcfg.loop_video      = loop_video;
  pcfg.use_mpp_encoder = true;
  pcfg.enc_codec       = stream_cfg.codec;
  pcfg.enc_bitrate     = stream_cfg.bitrate;
  pcfg.enc_callback    = [&muxer](const uint8_t *data, size_t size,
                                   int64_t pts, bool is_key) {
    muxer.WritePacket(data, size, pts, is_key);
  };

  Pipeline pipeline(pcfg);

  // 用 pipeline 实际分辨率打开 muxer
  mcfg.width  = pipeline.GetFrameWidth();
  mcfg.height = pipeline.GetFrameHeight();
  if (!muxer.Open(mcfg)) {
    fprintf(stderr, "[ERROR] Failed to open muxer\n");
    return 1;
  }

  // ====== 启动流水线 ======
  TimeDuration timer;
  pipeline.Start();

  printf("\n===== Pipeline streaming (MPP encode) =====\n");
  printf("  Threads: reader(1) + preprocess(1) + infer(%d) + writer(1)\n", thread_count);
  printf("  Infer every %d frames\n", infer_interval);
  printf("  Press Ctrl+C to stop\n\n");

  // ====== 主线程: 显示 / 等待 ======
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

  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      timer.DurationSinceLastTime());
  int total = pipeline.GetTotalFrames();
  int inferred = pipeline.GetInferredFrames();
  printf("[INFO ] Total: %ldms, frames: %d (inferred: %d, direct: %d), "
         "avg fps: %.2f\n",
         (long)ms.count(), total, inferred, total - inferred,
         total * 1000.0 / std::max((long)ms.count(), 1L));

  return 0;
}
