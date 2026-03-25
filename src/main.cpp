#include <cstdio>
#include <csignal>
#include <chrono>
#include <thread>
#include <atomic>
#include <string>
#include <opencv2/opencv.hpp>
#include "pipeline.h"
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

// ==================== FFmpeg管道推流 ====================
class FFmpegStreamer {
public:
  FFmpegStreamer() = default;
  ~FFmpegStreamer() { close(); }

  bool open(int width, int height, double fps, const StreamConfig &cfg) {
    std::string encoder = cfg.codec + "_rkmpp";
    int gop = cfg.gop > 0 ? cfg.gop : static_cast<int>(fps * 2);

    char cmd[2048];
    if (!cfg.rtsp_url.empty() && !cfg.rtmp_url.empty()) {
      snprintf(cmd, sizeof(cmd),
        "ffmpeg -hide_banner -loglevel warning "
        "-f rawvideo -pixel_format bgr24 -video_size %dx%d -framerate %.1f "
        "-i pipe:0 "
        "-c:v %s -rc_mode CBR -b:v %s -g %d -bf 0 "
        "-flags +low_delay -fflags +genpts "
        "-map 0:v "
        "-f tee '"
        "[f=rtsp:rtsp_transport=tcp]%s"
        "|[f=flv]%s"
        "'",
        width, height, fps,
        encoder.c_str(), cfg.bitrate.c_str(), gop,
        cfg.rtsp_url.c_str(), cfg.rtmp_url.c_str());
    } else if (!cfg.rtsp_url.empty()) {
      snprintf(cmd, sizeof(cmd),
        "ffmpeg -hide_banner -loglevel warning "
        "-f rawvideo -pixel_format bgr24 -video_size %dx%d -framerate %.1f "
        "-i pipe:0 "
        "-c:v %s -rc_mode CBR -b:v %s -g %d -bf 0 "
        "-flags +low_delay -fflags +genpts "
        "-rtsp_transport tcp -f rtsp %s",
        width, height, fps,
        encoder.c_str(), cfg.bitrate.c_str(), gop,
        cfg.rtsp_url.c_str());
    } else if (!cfg.rtmp_url.empty()) {
      snprintf(cmd, sizeof(cmd),
        "ffmpeg -hide_banner -loglevel warning "
        "-f rawvideo -pixel_format bgr24 -video_size %dx%d -framerate %.1f "
        "-i pipe:0 "
        "-c:v %s -rc_mode CBR -b:v %s -g %d -bf 0 "
        "-flags +low_delay -fflags +genpts "
        "-f flv %s",
        width, height, fps,
        encoder.c_str(), cfg.bitrate.c_str(), gop,
        cfg.rtmp_url.c_str());
    } else {
      return false;
    }

    printf("[STREAM] CMD: %s\n", cmd);
    pipe_ = popen(cmd, "w");
    if (!pipe_) {
      fprintf(stderr, "[STREAM] Failed to open ffmpeg pipe\n");
      return false;
    }
    printf("[STREAM] Started -> RTSP: %s, RTMP: %s\n",
           cfg.rtsp_url.c_str(), cfg.rtmp_url.c_str());
    return true;
  }

  bool write(const cv::Mat &frame) {
    if (!pipe_) return false;
    cv::Mat cont = frame.isContinuous() ? frame : frame.clone();
    size_t sz = cont.total() * cont.elemSize();
    return fwrite(cont.data, 1, sz, pipe_) == sz;
  }

  void close() {
    if (pipe_) {
      pclose(pipe_);
      pipe_ = nullptr;
      printf("[STREAM] Stopped\n");
    }
  }

private:
  FILE *pipe_ = nullptr;
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
    if (opt == "--no-display") { stream_cfg.enable_display = false; continue; }
    if (opt == "--loop")    { loop_video = true; continue; }
    if (i + 1 >= argc) { print_usage(argv[0]); return 1; }
    std::string val = argv[++i];
    if      (opt == "-m")        model_path = val;
    else if (opt == "-l")        label_path = val;
    else if (opt == "-i")        input_file = val;
    else if (opt == "-t")        thread_count = std::stoi(val);
    else if (opt == "-f")        {framerate = std::stod(val); stream_cfg.gop = 2*static_cast<int>(framerate);}
    else if (opt == "-n")        infer_interval = std::stoi(val);
    else if (opt == "--proto")   proto = val;
    else if (opt == "--rtsp")    stream_cfg.rtsp_url = val;
    else if (opt == "--rtmp")    stream_cfg.rtmp_url = val;
    else if (opt == "--bitrate") stream_cfg.bitrate = val;
    else { print_usage(argv[0]); return 1; }
  }

  if (proto == "rtsp")      stream_cfg.rtmp_url.clear();
  else if (proto == "rtmp") stream_cfg.rtsp_url.clear();
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

  // ====== 创建流水线 ======
  Pipeline::Config pcfg;
  pcfg.model_path       = model_path;
  pcfg.label_path       = label_path;
  pcfg.input_file       = input_file;
  pcfg.infer_threads    = thread_count;
  pcfg.infer_interval   = infer_interval;
  pcfg.framerate        = framerate;
  pcfg.loop_video       = loop_video;

  Pipeline pipeline(pcfg);

  // ====== 创建推流器 ======
  FFmpegStreamer streamer;
  if (!streamer.open(pipeline.GetFrameWidth(), pipeline.GetFrameHeight(),
                     framerate, stream_cfg)) {
    fprintf(stderr, "[ERROR] Failed to start streaming\n");
    return 1;
  }

  // ====== 启动流水线 ======
  TimeDuration timer;
  pipeline.Start([&streamer](const cv::Mat &frame) {
    return streamer.write(frame);
  });

  printf("\n===== Pipeline streaming =====\n");
  printf("  Threads: reader(1) + preprocess(1) + infer(%d) + writer(1)\n", thread_count);
  printf("  Infer every %d frames\n", infer_interval);
  printf("  Press Ctrl+C to stop\n\n");

  // ====== 主线程: 仅负责显示 (轻量) ======
  if (stream_cfg.enable_display) {
    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
    while (g_running) {
      std::shared_ptr<cv::Mat> frame;
      if (pipeline.GetDisplayFrame(frame))
        cv::imshow("Video", *frame);
      if (cv::waitKey(1) == 27) break;  // ESC退出
    }
    cv::destroyAllWindows();
  } else {
    while (g_running)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // ====== 停止 ======
  pipeline.Stop();
  streamer.close();

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
