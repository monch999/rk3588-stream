#include <csignal>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <atomic>
#include <cstdio>
#include "algorithm_interface.h"
#include "channel_pipeline.h"
#include "shared_clock.h"
#include "lens_model.h"
#include "zoom_controller.h"
#include "gimbal_state.h"
#include "angle_sink.h"

static std::atomic<bool> g_running{true};
void signal_handler(int) { g_running = false; }

// ==================== 帮助信息 ====================
void print_usage(const char *prog) {
  printf(
    "Usage: %s [options]\n"
    "\n"
    "  视频源:\n"
    "    --rgb-input <path>        RGB 视频源 (HDMI-RX: /dev/videoN, 或文件/RTSP)\n"
    "    --ms-input <path>         多光谱视频源\n"
    "    --thermal-input <path>    热红外视频源 (USB: /dev/videoN, UYVY)\n"
    "\n"
    "  YOLO 模型 (RGB 通道):\n"
    "    -m, --model <file>        RKNN 模型文件\n"
    "    -l, --labels <file>       标签文件\n"
    "    --npu-cores <N>           NPU 核心数 (默认: 3)\n"
    "\n"
    "  推流配置:\n"
    "    --rtsp-base <URL>         RTSP 基地址 (默认: rtsp://127.0.0.1:8554)\n"
    "    --bitrate <rate>          码率 (默认: 6M)\n"
    "    -f, --fps <rate>          帧率 (默认: 25)\n"
    "\n"
    "  推流模式:\n"
    "    --stream-mode <mode>      (默认: raw)\n"
    "        raw        仅转发原始视频\n"
    "        processed  推算法处理后的流\n"
    "        track      RGB 单目标追踪 (仅 RGB 通道, 忽略其他通道)\n"
    "\n"
    "  追踪模式选项:\n"
    "    --lens-config <file>      镜头参数 JSON\n"
    "    --tracking-config <file>  追踪+变焦参数 JSON\n"
    "    --gimbal-state <file>     云台初始位置 JSON\n"
    "    --track-target <id>       初始追踪目标 ID (默认: 1(person))\n"
    "\n"
    "  通用选项:\n"
    "    --loop                    视频循环播放\n"
    "    --display                 显示本地窗口\n"
    "    -h, --help                显示帮助\n",
    prog);
}

// ==================== 本地显示线程 ====================
void display_thread_func(std::vector<ChannelPipeline*> &display_channels,
                         std::atomic<bool> &running) {
  const int WIN_W = 1280, WIN_H = 720;
  const std::string win_name = "Multi-Stream Display";
  cv::namedWindow(win_name, cv::WINDOW_NORMAL);
  cv::resizeWindow(win_name, WIN_W, WIN_H);

  cv::Mat canvas(WIN_H, WIN_W, CV_8UC3, cv::Scalar(0, 0, 0));
  int n = static_cast<int>(display_channels.size());

  while (running) {
    canvas.setTo(cv::Scalar(0, 0, 0));

    std::vector<cv::Mat> frames(n);
    std::vector<bool> valid(n, false);
    for (int i = 0; i < n; i++)
      valid[i] = display_channels[i]->GetDisplayFrame(frames[i]);

    if (n == 1) {
      if (valid[0]) {
        cv::Mat resized;
        cv::resize(frames[0], resized, cv::Size(WIN_W, WIN_H));
        resized.copyTo(canvas);
      }
    } else if (n == 2) {
      int half_w = WIN_W / 2;
      for (int i = 0; i < 2; i++) {
        if (!valid[i]) continue;
        cv::Mat resized;
        cv::resize(frames[i], resized, cv::Size(half_w, WIN_H));
        resized.copyTo(canvas(cv::Rect(i * half_w, 0, half_w, WIN_H)));
      }
    } else if (n == 3) {
      int half_w = WIN_W / 2;
      int half_h = WIN_H / 2;
      if (valid[0]) {
        cv::Mat resized;
        cv::resize(frames[0], resized, cv::Size(half_w, WIN_H));
        resized.copyTo(canvas(cv::Rect(0, 0, half_w, WIN_H)));
      }
      if (valid[1]) {
        cv::Mat resized;
        cv::resize(frames[1], resized, cv::Size(half_w, half_h));
        resized.copyTo(canvas(cv::Rect(half_w, 0, half_w, half_h)));
      }
      if (valid[2]) {
        cv::Mat resized;
        cv::resize(frames[2], resized, cv::Size(half_w, half_h));
        resized.copyTo(canvas(cv::Rect(half_w, half_h, half_w, half_h)));
      }
    }

    auto put_label = [&](const std::string &text, int x, int y) {
      cv::putText(canvas, text, cv::Point(x + 10, y + 30),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    };
    if (n == 1) {
      put_label(display_channels[0]->GetName(), 0, 0);
    } else if (n == 2) {
      put_label(display_channels[0]->GetName(), 0, 0);
      put_label(display_channels[1]->GetName(), WIN_W / 2, 0);
    } else if (n == 3) {
      put_label(display_channels[0]->GetName(), 0, 0);
      put_label(display_channels[1]->GetName(), WIN_W / 2, 0);
      put_label(display_channels[2]->GetName(), WIN_W / 2, WIN_H / 2);
    }

    cv::imshow(win_name, canvas);
    int key = cv::waitKey(30);
    if (key == 27 || key == 'q') {
      running = false;
      break;
    }
  }
  cv::destroyAllWindows();
}

// ==================== 主程序 ====================
int main(int argc, char *argv[]) {
  // ---- 默认参数 ----
  std::string rgb_input, ms_input, thermal_input;
  std::string model_path, label_path;
  int  npu_cores       = 3;
  int  fps             = 25;
  std::string rtsp_base = "rtsp://127.0.0.1:8554";
  std::string bitrate   = "6M";
  std::string stream_mode = "raw";
  bool loop_video       = false;
  bool enable_display   = false;

  std::string lens_config_path;
  std::string tracking_config_path;
  std::string gimbal_state_path;
  int         initial_target_id = 1;

  // ---- 参数解析 ----
  for (int i = 1; i < argc; i++) {
    std::string opt = argv[i];
    if (opt == "-h" || opt == "--help")    { print_usage(argv[0]); return 0; }
    if (opt == "--loop")                   { loop_video = true; continue; }
    if (opt == "--display")                { enable_display = true; continue; }
    if (i + 1 >= argc) { print_usage(argv[0]); return 1; }
    std::string val = argv[++i];
    if      (opt == "--rgb-input")     rgb_input      = val;
    else if (opt == "--ms-input")      ms_input       = val;
    else if (opt == "--thermal-input") thermal_input   = val;
    else if (opt == "-m" || opt == "--model")  model_path = val;
    else if (opt == "-l" || opt == "--labels") label_path = val;
    else if (opt == "-f" || opt == "--fps")    fps = std::stoi(val);
    else if (opt == "--npu-cores")     npu_cores = std::stoi(val);
    else if (opt == "--rtsp-base")     rtsp_base = val;
    else if (opt == "--bitrate")       bitrate   = val;
    else if (opt == "--stream-mode")   stream_mode = val;
    else if (opt == "--lens-config")     lens_config_path     = val;
    else if (opt == "--tracking-config") tracking_config_path = val;
    else if (opt == "--gimbal-state")    gimbal_state_path    = val;
    else if (opt == "--track-target")    initial_target_id    = std::stoi(val);
    else { fprintf(stderr, "未知选项: %s\n", opt.c_str()); return 1; }
  }

  // ---- stream-mode 解析 ----
  bool enable_raw = false, enable_processed = false, enable_track = false;
  if      (stream_mode == "raw")       { enable_raw = true; }
  else if (stream_mode == "processed") { enable_processed = true; }
  else if (stream_mode == "track")     { enable_track = true; }
  else {
    fprintf(stderr, "错误: 无效的 --stream-mode '%s' (可选: raw, processed, track)\n",
            stream_mode.c_str());
    return 1;
  }

  // ---- 参数校验 ----
  if (rgb_input.empty() && ms_input.empty() && thermal_input.empty()) {
    fprintf(stderr, "错误: 至少需要指定一路视频输入\n");
    print_usage(argv[0]);
    return 1;
  }
  if (!rgb_input.empty() && enable_processed
      && (model_path.empty() || label_path.empty())) {
    fprintf(stderr, "错误: RGB 通道处理模式需要指定 -m <model> -l <labels>\n");
    return 1;
  }

  signal(SIGINT,  signal_handler);
  signal(SIGTERM, signal_handler);

  // ==================== 追踪模块创建 ====================
  std::unique_ptr<LensModel>       lens_model;
  std::unique_ptr<ZoomController>  zoom_ctrl;
  std::unique_ptr<GimbalState>     gimbal_state;
  std::unique_ptr<IAngleSink>      angle_sink;

  if (enable_track) {
    if (rgb_input.empty()) {
      fprintf(stderr, "错误: track 模式必须提供 --rgb-input\n"); return 1;
    }
    if (model_path.empty() || label_path.empty()) {
      fprintf(stderr, "错误: track 模式必须提供 -m <model> -l <labels>\n"); return 1;
    }
    if (lens_config_path.empty() || tracking_config_path.empty()
        || gimbal_state_path.empty()) {
      fprintf(stderr,
          "错误: track 模式必须提供 --lens-config / --tracking-config / --gimbal-state\n");
      return 1;
    }

    lens_model = std::make_unique<LensModel>();
    if (!lens_model->LoadFromFile(lens_config_path)) return 1;

    zoom_ctrl = std::make_unique<ZoomController>();
    if (!zoom_ctrl->LoadFromFile(tracking_config_path)) return 1;

    gimbal_state = std::make_unique<GimbalState>();
    if (!gimbal_state->LoadInitialFromFile(gimbal_state_path)) return 1;

    angle_sink = std::make_unique<StdoutAngleSink>();
    printf("[INFO ] Track mode initial target_id=%d\n", initial_target_id);
  }

  // ==================== 共享时钟 ====================
  SharedClock clock;

  // ==================== 创建处理器 ====================
  std::unique_ptr<YoloProcessor>      yolo_proc;
  std::unique_ptr<YoloTrackProcessor> track_proc;
  std::unique_ptr<NdviProcessor>      ndvi_proc;
  std::unique_ptr<FireDetector>       fire_proc;

  // ==================== 创建通道 ====================
  std::vector<std::unique_ptr<ChannelPipeline>> channels;
  std::vector<ChannelPipeline*> display_channels;

  // ---- RGB 通道 (HDMI-RX: /dev/videoN 或 文件/RTSP) ----
  if (!rgb_input.empty()) {
    IFrameProcessor* rgb_processor = nullptr;

    if (enable_processed) {
      // 探测分辨率
      auto probe = ChannelPipeline::CreateSource(rgb_input);
      YoloProcessor::Config ycfg;
      ycfg.model_path    = model_path;
      ycfg.label_path    = label_path;
      ycfg.num_cores     = npu_cores;
      ycfg.frame_w       = probe->get_frame_width();
      ycfg.frame_h       = probe->get_frame_height();
      ycfg.model_input_w = 640;
      ycfg.model_input_h = 384;
      yolo_proc = std::make_unique<YoloProcessor>(ycfg);
      if (!yolo_proc->Init()) {
        fprintf(stderr, "[ERROR] YoloProcessor init failed\n"); return 1;
      }
      rgb_processor = yolo_proc.get();
    } else if (enable_track) {
      auto probe = ChannelPipeline::CreateSource(rgb_input);
      YoloTrackProcessor::Config tcfg;
      tcfg.model_path        = model_path;
      tcfg.label_path        = label_path;
      tcfg.frame_w           = probe->get_frame_width();
      tcfg.frame_h           = probe->get_frame_height();
      tcfg.model_input_w     = 640;
      tcfg.model_input_h     = 384;
      tcfg.lens              = lens_model.get();
      tcfg.zoom              = zoom_ctrl.get();
      tcfg.gimbal            = gimbal_state.get();
      tcfg.sink              = angle_sink.get();
      tcfg.initial_target_id = initial_target_id;
      track_proc = std::make_unique<YoloTrackProcessor>(tcfg);
      if (!track_proc->Init()) {
        fprintf(stderr, "[ERROR] YoloTrackProcessor init failed\n"); return 1;
      }
      rgb_processor = track_proc.get();
    }

    ChannelPipeline::Config cfg;
    cfg.name             = "rgb";
    cfg.input_file       = rgb_input;
    cfg.raw_fps          = 25.0;
    cfg.processed_fps    = (enable_track ? 25.0 : fps);
    cfg.enable_raw       = enable_raw;
    cfg.enable_processed = enable_processed || enable_track;
    cfg.raw_rtsp_url     = rtsp_base + "/rgb_raw";
    cfg.processed_rtsp_url = enable_track
        ? (rtsp_base + "/rgb_track")
        : (rtsp_base + "/rgb_yolo");
    cfg.bitrate        = bitrate;
    cfg.loop_video     = loop_video;
    cfg.clock          = &clock;
    cfg.processor      = rgb_processor;
    cfg.enable_display = enable_display;

    channels.push_back(std::make_unique<ChannelPipeline>(cfg));
  }

  // ---- track 模式: 只启动 RGB 通道, 忽略热红外和多光谱 ----
  if (!enable_track) {

    // ---- 热红外通道 (USB: /dev/videoN UYVY, 或文件) ----
    if (!thermal_input.empty()) {
      if (enable_processed) {
        fire_proc = std::make_unique<FireDetector>();
        if (!fire_proc->Init()) {
          fprintf(stderr, "[ERROR] FireDetector init failed\n");
          return 1;
        }
      }

      ChannelPipeline::Config cfg;
      cfg.name               = "thermal";
      cfg.input_file         = thermal_input;
      cfg.is_thermal         = true;
      cfg.raw_fps            = 25.0;  // 热红外机芯通常 25fps
      cfg.processed_fps      = 25.0;
      cfg.enable_raw         = enable_raw;
      cfg.enable_processed   = enable_processed;
      cfg.raw_rtsp_url       = rtsp_base + "/thermal_raw";
      cfg.processed_rtsp_url = rtsp_base + "/thermal_fire";
      cfg.bitrate            = bitrate;
      cfg.loop_video         = loop_video;
      cfg.clock              = &clock;
      cfg.processor          = enable_processed ? fire_proc.get() : nullptr;
      cfg.enable_display     = enable_display;

      channels.push_back(std::make_unique<ChannelPipeline>(cfg));
    }

    // ---- 多光谱通道 ----
    if (!ms_input.empty()) {
      if (enable_processed) {
        ndvi_proc = std::make_unique<NdviProcessor>();
        if (!ndvi_proc->Init()) {
          fprintf(stderr, "[ERROR] NdviProcessor init failed\n");
          return 1;
        }
      }

      ChannelPipeline::Config cfg;
      cfg.name               = "ms";
      cfg.input_file         = ms_input;
      cfg.raw_fps            = 60.0;
      cfg.processed_fps      = 30.0;
      cfg.enable_raw         = enable_raw;
      cfg.enable_processed   = enable_processed;
      cfg.raw_rtsp_url       = rtsp_base + "/ms_raw";
      cfg.processed_rtsp_url = rtsp_base + "/ms_ndvi";
      cfg.bitrate            = bitrate;
      cfg.loop_video         = loop_video;
      cfg.clock              = &clock;
      cfg.processor          = enable_processed ? ndvi_proc.get() : nullptr;
      cfg.enable_display     = enable_display;

      channels.push_back(std::make_unique<ChannelPipeline>(cfg));
    }
  }

  // Raw 模式不支持本地显示
  if (enable_raw && enable_display) {
    printf("[INFO ] Raw 模式不支持本地显示，已忽略 --display\n");
    enable_display = false;
  }

  // ==================== 显示通道列表 (布局: 左=rgb, 右上=thermal, 右下=ms) ====================
  if (enable_display) {
    for (auto &ch : channels)
      if (ch->GetName() == "rgb")     display_channels.push_back(ch.get());
    for (auto &ch : channels)
      if (ch->GetName() == "thermal") display_channels.push_back(ch.get());
    for (auto &ch : channels)
      if (ch->GetName() == "ms")      display_channels.push_back(ch.get());
  }

  // ==================== 启动 ====================
  clock.Start();
  for (auto &ch : channels)
    ch->Start();

  printf("\n===== Multi-Stream Pipeline Running =====\n");
  printf("  Channels: %zu\n", channels.size());
  printf("  Stream mode: %s\n", stream_mode.c_str());
  printf("  Display: %s\n", enable_display ? "ON" : "OFF");
  printf("  Press Ctrl+C to stop\n\n");

  // ==================== 显示线程 ====================
  std::thread display_thread;
  if (enable_display && !display_channels.empty()) {
    display_thread = std::thread(display_thread_func,
                                 std::ref(display_channels),
                                 std::ref(g_running));
  }

  // ==================== 主线程等待退出 ====================
  while (g_running)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // ==================== 停止 ====================
  printf("\n[INFO ] Shutting down...\n");
  if (display_thread.joinable())
    display_thread.join();

  for (auto &ch : channels)
    ch->Stop();

  printf("\n===== Final Statistics =====\n");
  for (auto &ch : channels) {
    printf("  [%-8s] processed=%d frames\n",
           ch->GetName().c_str(), ch->GetProcessedFrames());
  }

  return 0;
}
