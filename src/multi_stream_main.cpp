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

static std::atomic<bool> g_running{true};
void signal_handler(int) { g_running = false; }

// ==================== 帮助信息 ====================
void print_usage(const char *prog) {
  printf(
    "Usage: %s [options]\n"
    "\n"
    "  三路视频源 (必需):\n"
    "    --rgb-input <file>        RGB 视频文件 (4K, 30fps)\n"
    "    --ms-input <file>         多光谱视频文件 (60fps)\n"
    "    --thermal-input <file>    热红外视频文件 (640x512, 30fps)\n"
    "\n"
    "  YOLO 模型 (RGB 通道):\n"
    "    -m, --model <file>        RKNN 模型文件\n"
    "    -l, --labels <file>       标签文件\n"
    "    --npu-cores <N>           NPU 核心数 (默认: 3)\n"
    "\n"
    "  推流配置:\n"
    "    --rtsp-base <URL>         RTSP 基地址 (默认: rtsp://127.0.0.1:8554)\n"
    "    --bitrate <rate>          码率 (默认: 4M)\n"
    "    --stream-mode <mode>      推流模式: raw / processed / both (默认: both)\n"
    "                                raw:       仅推原始流，不进行算法处理\n"
    "                                processed: 仅推处理后的流\n"
    "                                both:      同时推原始流和处理后的流\n"
    "\n"
    "  通用选项:\n"
    "    --loop                    视频循环播放\n"
    "    --no-display              不显示本地窗口\n"
    "    -h, --help                显示帮助\n"
    "\n"
    "  推流地址 (自动生成):\n"
    "    {base}/rgb_raw             RGB 原始流 (30fps)\n"
    "    {base}/rgb_yolo            RGB YOLO推理流 (15fps)\n"
    "    {base}/ms_raw              多光谱原始流 (60fps)\n"
    "    {base}/ms_ndvi             多光谱植被指数流 (30fps)\n"
    "    {base}/thermal_raw         热红外原始流 (30fps)\n"
    "    {base}/thermal_fire        热红外火点检测流 (30fps)\n",
    prog);
}

// ==================== 主程序 ====================
int main(int argc, char *argv[]) {
  // ---- 默认参数 ----
  std::string rgb_input, ms_input, thermal_input;
  std::string model_path, label_path;
  int  npu_cores     = 3;
  std::string rtsp_base = "rtsp://127.0.0.1:8554";
  std::string bitrate   = "4M";
  std::string stream_mode = "both";  // raw / processed / both
  bool loop_video       = false;
  bool enable_display   = true;

  // ---- 参数解析 ----
  for (int i = 1; i < argc; i++) {
    std::string opt = argv[i];
    if (opt == "-h" || opt == "--help")    { print_usage(argv[0]); return 0; }
    if (opt == "--loop")                   { loop_video = true; continue; }
    if (opt == "--no-display")             { enable_display = false; continue; }
    if (i + 1 >= argc) { print_usage(argv[0]); return 1; }
    std::string val = argv[++i];
    if      (opt == "--rgb-input")     rgb_input     = val;
    else if (opt == "--ms-input")      ms_input      = val;
    else if (opt == "--thermal-input") thermal_input  = val;
    else if (opt == "-m" || opt == "--model")  model_path = val;
    else if (opt == "-l" || opt == "--labels") label_path = val;
    else if (opt == "--npu-cores")     npu_cores = std::stoi(val);
    else if (opt == "--rtsp-base")     rtsp_base = val;
    else if (opt == "--bitrate")       bitrate   = val;
    else if (opt == "--stream-mode")  stream_mode = val;
    else { fprintf(stderr, "未知选项: %s\n", opt.c_str()); return 1; }
  }

  // ---- 参数校验 ----
  if (rgb_input.empty() && ms_input.empty() && thermal_input.empty()) {
    fprintf(stderr, "错误: 至少需要指定一路视频输入\n");
    print_usage(argv[0]);
    return 1;
  }
  if (stream_mode != "raw" && stream_mode != "processed" &&
      stream_mode != "both") {
    fprintf(stderr, "错误: --stream-mode 必须是 raw, processed 或 both\n");
    return 1;
  }

  bool enable_raw       = (stream_mode == "raw"  || stream_mode == "both");
  bool enable_processed = (stream_mode == "processed" || stream_mode == "both");

  // RGB 通道的处理需要模型 (仅当推处理流时)
  if (!rgb_input.empty() && enable_processed &&
      (model_path.empty() || label_path.empty())) {
    fprintf(stderr, "错误: RGB 通道推处理流需要指定 -m <model> -l <labels>\n");
    return 1;
  }

  signal(SIGINT,  signal_handler);
  signal(SIGTERM, signal_handler);

  // ==================== 共享时钟 ====================
  SharedClock clock;

  // ==================== 创建处理器 ====================
  std::unique_ptr<YoloProcessor>  yolo_proc;
  std::unique_ptr<NdviProcessor>  ndvi_proc;
  std::unique_ptr<FireDetector>   fire_proc;

  // ==================== 创建通道 ====================
  std::vector<std::unique_ptr<ChannelPipeline>> channels;

  // ---- RGB 通道 ----
  if (!rgb_input.empty()) {
    // 仅当需要处理流时初始化 YOLO 处理器
    if (enable_processed) {
      VideoFile probe(rgb_input);
      YoloProcessor::Config ycfg;
      ycfg.model_path      = model_path;
      ycfg.label_path      = label_path;
      ycfg.num_cores       = npu_cores;
      ycfg.frame_w         = probe.get_frame_width();
      ycfg.frame_h         = probe.get_frame_height();
      ycfg.model_input_w   = 640;
      ycfg.model_input_h   = 384;
      yolo_proc = std::make_unique<YoloProcessor>(ycfg);
      if (!yolo_proc->Init()) {
        fprintf(stderr, "[ERROR] YoloProcessor init failed\n");
        return 1;
      }
    }

    ChannelPipeline::Config cfg;
    cfg.name               = "rgb";
    cfg.input_file         = rgb_input;
    cfg.raw_fps            = 30.0;
    cfg.processed_fps      = 15.0;
    cfg.raw_rtsp_url       = rtsp_base + "/rgb_raw";
    cfg.processed_rtsp_url = rtsp_base + "/rgb_yolo";
    cfg.bitrate            = bitrate;
    cfg.loop_video         = loop_video;
    cfg.enable_raw         = enable_raw;
    cfg.enable_processed   = enable_processed;
    cfg.clock              = &clock;
    cfg.processor          = yolo_proc.get();

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
    cfg.raw_rtsp_url       = rtsp_base + "/ms_raw";
    cfg.processed_rtsp_url = rtsp_base + "/ms_ndvi";
    cfg.bitrate            = bitrate;
    cfg.loop_video         = loop_video;
    cfg.enable_raw         = enable_raw;
    cfg.enable_processed   = enable_processed;
    cfg.clock              = &clock;
    cfg.processor          = ndvi_proc.get();

    channels.push_back(std::make_unique<ChannelPipeline>(cfg));
  }

  // ---- 热红外通道 ----
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
    cfg.raw_fps            = 30.0;
    cfg.processed_fps      = 30.0;
    cfg.raw_rtsp_url       = rtsp_base + "/thermal_raw";
    cfg.processed_rtsp_url = rtsp_base + "/thermal_fire";
    cfg.bitrate            = bitrate;
    cfg.loop_video         = loop_video;
    cfg.enable_raw         = enable_raw;
    cfg.enable_processed   = enable_processed;
    cfg.clock              = &clock;
    cfg.processor          = fire_proc.get();

    channels.push_back(std::make_unique<ChannelPipeline>(cfg));
  }

  // ==================== 启动共享时钟 (所有通道 PTS 的零点) ====================
  clock.Start();

  // ==================== 启动所有通道 ====================
  for (auto &ch : channels)
    ch->Start();

  size_t n_raw  = enable_raw       ? channels.size() : 0;
  size_t n_proc = enable_processed ? channels.size() : 0;

  printf("\n===== Multi-Stream Pipeline Running =====\n");
  printf("  Channels:    %zu\n", channels.size());
  printf("  Stream mode: %s\n", stream_mode.c_str());
  printf("  Streams:     %zu raw + %zu processed = %zu total\n",
         n_raw, n_proc, n_raw + n_proc);
  printf("  Clock epoch: monotonic (steady_clock)\n");
  printf("  Press Ctrl+C to stop\n\n");

  // ==================== 主线程: 等待退出信号 ====================
  if (enable_display) {
    // TODO: 可选的多窗口显示（当前暂不实现，推流为主要输出）
    while (g_running)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  } else {
    while (g_running)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // ==================== 停止所有通道 ====================
  printf("\n[INFO ] Shutting down...\n");
  for (auto &ch : channels)
    ch->Stop();

  // ==================== 打印统计 ====================
  printf("\n===== Final Statistics =====\n");
  for (auto &ch : channels) {
    printf("  [%-8s] raw=%d frames, processed=%d frames\n",
           ch->GetName().c_str(),
           ch->GetRawFrames(), ch->GetProcessedFrames());
  }

  return 0;
}
