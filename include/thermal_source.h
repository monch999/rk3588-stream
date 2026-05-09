#pragma once
#include "video_source.h"
#include <atomic>
#include <functional>
#include <linux/videodev2.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// ==================== 热红外温度元数据 ====================
struct ThermalData {
    int64_t pts_us;        // 时间戳
    float   max_temp;      // 全局最高温
    float   min_temp;      // 全局最低温
    int     hot_x, hot_y;  // 最高温坐标 (相对于 640x512)
    int     cold_x, cold_y;// 最低温坐标 (相对于 640x512)
};

// ==================== 热红外专属视频源 ====================
class ThermalSource : public IVideoSource {
public:
    explicit ThermalSource(const std::string& dev_path, int buffer_count = 6);
    ~ThermalSource() override;

    ThermalSource(const ThermalSource&) = delete;
    ThermalSource& operator=(const ThermalSource&) = delete;

    std::shared_ptr<DrmFrame> GetNextDrmFrame() override;

    // 查询最新温度数据 (线程安全)
    ThermalData GetThermalData() const {
        std::lock_guard<std::mutex> lk(thermal_mtx_);
        return latest_thermal_;
    }
    
    // 注意：对外宣称的画面尺寸是裁剪后的 640x512，而不是原始的 1280x520
    int    get_frame_width()  const override { return 640; }
    int    get_frame_height() const override { return 512; }
    double get_fps()          const override { return fps_; }

private:
    bool Open();
    bool NegotiateAndStart();
    void StopAndReleaseBuffers();
    bool SetFormat();                
    bool RequestBuffers();           

    std::shared_ptr<DrmFrame> ProcessAndWrapBuffer(int buf_index, int64_t pts_us);
    void ReturnBuffer(int buf_index);
    
    // 解析底部的 8 行冗余数据
    void ExtractThermalData(uint8_t* raw_ptr, int64_t pts_us);

    struct V4l2Buffer {
        int    dmabuf_fd = -1;
        void* mmap_addr = nullptr;
        size_t mmap_size = 0;
        bool   queued    = false;
    };

    std::string dev_path_;
    int         buffer_count_ = 6;
    int         fd_ = -1;

    // 传感器原始物理尺寸
    const int raw_width_  = 1280;
    const int raw_height_ = 520;
    
    double fps_ = 25.0; // 热红外机芯通常是 25fps

    std::vector<V4l2Buffer> buffers_;
    std::mutex              return_mtx_;
    std::atomic<bool>       streaming_{false};
    
    mutable std::mutex      thermal_mtx_;
    ThermalData             latest_thermal_{};
};