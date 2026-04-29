#pragma once
#include <memory>
#include <mutex>
#include <vector>
#include <cstdio>
#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_buffer.h>

#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

// ==================== DRM Buffer 池 ====================
// 为 zero-copy 链路提供独立于 decoder 的 DRM buffer
// 用 MPP_BUFFER_TYPE_DRM (与 RGA / MPP encoder 互通)
//
// 用法:
//   auto& alloc = DrmAllocator::Instance();
//   auto buf = alloc.Acquire(NV12, 3840, 2160);  // 自动归还
//   int fd = mpp_buffer_get_fd(buf->mpp_buf);
//   void* va = mpp_buffer_get_ptr(buf->mpp_buf);
//
class DrmAllocator {
public:
  enum Format { NV12, RGB888 };

  // 池中的单块 buffer
  struct Buffer {
    MppBuffer mpp_buf = nullptr;
    int       fd      = -1;
    void     *vaddr   = nullptr;
    size_t    size    = 0;
    int       width   = 0;
    int       height  = 0;
    int       h_stride = 0;
    int       v_stride = 0;
    Format    format  = NV12;

    // CPU 写完后, encoder/RGA 读前, 调用以同步 cache
    void SyncEnd() {
      if (mpp_buf) mpp_buffer_sync_end(mpp_buf);
    }
    // RGA/MPP 写完后, CPU 读前, 调用以同步 cache
    void SyncBegin() {
      if (mpp_buf) mpp_buffer_sync_begin(mpp_buf);
    }
  };

  static DrmAllocator& Instance() {
    static DrmAllocator inst;
    return inst;
  }

  // 申请一块 buffer, shared_ptr 析构时自动归还
  std::shared_ptr<Buffer> Acquire(Format fmt, int width, int height) {
    int h_stride, v_stride;
    size_t size = ComputeSize(fmt, width, height, h_stride, v_stride);

    std::lock_guard<std::mutex> lk(mtx_);
    EnsureGroup();

    MppBuffer mb = nullptr;
    MPP_RET ret = mpp_buffer_get(group_, &mb, size);
    if (ret != MPP_OK || !mb) {
      fprintf(stderr, "[DRM_ALLOC] mpp_buffer_get failed: ret=%d size=%zu\n", ret, size);
      return nullptr;
    }

    auto* raw = new Buffer();
    raw->mpp_buf  = mb;
    raw->fd       = mpp_buffer_get_fd(mb);
    raw->vaddr    = mpp_buffer_get_ptr(mb);
    raw->size     = size;
    raw->width    = width;
    raw->height   = height;
    raw->h_stride = h_stride;
    raw->v_stride = v_stride;
    raw->format   = fmt;

    return std::shared_ptr<Buffer>(raw, [](Buffer* b) {
      if (b) {
        if (b->mpp_buf) mpp_buffer_put(b->mpp_buf);
        delete b;
      }
    });
  }

  // 全局清理 (程序退出时调用)
  void Shutdown() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (group_) {
      mpp_buffer_group_put(group_);
      group_ = nullptr;
    }
  }

  ~DrmAllocator() { Shutdown(); }

private:
  DrmAllocator() = default;
  DrmAllocator(const DrmAllocator&) = delete;
  DrmAllocator& operator=(const DrmAllocator&) = delete;

  void EnsureGroup() {
    if (group_) return;
    MPP_RET ret = mpp_buffer_group_get_internal(&group_, MPP_BUFFER_TYPE_DRM);
    if (ret != MPP_OK) {
      fprintf(stderr, "[DRM_ALLOC] create buffer group failed: %d\n", ret);
      group_ = nullptr;
    }
  }

  static size_t ComputeSize(Format fmt, int w, int h, int& h_stride, int& v_stride) {
    h_stride = MPP_ALIGN(w, 16);
    v_stride = MPP_ALIGN(h, 16);
    switch (fmt) {
      case NV12:   return (size_t)h_stride * v_stride * 3 / 2;
      case RGB888: return (size_t)h_stride * v_stride * 3;
    }
    return 0;
  }

  std::mutex mtx_;
  MppBufferGroup group_ = nullptr;
};
