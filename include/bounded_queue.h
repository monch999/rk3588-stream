#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class BoundedQueue {
public:
  explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

  // 阻塞push，shutdown后返回false
  bool push(T item) {
    std::unique_lock<std::mutex> lock(mtx_);
    not_full_.wait(lock, [&] { return queue_.size() < capacity_ || shutdown_; });
    if (shutdown_) return false;
    queue_.push(std::move(item));
    not_empty_.notify_one();
    return true;
  }

  // 非阻塞push，满或shutdown返回false
  bool try_push(T item) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (queue_.size() >= capacity_ || shutdown_) return false;
    queue_.push(std::move(item));
    not_empty_.notify_one();
    return true;
  }

  // 阻塞pop，shutdown且空时返回false
  bool pop(T& item) {
    std::unique_lock<std::mutex> lock(mtx_);
    not_empty_.wait(lock, [&] { return !queue_.empty() || shutdown_; });
    if (queue_.empty()) return false;
    item = std::move(queue_.front());
    queue_.pop();
    not_full_.notify_one();
    return true;
  }

  // 非阻塞pop
  bool try_pop(T& item) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (queue_.empty()) return false;
    item = std::move(queue_.front());
    queue_.pop();
    not_full_.notify_one();
    return true;
  }

  void shutdown() {
    std::lock_guard<std::mutex> lock(mtx_);
    shutdown_ = true;
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  size_t size() {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.size();
  }

private:
  std::queue<T> queue_;
  std::mutex mtx_;
  std::condition_variable not_empty_, not_full_;
  size_t capacity_;
  bool shutdown_ = false;
};
