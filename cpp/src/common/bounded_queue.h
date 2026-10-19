#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace vc {

// Blocking FIFO between pipeline threads. close() aborts it: waiting and later push() calls return
// false, pop() returns nullopt and queued items are discarded. close() is idempotent.
template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(size_t capacity) : capacity_(capacity == 0 ? 1 : capacity) {}

  // false means the queue was closed and the item was dropped; the caller must stop producing.
  [[nodiscard]] bool push(T item) {
    std::unique_lock lock(mutex_);
    notFull_.wait(lock, [&] { return closed_ || items_.size() < capacity_; });
    if (closed_) return false;
    items_.push_back(std::move(item));
    notEmpty_.notify_one();
    return true;
  }

  std::optional<T> pop() {
    std::unique_lock lock(mutex_);
    notEmpty_.wait(lock, [&] { return closed_ || !items_.empty(); });
    if (closed_) return std::nullopt;
    T item = std::move(items_.front());
    items_.pop_front();
    notFull_.notify_one();
    return item;
  }

  void close() {
    {
      std::lock_guard lock(mutex_);
      closed_ = true;
      items_.clear();
    }
    notFull_.notify_all();
    notEmpty_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable notFull_;
  std::condition_variable notEmpty_;
  std::deque<T> items_;
  size_t capacity_;
  bool closed_ = false;
};

}  // namespace vc
