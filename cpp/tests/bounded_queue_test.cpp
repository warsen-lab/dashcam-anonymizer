#include "common/bounded_queue.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <optional>
#include <thread>

namespace vc {
namespace {

TEST(BoundedQueue, IsFifo) {
  BoundedQueue<int> q(4);
  EXPECT_TRUE(q.push(1));
  EXPECT_TRUE(q.push(2));
  EXPECT_EQ(q.pop(), std::optional<int>(1));
  EXPECT_EQ(q.pop(), std::optional<int>(2));
}

TEST(BoundedQueue, PushBlocksWhileFull) {
  BoundedQueue<int> q(1);
  ASSERT_TRUE(q.push(1));
  std::atomic<bool> pushed{false};
  std::thread producer([&] {
    q.push(2);
    pushed = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(pushed.load());
  EXPECT_EQ(q.pop(), std::optional<int>(1));
  producer.join();
  EXPECT_TRUE(pushed.load());
  EXPECT_EQ(q.pop(), std::optional<int>(2));
}

TEST(BoundedQueue, CloseWakesWaitersAndRejectsPushes) {
  BoundedQueue<int> q(1);
  std::optional<int> got = 7;
  std::thread consumer([&] { got = q.pop(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  q.close();
  consumer.join();
  EXPECT_FALSE(got.has_value());
  EXPECT_FALSE(q.push(3));
}

}  // namespace
}  // namespace vc
