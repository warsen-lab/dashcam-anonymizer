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
  std::atomic<bool> accepted{false};
  std::thread producer([&] {
    accepted = q.push(2);
    pushed = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(pushed.load());
  EXPECT_EQ(q.pop(), std::optional<int>(1));
  producer.join();
  EXPECT_TRUE(pushed.load());
  EXPECT_TRUE(accepted.load());
  EXPECT_EQ(q.pop(), std::optional<int>(2));
}

TEST(BoundedQueue, CloseWakesWaitersAndRejectsPushes) {
  BoundedQueue<int> q(1);
  std::atomic<bool> consumerStarted{false};
  std::optional<int> got = 7;
  std::thread consumer([&] {
    consumerStarted = true;
    got = q.pop();
  });
  while (!consumerStarted.load()) std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));  // let pop() block
  q.close();
  consumer.join();
  EXPECT_FALSE(got.has_value());
  EXPECT_FALSE(q.push(3));
}

TEST(BoundedQueue, CloseWakesABlockedProducer) {
  BoundedQueue<int> q(1);
  ASSERT_TRUE(q.push(1));
  std::atomic<bool> producerStarted{false};
  std::atomic<bool> finished{false};
  std::atomic<bool> accepted{true};
  std::thread producer([&] {
    producerStarted = true;
    accepted = q.push(2);  // blocks: the queue is full
    finished = true;
  });
  while (!producerStarted.load()) std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));  // let push() block
  EXPECT_FALSE(finished.load());
  q.close();
  producer.join();
  EXPECT_TRUE(finished.load());
  EXPECT_FALSE(accepted.load());
}

TEST(BoundedQueue, CloseDiscardsQueuedItems) {
  BoundedQueue<int> q(4);
  ASSERT_TRUE(q.push(1));
  ASSERT_TRUE(q.push(2));
  q.close();
  EXPECT_FALSE(q.pop().has_value());
  EXPECT_FALSE(q.pop().has_value());
  EXPECT_FALSE(q.push(3));
}

TEST(BoundedQueue, CloseTwiceIsHarmless) {
  BoundedQueue<int> q(1);
  ASSERT_TRUE(q.push(1));
  q.close();
  q.close();
  EXPECT_FALSE(q.pop().has_value());
  EXPECT_FALSE(q.push(2));
}

}  // namespace
}  // namespace vc
