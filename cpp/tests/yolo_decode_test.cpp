#include "detect/yolo_decode.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <vector>

namespace vc {
namespace {

// columns: (cx, cy, w, h, face, plate) -> row-major [6][N]
std::vector<float> pred(const std::vector<std::array<float, 6>>& columns) {
  const size_t n = columns.size();
  std::vector<float> out(6 * n);
  for (size_t a = 0; a < n; ++a) {
    for (size_t row = 0; row < 6; ++row) out[row * n + a] = columns[a][row];
  }
  return out;
}

// Same input as py/tests/test_train_geometry.py::test_decode_maps_back_by_r_thresholds_and_suppresses_per_class
TEST(YoloDecode, MapsBackByRThresholdsAndSuppressesPerClass) {
  const std::vector<float> p = pred({{32, 16, 8, 4, 0.9f, 0.0f},
                                     {100, 100, 10, 10, 0.1f, 0.05f},
                                     {33, 16, 8, 4, 0.6f, 0.0f},
                                     {32, 16, 8, 4, 0.0f, 0.5f}});
  const std::vector<Box> boxes = decodeYolov8(p.data(), 2, 4, makeLetterbox(1000, 1000, 500, 500), {});
  ASSERT_EQ(boxes.size(), 2u);
  EXPECT_EQ(boxes[0].cls, 0);
  EXPECT_FLOAT_EQ(boxes[0].score, 0.9f);
  EXPECT_NEAR(boxes[0].x1, 56, 1e-3);
  EXPECT_NEAR(boxes[0].y1, 28, 1e-3);
  EXPECT_NEAR(boxes[0].x2, 72, 1e-3);
  EXPECT_NEAR(boxes[0].y2, 36, 1e-3);
  EXPECT_EQ(boxes[1].cls, 1);
  EXPECT_FLOAT_EQ(boxes[1].score, 0.5f);
}

TEST(YoloDecode, ClipsToTheFrame) {
  const std::vector<float> p = pred({{2, 2, 10, 10, 0.8f, 0.0f}});
  const std::vector<Box> boxes = decodeYolov8(p.data(), 2, 1, makeLetterbox(20, 20, 20, 20), {});
  ASSERT_EQ(boxes.size(), 1u);
  EXPECT_FLOAT_EQ(boxes[0].x1, 0);
  EXPECT_FLOAT_EQ(boxes[0].y1, 0);
  EXPECT_FLOAT_EQ(boxes[0].x2, 7);
  EXPECT_FLOAT_EQ(boxes[0].y2, 7);
}

TEST(YoloDecode, KeepsScoresEqualToTheThreshold) {
  const std::vector<float> p = pred({{10, 10, 4, 4, 0.15f, 0.0f}});
  EXPECT_EQ(decodeYolov8(p.data(), 2, 1, makeLetterbox(20, 20, 20, 20), {}).size(), 1u);
}

TEST(YoloDecode, DropsAnchorsWithNanScores) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> p = pred({{10, 10, 4, 4, nan, 0.9f},
                                     {10, 10, 4, 4, 0.9f, nan}});
  EXPECT_TRUE(decodeYolov8(p.data(), 2, 2, makeLetterbox(20, 20, 20, 20), {}).empty());
}

TEST(YoloDecode, ClipsEachAxisToItsOwnFrameSize) {
  const std::vector<float> p = pred({{38, 18, 10, 10, 0.8f, 0.0f}});
  const std::vector<Box> boxes = decodeYolov8(p.data(), 2, 1, makeLetterbox(40, 20, 40, 20), {});
  ASSERT_EQ(boxes.size(), 1u);
  EXPECT_FLOAT_EQ(boxes[0].x1, 33);
  EXPECT_FLOAT_EQ(boxes[0].y1, 13);
  EXPECT_FLOAT_EQ(boxes[0].x2, 40);
  EXPECT_FLOAT_EQ(boxes[0].y2, 20);
}

}  // namespace
}  // namespace vc
