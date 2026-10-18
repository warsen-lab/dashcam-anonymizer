#include "detect/box.h"

#include <gtest/gtest.h>

#include <vector>

namespace vc {
namespace {

Box box(int cls, float x1, float y1, float x2, float y2, float score) {
  return Box{cls, x1, y1, x2, y2, score};
}

TEST(Box, IouOfOverlappingAndDisjointBoxes) {
  EXPECT_FLOAT_EQ(iou(box(0, 0, 0, 10, 10, 1), box(0, 5, 0, 15, 10, 1)), 50.0f / 150.0f);
  EXPECT_FLOAT_EQ(iou(box(0, 0, 0, 10, 10, 1), box(0, 20, 20, 30, 30, 1)), 0.0f);
  EXPECT_FLOAT_EQ(iou(box(0, 0, 0, 0, 0, 1), box(0, 0, 0, 0, 0, 1)), 0.0f);
}

TEST(Box, NmsKeepsTheBestBoxPerClass) {
  const std::vector<Box> kept = nmsPerClass(
      {box(0, 0, 0, 10, 10, 0.6f), box(0, 1, 0, 11, 10, 0.9f), box(1, 0, 0, 10, 10, 0.5f),
       box(0, 50, 50, 60, 60, 0.3f)},
      0.5f);
  ASSERT_EQ(kept.size(), 3u);
  EXPECT_FLOAT_EQ(kept[0].score, 0.9f);
  EXPECT_EQ(kept[1].cls, 1);
  EXPECT_FLOAT_EQ(kept[2].score, 0.3f);
}

TEST(Box, NmsSuppressesAtExactlyTheThreshold) {
  // IoU 1/3 with threshold 1/3 is suppressed, like vc_label.boxes.nms (keeps only IoU < threshold).
  EXPECT_EQ(nmsPerClass({box(0, 0, 0, 10, 10, 0.9f), box(0, 5, 0, 15, 10, 0.8f)}, 50.0f / 150.0f).size(), 1u);
}

}  // namespace
}  // namespace vc
