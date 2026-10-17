#include "sampler/luma_thumb.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace vc {
namespace {

LumaPlane planeOf(const std::vector<uint8_t>& pixels, int width, int height, int stride) {
  return LumaPlane{pixels.data(), width, height, stride};
}

TEST(LumaThumb, UniformPlaneGivesUniformThumb) {
  const std::vector<uint8_t> pixels(320 * 180, 100);

  const LumaThumb thumb = makeLumaThumb(planeOf(pixels, 320, 180, 320));

  EXPECT_EQ(thumb.width, 160);
  EXPECT_EQ(thumb.height, 90);
  ASSERT_EQ(thumb.pixels.size(), 160u * 90u);
  for (uint8_t p : thumb.pixels) EXPECT_EQ(p, 100);
}

TEST(LumaThumb, AveragesEachBlock) {
  const std::vector<uint8_t> pixels = {0, 0, 255, 255,  //
                                       0, 0, 255, 255};

  const LumaThumb thumb = makeLumaThumb(planeOf(pixels, 4, 2, 4), 2, 1);

  EXPECT_EQ(thumb.pixels, (std::vector<uint8_t>{0, 255}));
}

TEST(LumaThumb, IgnoresRowPadding) {
  // width 2, stride 4: the last two bytes of each row are padding.
  const std::vector<uint8_t> pixels = {10, 20, 255, 255,  //
                                       30, 40, 255, 255};

  const LumaThumb thumb = makeLumaThumb(planeOf(pixels, 2, 2, 4), 1, 1);

  EXPECT_EQ(thumb.pixels, (std::vector<uint8_t>{25}));
}

TEST(LumaThumb, RejectsThumbLargerThanPlane) {
  const std::vector<uint8_t> pixels(4, 0);
  EXPECT_THROW(makeLumaThumb(planeOf(pixels, 2, 2, 2), 3, 1), std::invalid_argument);
}

TEST(LumaThumb, MeanAbsDiffIsNormalised) {
  const LumaThumb black{2, 1, {0, 0}};
  const LumaThumb white{2, 1, {255, 255}};
  const LumaThumb half{2, 1, {0, 255}};

  EXPECT_DOUBLE_EQ(meanAbsDiff(black, black), 0.0);
  EXPECT_DOUBLE_EQ(meanAbsDiff(black, white), 1.0);
  EXPECT_DOUBLE_EQ(meanAbsDiff(black, half), 0.5);
}

TEST(LumaThumb, MeanAbsDiffRejectsSizeMismatch) {
  const LumaThumb one{1, 1, {0}};
  const LumaThumb two{2, 1, {0, 0}};
  EXPECT_THROW(meanAbsDiff(one, two), std::invalid_argument);
}

}  // namespace
}  // namespace vc
