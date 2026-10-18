#include "detect/letterbox.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "common/vc_error.h"
#include "test_media.h"

namespace vc {
namespace {

float at(const std::vector<float>& t, const Letterbox& lb, int c, int y, int x) {
  return t[(static_cast<size_t>(c) * lb.inputHeight + y) * lb.inputWidth + x];
}

TEST(Letterbox, ScaleFitsTheLimitingSide) {
  const Letterbox a = makeLetterbox(1920, 1080, 1920, 1088);
  EXPECT_DOUBLE_EQ(a.r, 1.0);
  EXPECT_EQ(a.scaledWidth, 1920);
  EXPECT_EQ(a.scaledHeight, 1080);
  const Letterbox b = makeLetterbox(3840, 2160, 1920, 1088);
  EXPECT_DOUBLE_EQ(b.r, 0.5);
  EXPECT_EQ(b.scaledWidth, 1920);
  EXPECT_EQ(b.scaledHeight, 1080);
  const Letterbox c = makeLetterbox(100, 50, 64, 64);
  EXPECT_DOUBLE_EQ(c.r, 0.64);
  EXPECT_EQ(c.scaledWidth, 64);
  EXPECT_EQ(c.scaledHeight, 32);
  EXPECT_THROW(makeLetterbox(0, 10, 64, 64), std::invalid_argument);
}

TEST(Letterbox, FillsRgbAtTheTopLeftAndPadsWith114) {
  // BT.601 limited-range red
  const auto red = test::makeFrame(128, 64, AV_PIX_FMT_YUV420P, 81, 90, 240);
  const Letterbox lb = makeLetterbox(128, 64, 64, 64);
  std::vector<float> t(3 * 64 * 64);
  InputTensorFiller filler(lb);
  filler.fill(*red, t.data());
  for (int y : {0, 15, 31}) {
    for (int x : {0, 32, 63}) {
      EXPECT_GT(at(t, lb, 0, y, x), 0.85f);
      EXPECT_LT(at(t, lb, 1, y, x), 0.15f);
      EXPECT_LT(at(t, lb, 2, y, x), 0.15f);
    }
  }
  for (int c = 0; c < 3; ++c) {
    for (int y : {32, 50, 63}) EXPECT_FLOAT_EQ(at(t, lb, c, y, 10), 114.0f / 255.0f);
  }
}

TEST(Letterbox, KeepsChannelOrderForBlueNv12) {
  // BT.601 limited-range blue
  const auto blue = test::makeFrame(64, 64, AV_PIX_FMT_NV12, 41, 240, 110);
  const Letterbox lb = makeLetterbox(64, 64, 64, 64);
  std::vector<float> t(3 * 64 * 64);
  InputTensorFiller(lb).fill(*blue, t.data());
  EXPECT_LT(at(t, lb, 0, 10, 10), 0.15f);
  EXPECT_LT(at(t, lb, 1, 10, 10), 0.15f);
  EXPECT_GT(at(t, lb, 2, 10, 10), 0.85f);
}

TEST(Letterbox, RejectsFramesOfAnotherSize) {
  const auto frame = test::makeFrame(32, 32, AV_PIX_FMT_YUV420P, 128, 128, 128);
  std::vector<float> t(3 * 64 * 64);
  InputTensorFiller filler(makeLetterbox(64, 64, 64, 64));
  EXPECT_THROW(filler.fill(*frame, t.data()), VcError);
}

}  // namespace
}  // namespace vc
