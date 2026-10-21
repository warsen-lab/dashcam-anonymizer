#include "detect/verifier_crop.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "io/av_ptr.h"
#include "test_media.h"

extern "C" {
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

namespace vc {
namespace {

// The RGB24 conversion of the whole frame, the reference a crop at scale 1 must reproduce.
std::vector<uint8_t> fullRgb(const AVFrame& frame) {
  av::SwsPtr sws = av::allocSws();
  av::FramePtr rgb = av::allocFrame();
  rgb->format = AV_PIX_FMT_RGB24;
  rgb->width = frame.width;
  rgb->height = frame.height;
  EXPECT_GE(sws_scale_frame(sws.get(), rgb.get(), &frame), 0);
  std::vector<uint8_t> out(static_cast<size_t>(frame.width) * frame.height * 3);
  for (int y = 0; y < frame.height; ++y) {
    std::copy_n(rgb->data[0] + static_cast<ptrdiff_t>(y) * rgb->linesize[0], frame.width * 3,
                out.data() + static_cast<size_t>(y) * frame.width * 3);
  }
  return out;
}

TEST(VerifierCrop, WindowIsASquareAroundTheBoxCentre) {
  const CropWindow w = verifierWindow(Box{0, 100, 200, 140, 210, 0.5f}, 2.5, 32);  // 40 x 10
  EXPECT_DOUBLE_EQ(w.side, 100);  // 2.5 * 40
  EXPECT_DOUBLE_EQ(w.x, 70);      // centre 120 - 50
  EXPECT_DOUBLE_EQ(w.y, 155);     // centre 205 - 50
}

TEST(VerifierCrop, WindowIsAtLeastMinPx) {
  const CropWindow w = verifierWindow(Box{0, 10, 10, 16, 14, 0.5f}, 2.5, 32);  // 6 x 4: 2.5 * 6 = 15 < 32
  EXPECT_DOUBLE_EQ(w.side, 32);
  EXPECT_DOUBLE_EQ(w.x, 13 - 16);
  EXPECT_DOUBLE_EQ(w.y, 12 - 16);
}

// A window inside the frame at scale 1 (side == size) is the frame's own RGB conversion, pixel for pixel.
TEST(VerifierCrop, AtScaleOneTheCropIsTheFramesRgb) {
  auto frame = test::makeFrame(128, 96, AV_PIX_FMT_YUV420P, 0, 128, 128);
  for (int y = 0; y < 96; ++y) {  // a luma ramp, chroma neutral: no chroma edge to blur
    for (int x = 0; x < 128; ++x) frame->data[0][y * frame->linesize[0] + x] = static_cast<uint8_t>(16 + x + y);
  }
  const std::vector<uint8_t> ref = fullRgb(*frame);
  CropSampler sampler(CropParams{.size = 32});
  std::vector<uint8_t> rgb(32 * 32 * 3);
  sampler.fill(*frame, CropWindow{40, 20, 32}, rgb.data());
  for (int v = 0; v < 32; ++v) {
    for (int u = 0; u < 32; ++u) {
      for (int c = 0; c < 3; ++c) {
        ASSERT_EQ(rgb[(v * 32 + u) * 3 + c], ref[((20 + v) * 128 + 40 + u) * 3 + c]) << u << "," << v << "," << c;
      }
    }
  }
}

// What lies outside the frame is the pad value; a solid frame gives its own colour everywhere else.
TEST(VerifierCrop, OutsideTheFrameIsPadded) {
  auto frame = test::makeFrame(64, 64, AV_PIX_FMT_YUV420P, 180, 128, 128);
  const std::vector<uint8_t> ref = fullRgb(*frame);
  CropSampler sampler(CropParams{.size = 16, .padValue = 114});
  std::vector<uint8_t> rgb(16 * 16 * 3);
  sampler.fill(*frame, CropWindow{-32, -32, 64}, rgb.data());  // the frame covers the bottom-right quarter
  for (int v = 0; v < 16; ++v) {
    for (int u = 0; u < 16; ++u) {
      const bool inside = u >= 8 && v >= 8;
      for (int c = 0; c < 3; ++c) {
        ASSERT_EQ(rgb[(v * 16 + u) * 3 + c], inside ? ref[c] : 114) << u << "," << v;
      }
    }
  }
}

// NV12 keeps U and V interleaved on one plane: the view must start on a U, V pair, or the colours swap.
TEST(VerifierCrop, Nv12CropsKeepTheirColour) {
  auto frame = test::makeFrame(64, 64, AV_PIX_FMT_NV12, 120, 90, 200);
  const std::vector<uint8_t> ref = fullRgb(*frame);
  CropSampler sampler(CropParams{.size = 16});
  std::vector<uint8_t> rgb(16 * 16 * 3);
  sampler.fill(*frame, CropWindow{21, 13, 16}, rgb.data());  // odd corner: moved back onto the chroma grid
  for (int i = 0; i < 16 * 16; ++i) {
    for (int c = 0; c < 3; ++c) ASSERT_EQ(rgb[i * 3 + c], ref[c]) << i << "," << c;
  }
}

TEST(VerifierCrop, AWindowWhollyOutsideIsAllPad) {
  auto frame = test::makeFrame(64, 64, AV_PIX_FMT_YUV420P, 180, 128, 128);
  CropSampler sampler(CropParams{.size = 8});
  std::vector<uint8_t> rgb(8 * 8 * 3, 0);
  sampler.fill(*frame, CropWindow{200, 200, 40}, rgb.data());
  for (uint8_t b : rgb) ASSERT_EQ(b, 114);
}

TEST(VerifierCrop, ToPlanarFloatSplitsTheChannels) {
  const std::vector<uint8_t> rgb = {255, 0, 51};  // one pixel
  std::vector<float> out(3);
  toPlanarFloat(rgb.data(), 1, out.data());
  EXPECT_FLOAT_EQ(out[0], 1.0f);
  EXPECT_FLOAT_EQ(out[1], 0.0f);
  EXPECT_FLOAT_EQ(out[2], 51.0f / 255.0f);
}

}  // namespace
}  // namespace vc
