#include "mask/mosaic.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "common/vc_error.h"
#include "test_media.h"

extern "C" {
#include <libavutil/frame.h>
}

namespace vc {
namespace {

// Every block of `rect` (bw x bh, grid from its top-left) in the plane is uniform and equals the
// rounded mean of `before`; everything outside `rect` is unchanged.
void expectPixelated(const AVFrame& after, const AVFrame& before, int plane, int step, int offset,
                     PixelRect rect, int bw, int bh, int planeWidth, int planeHeight) {
  const auto px = [&](const AVFrame& f, int x, int y) {
    return f.data[plane][y * f.linesize[plane] + x * step + offset];
  };
  for (int y = 0; y < planeHeight; ++y) {
    for (int x = 0; x < planeWidth; ++x) {
      const bool inside = x >= rect.x1 && x < rect.x2 && y >= rect.y1 && y < rect.y2;
      if (!inside) {
        ASSERT_EQ(px(after, x, y), px(before, x, y)) << "plane " << plane << " outside at " << x << "," << y;
        continue;
      }
      const int bx = rect.x1 + (x - rect.x1) / bw * bw;
      const int by = rect.y1 + (y - rect.y1) / bh * bh;
      const int ex = std::min(bx + bw, rect.x2);
      const int ey = std::min(by + bh, rect.y2);
      uint64_t sum = 0;
      for (int yy = by; yy < ey; ++yy) {
        for (int xx = bx; xx < ex; ++xx) sum += px(before, xx, yy);
      }
      const uint64_t n = static_cast<uint64_t>(ex - bx) * static_cast<uint64_t>(ey - by);
      ASSERT_EQ(px(after, x, y), (sum + n / 2) / n) << "plane " << plane << " inside at " << x << "," << y;
    }
  }
}

TEST(Mosaic, BlockSizeFollowsTheLongSide) {
  EXPECT_EQ(mosaicBlockSize({0, 0, 40, 30}, MosaicConfig{}), 16);
  EXPECT_EQ(mosaicBlockSize({0, 0, 200, 100}, MosaicConfig{}), 25);
  EXPECT_THROW(mosaicBlockSize({0, 0, 10, 10}, MosaicConfig{0, 8}), std::invalid_argument);
}

TEST(Mosaic, CoveringRectRoundsOutwardAndClips) {
  const PixelRect r = coveringRect(10.2f, 5.7f, 20.1f, 9.0f, 100, 100);
  EXPECT_EQ(r.x1, 10);
  EXPECT_EQ(r.y1, 5);
  EXPECT_EQ(r.x2, 21);
  EXPECT_EQ(r.y2, 9);
  const PixelRect c = coveringRect(-5, -5, 10, 10, 8, 8);
  EXPECT_EQ(c.x1, 0);
  EXPECT_EQ(c.x2, 8);
  EXPECT_TRUE(coveringRect(200, 200, 300, 300, 100, 100).empty());
}

TEST(Mosaic, PixelatesYuv420pAndLeavesTheRestAlone) {
  auto frame = test::makeNoiseFrame(96, 64, AV_PIX_FMT_YUV420P, 1);
  const auto before = test::deepCopy(*frame);
  applyMosaic(*frame, {PixelRect{10, 10, 50, 40}}, MosaicConfig{});
  expectPixelated(*frame, *before, 0, 1, 0, {10, 10, 50, 40}, 16, 16, 96, 64);
  expectPixelated(*frame, *before, 1, 1, 0, {5, 5, 25, 20}, 8, 8, 48, 32);
  expectPixelated(*frame, *before, 2, 1, 0, {5, 5, 25, 20}, 8, 8, 48, 32);
}

TEST(Mosaic, PixelatesNv12InterleavedChroma) {
  auto frame = test::makeNoiseFrame(96, 64, AV_PIX_FMT_NV12, 2);
  const auto before = test::deepCopy(*frame);
  applyMosaic(*frame, {PixelRect{10, 10, 50, 40}}, MosaicConfig{});
  expectPixelated(*frame, *before, 0, 1, 0, {10, 10, 50, 40}, 16, 16, 96, 64);
  expectPixelated(*frame, *before, 1, 2, 0, {5, 5, 25, 20}, 8, 8, 48, 32);
  expectPixelated(*frame, *before, 1, 2, 1, {5, 5, 25, 20}, 8, 8, 48, 32);
}

TEST(Mosaic, ClipsRectsToTheFrame) {
  auto frame = test::makeNoiseFrame(96, 64, AV_PIX_FMT_YUV420P, 3);
  const auto before = test::deepCopy(*frame);
  applyMosaic(*frame, {PixelRect{90, 60, 120, 80}}, MosaicConfig{});
  expectPixelated(*frame, *before, 0, 1, 0, {90, 60, 96, 64}, 16, 16, 96, 64);
}

TEST(Mosaic, RefusesSharedBuffersAndOtherFormats) {
  auto frame = test::makeNoiseFrame(32, 32, AV_PIX_FMT_YUV420P, 4);
  av::FramePtr other = av::allocFrame();
  ASSERT_GE(av_frame_ref(other.get(), frame.get()), 0);
  EXPECT_THROW(applyMosaic(*frame, {PixelRect{0, 0, 16, 16}}, MosaicConfig{}), VcError);

  av::FramePtr rgb = av::allocFrame();
  rgb->format = AV_PIX_FMT_RGB24;
  rgb->width = 16;
  rgb->height = 16;
  ASSERT_GE(av_frame_get_buffer(rgb.get(), 0), 0);
  EXPECT_THROW(applyMosaic(*rgb, {PixelRect{0, 0, 8, 8}}, MosaicConfig{}), VcError);
}

}  // namespace
}  // namespace vc
