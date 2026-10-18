#include "mask/mosaic.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include "common/vc_error.h"
#include "test_media.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

namespace vc {
namespace {

// [begin, end) of the block that holds offset `o` on an axis of `len` pixels cut into blocks of `b`
// from offset 0. A trailing remainder narrower than half a block joins the block before it; an axis
// shorter than one block is a single block.
std::pair<int, int> blockSpan(int o, int len, int b) {
  const int full = len / b;
  const int rem = len % b;
  const bool merged = full >= 1 && rem > 0 && rem * 2 < b;
  const int blocks = (merged || rem == 0) ? full : full + 1;
  const int index = std::min(o / b, blocks - 1);
  return {index * b, index == blocks - 1 ? len : (index + 1) * b};
}

// Every block of `rect` (bw x bh, grid from its top-left, thin trailing blocks merged) in the plane
// is uniform and equals the rounded mean of `before`; everything outside `rect` is unchanged.
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
      const auto [spanX1, spanX2] = blockSpan(x - rect.x1, rect.x2 - rect.x1, bw);
      const auto [spanY1, spanY2] = blockSpan(y - rect.y1, rect.y2 - rect.y1, bh);
      uint64_t sum = 0;
      for (int yy = rect.y1 + spanY1; yy < rect.y1 + spanY2; ++yy) {
        for (int xx = rect.x1 + spanX1; xx < rect.x1 + spanX2; ++xx) sum += px(before, xx, yy);
      }
      const uint64_t n = static_cast<uint64_t>(spanX2 - spanX1) * static_cast<uint64_t>(spanY2 - spanY1);
      ASSERT_EQ(px(after, x, y), (sum + n / 2) / n) << "plane " << plane << " inside at " << x << "," << y;
    }
  }
}

// Checks all three planes of a planar frame: luma pixelated over `luma` with `block` x `block`
// blocks, both chroma planes over `chroma` with `chromaBw` x `chromaBh` blocks.
void expectPlanar(const AVFrame& after, const AVFrame& before, PixelRect luma, int block, PixelRect chroma,
                  int chromaBw, int chromaBh) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(after.format));
  const int cw = (after.width + (1 << desc->log2_chroma_w) - 1) >> desc->log2_chroma_w;
  const int ch = (after.height + (1 << desc->log2_chroma_h) - 1) >> desc->log2_chroma_h;
  expectPixelated(after, before, 0, 1, 0, luma, block, block, after.width, after.height);
  expectPixelated(after, before, 1, 1, 0, chroma, chromaBw, chromaBh, cw, ch);
  expectPixelated(after, before, 2, 1, 0, chroma, chromaBw, chromaBh, cw, ch);
}

// Applies one rect to a noise frame and checks it with expectPlanar.
void expectPlanarMosaic(AVPixelFormat format, int width, int height, PixelRect luma, int block, PixelRect chroma,
                        int chromaBw, int chromaBh, unsigned seed) {
  auto frame = test::makeNoiseFrame(width, height, format, seed);
  const auto before = test::deepCopy(*frame);
  applyMosaic(*frame, {luma}, MosaicConfig{});
  expectPlanar(*frame, *before, luma, block, chroma, chromaBw, chromaBh);
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

TEST(Mosaic, CoveringRectRoundsAFractionalBottomEdgeUp) {
  const PixelRect r = coveringRect(10.2f, 5.7f, 20.1f, 9.3f, 100, 100);
  EXPECT_EQ(r.x1, 10);
  EXPECT_EQ(r.y1, 5);
  EXPECT_EQ(r.x2, 21);
  EXPECT_EQ(r.y2, 10);
}

TEST(Mosaic, CoveringRectClampsHugeValuesBeforeConverting) {
  const float huge = 1e30f;
  const PixelRect all = coveringRect(-huge, -huge, huge, huge, 100, 80);
  EXPECT_EQ(all.x1, 0);
  EXPECT_EQ(all.y1, 0);
  EXPECT_EQ(all.x2, 100);
  EXPECT_EQ(all.y2, 80);
  EXPECT_TRUE(coveringRect(huge, huge, huge, huge, 100, 80).empty());
  EXPECT_TRUE(coveringRect(-huge, -huge, -huge, -huge, 100, 80).empty());
}

TEST(Mosaic, CoveringRectRefusesNonFiniteCoordinates) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (const float bad : {nan, inf, -inf}) {
    EXPECT_THROW(coveringRect(bad, 10, 20, 20, 100, 100), VcError);
    EXPECT_THROW(coveringRect(10, bad, 20, 20, 100, 100), VcError);
    EXPECT_THROW(coveringRect(10, 10, bad, 20, 100, 100), VcError);
    EXPECT_THROW(coveringRect(10, 10, 20, bad, 100, 100), VcError);
  }
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

TEST(Mosaic, PixelatesAnOddOriginAndExtent) {
  // {11, 9, 50, 41}: 39 x 32 luma, block 16 (x: 16, 16, 7 merged into the last: 16 + 7 wide); chroma
  // {11 >> 1, 9 >> 1, ceil(50 / 2), ceil(41 / 2)} = {5, 4, 25, 21}, blocks 8 (y: 8, 8, 1 merged).
  const PixelRect rect{11, 9, 50, 41};
  expectPlanarMosaic(AV_PIX_FMT_YUV420P, 96, 64, rect, 16, {5, 4, 25, 21}, 8, 8, 5);

  auto frame = test::makeNoiseFrame(96, 64, AV_PIX_FMT_NV12, 6);
  const auto before = test::deepCopy(*frame);
  applyMosaic(*frame, {rect}, MosaicConfig{});
  expectPixelated(*frame, *before, 0, 1, 0, rect, 16, 16, 96, 64);
  expectPixelated(*frame, *before, 1, 2, 0, {5, 4, 25, 21}, 8, 8, 48, 32);
  expectPixelated(*frame, *before, 1, 2, 1, {5, 4, 25, 21}, 8, 8, 48, 32);
}

TEST(Mosaic, PixelatesAWideRectWithOddChromaBlocks) {
  // 203 px wide: block ceil(203 / 8) = 26, chroma block 13. 27 high: the last row of 1 px is merged
  // into the block above it, in luma (26 + 1) and in chroma ({0, 0, 102, 14}: 13 + 1).
  expectPlanarMosaic(AV_PIX_FMT_YUV420P, 208, 64, {0, 0, 203, 27}, 26, {0, 0, 102, 14}, 13, 13, 7);
  // Without a thin remainder in y: 7 blocks of 26 and a last one of 21 (chroma 7 x 13 and 11).
  expectPlanarMosaic(AV_PIX_FMT_YUV420P, 208, 64, {0, 0, 203, 52}, 26, {0, 0, 102, 26}, 13, 13, 8);
}

TEST(Mosaic, PixelatesAnOddFrameSize) {
  // 97 x 63: chroma planes are 49 x 32. {48, 15, 97, 63} runs to the right and bottom edges:
  // 49 x 48 luma (last column merged), chroma {24, 7, 49, 32} is 25 x 25 (both last lines merged).
  expectPlanarMosaic(AV_PIX_FMT_YUV420P, 97, 63, {48, 15, 97, 63}, 16, {24, 7, 49, 32}, 8, 8, 9);
}

TEST(Mosaic, PixelatesYuv422pAndYuv444p) {
  const PixelRect rect{11, 9, 50, 42};  // 39 x 33: both last lines are merged in luma
  // 4:2:2: chroma is half as wide only: {5, 9, 25, 42}, blocks 8 x 16.
  expectPlanarMosaic(AV_PIX_FMT_YUV422P, 96, 64, rect, 16, {5, 9, 25, 42}, 8, 16, 10);
  // 4:4:4: chroma is as large as luma.
  expectPlanarMosaic(AV_PIX_FMT_YUV444P, 96, 64, rect, 16, rect, 16, 16, 11);
}

TEST(Mosaic, AppliesOverlappingRectsInOrder) {
  const PixelRect first{10, 10, 50, 40};
  const PixelRect second{30, 20, 80, 60};
  auto stepwise = test::makeNoiseFrame(96, 64, AV_PIX_FMT_YUV420P, 12);
  const auto original = test::deepCopy(*stepwise);
  applyMosaic(*stepwise, {first}, MosaicConfig{});
  const auto afterFirst = test::deepCopy(*stepwise);
  applyMosaic(*stepwise, {second}, MosaicConfig{});
  // The second rect works on what the first one left behind.
  expectPlanar(*stepwise, *afterFirst, second, 16, {15, 10, 40, 30}, 8, 8);

  auto together = test::deepCopy(*original);
  applyMosaic(*together, {first, second}, MosaicConfig{});
  for (int plane = 0; plane < 3; ++plane) {
    const int w = plane == 0 ? 96 : 48;
    const int h = plane == 0 ? 64 : 32;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        ASSERT_EQ(together->data[plane][y * together->linesize[plane] + x],
                  stepwise->data[plane][y * stepwise->linesize[plane] + x])
            << "plane " << plane << " at " << x << "," << y;
      }
    }
  }
}

TEST(Mosaic, MergesAOnePixelRemainderIntoThePreviousBlock) {
  // 33 x 32 at (10, 10): blocks 16, 16 and a remainder of 1 px, which must not stay a column of
  // near-original detail but share the mean of the block before it (x 26..42).
  const PixelRect rect{10, 10, 43, 42};
  auto frame = test::makeNoiseFrame(96, 64, AV_PIX_FMT_YUV420P, 13);
  const auto before = test::deepCopy(*frame);
  applyMosaic(*frame, {rect}, MosaicConfig{});
  for (int y = 10; y < 26; ++y) {
    EXPECT_EQ(frame->data[0][y * frame->linesize[0] + 42], frame->data[0][y * frame->linesize[0] + 26]);
  }
  expectPlanar(*frame, *before, rect, 16, {5, 5, 22, 21}, 8, 8);
}

TEST(Mosaic, KeepsAHalfBlockRemainderAsItsOwnBlock) {
  // 40 x 30: x is 16 + 16 + 8 and 8 is exactly half a block, so it stays a block of its own.
  expectPlanarMosaic(AV_PIX_FMT_YUV420P, 96, 64, {10, 10, 50, 40}, 16, {5, 5, 25, 20}, 8, 8, 14);
}

TEST(Mosaic, RejectsAnInvalidConfigBeforeTouchingAnyPixel) {
  auto frame = test::makeNoiseFrame(96, 64, AV_PIX_FMT_YUV420P, 15);
  const auto before = test::deepCopy(*frame);
  for (const MosaicConfig& bad : {MosaicConfig{0, 8}, MosaicConfig{16, 0}, MosaicConfig{-1, -1}}) {
    EXPECT_THROW(applyMosaic(*frame, {}, bad), std::invalid_argument);  // even with nothing to do
    EXPECT_THROW(applyMosaic(*frame, {PixelRect{0, 0, 0, 0}}, bad), std::invalid_argument);
    EXPECT_THROW(applyMosaic(*frame, {PixelRect{10, 10, 50, 40}, PixelRect{60, 10, 90, 40}}, bad),
                 std::invalid_argument);
  }
  expectPlanar(*frame, *before, {0, 0, 0, 0}, 16, {0, 0, 0, 0}, 8, 8);  // nothing changed
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
