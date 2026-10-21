#include "mask/mosaic.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "common/vc_error.h"
#include "test_media.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

namespace vc {
namespace {

// Every cellW x cellH cell of the grid anchored at the plane's origin, inside `area`, is uniform and equals the
// rounded mean of `before` over the cell's part inside `area`; everything outside `area` is unchanged.
void expectGrid(const AVFrame& after, const AVFrame& before, int plane, int step, int offset, PixelRect area,
                int cellW, int cellH, int planeWidth, int planeHeight) {
  const auto px = [&](const AVFrame& f, int x, int y) {
    return f.data[plane][y * f.linesize[plane] + x * step + offset];
  };
  for (int y = 0; y < planeHeight; ++y) {
    for (int x = 0; x < planeWidth; ++x) {
      const bool inside = x >= area.x1 && x < area.x2 && y >= area.y1 && y < area.y2;
      if (!inside) {
        ASSERT_EQ(px(after, x, y), px(before, x, y)) << "plane " << plane << " outside at " << x << "," << y;
        continue;
      }
      const int gx = x / cellW * cellW;
      const int gy = y / cellH * cellH;
      const int cx1 = std::max(gx, area.x1);
      const int cy1 = std::max(gy, area.y1);
      const int cx2 = std::min(gx + cellW, area.x2);
      const int cy2 = std::min(gy + cellH, area.y2);
      uint64_t sum = 0;
      for (int yy = cy1; yy < cy2; ++yy) {
        for (int xx = cx1; xx < cx2; ++xx) sum += px(before, xx, yy);
      }
      const uint64_t n = static_cast<uint64_t>(cx2 - cx1) * static_cast<uint64_t>(cy2 - cy1);
      ASSERT_EQ(px(after, x, y), (sum + n / 2) / n) << "plane " << plane << " inside at " << x << "," << y;
    }
  }
}

// All three planes of a planar frame: luma over `luma` in block x block cells, both chroma planes over `chroma`
// in chromaW x chromaH cells.
void expectPlanarGrid(const AVFrame& after, const AVFrame& before, PixelRect luma, int block, PixelRect chroma,
                      int chromaW, int chromaH) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(after.format));
  const int cw = (after.width + (1 << desc->log2_chroma_w) - 1) >> desc->log2_chroma_w;
  const int ch = (after.height + (1 << desc->log2_chroma_h) - 1) >> desc->log2_chroma_h;
  expectGrid(after, before, 0, 1, 0, luma, block, block, after.width, after.height);
  expectGrid(after, before, 1, 1, 0, chroma, chromaW, chromaH, cw, ch);
  expectGrid(after, before, 2, 1, 0, chroma, chromaW, chromaH, cw, ch);
}

// One rect with one block on a noise frame, checked with expectPlanarGrid.
void expectPlanarGridMosaic(AVPixelFormat format, int width, int height, PixelRect rect, int block, PixelRect luma,
                            PixelRect chroma, int chromaW, int chromaH, unsigned seed) {
  auto frame = test::makeNoiseFrame(width, height, format, seed);
  const auto before = test::deepCopy(*frame);
  applyMosaic(*frame, {MosaicRect{rect, block}});
  expectPlanarGrid(*frame, *before, luma, block, chroma, chromaW, chromaH);
}

// Every pixel of every plane of two YUV 4:2:0 frames of the same size is the same.
bool samePixels420(const AVFrame& a, const AVFrame& b) {
  for (int plane = 0; plane < 3; ++plane) {
    const int w = plane == 0 ? a.width : (a.width + 1) / 2;
    const int h = plane == 0 ? a.height : (a.height + 1) / 2;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (a.data[plane][y * a.linesize[plane] + x] != b.data[plane][y * b.linesize[plane] + x]) return false;
      }
    }
  }
  return true;
}

// Every pixel of every plane of two planar frames of the same format and size is the same.
bool samePlanarPixels(const AVFrame& a, const AVFrame& b) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(a.format));
  for (int plane = 0; plane < 3; ++plane) {
    const int w = plane == 0 ? a.width : (a.width + (1 << desc->log2_chroma_w) - 1) >> desc->log2_chroma_w;
    const int h = plane == 0 ? a.height : (a.height + (1 << desc->log2_chroma_h) - 1) >> desc->log2_chroma_h;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        if (a.data[plane][y * a.linesize[plane] + x] != b.data[plane][y * b.linesize[plane] + x]) return false;
      }
    }
  }
  return true;
}

TEST(Mosaic, BlockSizeFollowsTheLongSide) {
  EXPECT_EQ(mosaicBlockSize({0, 0, 40, 30}, MosaicConfig{}), 16);
  EXPECT_EQ(mosaicBlockSize({0, 0, 200, 100}, MosaicConfig{}), 25);
  EXPECT_THROW(mosaicBlockSize({0, 0, 10, 10}, MosaicConfig{0, 8}), std::invalid_argument);
}

TEST(Mosaic, LadderBlockIsTheFirstStepNotBelowTheWantedSize) {
  EXPECT_EQ(ladderBlock(1), 16);
  EXPECT_EQ(ladderBlock(16), 16);
  EXPECT_EQ(ladderBlock(17), 24);
  EXPECT_EQ(ladderBlock(25), 32);
  EXPECT_EQ(ladderBlock(100), 128);
  EXPECT_EQ(ladderBlock(256), 256);
  EXPECT_EQ(ladderBlock(1000), 256);  // the top of the ladder
}

TEST(Mosaic, ATrackMovesUpTheLadderOnlyPastOneAndAHalfTimesItsBlock) {
  EXPECT_EQ(nextTrackBlock(0, 20), 24);   // a new track: the first step >= 20
  EXPECT_EQ(nextTrackBlock(24, 36), 24);  // 36 is 1.5 x 24, not past it
  EXPECT_EQ(nextTrackBlock(24, 37), 48);  // past it: the first step >= 37
  EXPECT_EQ(nextTrackBlock(48, 10), 48);  // never down
  EXPECT_EQ(nextTrackBlock(256, 1000), 256);
}

TEST(Mosaic, TrackBlocksFollowEachTrackAndForgetEndedOnes) {
  TrackBlocks blocks{MosaicConfig{}};
  EXPECT_EQ(blocks.blockFor(1, {0, 0, 100, 50}), 16);  // ceil(100 / 8) = 13, at least 16
  EXPECT_EQ(blocks.blockFor(2, {0, 0, 300, 50}), 48);  // ceil(300 / 8) = 38
  blocks.endFrame();
  EXPECT_EQ(blocks.blockFor(1, {0, 0, 180, 50}), 16);  // 23 is not past 1.5 x 16 = 24
  EXPECT_EQ(blocks.blockFor(2, {0, 0, 100, 50}), 48);  // never down
  blocks.endFrame();
  EXPECT_EQ(blocks.blockFor(1, {0, 0, 200, 50}), 32);  // 25 is past 24
  blocks.endFrame();                                   // track 2 had no mask on this frame: forgotten
  EXPECT_EQ(blocks.blockFor(2, {0, 0, 100, 50}), 16);  // so it starts afresh
  EXPECT_THROW(TrackBlocks(MosaicConfig{0, 8}), std::invalid_argument);
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

// {10, 10, 50, 40} with 16 px blocks covers the frame's cells [0, 64) x [0, 48); chroma, 8 px cells, [0, 32) x [0, 24).
TEST(Mosaic, PixelatesOnTheGridOfTheWholeFrame) {
  expectPlanarGridMosaic(AV_PIX_FMT_YUV420P, 96, 64, {10, 10, 50, 40}, 16, {0, 0, 64, 48}, {0, 0, 32, 24}, 8, 8, 1);
}

TEST(Mosaic, PixelatesNv12OnTheSameGrid) {
  auto frame = test::makeNoiseFrame(96, 64, AV_PIX_FMT_NV12, 2);
  const auto before = test::deepCopy(*frame);
  applyMosaic(*frame, {MosaicRect{{10, 10, 50, 40}, 16}});
  expectGrid(*frame, *before, 0, 1, 0, {0, 0, 64, 48}, 16, 16, 96, 64);
  expectGrid(*frame, *before, 1, 2, 0, {0, 0, 32, 24}, 8, 8, 48, 32);
  expectGrid(*frame, *before, 1, 2, 1, {0, 0, 32, 24}, 8, 8, 48, 32);
}

// The pattern is anchored to the picture, not to the mask (spec 2026-10-19 §4): a mask that moves within the
// same cells changes nothing, and the cells two positions share keep their values.
TEST(Mosaic, AMovingMaskKeepsItsPattern) {
  auto source = test::makeNoiseFrame(96, 64, AV_PIX_FMT_YUV420P, 3);
  auto a = test::deepCopy(*source);
  auto b = test::deepCopy(*source);
  auto c = test::deepCopy(*source);
  applyMosaic(*a, {MosaicRect{{10, 10, 50, 40}, 16}});
  applyMosaic(*b, {MosaicRect{{14, 12, 54, 42}, 16}});  // 4 px right, 2 px down: the same cells
  EXPECT_TRUE(samePixels420(*a, *b));
  applyMosaic(*c, {MosaicRect{{26, 10, 66, 40}, 16}});  // the cells [16, 80) x [0, 48)
  for (int y = 0; y < 48; ++y) {
    for (int x = 16; x < 64; ++x) {
      ASSERT_EQ(a->data[0][y * a->linesize[0] + x], c->data[0][y * c->linesize[0] + x]) << x << "," << y;
    }
  }
}

TEST(Mosaic, PixelatesYuv422pAndYuv444pOnTheGrid) {
  // 4:2:2: chroma half as wide only, cells 8 x 16 over [0, 32) x [0, 48); 4:4:4: chroma as luma.
  expectPlanarGridMosaic(AV_PIX_FMT_YUV422P, 96, 64, {11, 9, 50, 42}, 16, {0, 0, 64, 48}, {0, 0, 32, 48}, 8, 16, 10);
  expectPlanarGridMosaic(AV_PIX_FMT_YUV444P, 96, 64, {11, 9, 50, 42}, 16, {0, 0, 64, 48}, {0, 0, 64, 48}, 16, 16, 11);
}

// The cells the right and bottom edges of a 97 x 63 frame cut average the pixels that are there: {80, 40, 97, 63}
// with 16 px blocks is [80, 97) x [32, 63), cells 16 and 1 px wide, 16 and 15 px high; chroma (49 x 32) is
// [40, 49) x [16, 32), cells 8 and 1 wide.
TEST(Mosaic, CellsCutByTheFrameEdgeAverageWhatIsThere) {
  expectPlanarGridMosaic(AV_PIX_FMT_YUV420P, 97, 63, {80, 40, 97, 63}, 16, {80, 32, 97, 63}, {40, 16, 49, 32}, 8, 8, 9);
}

// A rect that is not snapped (vc_redact's plates) is pixelated only inside itself, on the same frame grid: the
// cells its edges cut average their part inside the rect. Luma [10, 50) x [10, 40) in 16 px cells, chroma
// [5, 25) x [5, 20) in 8 px cells.
TEST(Mosaic, AnUnsnappedRectIsPixelatedOnlyInsideItselfOnTheFrameGrid) {
  auto frame = test::makeNoiseFrame(96, 64, AV_PIX_FMT_YUV420P, 12);
  const auto before = test::deepCopy(*frame);
  applyMosaic(*frame, {MosaicRect{{10, 10, 50, 40}, 16, false}});
  expectPlanarGrid(*frame, *before, {10, 10, 50, 40}, 16, {5, 5, 25, 20}, 8, 8);
}

TEST(Mosaic, AnUnsnappedNv12RectIsPixelatedOnlyInsideItself) {
  auto frame = test::makeNoiseFrame(96, 64, AV_PIX_FMT_NV12, 13);
  const auto before = test::deepCopy(*frame);
  applyMosaic(*frame, {MosaicRect{{11, 9, 51, 41}, 16, false}});
  // Odd luma edges: chroma covers every sample a luma pixel inside shares, [5, 26) x [4, 21).
  expectGrid(*frame, *before, 0, 1, 0, {11, 9, 51, 41}, 16, 16, 96, 64);
  expectGrid(*frame, *before, 1, 2, 0, {5, 4, 26, 21}, 8, 8, 48, 32);
  expectGrid(*frame, *before, 1, 2, 1, {5, 4, 26, 21}, 8, 8, 48, 32);
}

// The whole cells of an unsnapped rect are the cells a snapped one gives there: the pattern stays on the frame grid.
TEST(Mosaic, AnUnsnappedRectSharesItsWholeCellsWithTheSnappedOne) {
  auto source = test::makeNoiseFrame(96, 64, AV_PIX_FMT_YUV420P, 14);
  auto snapped = test::deepCopy(*source);
  auto clipped = test::deepCopy(*source);
  applyMosaic(*snapped, {MosaicRect{{10, 10, 50, 40}, 16}});
  applyMosaic(*clipped, {MosaicRect{{10, 10, 50, 40}, 16, false}});
  for (int y = 16; y < 32; ++y) {
    for (int x = 16; x < 48; ++x) {
      ASSERT_EQ(snapped->data[0][y * snapped->linesize[0] + x], clipped->data[0][y * clipped->linesize[0] + x])
          << x << "," << y;
    }
  }
}

// Overlapping masks are pixelated largest block first, so the finer mosaic drawn last is what shows where they
// overlap, whatever their order in the list. 16 px cells straddle the 24 px ones, so the order is visible.
TEST(Mosaic, AppliesLargerBlocksFirst) {
  const MosaicRect coarse{{0, 0, 72, 48}, 24};
  const MosaicRect fine{{16, 16, 48, 48}, 16};
  auto together = test::makeNoiseFrame(96, 64, AV_PIX_FMT_YUV420P, 12);
  auto coarseFirst = test::deepCopy(*together);
  auto fineFirst = test::deepCopy(*together);
  applyMosaic(*together, {fine, coarse});
  applyMosaic(*coarseFirst, {coarse});
  const auto afterCoarse = test::deepCopy(*coarseFirst);
  applyMosaic(*coarseFirst, {fine});
  expectPlanarGrid(*coarseFirst, *afterCoarse, {16, 16, 48, 48}, 16, {8, 8, 24, 24}, 8, 8);
  EXPECT_TRUE(samePixels420(*together, *coarseFirst));
  applyMosaic(*fineFirst, {fine});
  applyMosaic(*fineFirst, {coarse});
  EXPECT_FALSE(samePixels420(*together, *fineFirst));
}

TEST(Mosaic, RejectsAnInvalidBlockBeforeTouchingAnyPixel) {
  auto frame = test::makeNoiseFrame(96, 64, AV_PIX_FMT_YUV420P, 15);
  const auto before = test::deepCopy(*frame);
  for (const int bad : {0, -16, 15, 1}) {
    EXPECT_THROW(applyMosaic(*frame, {MosaicRect{{10, 10, 50, 40}, 16}, MosaicRect{{60, 10, 90, 40}, bad}}),
                 std::invalid_argument)
        << bad;
  }
  EXPECT_TRUE(samePixels420(*frame, *before));
}

// 4:1:1 and 4:1:0 chroma is a quarter as wide as luma: a block of 2 would make a chroma cell 0 px wide (the grid
// would never advance) and one of 6 a chroma cell of 1.5 px, off the luma grid. A block must be a multiple of 4
// there, and the frame is untouched when it is not.
TEST(Mosaic, RejectsABlockThatIsNotWholeChromaCellsBeforeTouchingAnyPixel) {
  for (const AVPixelFormat format : {AV_PIX_FMT_YUV411P, AV_PIX_FMT_YUV410P}) {
    auto frame = test::makeNoiseFrame(96, 64, format, 16);
    const auto before = test::deepCopy(*frame);
    for (const int bad : {2, 6, 18}) {
      EXPECT_THROW(applyMosaic(*frame, {MosaicRect{{10, 10, 50, 40}, 16}, MosaicRect{{60, 10, 90, 40}, bad}}),
                   std::invalid_argument)
          << av_get_pix_fmt_name(format) << " block " << bad;
    }
    EXPECT_TRUE(samePlanarPixels(*frame, *before)) << av_get_pix_fmt_name(format);
  }
}

// ... and a multiple of 4 works: 4:1:1 chroma cells are block/4 wide and block high, 4:1:0 block/4 by block/4.
TEST(Mosaic, PixelatesYuv411pAndYuv410pOnTheGrid) {
  // {10, 10, 50, 40} with 16 px blocks: luma [0, 64) x [0, 48); chroma [0, 16) x [0, 48) (4:1:1), [0, 16) x [0, 12) (4:1:0).
  expectPlanarGridMosaic(AV_PIX_FMT_YUV411P, 96, 64, {10, 10, 50, 40}, 16, {0, 0, 64, 48}, {0, 0, 16, 48}, 4, 16, 17);
  expectPlanarGridMosaic(AV_PIX_FMT_YUV410P, 96, 64, {10, 10, 50, 40}, 16, {0, 0, 64, 48}, {0, 0, 16, 12}, 4, 4, 18);
  // The smallest block: 4 px, [8, 52) x [8, 40); 4:1:1 chroma cells 1 x 4 over [2, 13) x [8, 40).
  expectPlanarGridMosaic(AV_PIX_FMT_YUV411P, 96, 64, {10, 10, 50, 40}, 4, {8, 8, 52, 40}, {2, 8, 13, 40}, 1, 4, 19);
}

TEST(Mosaic, RefusesSharedBuffersAndOtherFormats) {
  auto frame = test::makeNoiseFrame(32, 32, AV_PIX_FMT_YUV420P, 4);
  av::FramePtr other = av::allocFrame();
  ASSERT_GE(av_frame_ref(other.get(), frame.get()), 0);
  EXPECT_THROW(applyMosaic(*frame, {MosaicRect{{0, 0, 16, 16}, 16}}), VcError);

  av::FramePtr rgb = av::allocFrame();
  rgb->format = AV_PIX_FMT_RGB24;
  rgb->width = 16;
  rgb->height = 16;
  ASSERT_GE(av_frame_get_buffer(rgb.get(), 0), 0);
  EXPECT_THROW(applyMosaic(*rgb, {MosaicRect{{0, 0, 8, 8}, 16}}), VcError);
}

}  // namespace
}  // namespace vc
