#include "mask/mosaic_grid.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "mask/mosaic.h"
#include "test_media.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

namespace vc {
namespace {

bool sameRect(const PixelRect& a, const PixelRect& b) { return a.x1 == b.x1 && a.y1 == b.y1 && a.x2 == b.x2 && a.y2 == b.y2; }

std::string text(const PixelRect& r) {
  return "[" + std::to_string(r.x1) + ", " + std::to_string(r.y1) + ")-(" + std::to_string(r.x2) + ", " +
         std::to_string(r.y2) + ")";
}

TEST(MosaicGrid, ClipsToThePlane) {
  EXPECT_TRUE(sameRect(grid::clipRect({-5, -1, 700, 400}, 640, 360), PixelRect{0, 0, 640, 360}));
  EXPECT_TRUE(sameRect(grid::clipRect({10, 20, 30, 40}, 640, 360), PixelRect{10, 20, 30, 40}));
  EXPECT_TRUE(grid::clipRect({700, 10, 800, 20}, 640, 360).empty());
}

// The grid is anchored at the frame's top-left corner: edges move out to multiples of the block, then the frame
// clips them.
TEST(MosaicGrid, SnapsToTheFramesGrid) {
  EXPECT_TRUE(sameRect(grid::snapRect({85, 88, 215, 192}, 24, 640, 360), PixelRect{72, 72, 216, 192}));
  EXPECT_TRUE(sameRect(grid::snapRect({96, 96, 120, 120}, 24, 640, 360), PixelRect{96, 96, 120, 120}));
  EXPECT_TRUE(sameRect(grid::snapRect({630, 350, 640, 360}, 16, 640, 360), PixelRect{624, 336, 640, 360}));
  EXPECT_TRUE(sameRect(grid::snapRect({0, 0, 1, 1}, 256, 640, 360), PixelRect{0, 0, 256, 256}));
}

TEST(MosaicGrid, CeilShiftRoundsUp) {
  EXPECT_EQ(grid::ceilShift(0, 1), 0);
  EXPECT_EQ(grid::ceilShift(5, 1), 3);
  EXPECT_EQ(grid::ceilShift(6, 1), 3);
  EXPECT_EQ(grid::ceilShift(5, 2), 2);
  EXPECT_EQ(grid::ceilShift(7, 0), 7);
}

// A snapped 101 x 50 area in 16 px cells: 7 columns (the last 5 px wide, cut by the plane's edge) and 4 rows.
TEST(MosaicGrid, CellsOfASnappedAreaStartOnItsEdges) {
  const grid::CellGrid g = grid::cellGrid({32, 16, 133, 66}, 16, 16);
  EXPECT_EQ(g.col0, 2);
  EXPECT_EQ(g.row0, 1);
  EXPECT_EQ(g.cols, 7);
  EXPECT_EQ(g.rows, 4);
  EXPECT_EQ(g.cells(), 28);
  EXPECT_TRUE(sameRect(grid::cellRect(g, 0, 0), PixelRect{32, 16, 48, 32}));
  EXPECT_TRUE(sameRect(grid::cellRect(g, 6, 3), PixelRect{128, 64, 133, 66}));
  EXPECT_EQ(grid::cellGrid({10, 10, 10, 20}, 8, 8).cells(), 0);
}

// An area off the grid (an unsnapped plate) touches the frame grid's cells and keeps only their parts inside it:
// [10, 50) x [10, 40) in 16 px cells is 4 columns ([10, 16) ... [48, 50)) and 3 rows ([10, 16) ... [32, 40)).
TEST(MosaicGrid, CellsOfAnUnsnappedAreaAreCutByItsEdges) {
  const grid::CellGrid g = grid::cellGrid({10, 10, 50, 40}, 16, 16);
  EXPECT_EQ(g.col0, 0);
  EXPECT_EQ(g.row0, 0);
  EXPECT_EQ(g.cols, 4);
  EXPECT_EQ(g.rows, 3);
  EXPECT_TRUE(sameRect(grid::cellRect(g, 0, 0), PixelRect{10, 10, 16, 16}));
  EXPECT_TRUE(sameRect(grid::cellRect(g, 1, 1), PixelRect{16, 16, 32, 32}));
  EXPECT_TRUE(sameRect(grid::cellRect(g, 3, 2), PixelRect{48, 32, 50, 40}));
}

TEST(MosaicGrid, MeanRoundsHalfUp) {
  EXPECT_EQ(grid::roundedMean(5, 2), 3);  // 2.5
  EXPECT_EQ(grid::roundedMean(4, 3), 1);  // 1.33
  EXPECT_EQ(grid::roundedMean(5, 3), 2);  // 1.67
  EXPECT_EQ(grid::roundedMean(255 * 256, 256), 255);
  EXPECT_EQ(grid::roundedMean(0, 7), 0);
}

// 4:2:0 chroma: half the area (rounded out) in half-size cells; 4:2:2 halves only the width; 4:4:4 keeps both; an
// unsnapped rect keeps its own edges.
TEST(MosaicGrid, MaskCellsPerChromaLayoutAndSnap) {
  const PixelRect rect{85, 88, 215, 192};
  const grid::MaskCells c420 = grid::maskCells(rect, 24, true, 640, 360, 1, 1);
  EXPECT_TRUE(sameRect(c420.luma.area, PixelRect{72, 72, 216, 192}));
  EXPECT_EQ(c420.luma.cellW, 24);
  EXPECT_TRUE(sameRect(c420.chroma.area, PixelRect{36, 36, 108, 96}));
  EXPECT_EQ(c420.chroma.cellW, 12);
  EXPECT_EQ(c420.chroma.cellH, 12);
  EXPECT_EQ(c420.chroma.cells(), c420.luma.cells());
  const grid::MaskCells c422 = grid::maskCells(rect, 24, true, 640, 360, 1, 0);
  EXPECT_TRUE(sameRect(c422.chroma.area, PixelRect{36, 72, 108, 192}));
  EXPECT_EQ(c422.chroma.cellW, 12);
  EXPECT_EQ(c422.chroma.cellH, 24);
  const grid::MaskCells c444 = grid::maskCells(rect, 24, true, 640, 360, 0, 0);
  EXPECT_TRUE(sameRect(c444.chroma.area, c444.luma.area));
  // The unsnapped NV12 case of Mosaic.AnUnsnappedNv12RectIsPixelatedOnlyInsideItself: chroma [5, 26) x [4, 21).
  const grid::MaskCells plate = grid::maskCells({11, 9, 51, 41}, 16, false, 96, 64, 1, 1);
  EXPECT_TRUE(sameRect(plate.luma.area, PixelRect{11, 9, 51, 41}));
  EXPECT_TRUE(sameRect(plate.chroma.area, PixelRect{5, 4, 26, 21}));
  // An odd-sized 4:2:0 frame: the chroma plane is ceil(width / 2) wide, and so is the area at the right edge.
  const grid::MaskCells odd = grid::maskCells({620, 0, 641, 16}, 16, true, 641, 361, 1, 1);
  EXPECT_TRUE(sameRect(odd.luma.area, PixelRect{608, 0, 641, 16}));
  EXPECT_TRUE(sameRect(odd.chroma.area, PixelRect{304, 0, 321, 8}));
  for (const bool snap : {true, false}) {  // outside the frame: nothing
    EXPECT_EQ(grid::maskCells({700, 10, 800, 20}, 16, snap, 640, 360, 1, 1).luma.cells(), 0);
    EXPECT_EQ(grid::maskCells({700, 10, 800, 20}, 16, snap, 640, 360, 1, 1).chroma.cells(), 0);
  }
}

// Every pixel of the area lies in exactly one cell and no cell is empty, for random masks, blocks and snap flags.
TEST(MosaicGrid, CellsPartitionTheArea) {
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> coord(-40, 700);
  for (int trial = 0; trial < 200; ++trial) {
    const int block = kMosaicBlockLadder[static_cast<size_t>(trial) % kMosaicBlockLadder.size()];
    const bool snap = trial % 2 == 0;
    const PixelRect rect{coord(rng), coord(rng), coord(rng), coord(rng)};
    const grid::CellGrid g = grid::maskCells(rect, block, snap, 641, 361, 1, 1).luma;
    std::vector<int> hits(641 * 361, 0);
    for (int row = 0; row < g.rows; ++row) {
      for (int col = 0; col < g.cols; ++col) {
        const PixelRect cell = grid::cellRect(g, col, row);
        ASSERT_FALSE(cell.empty()) << text(rect) << " block " << block << " snap " << snap;
        for (int y = cell.y1; y < cell.y2; ++y) {
          for (int x = cell.x1; x < cell.x2; ++x) ++hits[static_cast<size_t>(y) * 641 + static_cast<size_t>(x)];
        }
      }
    }
    for (int y = 0; y < 361; ++y) {
      for (int x = 0; x < 641; ++x) {
        const bool inside = x >= g.area.x1 && x < g.area.x2 && y >= g.area.y1 && y < g.area.y2;
        ASSERT_EQ(hits[static_cast<size_t>(y) * 641 + static_cast<size_t>(x)], inside ? 1 : 0)
            << text(rect) << " block " << block << " snap " << snap << " at " << x << "," << y;
      }
    }
  }
}

// The mosaic as it was before it used mosaic_grid.h (commit 0f2aa85), kept here as the reference: applyMosaic must
// produce the same bytes.
void legacyPixelate(uint8_t* data, int stride, const PixelRect& area, int cellW, int cellH, int step, int offset) {
  for (int gy = area.y1 / cellH * cellH; gy < area.y2; gy += cellH) {
    const int by = std::max(gy, area.y1);
    const int ey = std::min(gy + cellH, area.y2);
    for (int gx = area.x1 / cellW * cellW; gx < area.x2; gx += cellW) {
      const int bx = std::max(gx, area.x1);
      const int ex = std::min(gx + cellW, area.x2);
      uint64_t sum = 0;
      for (int y = by; y < ey; ++y) {
        const uint8_t* row = data + static_cast<ptrdiff_t>(y) * stride + offset;
        for (int x = bx; x < ex; ++x) sum += row[x * step];
      }
      const uint64_t count = static_cast<uint64_t>(ey - by) * static_cast<uint64_t>(ex - bx);
      const auto mean = static_cast<uint8_t>((sum + count / 2) / count);
      for (int y = by; y < ey; ++y) {
        uint8_t* row = data + static_cast<ptrdiff_t>(y) * stride + offset;
        for (int x = bx; x < ex; ++x) row[x * step] = mean;
      }
    }
  }
}

void legacyMosaic(AVFrame& frame, const std::vector<MosaicRect>& rects) {
  const auto format = static_cast<AVPixelFormat>(frame.format);
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(format);
  const bool nv12 = format == AV_PIX_FMT_NV12;
  const int sw = desc->log2_chroma_w;
  const int sh = desc->log2_chroma_h;
  const auto ceilShift = [](int value, int shift) { return -((-value) >> shift); };
  const auto clipped = [&](const PixelRect& r) {
    return PixelRect{std::clamp(r.x1, 0, frame.width), std::clamp(r.y1, 0, frame.height), std::clamp(r.x2, 0, frame.width),
                     std::clamp(r.y2, 0, frame.height)};
  };
  std::vector<MosaicRect> ordered = rects;
  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const MosaicRect& a, const MosaicRect& b) { return a.block > b.block; });
  for (const MosaicRect& m : ordered) {
    const PixelRect r = clipped(m.rect);
    if (r.empty()) continue;
    const int b = m.block;
    const auto up = [b](int v) { return static_cast<int>((static_cast<int64_t>(v) + b - 1) / b * b); };
    const PixelRect area = m.snap ? clipped({r.x1 / b * b, r.y1 / b * b, up(r.x2), up(r.y2)}) : r;
    legacyPixelate(frame.data[0], frame.linesize[0], area, b, b, 1, 0);
    const PixelRect c{area.x1 >> sw, area.y1 >> sh, ceilShift(area.x2, sw), ceilShift(area.y2, sh)};
    if (nv12) {
      legacyPixelate(frame.data[1], frame.linesize[1], c, b >> sw, b >> sh, 2, 0);
      legacyPixelate(frame.data[1], frame.linesize[1], c, b >> sw, b >> sh, 2, 1);
    } else {
      legacyPixelate(frame.data[1], frame.linesize[1], c, b >> sw, b >> sh, 1, 0);
      legacyPixelate(frame.data[2], frame.linesize[2], c, b >> sw, b >> sh, 1, 0);
    }
  }
}

// Every byte of every plane of two frames of the same format and size.
bool sameBytes(const AVFrame& a, const AVFrame& b) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(a.format));
  const bool nv12 = a.format == AV_PIX_FMT_NV12;
  for (int plane = 0; plane < (nv12 ? 2 : 3); ++plane) {
    const int cw = (a.width + (1 << desc->log2_chroma_w) - 1) >> desc->log2_chroma_w;
    const int w = plane == 0 ? a.width : (nv12 ? 2 * cw : cw);
    const int h = plane == 0 ? a.height : (a.height + (1 << desc->log2_chroma_h) - 1) >> desc->log2_chroma_h;
    for (int y = 0; y < h; ++y) {
      if (!std::equal(a.data[plane] + static_cast<ptrdiff_t>(y) * a.linesize[plane],
                      a.data[plane] + static_cast<ptrdiff_t>(y) * a.linesize[plane] + w,
                      b.data[plane] + static_cast<ptrdiff_t>(y) * b.linesize[plane])) {
        return false;
      }
    }
  }
  return true;
}

// spec 2026-10-20 §4: the CPU mosaic on the shared grid is byte for byte what it was, for every layout the reader
// emits, odd frame sizes, snapped and unsnapped rects, overlapping rects of mixed blocks included.
TEST(MosaicGrid, ApplyMosaicMatchesThePreviousImplementationByteForByte) {
  std::mt19937 rng(2026);
  for (const AVPixelFormat format : {AV_PIX_FMT_YUV420P, AV_PIX_FMT_NV12, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P,
                                     AV_PIX_FMT_YUV411P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUV440P,
                                     AV_PIX_FMT_YUV410P}) {
    for (const auto& [width, height] : {std::pair{640, 360}, std::pair{641, 361}, std::pair{1920, 1080}}) {
      // A corner (it may lie outside the frame) and a positive size, so that no rect starts out empty and the
      // rects overlap (and cross the frame edges) often enough to cover the draw order.
      std::uniform_int_distribution<int> x(-50, width + 50);
      std::uniform_int_distribution<int> y(-50, height + 50);
      std::uniform_int_distribution<int> w(1, width / 2);
      std::uniform_int_distribution<int> h(1, height / 2);
      std::uniform_int_distribution<size_t> step(0, kMosaicBlockLadder.size() - 1);
      for (int trial = 0; trial < 6; ++trial) {
        std::vector<MosaicRect> rects;
        const auto randomRect = [&](int block, bool snap) {
          const int x1 = x(rng);
          const int y1 = y(rng);
          return MosaicRect{{x1, y1, x1 + w(rng), y1 + h(rng)}, block, snap};
        };
        for (int k = 0; k < 4; ++k) rects.push_back(randomRect(kMosaicBlockLadder[step(rng)], k % 2 == 0));
        rects.push_back(randomRect(16, trial % 2 == 0));
        av::FramePtr shared = test::makeNoiseFrame(width, height, format, static_cast<unsigned>(trial + width));
        av::FramePtr legacy = test::deepCopy(*shared);
        applyMosaic(*shared, rects);
        legacyMosaic(*legacy, rects);
        ASSERT_TRUE(sameBytes(*shared, *legacy))
            << av_get_pix_fmt_name(format) << " " << width << "x" << height << " trial " << trial;
      }
    }
  }
}

}  // namespace
}  // namespace vc
