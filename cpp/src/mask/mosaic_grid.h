#pragma once

// The mosaic grid as pure functions (spec 2026-10-20 §4): which cells a mask covers and the pixels of each, on the
// luma and on a subsampled chroma plane. applyMosaic (mosaic.cpp) uses them on the CPU; the CUDA mosaic (stage 2b)
// calls the same functions in its kernels, so both pixelate exactly the same cells. Plain C++ without the standard
// library (device code cannot call it), every function VC_HOST_DEVICE.

#include <cstdint>

#include "common/host_device.h"

namespace vc {

// Half-open pixel rectangle [x1, x2) x [y1, y2).
struct PixelRect {
  int x1 = 0;
  int y1 = 0;
  int x2 = 0;
  int y2 = 0;

  VC_HOST_DEVICE bool empty() const { return x2 <= x1 || y2 <= y1; }
};

namespace grid {

VC_HOST_DEVICE inline int clampInt(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

VC_HOST_DEVICE inline int minInt(int a, int b) { return a < b ? a : b; }

VC_HOST_DEVICE inline int maxInt(int a, int b) { return a > b ? a : b; }

// ceil(value / 2^shift) for any int (an arithmetic right shift rounds down).
VC_HOST_DEVICE inline int ceilShift(int value, int shift) { return -((-value) >> shift); }

// `r` clipped to a width x height plane.
VC_HOST_DEVICE inline PixelRect clipRect(const PixelRect& r, int width, int height) {
  return {clampInt(r.x1, 0, width), clampInt(r.y1, 0, height), clampInt(r.x2, 0, width), clampInt(r.y2, 0, height)};
}

// `r` (inside the frame) with its edges moved out to multiples of `block`, clipped to the frame: the grid is
// anchored at the frame's top-left corner, not at the mask.
VC_HOST_DEVICE inline PixelRect snapRect(const PixelRect& r, int block, int width, int height) {
  const auto down = [block](int v) { return v / block * block; };
  const auto up = [block](int v) { return static_cast<int>((static_cast<int64_t>(v) + block - 1) / block * block); };
  return clipRect(PixelRect{down(r.x1), down(r.y1), up(r.x2), up(r.y2)}, width, height);
}

// The cells of one plane that `area` (plane pixels, inside the plane) touches, on the cellW x cellH grid anchored
// at the plane's origin: cols x rows cells from grid cell (col0, row0). A cell the area's edges cut keeps only its
// part inside the area (a snapped area has no such cells except at the plane's right and bottom edges).
struct CellGrid {
  PixelRect area{};
  int cellW = 1;
  int cellH = 1;
  int col0 = 0;
  int row0 = 0;
  int cols = 0;
  int rows = 0;

  VC_HOST_DEVICE int cells() const { return cols * rows; }
};

VC_HOST_DEVICE inline CellGrid cellGrid(const PixelRect& area, int cellW, int cellH) {
  CellGrid g;
  g.area = area;
  g.cellW = cellW;
  g.cellH = cellH;
  if (!area.empty()) {
    g.col0 = area.x1 / cellW;
    g.row0 = area.y1 / cellH;
    g.cols = (area.x2 - 1) / cellW - g.col0 + 1;
    g.rows = (area.y2 - 1) / cellH - g.row0 + 1;
  }
  return g;
}

// The pixels of cell (col, row) of `g` inside its area, 0 <= col < cols, 0 <= row < rows.
VC_HOST_DEVICE inline PixelRect cellRect(const CellGrid& g, int col, int row) {
  const int gx = (g.col0 + col) * g.cellW;
  const int gy = (g.row0 + row) * g.cellH;
  return {maxInt(gx, g.area.x1), maxInt(gy, g.area.y1), minInt(gx + g.cellW, g.area.x2),
          minInt(gy + g.cellH, g.area.y2)};
}

// The mean of `count` samples summing to `sum`, rounded half up.
VC_HOST_DEVICE inline uint8_t roundedMean(uint64_t sum, uint64_t count) {
  return static_cast<uint8_t>((sum + count / 2) / count);
}

// The luma and chroma cells one mask covers: `rect` (frame pixels) clipped to the frame, snapped out to the grid
// of `block` when `snap` (vc_redact's faces; plates keep their edges), its chroma area the luma area shifted by the
// format's chroma subsampling (log2, rounded outwards) with (block >> log2) cells. Both grids are empty when the
// clipped rect is.
struct MaskCells {
  CellGrid luma{};
  CellGrid chroma{};
};

VC_HOST_DEVICE inline MaskCells maskCells(const PixelRect& rect, int block, bool snap, int width, int height,
                                          int log2ChromaW, int log2ChromaH) {
  MaskCells m;
  const PixelRect r = clipRect(rect, width, height);
  if (r.empty()) return m;
  const PixelRect area = snap ? snapRect(r, block, width, height) : r;
  m.luma = cellGrid(area, block, block);
  const PixelRect chroma{area.x1 >> log2ChromaW, area.y1 >> log2ChromaH, ceilShift(area.x2, log2ChromaW),
                         ceilShift(area.y2, log2ChromaH)};
  m.chroma = cellGrid(chroma, block >> log2ChromaW, block >> log2ChromaH);
  return m;
}

}  // namespace grid
}  // namespace vc
