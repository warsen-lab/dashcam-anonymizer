#pragma once

#include <vector>

struct AVFrame;

namespace vc {

struct MosaicConfig {
  int minBlockPx = 16;         // no block is smaller than this
  int blocksPerLongSide = 8;   // at most this many blocks along a mask's long side
};

// Half-open pixel rectangle [x1, x2) x [y1, y2).
struct PixelRect {
  int x1 = 0;
  int y1 = 0;
  int x2 = 0;
  int y2 = 0;

  bool empty() const { return x2 <= x1 || y2 <= y1; }
};

// Smallest pixel rectangle covering the box, clipped to the frame.
PixelRect coveringRect(float x1, float y1, float x2, float y2, int frameWidth, int frameHeight);

// max(minBlockPx, ceil(long side / blocksPerLongSide)); throws std::invalid_argument when a
// setting is below 1.
int mosaicBlockSize(const PixelRect& rect, const MosaicConfig& config);

// Replaces every block of every rect (grid from the rect's top-left) by its mean: luma at full
// resolution, chroma at chroma resolution with correspondingly smaller blocks. Supports 8-bit
// planar YUV and NV12. Throws VcError for other pixel formats or when the frame's buffers are
// shared (call av_frame_make_writable first).
void applyMosaic(AVFrame& frame, const std::vector<PixelRect>& rects, const MosaicConfig& config);

}  // namespace vc
