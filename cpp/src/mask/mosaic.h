#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

struct AVFrame;

namespace vc {

struct MosaicConfig {
  int minBlockPx = 16;        // no block is asked for below this
  int blocksPerLongSide = 8;  // a mask asks for about this many blocks along its long side
};

// Half-open pixel rectangle [x1, x2) x [y1, y2).
struct PixelRect {
  int x1 = 0;
  int y1 = 0;
  int x2 = 0;
  int y2 = 0;

  bool empty() const { return x2 <= x1 || y2 <= y1; }
};

// Smallest pixel rectangle covering the box, clipped to the frame. Throws VcError when a
// coordinate is not finite (a broken box must stop the run, not become a mask).
PixelRect coveringRect(float x1, float y1, float x2, float y2, int frameWidth, int frameHeight);

// The block a mask asks for: max(minBlockPx, ceil(long side / blocksPerLongSide)). Throws
// std::invalid_argument when a setting is below 1.
int mosaicBlockSize(const PixelRect& rect, const MosaicConfig& config);

// The block sizes masks are pixelated with (spec 2026-10-19 §4).
inline constexpr std::array<int, 9> kMosaicBlockLadder = {16, 24, 32, 48, 64, 96, 128, 192, 256};

// The smallest ladder step >= `wanted`; the top step for anything larger.
int ladderBlock(int wanted);

// The block of a track's next mask: ladderBlock(wanted) for a new track (`current` 0); afterwards
// `current`, moved up to ladderBlock(wanted) only when `wanted` exceeds 1.5 times `current`. It never
// moves down.
int nextTrackBlock(int current, int wanted);

// The blocks of every track over consecutive frames (spec §4).
class TrackBlocks {
 public:
  // Throws std::invalid_argument when a setting is below 1.
  explicit TrackBlocks(const MosaicConfig& config);
  // The block for `track`'s mask `rect` on the current frame.
  int blockFor(int64_t track, const PixelRect& rect);
  // Ends the current frame: a track without a mask on it is forgotten (BoxTimeline masks a track on
  // consecutive frames until it ends).
  void endFrame();

 private:
  MosaicConfig config_;
  std::unordered_map<int64_t, int> previous_;
  std::unordered_map<int64_t, int> current_;
};

// A rect and the block it is pixelated with.
struct MosaicRect {
  PixelRect rect;
  int block = kMosaicBlockLadder[0];
};

// Pixelates each rect on the grid of its block size b anchored at the frame's top-left corner (spec §4):
// the rect's edges move out to multiples of b (x1 = floor(x1 / b) * b, x2 = ceil(x2 / b) * b), are
// clipped to the frame, and every b x b cell inside becomes the rounded mean of its pixels; a cell the
// frame's right or bottom edge cuts averages the pixels that are there. Chroma uses (b >> log2_chroma_w)
// x (b >> log2_chroma_h) cells on the same grid (b / 2 for 4:2:0). Rects are applied largest block first
// (equal blocks in list order), so where masks overlap the finer mosaic drawn last shows. Supports 8-bit
// planar YUV and NV12. Throws std::invalid_argument, before any pixel is touched, for a block that is not
// an even number >= 2 or not a multiple of the frame's chroma subsampling factor (so a whole number of
// chroma samples: 4 for 4:1:1 and 4:1:0, where an even block such as 2 or 6 would leave the chroma grid
// empty or off the luma grid), and VcError for other pixel formats or when the frame's buffers are shared
// (call av_frame_make_writable first).
void applyMosaic(AVFrame& frame, const std::vector<MosaicRect>& rects);

}  // namespace vc
