#include "mask/mosaic.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/vc_error.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

namespace vc {
namespace {

// A track's block moves up the ladder only when its mask asks for more than this many times the block.
constexpr double kBlockUpgradeFactor = 1.5;

int ceilShift(int value, int shift) { return -((-value) >> shift); }

PixelRect clipped(const PixelRect& r, int width, int height) {
  return {std::clamp(r.x1, 0, width), std::clamp(r.y1, 0, height), std::clamp(r.x2, 0, width),
          std::clamp(r.y2, 0, height)};
}

// `r` (inside the frame) with its edges moved out to multiples of `block`, clipped to the frame.
PixelRect snapped(const PixelRect& r, int block, int width, int height) {
  const auto down = [block](int v) { return v / block * block; };
  const auto up = [block](int v) {
    return static_cast<int>((static_cast<int64_t>(v) + block - 1) / block * block);
  };
  return clipped({down(r.x1), down(r.y1), up(r.x2), up(r.y2)}, width, height);
}

// Sets every cell of the cellW x cellH grid anchored at the plane's origin inside `area` (plane pixels,
// its edges on the grid or at the plane's edge) to the cell's rounded mean. `step` is the byte distance
// between neighbouring samples, `offset` the first sample's byte (NV12 chroma: 2, 0 / 1).
void pixelateGrid(uint8_t* data, int stride, const PixelRect& area, int cellW, int cellH, int step, int offset) {
  for (int by = area.y1; by < area.y2; by += cellH) {
    const int ey = std::min(by + cellH, area.y2);
    for (int bx = area.x1; bx < area.x2; bx += cellW) {
      const int ex = std::min(bx + cellW, area.x2);
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

void requireValidConfig(const MosaicConfig& config) {
  if (config.minBlockPx < 1 || config.blocksPerLongSide < 1) {
    throw std::invalid_argument("mosaic block settings must be >= 1");
  }
}

// Even, so that the chroma grid of 4:2:0 and 4:2:2 (half the block) lines up with the luma grid.
void requireValidBlock(int block) {
  if (block < 2 || block % 2 != 0) {
    throw std::invalid_argument("mosaic block sizes must be even and >= 2, got " + std::to_string(block));
  }
}

}  // namespace

PixelRect coveringRect(float x1, float y1, float x2, float y2, int frameWidth, int frameHeight) {
  if (!std::isfinite(x1) || !std::isfinite(y1) || !std::isfinite(x2) || !std::isfinite(y2)) {
    throw VcError("mask box has non-finite coordinates");
  }
  // Clamp while still a floating-point number: converting an out-of-range value to int is undefined.
  const auto edge = [](double v, int size) {
    return static_cast<int>(std::clamp(v, 0.0, static_cast<double>(size)));
  };
  return {edge(std::floor(x1), frameWidth), edge(std::floor(y1), frameHeight), edge(std::ceil(x2), frameWidth),
          edge(std::ceil(y2), frameHeight)};
}

int mosaicBlockSize(const PixelRect& rect, const MosaicConfig& config) {
  requireValidConfig(config);
  const int64_t longSide = std::max(rect.x2 - rect.x1, rect.y2 - rect.y1);
  const int64_t perBlock = (longSide + config.blocksPerLongSide - 1) / config.blocksPerLongSide;
  return static_cast<int>(std::max<int64_t>(config.minBlockPx, perBlock));
}

int ladderBlock(int wanted) {
  for (const int step : kMosaicBlockLadder) {
    if (step >= wanted) return step;
  }
  return kMosaicBlockLadder.back();
}

int nextTrackBlock(int current, int wanted) {
  if (current <= 0) return ladderBlock(wanted);
  if (static_cast<double>(wanted) > kBlockUpgradeFactor * current) return std::max(current, ladderBlock(wanted));
  return current;
}

TrackBlocks::TrackBlocks(const MosaicConfig& config) : config_(config) { requireValidConfig(config); }

int TrackBlocks::blockFor(int64_t track, const PixelRect& rect) {
  const auto it = previous_.find(track);
  const int block = nextTrackBlock(it == previous_.end() ? 0 : it->second, mosaicBlockSize(rect, config_));
  current_[track] = block;
  return block;
}

void TrackBlocks::endFrame() {
  previous_ = std::move(current_);
  current_.clear();
}

void applyMosaic(AVFrame& frame, const std::vector<MosaicRect>& rects) {
  for (const MosaicRect& r : rects) requireValidBlock(r.block);  // before any pixel is touched
  const auto format = static_cast<AVPixelFormat>(frame.format);
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(format);
  const bool nv12 = format == AV_PIX_FMT_NV12;
  const bool planar =
      desc != nullptr && (desc->flags & AV_PIX_FMT_FLAG_PLANAR) != 0 &&
      (desc->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_HWACCEL | AV_PIX_FMT_FLAG_PAL)) == 0 &&
      desc->nb_components == 3 && desc->comp[0].depth == 8 && desc->comp[1].plane == 1 &&
      desc->comp[2].plane == 2;
  if (!nv12 && !planar) {
    throw VcError(std::string("mosaic: unsupported pixel format ") + (desc ? desc->name : "unknown"));
  }
  if (!av_frame_is_writable(&frame)) {
    throw VcError("mosaic: frame buffers are shared; make the frame writable first");
  }
  const int sw = desc->log2_chroma_w;
  const int sh = desc->log2_chroma_h;
  // Largest blocks first: where masks overlap, the finer mosaic drawn last is what shows.
  std::vector<MosaicRect> ordered = rects;
  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const MosaicRect& a, const MosaicRect& b) { return a.block > b.block; });
  for (const MosaicRect& m : ordered) {
    const PixelRect r = clipped(m.rect, frame.width, frame.height);
    if (r.empty()) continue;
    const PixelRect area = snapped(r, m.block, frame.width, frame.height);
    pixelateGrid(frame.data[0], frame.linesize[0], area, m.block, m.block, 1, 0);
    const PixelRect c{area.x1 >> sw, area.y1 >> sh, ceilShift(area.x2, sw), ceilShift(area.y2, sh)};
    const int cw = m.block >> sw;
    const int ch = m.block >> sh;
    if (nv12) {
      pixelateGrid(frame.data[1], frame.linesize[1], c, cw, ch, 2, 0);
      pixelateGrid(frame.data[1], frame.linesize[1], c, cw, ch, 2, 1);
    } else {
      pixelateGrid(frame.data[1], frame.linesize[1], c, cw, ch, 1, 0);
      pixelateGrid(frame.data[2], frame.linesize[2], c, cw, ch, 1, 0);
    }
  }
}

}  // namespace vc
