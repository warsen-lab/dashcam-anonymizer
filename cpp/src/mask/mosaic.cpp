#include "mask/mosaic.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/vc_error.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

namespace vc {
namespace {

int ceilShift(int value, int shift) { return -((-value) >> shift); }

PixelRect clipped(const PixelRect& r, int width, int height) {
  return {std::clamp(r.x1, 0, width), std::clamp(r.y1, 0, height), std::clamp(r.x2, 0, width),
          std::clamp(r.y2, 0, height)};
}

// Sets every blockW x blockH block of `rect` (plane pixels) to its rounded mean. `step` is the
// byte distance between neighbouring samples, `offset` the first sample's byte (NV12 chroma: 2, 0/1).
void pixelate(uint8_t* data, int stride, const PixelRect& rect, int blockW, int blockH, int step, int offset) {
  for (int by = rect.y1; by < rect.y2; by += blockH) {
    const int ey = std::min(by + blockH, rect.y2);
    for (int bx = rect.x1; bx < rect.x2; bx += blockW) {
      const int ex = std::min(bx + blockW, rect.x2);
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

}  // namespace

PixelRect coveringRect(float x1, float y1, float x2, float y2, int frameWidth, int frameHeight) {
  const PixelRect r{static_cast<int>(std::floor(x1)), static_cast<int>(std::floor(y1)),
                    static_cast<int>(std::ceil(x2)), static_cast<int>(std::ceil(y2))};
  return clipped(r, frameWidth, frameHeight);
}

int mosaicBlockSize(const PixelRect& rect, const MosaicConfig& config) {
  if (config.minBlockPx < 1 || config.blocksPerLongSide < 1) {
    throw std::invalid_argument("mosaic block settings must be >= 1");
  }
  const int longSide = std::max(rect.x2 - rect.x1, rect.y2 - rect.y1);
  return std::max(config.minBlockPx, (longSide + config.blocksPerLongSide - 1) / config.blocksPerLongSide);
}

void applyMosaic(AVFrame& frame, const std::vector<PixelRect>& rects, const MosaicConfig& config) {
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
  for (const PixelRect& raw : rects) {
    const PixelRect r = clipped(raw, frame.width, frame.height);
    if (r.empty()) continue;
    const int block = mosaicBlockSize(r, config);
    pixelate(frame.data[0], frame.linesize[0], r, block, block, 1, 0);
    const PixelRect c{r.x1 >> sw, r.y1 >> sh, ceilShift(r.x2, sw), ceilShift(r.y2, sh)};
    const int bw = std::max(1, ceilShift(block, sw));
    const int bh = std::max(1, ceilShift(block, sh));
    if (nv12) {
      pixelate(frame.data[1], frame.linesize[1], c, bw, bh, 2, 0);
      pixelate(frame.data[1], frame.linesize[1], c, bw, bh, 2, 1);
    } else {
      pixelate(frame.data[1], frame.linesize[1], c, bw, bh, 1, 0);
      pixelate(frame.data[2], frame.linesize[2], c, bw, bh, 1, 0);
    }
  }
}

}  // namespace vc
