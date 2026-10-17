#include "sampler/luma_thumb.h"

#include <cstdlib>
#include <stdexcept>

namespace vc {

LumaThumb makeLumaThumb(const LumaPlane& plane, int outWidth, int outHeight) {
  if (plane.data == nullptr || plane.width <= 0 || plane.height <= 0 ||
      plane.stride < plane.width) {
    throw std::invalid_argument("makeLumaThumb: invalid luma plane");
  }
  if (outWidth <= 0 || outHeight <= 0 || outWidth > plane.width ||
      outHeight > plane.height) {
    throw std::invalid_argument("makeLumaThumb: invalid thumbnail size");
  }
  LumaThumb thumb{outWidth, outHeight,
                  std::vector<uint8_t>(static_cast<size_t>(outWidth) * outHeight)};
  for (int oy = 0; oy < outHeight; ++oy) {
    const int y0 = oy * plane.height / outHeight;
    const int y1 = (oy + 1) * plane.height / outHeight;
    for (int ox = 0; ox < outWidth; ++ox) {
      const int x0 = ox * plane.width / outWidth;
      const int x1 = (ox + 1) * plane.width / outWidth;
      uint64_t sum = 0;
      for (int y = y0; y < y1; ++y) {
        const uint8_t* row = plane.data + static_cast<size_t>(y) * plane.stride;
        for (int x = x0; x < x1; ++x) {
          sum += row[x];
        }
      }
      const uint64_t count = static_cast<uint64_t>(y1 - y0) * static_cast<uint64_t>(x1 - x0);
      thumb.pixels[static_cast<size_t>(oy) * outWidth + ox] =
          static_cast<uint8_t>((sum + count / 2) / count);
    }
  }
  return thumb;
}

double meanAbsDiff(const LumaThumb& a, const LumaThumb& b) {
  if (a.width != b.width || a.height != b.height || a.pixels.size() != b.pixels.size() ||
      a.pixels.empty()) {
    throw std::invalid_argument("meanAbsDiff: thumbnail size mismatch");
  }
  uint64_t sum = 0;
  for (size_t i = 0; i < a.pixels.size(); ++i) {
    sum += static_cast<uint64_t>(std::abs(int{a.pixels[i]} - int{b.pixels[i]}));
  }
  return static_cast<double>(sum) / (255.0 * static_cast<double>(a.pixels.size()));
}

}  // namespace vc
