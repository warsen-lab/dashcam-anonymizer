#pragma once

#include <cstdint>
#include <vector>

#include "common/luma_plane.h"

namespace vc {

// Small grayscale thumbnail used for motion and near-duplicate scoring.
struct LumaThumb {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;  // row-major, width * height
};

inline constexpr int kThumbWidth = 160;
inline constexpr int kThumbHeight = 90;

// Box-average downscale. Throws std::invalid_argument for an invalid plane or
// an output size that is non-positive or larger than the input.
LumaThumb makeLumaThumb(const LumaPlane& plane, int outWidth = kThumbWidth,
                        int outHeight = kThumbHeight);

// Mean absolute pixel difference normalised to [0, 1]. Throws
// std::invalid_argument when the sizes differ or the thumbnails are empty.
double meanAbsDiff(const LumaThumb& a, const LumaThumb& b);

}  // namespace vc
