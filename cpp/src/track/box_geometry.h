#pragma once

#include <algorithm>
#include <limits>

#include "detect/box.h"

namespace vc {

// Box arithmetic shared by the track code. Sizes are computed in double: the float difference of two finite
// coordinates can overflow to inf, and inf later turns into a NaN margin (0 * inf) that would drop a mask silently.

struct Point {
  double x = 0;
  double y = 0;
};

inline Point centerOf(const Box& b) {
  return {(static_cast<double>(b.x1) + b.x2) / 2.0, (static_cast<double>(b.y1) + b.y2) / 2.0};
}

inline double widthOf(const Box& b) { return b.x2 > b.x1 ? static_cast<double>(b.x2) - b.x1 : 0.0; }
inline double heightOf(const Box& b) { return b.y2 > b.y1 ? static_cast<double>(b.y2) - b.y1 : 0.0; }

// The nearest float, saturating instead of overflowing to inf.
inline float toFloat(double v) {
  constexpr double kMax = std::numeric_limits<float>::max();
  return static_cast<float>(std::clamp(v, -kMax, kMax));
}

inline Box shifted(const Box& b, double dx, double dy) {
  Box s = b;
  s.x1 = toFloat(b.x1 + dx);
  s.x2 = toFloat(b.x2 + dx);
  s.y1 = toFloat(b.y1 + dy);
  s.y2 = toFloat(b.y2 + dy);
  return s;
}

// Each coordinate a fraction t of its way from a's to b's; the score is the higher of the two.
inline Box lerp(const Box& a, const Box& b, double t) {
  const auto mix = [t](float p, float q) {
    return toFloat(static_cast<double>(p) + (static_cast<double>(q) - static_cast<double>(p)) * t);
  };
  Box out = a;
  out.x1 = mix(a.x1, b.x1);
  out.y1 = mix(a.y1, b.y1);
  out.x2 = mix(a.x2, b.x2);
  out.y2 = mix(a.y2, b.y2);
  out.score = std::max(a.score, b.score);
  return out;
}

}  // namespace vc
