#include "detect/box.h"

#include <algorithm>

namespace vc {

float iou(const Box& a, const Box& b) {
  const float w = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  const float h = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (w <= 0.0f || h <= 0.0f) return 0.0f;
  const float inter = w * h;
  const float uni = a.area() + b.area() - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<Box> nmsPerClass(std::vector<Box> boxes, float iouThreshold) {
  std::stable_sort(boxes.begin(), boxes.end(),
                   [](const Box& a, const Box& b) { return a.score > b.score; });
  std::vector<Box> kept;
  for (const Box& b : boxes) {
    const bool suppressed = std::any_of(kept.begin(), kept.end(), [&](const Box& k) {
      return k.cls == b.cls && iou(k, b) >= iouThreshold;
    });
    if (!suppressed) kept.push_back(b);
  }
  return kept;
}

}  // namespace vc
