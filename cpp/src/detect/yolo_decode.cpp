#include "detect/yolo_decode.h"

#include <algorithm>
#include <utility>

namespace vc {

std::vector<Box> decodeYolov8(const float* pred, int numClasses, int64_t numAnchors,
                              const Letterbox& lb, const DecodeParams& params) {
  const auto value = [&](int row, int64_t anchor) {
    return pred[static_cast<int64_t>(row) * numAnchors + anchor];
  };
  const float inv = static_cast<float>(1.0 / lb.r);
  const float fw = static_cast<float>(lb.frameWidth);
  const float fh = static_cast<float>(lb.frameHeight);
  std::vector<Box> boxes;
  for (int64_t a = 0; a < numAnchors; ++a) {
    int best = 0;
    float score = value(4, a);
    for (int c = 1; c < numClasses; ++c) {
      const float s = value(4 + c, a);
      if (s > score) {
        score = s;
        best = c;
      }
    }
    if (score < params.scoreThreshold) continue;
    const float cx = value(0, a);
    const float cy = value(1, a);
    const float w = value(2, a);
    const float h = value(3, a);
    Box b;
    b.cls = best;
    b.score = score;
    b.x1 = std::clamp((cx - w / 2) * inv, 0.0f, fw);
    b.y1 = std::clamp((cy - h / 2) * inv, 0.0f, fh);
    b.x2 = std::clamp((cx + w / 2) * inv, 0.0f, fw);
    b.y2 = std::clamp((cy + h / 2) * inv, 0.0f, fh);
    boxes.push_back(b);
  }
  return nmsPerClass(std::move(boxes), params.nmsIou);
}

}  // namespace vc
