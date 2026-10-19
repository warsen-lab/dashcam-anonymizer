#include "detect/yolo_decode.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "common/vc_error.h"

namespace vc {

std::vector<Box> decodeYolov8(const float* pred, int numClasses, int64_t numAnchors,
                              const Letterbox& lb, const DecodeParams& params) {
  const auto value = [&](int row, int64_t anchor) {
    return pred[static_cast<int64_t>(row) * numAnchors + anchor];
  };
  // Divide by the float32 r, as numpy does for the float32 coordinates in the Python decoder.
  const float r = static_cast<float>(lb.r);
  const float fw = static_cast<float>(lb.frameWidth);
  const float fh = static_cast<float>(lb.frameHeight);
  std::vector<Box> boxes;
  for (int64_t a = 0; a < numAnchors; ++a) {
    int best = 0;
    float score = value(4, a);
    bool finite = std::isfinite(score);
    for (int c = 1; c < numClasses; ++c) {
      const float s = value(4 + c, a);
      finite = finite && std::isfinite(s);
      if (s > score) {
        score = s;
        best = c;
      }
    }
    // A healthy model never emits a NaN / inf score. Such an anchor would fail every threshold test
    // and silently vanish, and a run that loses all its detections this way is an unmasked video.
    if (!finite) throw VcError("model output has non-finite scores");
    if (!(score >= params.scoreThreshold)) continue;
    const float cx = value(0, a);
    const float cy = value(1, a);
    const float w = value(2, a);
    const float h = value(3, a);
    // A kept anchor with a broken box is a broken model (or output): stop rather than mask less.
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h)) {
      throw VcError("model output has non-finite box coordinates");
    }
    Box b;
    b.cls = best;
    b.score = score;
    b.x1 = std::clamp((cx - w / 2) / r, 0.0f, fw);
    b.y1 = std::clamp((cy - h / 2) / r, 0.0f, fh);
    b.x2 = std::clamp((cx + w / 2) / r, 0.0f, fw);
    b.y2 = std::clamp((cy + h / 2) / r, 0.0f, fh);
    boxes.push_back(b);
  }
  return nmsPerClass(std::move(boxes), params.nmsIou);
}

}  // namespace vc
