#include "detect/yunet_decode.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>

#include "common/vc_error.h"

namespace vc {
namespace {

// int(float) of a value beyond the int range is undefined behaviour; absurd boxes saturate here first
// (vc_train.yunet does the same).
constexpr double kIntLimit = 1 << 30;

struct Candidate {
  float x = 0;  // left
  float y = 0;  // top
  float w = 0;
  float h = 0;
  float score = 0;
};

// cv::Rect2i of a candidate: each value truncated toward zero.
struct IntRect {
  int64_t x = 0;
  int64_t y = 0;
  int64_t w = 0;
  int64_t h = 0;
};

int64_t truncated(float v) { return static_cast<int64_t>(std::clamp(static_cast<double>(v), -kIntLimit, kIntLimit)); }

IntRect intRect(const Candidate& c) { return IntRect{truncated(c.x), truncated(c.y), truncated(c.w), truncated(c.h)}; }

// OpenCV's rectOverlap for integer rectangles: 1.f - float(jaccardDistance(a, b)), where the distance
// is 0 when both areas are 0.
float rectOverlap(const IntRect& a, const IntRect& b) {
  const int64_t areaA = a.w * a.h;
  const int64_t areaB = b.w * b.h;
  if (areaA + areaB <= 0) return 1.0f;
  int64_t inter = 0;
  if (a.w > 0 && a.h > 0 && b.w > 0 && b.h > 0) {
    const int64_t iw = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
    const int64_t ih = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);
    if (iw > 0 && ih > 0) inter = iw * ih;
  }
  const double distance = 1.0 - static_cast<double>(inter) / static_cast<double>(areaA + areaB - inter);
  return 1.0f - static_cast<float>(distance);
}

// Every anchor of one level at or above the threshold, in index order.
void collect(const YuNetLevel& level, int stride, int padWidth, int padHeight, float threshold,
             std::vector<Candidate>& out) {
  const int cols = padWidth / stride;
  const int rows = padHeight / stride;
  const auto s = static_cast<float>(stride);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
      const float cls = level.cls[idx];
      const float obj = level.obj[idx];
      // A NaN / inf score would fail every threshold test and vanish: an unmasked face.
      if (!std::isfinite(cls) || !std::isfinite(obj)) throw VcError("model output has non-finite scores");
      const float score = std::sqrt(std::clamp(cls, 0.0f, 1.0f) * std::clamp(obj, 0.0f, 1.0f));
      if (!(score >= threshold)) continue;
      const float* b = level.bbox + 4 * idx;
      // The raw values too, not only the box they give: -inf in a size makes exp() 0, a finite box that
      // says nothing about the face (vc_train.yunet refuses it as well).
      if (!std::isfinite(b[0]) || !std::isfinite(b[1]) || !std::isfinite(b[2]) || !std::isfinite(b[3])) {
        throw VcError("model output has non-finite box coordinates");
      }
      const float cx = (static_cast<float>(c) + b[0]) * s;
      const float cy = (static_cast<float>(r) + b[1]) * s;
      const float w = std::exp(b[2]) * s;
      const float h = std::exp(b[3]) * s;
      const Candidate k{cx - w / 2.0f, cy - h / 2.0f, w, h, score};
      if (!std::isfinite(k.x) || !std::isfinite(k.y) || !std::isfinite(k.w) || !std::isfinite(k.h)) {
        throw VcError("model output has non-finite box coordinates");
      }
      out.push_back(k);
    }
  }
}

// Indices of the candidates NMS keeps, highest score first.
std::vector<size_t> nms(const std::vector<Candidate>& candidates, float nmsIou, int topK) {
  std::vector<size_t> order(candidates.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return candidates[a].score > candidates[b].score; });
  if (topK > 0 && order.size() > static_cast<size_t>(topK)) order.resize(static_cast<size_t>(topK));
  std::vector<IntRect> rects;
  rects.reserve(candidates.size());
  for (const Candidate& c : candidates) rects.push_back(intRect(c));
  std::vector<size_t> kept;
  for (const size_t i : order) {
    const bool keep = std::all_of(kept.begin(), kept.end(),
                                  [&](size_t k) { return rectOverlap(rects[i], rects[k]) <= nmsIou; });
    if (keep) kept.push_back(i);
  }
  return kept;
}

}  // namespace

std::vector<Box> decodeYuNet(const std::array<YuNetLevel, 3>& levels, int padWidth, int padHeight, int frameWidth,
                             int frameHeight, const YuNetParams& params) {
  std::vector<Candidate> candidates;
  for (size_t i = 0; i < levels.size(); ++i) {
    collect(levels[i], kYuNetStrides[i], padWidth, padHeight, params.scoreThreshold, candidates);
  }
  const auto fw = static_cast<float>(frameWidth);
  const auto fh = static_cast<float>(frameHeight);
  std::vector<Box> boxes;
  for (const size_t i : nms(candidates, params.nmsIou, params.topK)) {
    const Candidate& c = candidates[i];
    Box b;
    b.cls = 0;
    b.score = c.score;
    b.x1 = std::clamp(c.x, 0.0f, fw);
    b.y1 = std::clamp(c.y, 0.0f, fh);
    b.x2 = std::clamp(c.x + c.w, 0.0f, fw);
    b.y2 = std::clamp(c.y + c.h, 0.0f, fh);
    boxes.push_back(b);
  }
  return boxes;
}

}  // namespace vc
