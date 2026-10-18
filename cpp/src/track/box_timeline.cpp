#include "track/box_timeline.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vc {
namespace {

struct Point {
  double x = 0;
  double y = 0;
};

Point centerOf(const Box& b) { return {(b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0}; }

Box shifted(const Box& b, double dx, double dy) {
  Box s = b;
  s.x1 = static_cast<float>(b.x1 + dx);
  s.x2 = static_cast<float>(b.x2 + dx);
  s.y1 = static_cast<float>(b.y1 + dy);
  s.y2 = static_cast<float>(b.y2 + dy);
  return s;
}

Box lerp(const Box& a, const Box& b, double t) {
  const auto mix = [t](float p, float q) { return static_cast<float>(p + (q - p) * t); };
  Box out = a;
  out.x1 = mix(a.x1, b.x1);
  out.y1 = mix(a.y1, b.y1);
  out.x2 = mix(a.x2, b.x2);
  out.y2 = mix(a.y2, b.y2);
  out.score = std::max(a.score, b.score);
  return out;
}

void growTo(double& lo, double& hi, double minimum) {
  if (hi - lo >= minimum) return;
  const double c = (lo + hi) / 2.0;
  lo = c - minimum / 2.0;
  hi = c + minimum / 2.0;
}

// Position of each frame between two keyframes in [0, 1): by PTS when they increase strictly
// inside the interval, otherwise by decode index.
std::vector<double> fractions(const std::vector<FrameTime>& frames, const FrameTime& from, const FrameTime& to) {
  bool usePts = to.ptsMs > from.ptsMs;
  for (size_t k = 0; usePts && k < frames.size(); ++k) {
    const double pts = frames[k].ptsMs;
    if (pts < from.ptsMs || pts >= to.ptsMs || (k > 0 && pts <= frames[k - 1].ptsMs)) usePts = false;
  }
  std::vector<double> t;
  t.reserve(frames.size());
  for (const FrameTime& f : frames) {
    t.push_back(usePts ? (f.ptsMs - from.ptsMs) / (to.ptsMs - from.ptsMs)
                       : static_cast<double>(f.index - from.index) / static_cast<double>(to.index - from.index));
  }
  return t;
}

}  // namespace

const char* toString(MaskOrigin origin) {
  switch (origin) {
    case MaskOrigin::Detected:
      return "detected";
    case MaskOrigin::Interpolated:
      return "interpolated";
    case MaskOrigin::Extended:
      return "extended";
  }
  return "detected";
}

BoxTimeline::BoxTimeline(const TimelineConfig& config, int frameWidth, int frameHeight)
    : config_(config), width_(frameWidth), height_(frameHeight) {
  if (frameWidth <= 0 || frameHeight <= 0) throw std::invalid_argument("BoxTimeline: frame size must be positive");
  if (!(config.matchMaxDistance > 0) || config.expandFraction < 0 || config.motionExpandFraction < 0 ||
      config.minBoxPx < 0) {
    throw std::invalid_argument("BoxTimeline: invalid config");
  }
}

std::vector<std::pair<size_t, size_t>> BoxTimeline::match(const Keyframe& next, double steps) const {
  struct Candidate {
    double cost;
    size_t i;
    size_t j;
  };
  const Keyframe& prev = *prev_;
  std::vector<Candidate> candidates;
  for (size_t i = 0; i < prev.boxes.size(); ++i) {
    for (size_t j = 0; j < next.boxes.size(); ++j) {
      const Box& a = prev.boxes[i];
      const Box& b = next.boxes[j];
      if (a.cls != b.cls) continue;
      Point predicted = centerOf(a);
      if (velocity_[i]) {
        predicted.x += velocity_[i]->x * steps;
        predicted.y += velocity_[i]->y * steps;
      }
      const Point actual = centerOf(b);
      const double size = std::max(1.0, (a.width() + a.height() + b.width() + b.height()) / 4.0);
      const double cost = std::hypot(actual.x - predicted.x, actual.y - predicted.y) / size;
      if (cost <= config_.matchMaxDistance) candidates.push_back({cost, i, j});
    }
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const Candidate& p, const Candidate& q) { return p.cost < q.cost; });
  std::vector<bool> usedPrev(prev.boxes.size(), false);
  std::vector<bool> usedNext(next.boxes.size(), false);
  std::vector<std::pair<size_t, size_t>> pairs;
  for (const Candidate& c : candidates) {
    if (usedPrev[c.i] || usedNext[c.j]) continue;
    usedPrev[c.i] = true;
    usedNext[c.j] = true;
    pairs.emplace_back(c.i, c.j);
  }
  return pairs;
}

void BoxTimeline::add(std::vector<MaskBox>& out, const Box& box, double motionX, double motionY,
                      MaskOrigin origin) const {
  const double mx = config_.expandFraction * box.width() + config_.motionExpandFraction * motionX;
  const double my = config_.expandFraction * box.height() + config_.motionExpandFraction * motionY;
  double x1 = box.x1 - mx;
  double x2 = box.x2 + mx;
  double y1 = box.y1 - my;
  double y2 = box.y2 + my;
  growTo(x1, x2, config_.minBoxPx);
  growTo(y1, y2, config_.minBoxPx);
  x1 = std::clamp(x1, 0.0, static_cast<double>(width_));
  x2 = std::clamp(x2, 0.0, static_cast<double>(width_));
  y1 = std::clamp(y1, 0.0, static_cast<double>(height_));
  y2 = std::clamp(y2, 0.0, static_cast<double>(height_));
  if (x2 <= x1 || y2 <= y1) return;
  out.push_back(MaskBox{box.cls, static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(x2),
                        static_cast<float>(y2), origin});
}

std::vector<std::vector<MaskBox>> BoxTimeline::advance(const std::vector<FrameTime>& frames,
                                                       const Keyframe& next) {
  if (!prev_) {
    if (!frames.empty()) throw std::invalid_argument("BoxTimeline: frames before the first keyframe");
    prev_ = next;
    velocity_.assign(next.boxes.size(), std::nullopt);
    return {};
  }
  const Keyframe& prev = *prev_;
  if (frames.empty() || frames.front().index != prev.time.index) {
    throw std::invalid_argument("BoxTimeline: frames must start at the previous keyframe");
  }
  if (next.time.index <= frames.back().index) {
    throw std::invalid_argument("BoxTimeline: the next keyframe must follow the frames");
  }
  const double steps = static_cast<double>(next.time.index - prev.time.index);
  const std::vector<std::pair<size_t, size_t>> pairs = match(next, steps);
  std::vector<bool> prevUsed(prev.boxes.size(), false);
  std::vector<bool> nextUsed(next.boxes.size(), false);
  for (const auto& [i, j] : pairs) {
    prevUsed[i] = true;
    nextUsed[j] = true;
  }
  const std::vector<double> t = fractions(frames, prev.time, next.time);
  std::vector<std::vector<MaskBox>> masks(frames.size());
  for (size_t k = 0; k < frames.size(); ++k) {
    const bool atKey = frames[k].index == prev.time.index;
    const double since = static_cast<double>(frames[k].index - prev.time.index);
    for (const auto& [i, j] : pairs) {
      const Point a = centerOf(prev.boxes[i]);
      const Point b = centerOf(next.boxes[j]);
      add(masks[k], lerp(prev.boxes[i], next.boxes[j], t[k]), std::abs(b.x - a.x), std::abs(b.y - a.y),
          atKey ? MaskOrigin::Detected : MaskOrigin::Interpolated);
    }
    for (size_t i = 0; i < prev.boxes.size(); ++i) {
      if (prevUsed[i]) continue;
      const Velocity v = velocity_[i].value_or(Velocity{});
      add(masks[k], shifted(prev.boxes[i], v.x * since, v.y * since), std::abs(v.x) * steps,
          std::abs(v.y) * steps, atKey ? MaskOrigin::Detected : MaskOrigin::Extended);
    }
    for (size_t j = 0; j < next.boxes.size(); ++j) {
      if (!nextUsed[j]) add(masks[k], next.boxes[j], 0.0, 0.0, MaskOrigin::Extended);
    }
  }
  std::vector<std::optional<Velocity>> velocity(next.boxes.size());
  for (const auto& [i, j] : pairs) {
    const Point a = centerOf(prev.boxes[i]);
    const Point b = centerOf(next.boxes[j]);
    velocity[j] = Velocity{(b.x - a.x) / steps, (b.y - a.y) / steps};
  }
  velocity_ = std::move(velocity);
  lastSteps_ = steps;
  prev_ = next;
  return masks;
}

std::vector<std::vector<MaskBox>> BoxTimeline::finish(const std::vector<FrameTime>& frames) {
  if (!prev_) {
    if (!frames.empty()) throw std::invalid_argument("BoxTimeline: frames before the first keyframe");
    return {};
  }
  const Keyframe& prev = *prev_;
  if (!frames.empty() && frames.front().index != prev.time.index) {
    throw std::invalid_argument("BoxTimeline: frames must start at the previous keyframe");
  }
  const double steps = lastSteps_ > 0 ? lastSteps_ : 1.0;
  std::vector<std::vector<MaskBox>> masks(frames.size());
  for (size_t k = 0; k < frames.size(); ++k) {
    const bool atKey = frames[k].index == prev.time.index;
    const double since = static_cast<double>(frames[k].index - prev.time.index);
    for (size_t i = 0; i < prev.boxes.size(); ++i) {
      const Velocity v = velocity_[i].value_or(Velocity{});
      add(masks[k], shifted(prev.boxes[i], v.x * since, v.y * since), std::abs(v.x) * steps,
          std::abs(v.y) * steps, atKey ? MaskOrigin::Detected : MaskOrigin::Extended);
    }
  }
  prev_.reset();
  velocity_.clear();
  lastSteps_ = 0;
  return masks;
}

}  // namespace vc
