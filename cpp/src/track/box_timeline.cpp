#include "track/box_timeline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vc {
namespace {

// A box may be missed at this many keyframes in a row and still be matched later (it lives on as a
// ghost track); one more miss drops it.
constexpr int kMaxMissedKeyframes = 1;

struct Point {
  double x = 0;
  double y = 0;
};

Point centerOf(const Box& b) { return {(static_cast<double>(b.x1) + b.x2) / 2.0, (static_cast<double>(b.y1) + b.y2) / 2.0}; }

// Box sizes are computed in double: the float difference of two finite coordinates can overflow to
// inf, and inf later turns into a NaN margin (0 * inf) that would drop the mask silently.
double widthOf(const Box& b) { return b.x2 > b.x1 ? static_cast<double>(b.x2) - b.x1 : 0.0; }
double heightOf(const Box& b) { return b.y2 > b.y1 ? static_cast<double>(b.y2) - b.y1 : 0.0; }

// The nearest float, saturating instead of overflowing to inf.
float toFloat(double v) {
  constexpr double kMax = std::numeric_limits<float>::max();
  return static_cast<float>(std::clamp(v, -kMax, kMax));
}

Box shifted(const Box& b, double dx, double dy) {
  Box s = b;
  s.x1 = toFloat(b.x1 + dx);
  s.x2 = toFloat(b.x2 + dx);
  s.y1 = toFloat(b.y1 + dy);
  s.y2 = toFloat(b.y2 + dy);
  return s;
}

Box lerp(const Box& a, const Box& b, double t) {
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

void growTo(double& lo, double& hi, double minimum) {
  if (hi - lo >= minimum) return;
  const double c = (lo + hi) / 2.0;
  lo = c - minimum / 2.0;
  hi = c + minimum / 2.0;
}

// Position of each frame between two keyframes in [0, 1): by PTS when they increase strictly
// inside the interval, otherwise by decode index. Every test is written so that a NaN fails it.
std::vector<double> fractions(const std::vector<FrameTime>& frames, const FrameTime& from, const FrameTime& to) {
  bool usePts = to.ptsMs > from.ptsMs;
  for (size_t k = 0; usePts && k < frames.size(); ++k) {
    const double pts = frames[k].ptsMs;
    const bool inRange = pts >= from.ptsMs && pts < to.ptsMs;
    const bool increasing = k == 0 || pts > frames[k - 1].ptsMs;
    if (!(inRange && increasing)) usePts = false;
  }
  std::vector<double> t;
  t.reserve(frames.size());
  for (const FrameTime& f : frames) {
    t.push_back(usePts ? (f.ptsMs - from.ptsMs) / (to.ptsMs - from.ptsMs)
                       : static_cast<double>(f.index - from.index) / static_cast<double>(to.index - from.index));
  }
  return t;
}

// Frame indices must be strictly increasing.
void requireIncreasing(const std::vector<FrameTime>& frames) {
  for (size_t k = 1; k < frames.size(); ++k) {
    if (frames[k].index <= frames[k - 1].index) {
      throw std::invalid_argument("BoxTimeline: frame indices must be strictly increasing");
    }
  }
}

// A broken box or timestamp must stop the run: comparisons with NaN are false, so it would otherwise
// slip through matching and clamping into a mask that covers nothing.
void requireFinite(const Keyframe& key) {
  if (!std::isfinite(key.time.ptsMs)) throw std::invalid_argument("BoxTimeline: keyframe timestamp is not finite");
  for (const Box& b : key.boxes) {
    if (!std::isfinite(b.x1) || !std::isfinite(b.y1) || !std::isfinite(b.x2) || !std::isfinite(b.y2)) {
      throw std::invalid_argument("BoxTimeline: keyframe box has non-finite coordinates");
    }
  }
}

bool validConfig(const TimelineConfig& c) {
  return std::isfinite(c.matchMaxDistance) && c.matchMaxDistance > 0 && std::isfinite(c.expandFraction) &&
         c.expandFraction >= 0 && std::isfinite(c.motionExpandFraction) && c.motionExpandFraction >= 0 &&
         std::isfinite(c.minBoxPx) && c.minBoxPx >= 0;
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
  if (!validConfig(config)) throw std::invalid_argument("BoxTimeline: invalid config");
}

std::vector<std::pair<size_t, size_t>> BoxTimeline::match(const Keyframe& next, double steps) const {
  struct Candidate {
    double cost;
    size_t i;
    size_t j;
  };
  std::vector<Candidate> candidates;
  for (size_t i = 0; i < tracks_.size(); ++i) {
    const Box& a = tracks_[i].box;
    const Point from = centerOf(a);
    for (size_t j = 0; j < next.boxes.size(); ++j) {
      const Box& b = next.boxes[j];
      if (a.cls != b.cls) continue;
      const Point actual = centerOf(b);
      // A target that stops still matches: take the better of "keeps moving" and "stayed put".
      double distance = std::hypot(actual.x - from.x, actual.y - from.y);
      if (const auto& v = tracks_[i].velocity) {
        distance = std::min(distance, std::hypot(actual.x - (from.x + v->x * steps),
                                                 actual.y - (from.y + v->y * steps)));
      }
      const double size = std::max(1.0, (widthOf(a) + heightOf(a) + widthOf(b) + heightOf(b)) / 4.0);
      const double cost = distance / size;
      if (cost <= config_.matchMaxDistance) candidates.push_back({cost, i, j});
    }
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const Candidate& p, const Candidate& q) { return p.cost < q.cost; });
  std::vector<bool> usedTrack(tracks_.size(), false);
  std::vector<bool> usedNext(next.boxes.size(), false);
  std::vector<std::pair<size_t, size_t>> pairs;
  for (const Candidate& c : candidates) {
    if (usedTrack[c.i] || usedNext[c.j]) continue;
    usedTrack[c.i] = true;
    usedNext[c.j] = true;
    pairs.emplace_back(c.i, c.j);
  }
  return pairs;
}

void BoxTimeline::add(std::vector<MaskBox>& out, const Box& box, double motionX, double motionY,
                      MaskOrigin origin) const {
  const double mx = config_.expandFraction * widthOf(box) + config_.motionExpandFraction * motionX;
  const double my = config_.expandFraction * heightOf(box) + config_.motionExpandFraction * motionY;
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
  if (!(x2 > x1) || !(y2 > y1)) return;
  out.push_back(MaskBox{box.cls, static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(x2),
                        static_cast<float>(y2), origin});
}

std::vector<std::vector<MaskBox>> BoxTimeline::advance(const std::vector<FrameTime>& frames,
                                                       const Keyframe& next) {
  requireFinite(next);
  if (!keyTime_) {
    if (!frames.empty()) throw std::invalid_argument("BoxTimeline: frames before the first keyframe");
    std::vector<Track> tracks;
    for (const Box& box : next.boxes) tracks.push_back({box, std::nullopt, 0, box, 0.0});
    keyTime_ = next.time;
    tracks_ = std::move(tracks);
    return {};
  }
  const FrameTime& prevTime = *keyTime_;
  if (frames.empty() || frames.front().index != prevTime.index) {
    throw std::invalid_argument("BoxTimeline: frames must start at the previous keyframe");
  }
  requireIncreasing(frames);
  if (next.time.index <= frames.back().index) {
    throw std::invalid_argument("BoxTimeline: the next keyframe must follow the frames");
  }
  const double steps = static_cast<double>(next.time.index - prevTime.index);
  const std::vector<std::pair<size_t, size_t>> pairs = match(next, steps);
  // Per track, the box it was matched with (or none), and per next box whether it was claimed.
  std::vector<std::optional<size_t>> matched(tracks_.size());
  std::vector<bool> nextUsed(next.boxes.size(), false);
  for (const auto& [i, j] : pairs) {
    matched[i] = j;
    nextUsed[j] = true;
  }
  const std::vector<double> t = fractions(frames, prevTime, next.time);
  std::vector<std::vector<MaskBox>> masks(frames.size());
  for (size_t k = 0; k < frames.size(); ++k) {
    const bool atKey = frames[k].index == prevTime.index;
    const double since = static_cast<double>(frames[k].index - prevTime.index);
    for (size_t i = 0; i < tracks_.size(); ++i) {
      const Track& track = tracks_[i];
      const MaskOrigin keyOrigin = track.missed == 0 ? MaskOrigin::Detected : MaskOrigin::Extended;
      if (matched[i]) {
        const Box& to = next.boxes[*matched[i]];
        const Point a = centerOf(track.box);
        const Point b = centerOf(to);
        add(masks[k], lerp(track.box, to, t[k]), std::abs(b.x - a.x), std::abs(b.y - a.y),
            atKey ? keyOrigin : MaskOrigin::Interpolated);
        // A ghost's position was a guess: hold the box it was matched with over the whole interval too,
        // so a target that stopped after the missed keyframe is covered where it really is.
        if (track.missed > 0) add(masks[k], to, 0.0, 0.0, MaskOrigin::Extended);
      } else {
        const Velocity v = track.velocity.value_or(Velocity{});
        add(masks[k], shifted(track.box, v.x * since, v.y * since), std::abs(v.x) * steps,
            std::abs(v.y) * steps, atKey ? keyOrigin : MaskOrigin::Extended);
      }
    }
    for (size_t j = 0; j < next.boxes.size(); ++j) {
      if (!nextUsed[j]) add(masks[k], next.boxes[j], 0.0, 0.0, MaskOrigin::Extended);
    }
  }
  // Detections first (matched ones carry their velocity), then the ghosts of tracks that were missed.
  // The velocity runs from the target's last real detection, not from a ghost's guessed position.
  std::vector<Track> tracks;
  tracks.reserve(next.boxes.size());
  for (const Box& box : next.boxes) tracks.push_back({box, std::nullopt, 0, box, 0.0});
  for (const auto& [i, j] : pairs) {
    const Track& track = tracks_[i];
    const Point a = centerOf(track.anchor);
    const Point b = centerOf(next.boxes[j]);
    const double span = track.anchorSteps + steps;
    tracks[j].velocity = Velocity{(b.x - a.x) / span, (b.y - a.y) / span};
  }
  for (size_t i = 0; i < tracks_.size(); ++i) {
    const Track& track = tracks_[i];
    if (matched[i] || track.missed >= kMaxMissedKeyframes) continue;
    const Velocity v = track.velocity.value_or(Velocity{});
    tracks.push_back({shifted(track.box, v.x * steps, v.y * steps), track.velocity, track.missed + 1, track.anchor,
                      track.anchorSteps + steps});
  }
  keyTime_ = next.time;
  tracks_ = std::move(tracks);
  lastSteps_ = steps;
  return masks;
}

std::vector<std::vector<MaskBox>> BoxTimeline::finish(const std::vector<FrameTime>& frames) {
  if (!keyTime_) {
    if (!frames.empty()) throw std::invalid_argument("BoxTimeline: frames before the first keyframe");
    return {};
  }
  const FrameTime& prevTime = *keyTime_;
  if (!frames.empty() && frames.front().index != prevTime.index) {
    throw std::invalid_argument("BoxTimeline: frames must start at the previous keyframe");
  }
  requireIncreasing(frames);
  // Extrapolating further than one interval would only drift away from the target.
  const double cap = lastSteps_ > 0 ? lastSteps_ : 1.0;
  std::vector<std::vector<MaskBox>> masks(frames.size());
  for (size_t k = 0; k < frames.size(); ++k) {
    const bool atKey = frames[k].index == prevTime.index;
    const double since = std::min(static_cast<double>(frames[k].index - prevTime.index), cap);
    for (const Track& track : tracks_) {
      const Velocity v = track.velocity.value_or(Velocity{});
      const MaskOrigin keyOrigin = track.missed == 0 ? MaskOrigin::Detected : MaskOrigin::Extended;
      add(masks[k], shifted(track.box, v.x * since, v.y * since), std::abs(v.x) * cap, std::abs(v.y) * cap,
          atKey ? keyOrigin : MaskOrigin::Extended);
    }
  }
  keyTime_.reset();
  tracks_.clear();
  lastSteps_ = 0;
  return masks;
}

}  // namespace vc
