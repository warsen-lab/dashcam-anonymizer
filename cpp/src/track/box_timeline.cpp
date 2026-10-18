#include "track/box_timeline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vc {
namespace {

// A box may be missed at this many keyframes in a row and still be matched later (it lives on as a
// ghost track); one more miss drops it.
constexpr int kMaxMissedKeyframes = 1;

// The second matching pass pairs boxes only when each side is within this factor of the other's.
constexpr double kMaxLooseSizeRatio = 3.0;

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

// The bounding box of both boxes; class and score of `a`.
Box unionOf(const Box& a, const Box& b) {
  Box out = a;
  out.x1 = std::min(a.x1, b.x1);
  out.y1 = std::min(a.y1, b.y1);
  out.x2 = std::max(a.x2, b.x2);
  out.y2 = std::max(a.y2, b.y2);
  return out;
}

bool similarSizes(const Box& a, const Box& b) {
  const auto within = [](double p, double q) { return p <= kMaxLooseSizeRatio * q && q <= kMaxLooseSizeRatio * p; };
  return within(widthOf(a), widthOf(b)) && within(heightOf(a), heightOf(b));
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
         std::isfinite(c.minBoxPx) && c.minBoxPx >= 0 && std::isfinite(c.matchMaxDistanceLoose) &&
         c.matchMaxDistanceLoose > 0;
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

std::vector<BoxTimeline::Match> BoxTimeline::match(const Keyframe& next, double steps) const {
  struct Candidate {
    double cost;
    size_t i;
    size_t j;
  };
  // The cheaper of "stayed put" and "kept its velocity" (a target that stops still matches).
  const auto cheapest = [&](const Track& track, const auto& costFrom) {
    const Point from = centerOf(track.box);
    double cost = costFrom(from);
    if (const auto& v = track.velocity) {
      cost = std::min(cost, costFrom(Point{from.x + v->x * steps, from.y + v->y * steps}));
    }
    return cost;
  };
  std::vector<bool> usedTrack(tracks_.size(), false);
  std::vector<bool> usedNext(next.boxes.size(), false);
  std::vector<Match> pairs;
  const auto takeGreedily = [&](std::vector<Candidate> candidates, bool loose) {
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Candidate& p, const Candidate& q) { return p.cost < q.cost; });
    for (const Candidate& c : candidates) {
      if (usedTrack[c.i] || usedNext[c.j]) continue;
      usedTrack[c.i] = true;
      usedNext[c.j] = true;
      pairs.push_back(Match{c.i, c.j, loose});
    }
  };

  // First pass: centre distance in mean box sides.
  std::vector<Candidate> strict;
  for (size_t i = 0; i < tracks_.size(); ++i) {
    const Box& a = tracks_[i].box;
    for (size_t j = 0; j < next.boxes.size(); ++j) {
      const Box& b = next.boxes[j];
      if (a.cls != b.cls) continue;
      const Point actual = centerOf(b);
      const double size = std::max(1.0, (widthOf(a) + heightOf(a) + widthOf(b) + heightOf(b)) / 4.0);
      const double cost =
          cheapest(tracks_[i], [&](Point p) { return std::hypot(actual.x - p.x, actual.y - p.y) / size; });
      if (cost <= config_.matchMaxDistance) strict.push_back({cost, i, j});
    }
  }
  takeGreedily(std::move(strict), false);

  // Second pass, over what is left: both axes in the largest side of the two boxes, similar sizes only.
  std::vector<Candidate> loose;
  for (size_t i = 0; i < tracks_.size(); ++i) {
    if (usedTrack[i]) continue;
    const Box& a = tracks_[i].box;
    for (size_t j = 0; j < next.boxes.size(); ++j) {
      const Box& b = next.boxes[j];
      if (usedNext[j] || a.cls != b.cls || !similarSizes(a, b)) continue;
      const Point actual = centerOf(b);
      const double side = std::max({1.0, widthOf(a), heightOf(a), widthOf(b), heightOf(b)});
      const double cost = cheapest(tracks_[i], [&](Point p) {
        return std::max(std::abs(actual.x - p.x), std::abs(actual.y - p.y)) / side;
      });
      if (cost <= config_.matchMaxDistanceLoose) loose.push_back({cost, i, j});
    }
  }
  takeGreedily(std::move(loose), true);
  return pairs;
}

void BoxTimeline::add(std::vector<MaskBox>& out, const Box& region, const Box& sized, double motionX,
                      double motionY, MaskOrigin origin) const {
  const double mx = config_.expandFraction * widthOf(sized) + config_.motionExpandFraction * motionX;
  const double my = config_.expandFraction * heightOf(sized) + config_.motionExpandFraction * motionY;
  double x1 = region.x1 - mx;
  double x2 = region.x2 + mx;
  double y1 = region.y1 - my;
  double y2 = region.y2 + my;
  growTo(x1, x2, config_.minBoxPx);
  growTo(y1, y2, config_.minBoxPx);
  x1 = std::clamp(x1, 0.0, static_cast<double>(width_));
  x2 = std::clamp(x2, 0.0, static_cast<double>(width_));
  y1 = std::clamp(y1, 0.0, static_cast<double>(height_));
  y2 = std::clamp(y2, 0.0, static_cast<double>(height_));
  if (!(x2 > x1) || !(y2 > y1)) return;
  out.push_back(MaskBox{region.cls, static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(x2),
                        static_cast<float>(y2), origin});
}

void BoxTimeline::sweep(std::vector<MaskBox>& out, const Track& track, double since, double span,
                        MaskOrigin origin) const {
  const Velocity v = track.velocity.value_or(Velocity{});
  const Box ahead = shifted(track.box, v.x * since, v.y * since);
  // A target that turned round right after its last detection: as far back from it as `ahead` is
  // forward (for a ghost, the frames since the detection include the missed interval).
  const double sinceDetection = track.anchorSteps + since;
  const Box behind = shifted(track.anchor, -v.x * sinceDetection, -v.y * sinceDetection);
  add(out, unionOf(unionOf(track.anchor, ahead), behind), track.box, std::abs(v.x) * span, std::abs(v.y) * span,
      origin);
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
  const std::vector<Match> pairs = match(next, steps);
  // Per track, its match (or none), and per next box whether it was claimed.
  std::vector<std::optional<Match>> matched(tracks_.size());
  std::vector<bool> nextUsed(next.boxes.size(), false);
  for (const Match& m : pairs) {
    matched[m.track] = m;
    nextUsed[m.next] = true;
  }
  const std::vector<double> t = fractions(frames, prevTime, next.time);
  std::vector<std::vector<MaskBox>> masks(frames.size());
  for (size_t k = 0; k < frames.size(); ++k) {
    const bool atKey = frames[k].index == prevTime.index;
    const double since = static_cast<double>(frames[k].index - prevTime.index);
    for (size_t i = 0; i < tracks_.size(); ++i) {
      const Track& track = tracks_[i];
      const MaskOrigin keyOrigin = track.missed == 0 ? MaskOrigin::Detected : MaskOrigin::Extended;
      if (!matched[i]) {
        sweep(masks[k], track, since, steps, atKey ? keyOrigin : MaskOrigin::Extended);
        continue;
      }
      const Box& to = next.boxes[matched[i]->next];
      const Point a = centerOf(track.box);
      const Point b = centerOf(to);
      add(masks[k], lerp(track.box, to, t[k]), std::abs(b.x - a.x), std::abs(b.y - a.y),
          atKey ? keyOrigin : MaskOrigin::Interpolated);
      // A loose pairing may be wrong, and so may a first one (no velocity to check it against): keep
      // what the track would have masked unmatched.
      const bool unsure = matched[i]->loose || !track.velocity;
      if (unsure) sweep(masks[k], track, since, steps, MaskOrigin::Extended);
      // ... and, as for a ghost (its position was a guess), hold the box it was matched with over the
      // whole interval, so a target that stopped is covered where it really is.
      if (unsure || track.missed > 0) add(masks[k], to, 0.0, 0.0, MaskOrigin::Extended);
      // A ghost is also interpolated from the target's last real detection: one that turned round
      // after it is not where the ghost guessed.
      if (track.missed > 0) {
        const Point c = centerOf(track.anchor);
        const double span = track.anchorSteps + steps;
        add(masks[k], lerp(track.anchor, to, (track.anchorSteps + since) / span), std::abs(b.x - c.x),
            std::abs(b.y - c.y), MaskOrigin::Interpolated);
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
  for (const Match& m : pairs) {
    const Track& track = tracks_[m.track];
    const Point a = centerOf(track.anchor);
    const Point b = centerOf(next.boxes[m.next]);
    const double span = track.anchorSteps + steps;
    tracks[m.next].velocity = Velocity{(b.x - a.x) / span, (b.y - a.y) / span};
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
      const MaskOrigin keyOrigin = track.missed == 0 ? MaskOrigin::Detected : MaskOrigin::Extended;
      sweep(masks[k], track, since, cap, atKey ? keyOrigin : MaskOrigin::Extended);
    }
  }
  keyTime_.reset();
  tracks_.clear();
  lastSteps_ = 0;
  return masks;
}

}  // namespace vc
