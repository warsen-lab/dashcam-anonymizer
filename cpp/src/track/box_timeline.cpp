#include "track/box_timeline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "track/box_geometry.h"

namespace vc {
namespace {

// A held box moves along its velocity by at most this fraction of its width (x) and height (y) (spec
// 2026-10-19 §3.3). A ghost lives for exactly one keyframe (the old kMaxMissedKeyframes = 1).
constexpr double kMaxHeldShift = 0.5;

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
  const bool numbers =
      std::isfinite(c.matchMaxDistance) && c.matchMaxDistance > 0 && std::isfinite(c.expandFraction) &&
      c.expandFraction >= 0 && std::isfinite(c.motionExpandFraction) && c.motionExpandFraction >= 0 &&
      std::isfinite(c.minBoxPx) && c.minBoxPx >= 0 && std::isfinite(c.matchMaxDistanceLoose) &&
      c.matchMaxDistanceLoose > 0 && std::isfinite(c.shrinkPerFrame) && c.shrinkPerFrame > 0 &&
      c.shrinkPerFrame <= 1 && std::isfinite(c.maxMaskAreaRatio) && c.maxMaskAreaRatio >= 1;
  // NaN fails both comparisons.
  return numbers && std::all_of(c.confirmScore.begin(), c.confirmScore.end(),
                                [](float s) { return s >= 0.0f && s <= 1.0f; });
}

}  // namespace

const char* toString(MaskOrigin origin) {
  switch (origin) {
    case MaskOrigin::Detected:
      return "detected";
    case MaskOrigin::Interpolated:
      return "interpolated";
    case MaskOrigin::Held:
      return "held";
  }
  return "detected";
}

BoxTimeline::BoxTimeline(const TimelineConfig& config, int frameWidth, int frameHeight)
    : config_(config), width_(frameWidth), height_(frameHeight) {
  if (frameWidth <= 0 || frameHeight <= 0) throw std::invalid_argument("BoxTimeline: frame size must be positive");
  if (!validConfig(config)) throw std::invalid_argument("BoxTimeline: invalid config");
}

std::vector<MatchPair> BoxTimeline::match(const Keyframe& next, double steps) const {
  std::vector<MatchTrack> tracks;
  tracks.reserve(tracks_.size());
  for (const Track& t : tracks_) {
    std::optional<Velocity> velocity;
    if (t.velocity) velocity = Velocity{t.velocity->x, t.velocity->y};
    tracks.push_back(MatchTrack{t.box, t.anchor, velocity});
  }
  return matchBoxes(tracks, next.boxes, steps,
                    MatchParams{config_.matchMaxDistance, config_.matchMaxDistanceLoose, config_.confirmScore});
}

BoxTimeline::Size BoxTimeline::smoothed(Size shown, Size raw, double shrink, bool held) {
  const auto side = [shrink, held](double previous, double wanted) {
    if (wanted >= previous) return wanted;
    return held ? previous : std::max(wanted, previous * (1 - shrink));
  };
  return Size{side(shown.w, raw.w), side(shown.h, raw.h)};
}

BoxTimeline::Size BoxTimeline::cappedArea(Size size, Size raw, double ratio) {
  const double limit = ratio * raw.w * raw.h;
  if (size.w * size.h <= limit) return size;
  // Both sides a fraction f of their way out from raw's, (w + f a) x (h + f b) with a = W - w, b = H - h, is an
  // area that grows with f from w h <= limit (ratio >= 1) to W H > limit. It equals the limit at the positive root
  // of a b f^2 + p f - q = 0, p = a h + b w > 0, q = limit - w h >= 0, written in a form that also holds for a b = 0.
  const double a = size.w - raw.w;
  const double b = size.h - raw.h;
  const double p = a * raw.h + b * raw.w;
  const double q = limit - raw.w * raw.h;
  const double f = std::clamp(2 * q / (p + std::sqrt(p * p + 4 * a * b * q)), 0.0, 1.0);
  return Size{raw.w + f * a, raw.h + f * b};
}

bool BoxTimeline::needsConfirmation(const Box& box) const {
  return ::vc::needsConfirmation(box, config_.confirmScore);
}

BoxTimeline::Offset BoxTimeline::heldShift(const Track& track, double since) const {
  if (!track.velocity) return {};
  const double capX = kMaxHeldShift * widthOf(track.box);
  const double capY = kMaxHeldShift * heightOf(track.box);
  return {std::clamp(track.velocity->x * since, -capX, capX), std::clamp(track.velocity->y * since, -capY, capY)};
}

void BoxTimeline::emit(std::vector<MaskBox>& out, const Track& track, const Box& region, Offset motion,
                       MaskOrigin origin) {
  const double mx = config_.expandFraction * widthOf(region) + config_.motionExpandFraction * motion.x;
  const double my = config_.expandFraction * heightOf(region) + config_.motionExpandFraction * motion.y;
  double x1 = region.x1 - mx;
  double x2 = region.x2 + mx;
  double y1 = region.y1 - my;
  double y2 = region.y2 + my;
  growTo(x1, x2, config_.minBoxPx);
  growTo(y1, y2, config_.minBoxPx);
  const auto frameW = static_cast<double>(width_);
  const auto frameH = static_cast<double>(height_);
  if (!(std::min(x2, frameW) > std::max(x1, 0.0)) || !(std::min(y2, frameH) > std::max(y1, 0.0))) {
    shown_.erase(track.id);  // off the frame: a mask that comes back starts afresh
    return;
  }
  const Size raw{x2 - x1, y2 - y1};
  const auto it = shown_.find(track.id);
  const Size size = it == shown_.end() ? raw
                                       : cappedArea(smoothed(it->second, raw, config_.shrinkPerFrame,
                                                             origin == MaskOrigin::Held),
                                                    raw, config_.maxMaskAreaRatio);
  shown_[track.id] = size;
  const double cx = (x1 + x2) / 2.0;
  const double cy = (y1 + y2) / 2.0;
  out.push_back(MaskBox{region.cls, static_cast<float>(std::clamp(cx - size.w / 2.0, 0.0, frameW)),
                        static_cast<float>(std::clamp(cy - size.h / 2.0, 0.0, frameH)),
                        static_cast<float>(std::clamp(cx + size.w / 2.0, 0.0, frameW)),
                        static_cast<float>(std::clamp(cy + size.h / 2.0, 0.0, frameH)), origin, track.id});
}

void BoxTimeline::hold(std::vector<std::vector<MaskBox>>& masks, const std::vector<FrameTime>& frames,
                       const FrameTime& from, const Track& track, double span) {
  const Offset reach = heldShift(track, span);
  const Offset motion{std::abs(reach.x), std::abs(reach.y)};
  for (size_t k = 0; k < frames.size(); ++k) {
    const Offset s = heldShift(track, static_cast<double>(frames[k].index - from.index));
    emit(masks[k], track, shifted(track.box, s.x, s.y), motion,
         frames[k].index == from.index ? MaskOrigin::Detected : MaskOrigin::Held);
  }
}

void BoxTimeline::holdGhost(std::vector<std::vector<MaskBox>>& masks, const std::vector<FrameTime>& frames,
                            const Track& track) {
  for (size_t k = 0; k < frames.size(); ++k) emit(masks[k], track, track.box, Offset{}, MaskOrigin::Held);
}

TimelineOutput BoxTimeline::advance(const std::vector<FrameTime>& frames, const Keyframe& next) {
  requireFinite(next);
  if (!keyTime_) {
    if (!frames.empty()) throw std::invalid_argument("BoxTimeline: frames before the first keyframe");
    tracks_.clear();
    for (size_t j = 0; j < next.boxes.size(); ++j) {
      const Box& box = next.boxes[j];
      tracks_.push_back(
          Track{nextId_++, box, std::nullopt, false, box, 0.0, static_cast<int>(j), !needsConfirmation(box)});
    }
    keyTime_ = next.time;
    keyBoxes_ = next.boxes.size();
    return {};
  }
  const FrameTime prevTime = *keyTime_;
  if (frames.empty() || frames.front().index != prevTime.index) {
    throw std::invalid_argument("BoxTimeline: frames must start at the previous keyframe");
  }
  requireIncreasing(frames);
  if (next.time.index <= frames.back().index) {
    throw std::invalid_argument("BoxTimeline: the next keyframe must follow the frames");
  }
  const double steps = static_cast<double>(next.time.index - prevTime.index);
  std::vector<std::optional<size_t>> matchedTo(tracks_.size());
  std::vector<std::optional<size_t>> matchedFrom(next.boxes.size());
  for (const MatchPair& m : match(next, steps)) {
    matchedTo[m.track] = m.next;
    matchedFrom[m.next] = m.track;
  }

  TimelineOutput out;
  out.masks.resize(frames.size());
  out.confirmed.assign(keyBoxes_, false);
  const std::vector<double> t = fractions(frames, prevTime, next.time);
  for (size_t i = 0; i < tracks_.size(); ++i) {
    const Track& track = tracks_[i];
    // A match confirms a track: two consecutive keyframes, or a ghost (always confirmed) found again.
    const bool masked = track.confirmed || matchedTo[i].has_value();
    if (track.keyBox >= 0) out.confirmed[static_cast<size_t>(track.keyBox)] = masked;
    if (matchedTo[i]) {
      const Box& to = next.boxes[*matchedTo[i]];
      const Point a = centerOf(track.box);
      const Point b = centerOf(to);
      const Offset motion{std::abs(b.x - a.x), std::abs(b.y - a.y)};
      for (size_t k = 0; k < frames.size(); ++k) {
        const MaskOrigin origin =
            k > 0 ? MaskOrigin::Interpolated : track.ghost ? MaskOrigin::Held : MaskOrigin::Detected;
        emit(out.masks[k], track, lerp(track.box, to, t[k]), motion, origin);
      }
    } else if (!track.ghost && masked) {
      hold(out.masks, frames, prevTime, track, steps);
    } else if (track.ghost && config_.holdLostGhosts) {
      holdGhost(out.masks, frames, track);  // lost for the second keyframe in a row: held, then it ends (spec §3.7)
    }
    // Otherwise an unmatched ghost, or an unconfirmed track seen once, ends here unmasked.
  }

  // The boxes of `next` in order (a matched one continues its track, confirmed), then the ghosts of the
  // confirmed real detections that were missed, at the end of their hold.
  std::vector<Track> tracks;
  tracks.reserve(next.boxes.size() + tracks_.size());
  for (size_t j = 0; j < next.boxes.size(); ++j) {
    const Box& box = next.boxes[j];
    Track fresh{0, box, std::nullopt, false, box, 0.0, static_cast<int>(j), true};
    if (const auto from = matchedFrom[j]) {
      const Track& old = tracks_[*from];
      const Point a = centerOf(old.anchor);
      const Point b = centerOf(box);
      const double span = old.anchorSteps + steps;
      fresh.id = old.id;
      fresh.velocity = Offset{(b.x - a.x) / span, (b.y - a.y) / span};
    } else {
      fresh.id = nextId_++;
      fresh.confirmed = !needsConfirmation(box);
    }
    tracks.push_back(fresh);
  }
  for (size_t i = 0; i < tracks_.size(); ++i) {
    const Track& track = tracks_[i];
    if (matchedTo[i] || track.ghost || !track.confirmed) continue;
    const Offset s = heldShift(track, steps);
    tracks.push_back(Track{track.id, shifted(track.box, s.x, s.y), track.velocity, true, track.anchor,
                           track.anchorSteps + steps, -1, true});
  }
  // Tracks that ended forget their masks.
  std::erase_if(shown_, [&tracks](const auto& entry) {
    return std::none_of(tracks.begin(), tracks.end(), [&entry](const Track& t) { return t.id == entry.first; });
  });
  keyTime_ = next.time;
  tracks_ = std::move(tracks);
  keyBoxes_ = next.boxes.size();
  return out;
}

TimelineOutput BoxTimeline::finish(const std::vector<FrameTime>& frames) {
  if (!keyTime_) {
    if (!frames.empty()) throw std::invalid_argument("BoxTimeline: frames before the first keyframe");
    return {};
  }
  const FrameTime prevTime = *keyTime_;
  if (!frames.empty() && frames.front().index != prevTime.index) {
    throw std::invalid_argument("BoxTimeline: frames must start at the previous keyframe");
  }
  requireIncreasing(frames);
  TimelineOutput out;
  out.masks.resize(frames.size());
  out.confirmed.assign(keyBoxes_, false);
  // The motion margin of the tail: the shift one frame past its end, as for an interval's next keyframe.
  const double span = frames.empty() ? 0.0 : static_cast<double>(frames.back().index - prevTime.index + 1);
  for (const Track& track : tracks_) {
    if (track.keyBox >= 0) out.confirmed[static_cast<size_t>(track.keyBox)] = track.confirmed;
    if (!track.ghost && track.confirmed) {
      hold(out.masks, frames, prevTime, track, span);
    } else if (track.ghost && config_.holdLostGhosts) {
      holdGhost(out.masks, frames, track);
    }
  }
  keyTime_.reset();
  tracks_.clear();
  shown_.clear();
  keyBoxes_ = 0;
  nextId_ = 1;
  return out;
}

}  // namespace vc
