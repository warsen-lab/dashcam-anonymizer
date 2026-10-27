#include "track/track_masker.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace vc {

TimelineConfig TrackMasker::timelineConfig(const TrackMaskerConfig& c) {
  if (c.lookahead < 0 || c.backfill < 0 || c.backfill > std::max(0, c.lookahead - 1)) {
    throw std::invalid_argument("TrackMasker: lookahead must be >= 0 and backfill in [0, max(0, lookahead - 1)]");
  }
  TimelineConfig t = c.timeline;
  t.linked = c.lookahead > 0;
  return t;
}

TrackMasker::TrackMasker(const TrackMaskerConfig& c, int frameWidth, int frameHeight)
    : timeline_(timelineConfig(c), frameWidth, frameHeight) {
  if (c.lookahead > 0) {
    linker_.emplace(LinkerConfig{
        MatchParams{c.timeline.matchMaxDistance, c.timeline.matchMaxDistanceLoose, c.timeline.confirmScore},
        c.lookahead, c.backfill});
  }
}

TrackingOutput TrackMasker::advance(const std::vector<FrameTime>& frames, const Keyframe& next) {
  TrackingOutput out;
  if (!linker_) {
    TimelineOutput step = timeline_.advance(frames, next);
    out.masks = std::move(step.masks);
    if (lastKey_) out.confirmed.push_back(std::move(step.confirmed));
    lastKey_ = next.time.index;
    return out;
  }
  if (!lastKey_ && !frames.empty()) throw std::invalid_argument("TrackMasker: frames before the first keyframe");
  if (lastKey_ && (frames.empty() || frames.front().index != *lastKey_)) {
    throw std::invalid_argument("TrackMasker: frames must start at the previous keyframe");
  }
  std::vector<Keyframe> settled = linker_->push(next);  // throws before anything here changes
  if (lastKey_) intervals_.back() = frames;
  intervals_.emplace_back();
  lastKey_ = next.time.index;
  for (Keyframe& k : settled) feed(std::move(k), out);
  return out;
}

TrackingOutput TrackMasker::finish(const std::vector<FrameTime>& frames) {
  TrackingOutput out;
  if (!linker_) {
    TimelineOutput step = timeline_.finish(frames);
    out.masks = std::move(step.masks);
    if (lastKey_) out.confirmed.push_back(std::move(step.confirmed));
    lastKey_.reset();
    return out;
  }
  if (!lastKey_) {
    if (!frames.empty()) throw std::invalid_argument("TrackMasker: frames before the first keyframe");
    return out;
  }
  if (!frames.empty() && frames.front().index != *lastKey_) {
    throw std::invalid_argument("TrackMasker: frames must start at the previous keyframe");
  }
  intervals_.back() = frames;
  for (Keyframe& k : linker_->finish()) feed(std::move(k), out);
  TimelineOutput step = timeline_.finish(intervals_.front());
  for (auto& m : step.masks) out.masks.push_back(std::move(m));
  intervals_.clear();
  fed_ = false;
  lastKey_.reset();
  return out;
}

void TrackMasker::feed(Keyframe key, TrackingOutput& out) {
  std::vector<bool> flags;
  for (const BoxLink& link : key.links) {
    if (link.kind == LinkKind::Real) flags.push_back(link.masked);  // the real boxes come first, in order
  }
  if (!fed_) {
    timeline_.advance({}, key);
    fed_ = true;
  } else {
    const std::vector<FrameTime> frames = std::move(intervals_.front());
    intervals_.pop_front();
    TimelineOutput step = timeline_.advance(frames, key);
    for (auto& m : step.masks) out.masks.push_back(std::move(m));
  }
  out.confirmed.push_back(std::move(flags));
}

}  // namespace vc
