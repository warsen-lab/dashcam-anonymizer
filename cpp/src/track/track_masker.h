#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include "track/box_timeline.h"
#include "track/keyframe_linker.h"

namespace vc {

struct TrackMaskerConfig {
  TimelineConfig timeline{};  // its `linked` is set by TrackMasker
  int lookahead = 0;  // W: 0 = BoxTimeline alone (as before spec 2026-10-27); >= 1 = KeyframeLinker in front of it
  int backfill = 0;   // B in [0, max(0, W - 1)]
};

struct TrackingOutput {
  // The masks of the oldest frames passed in that had none yet, in order: over a whole video, advance() and
  // finish() return one entry per frame, in the order the frames came.
  std::vector<std::vector<MaskBox>> masks{};
  // One entry per keyframe now settled, oldest first: whether each box it came with is masked, in order.
  std::vector<std::vector<bool>> confirmed{};
};

// Keyframes to masks (spec 2026-10-27 §3). With lookahead 0 this is BoxTimeline: the masks of each interval come
// with the next keyframe and its keyframe's flags with the one after. With lookahead W >= 1 the keyframes go through
// KeyframeLinker first and into a linked BoxTimeline W keyframes later, so masks come out W keyframes later too
// (the caller keeps those frames); a keyframe's flags come out when it leaves the linker.
class TrackMasker {
 public:
  // Throws std::invalid_argument for a negative lookahead, a backfill outside [0, max(0, lookahead - 1)], and what
  // BoxTimeline's constructor throws.
  TrackMasker(const TrackMaskerConfig& config, int frameWidth, int frameHeight);

  // As BoxTimeline::advance: `frames` from the previous keyframe (inclusive) to `next` (exclusive), empty on the
  // first call. Invalid arguments throw std::invalid_argument and leave the masker unchanged; any other exception
  // (from inside the timeline) leaves it unusable.
  TrackingOutput advance(const std::vector<FrameTime>& frames, const Keyframe& next);

  // As BoxTimeline::finish: the frames from the last keyframe to the end of the video. Returns every mask and flag
  // still owed and resets. Invalid arguments throw std::invalid_argument and leave the masker unchanged; any other
  // exception (from inside the timeline) leaves it unusable.
  TrackingOutput finish(const std::vector<FrameTime>& frames);

 private:
  static TimelineConfig timelineConfig(const TrackMaskerConfig& config);
  // Linked path: throws std::invalid_argument, touching nothing, unless `frames` start at the previous keyframe
  // (or are empty before the first one, or in finish() with no keyframes), have strictly increasing indices and,
  // when `next` is given, end before it.
  void requireFrames(const std::vector<FrameTime>& frames, const std::optional<FrameTime>& next) const;
  // Puts a keyframe the linker settled into the timeline, adding what comes out to `out`.
  void feed(Keyframe key, TrackingOutput& out);

  BoxTimeline timeline_;
  std::optional<KeyframeLinker> linker_;
  std::optional<int64_t> lastKey_;  // the index of the keyframe passed in last
  bool fed_ = false;                // linked: a keyframe went into the timeline
  // Linked: the frames of each keyframe from the last one fed to the timeline on, oldest first (the newest
  // keyframe's come with the next advance() or finish()).
  std::deque<std::vector<FrameTime>> intervals_;
};

}  // namespace vc
