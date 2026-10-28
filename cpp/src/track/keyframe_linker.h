#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "track/box_matching.h"
#include "track/box_timeline.h"

namespace vc {

struct LinkerConfig {
  MatchParams match{};
  int lookahead = 1;  // W >= 1: keyframes come out W keyframes late; a track is found again across up to W misses
  int backfill = 0;   // B in [0, W - 1]: keyframes filled in before a masked track's first detection
};

// Links the detections on keyframes into tracks, looking W keyframes ahead (spec 2026-10-27 §2), for a BoxTimeline
// with TimelineConfig::linked.
//
// Boxes are matched to the tracks with matchBoxes, each track where it would be on the previous keyframe: its last
// detection moved along its velocity. A track is confirmed (masked) at once when its first box scores at least its
// class's confirm score, otherwise when it is matched on the very next keyframe; an unconfirmed track that misses a
// keyframe ends. A confirmed track lives on unmatched for up to W keyframes: matched again, each keyframe it missed
// gets a box interpolated by frame index between its last detection and the new one (LinkKind::Bridged). When a
// track is confirmed, up to B keyframes before its first detection get its first box (LinkKind::Backfilled, never
// before the first keyframe), moved back along its velocity once it has one (in place until then) by at most half
// its width and height per keyframe interval; when the velocity first becomes known, which may be only after a gap,
// only the backfilled boxes on keyframes not yet returned are moved back along it, those already returned stay
// where they were. Filled-in boxes never set a velocity or confirm anything; a velocity
// runs between a track's last two detections.
//
// Keyframes come out once nothing can change them any more: real boxes first, in the order given, then the
// filled-in ones, with one BoxLink per box (track ids from 1).
class KeyframeLinker {
 public:
  // Throws std::invalid_argument for lookahead < 1 or backfill outside [0, lookahead - 1].
  explicit KeyframeLinker(const LinkerConfig& config);

  // Takes the next keyframe; returns the keyframes now settled, oldest first: none for the first W calls, then
  // one per call. Throws std::invalid_argument when its index does not follow the previous keyframe's, or its
  // timestamp or a box coordinate is not finite (the linker is left untouched).
  std::vector<Keyframe> push(const Keyframe& key);

  // Every keyframe not returned yet, oldest first; resets the linker (ids start from 1 again).
  std::vector<Keyframe> finish();

 private:
  struct Track {
    int64_t id = 0;
    bool confirmed = false;
    Box first{};             // its first detection
    int64_t firstSeq = 0;    // ... on this keyframe (pushes counted from 0)
    size_t firstBox = 0;     // ... at this index of that keyframe's boxes
    int64_t firstIndex = 0;  // ... at this frame index
    Box last{};              // its last detection
    int64_t lastSeq = 0;
    int64_t lastIndex = 0;
    std::optional<Velocity> velocity{};  // px per frame, between its last two detections
    int missed = 0;                      // keyframes since lastSeq
    std::vector<std::pair<int64_t, size_t>> backfills{};  // (keyframe, box index) of its backfilled boxes
  };

  Keyframe& at(int64_t seq);
  // Fills the keyframes `track` missed with boxes on the way from its last detection to `to` at frame `toIndex`.
  void bridge(const Track& track, const Box& to, int64_t toIndex);
  // Marks `track` and its first box masked and backfills before it.
  void confirm(Track& track);
  // Moves the backfilled boxes of `track` still pending to where its velocity puts them.
  void placeBackfills(const Track& track);
  Box backfillBox(const Track& track, int64_t seq);

  LinkerConfig config_;
  std::deque<Keyframe> pending_;  // pushed, not returned yet, oldest first
  int64_t baseSeq_ = 0;           // the push number of pending_.front()
  int64_t nextSeq_ = 0;
  std::optional<FrameTime> prevTime_;  // the keyframe pushed last
  std::vector<Track> tracks_;
  int64_t nextId_ = 1;
};

}  // namespace vc
