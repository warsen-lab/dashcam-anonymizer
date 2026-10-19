#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "detect/box.h"

namespace vc {

enum class MaskOrigin { Detected, Interpolated, Held };

// "detected" | "interpolated" | "held" — the values written to masks.jsonl.
const char* toString(MaskOrigin origin);

struct MaskBox {
  int cls = 0;
  float x1 = 0;
  float y1 = 0;
  float x2 = 0;
  float y2 = 0;
  MaskOrigin origin = MaskOrigin::Detected;
  int64_t track = 0;  // the track the mask is for (ids from 1); a track has at most one mask per frame
};

struct FrameTime {
  int64_t index = 0;
  double ptsMs = 0;
};

struct Keyframe {
  FrameTime time;
  std::vector<Box> boxes;
};

struct TimelineConfig {
  double matchMaxDistance = 1.0;      // first pass: max |predicted centre - centre| / mean box side
  double expandFraction = 0.15;       // margin per side, fraction of the box width / height
  double motionExpandFraction = 0.5;  // extra margin per side, fraction of the displacement per interval
  double minBoxPx = 16.0;             // masks grow to at least this width and height
  double matchMaxDistanceLoose = 6.0; // second pass: max |dx| and |dy| / the largest side of the two boxes
};

// What advance() and finish() return.
struct TimelineOutput {
  // The masks of each frame passed in, in the same order.
  std::vector<std::vector<MaskBox>> masks{};
  // One flag per box of the keyframe those frames start at, in Keyframe::boxes order: whether that box is
  // masked. Empty for the first advance() call.
  std::vector<bool> confirmed{};
};

// Turns detections on keyframes into one mask per target and frame (spec 2026-10-19 §3). Boxes on
// consecutive keyframes are matched into tracks; a track has at most one mask per frame.
//
// Matching runs in two greedy, one-to-one passes over same-class pairs, cheapest first; both try the
// target staying put and keeping its velocity. The first pass takes pairs whose centres are at most
// matchMaxDistance mean box sides apart. The second pass, over what is left, takes pairs of similar size
// (each side within a factor of 3 of the other's) whose centres are at most matchMaxDistanceLoose times
// the largest side of the two boxes apart along each axis.
//
// A matched track is interpolated between its two keyframe boxes (centre and size, linearly in time). A
// track missed at the next keyframe is held at its box, moved along its velocity by at most half the
// box's width along x and half its height along y; it lives on for one keyframe as a ghost at the end of
// that hold. A ghost matched at the following keyframe is interpolated from where it was held to the new
// box; an unmatched ghost ends, and the interval after its last hold is not masked. A box that starts a
// track is masked from its own keyframe on, never before. Every mask is grown by expandFraction of its box
// plus motionExpandFraction of the interval's displacement along each axis (for a hold, of its capped
// shift), to at least minBoxPx, and clipped to the frame. A track's velocity runs from its last real
// detection. Feed keyframes in order; frames are identified by decode index.
class BoxTimeline {
 public:
  // Throws std::invalid_argument for a non-positive frame size or invalid config.
  BoxTimeline(const TimelineConfig& config, int frameWidth, int frameHeight);

  // `frames`: every frame from the previous keyframe (inclusive) to `next` (exclusive), in order.
  // Returns their masks in the same order. The first call records `next` only: `frames` must be
  // empty and the result is empty. Throws std::invalid_argument when the call order is violated,
  // when a box coordinate or the timestamp of `next` is not finite (the timeline is left
  // untouched). A non-finite or out-of-range pts of one of `frames` is not an error: the
  // positions in that interval then come from the decode indices.
  TimelineOutput advance(const std::vector<FrameTime>& frames, const Keyframe& next);

  // Masks for the frames from the last keyframe (inclusive) to the end of the video: every track
  // detected on that keyframe is held there as above; ghosts end. Resets the timeline.
  TimelineOutput finish(const std::vector<FrameTime>& frames);

 private:
  // A velocity (px per frame) or a shift (px).
  struct Offset {
    double x = 0;
    double y = 0;
  };

  // A target followed from keyframe to keyframe.
  struct Track {
    int64_t id = 0;
    Box box{};                         // on the last keyframe: the detection, or where a ghost was held to
    std::optional<Offset> velocity{};  // px per frame index
    bool ghost = false;                // missed on the last keyframe (lives for one keyframe)
    Box anchor{};                      // the target's last real detection (the box itself unless a ghost)
    double anchorSteps = 0;            // frames from `anchor` to the last keyframe (0 unless a ghost)
    int keyBox = -1;                   // index of `box` in the last keyframe's boxes; -1 for a ghost
  };

  struct Match {
    size_t track = 0;
    size_t next = 0;
  };

  std::vector<Match> match(const Keyframe& next, double steps) const;
  // How far a held track has moved `since` frames after its keyframe: along its velocity, at most half
  // its box's width along x and half its height along y.
  Offset heldShift(const Track& track, double since) const;
  // Appends the mask of `track` for `region`, grown by expandFraction of its size plus
  // motionExpandFraction of `motion` per axis, to at least minBoxPx, clipped to the frame; nothing when
  // it lies outside the frame.
  void emit(std::vector<MaskBox>& out, const Track& track, const Box& region, Offset motion,
            MaskOrigin origin) const;
  // The masks of `track` held from its keyframe `from` over `frames` (which start there); the motion
  // margin is the shift reached `span` frames after the keyframe.
  void hold(std::vector<std::vector<MaskBox>>& masks, const std::vector<FrameTime>& frames, const FrameTime& from,
            const Track& track, double span) const;

  TimelineConfig config_;
  int width_;
  int height_;
  std::optional<FrameTime> keyTime_;  // the last keyframe
  std::vector<Track> tracks_;         // its boxes in order, then its ghosts
  size_t keyBoxes_ = 0;               // number of boxes on the last keyframe
  int64_t nextId_ = 1;
};

}  // namespace vc
