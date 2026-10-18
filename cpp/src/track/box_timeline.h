#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "detect/box.h"

namespace vc {

enum class MaskOrigin { Detected, Interpolated, Extended };

// "detected" | "interpolated" | "extended" — the values written to masks.jsonl.
const char* toString(MaskOrigin origin);

struct MaskBox {
  int cls = 0;
  float x1 = 0;
  float y1 = 0;
  float x2 = 0;
  float y2 = 0;
  MaskOrigin origin = MaskOrigin::Detected;
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
  double matchMaxDistance = 1.0;      // max |predicted centre - centre| / mean box side
  double expandFraction = 0.15;       // margin per side, fraction of the box width / height
  double motionExpandFraction = 0.5;  // extra margin per side, fraction of the displacement per interval
  double minBoxPx = 16.0;             // masks grow to at least this width and height
};

// Turns detections on keyframes into masks for every frame (main spec §7.2): boxes matched across
// keyframes are interpolated, unmatched ones are extended, and every mask is expanded. A box missed
// at one keyframe is kept for one more interval as a ghost track (moved along its velocity, masked
// as `extended`) and can still be matched at the next keyframe; a second miss drops it. A ghost
// that is matched also gets the new box held in place over the whole interval (`extended`, next to
// the interpolated mask): its position was a guess, and a target that stopped after the missed
// keyframe must stay covered. The velocity of such a match is measured from the target's last real
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
  std::vector<std::vector<MaskBox>> advance(const std::vector<FrameTime>& frames, const Keyframe& next);

  // Masks for the frames from the last keyframe (inclusive) to the end, extending every box
  // (ghosts included) forward along its velocity; the shift stops growing after one keyframe
  // interval. Resets the timeline.
  std::vector<std::vector<MaskBox>> finish(const std::vector<FrameTime>& frames);

 private:
  struct Velocity {
    double x = 0;
    double y = 0;
  };

  // A box on the last keyframe: a real detection (missed == 0) or a ghost (missed keyframes in a row).
  struct Track {
    Box box;
    std::optional<Velocity> velocity;  // px per frame index
    int missed = 0;
    Box anchor;                        // the target's last real detection (the box itself when missed == 0)
    double anchorSteps = 0;            // frames from `anchor` to the last keyframe (0 when missed == 0)
  };

  std::vector<std::pair<size_t, size_t>> match(const Keyframe& next, double steps) const;
  void add(std::vector<MaskBox>& out, const Box& box, double motionX, double motionY,
           MaskOrigin origin) const;

  TimelineConfig config_;
  int width_;
  int height_;
  std::optional<FrameTime> keyTime_;  // the last keyframe
  std::vector<Track> tracks_;         // its boxes, then its ghosts
  double lastSteps_ = 0;              // frames in the last interval
};

}  // namespace vc
