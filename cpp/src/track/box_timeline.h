#pragma once

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
// keyframes are interpolated, unmatched ones are extended over one keyframe interval, and every
// mask is expanded. Feed keyframes in order; frames are identified by decode index.
class BoxTimeline {
 public:
  // Throws std::invalid_argument for a non-positive frame size or invalid config.
  BoxTimeline(const TimelineConfig& config, int frameWidth, int frameHeight);

  // `frames`: every frame from the previous keyframe (inclusive) to `next` (exclusive), in order.
  // Returns their masks in the same order. The first call records `next` only: `frames` must be
  // empty and the result is empty. Throws std::invalid_argument when the call order is violated.
  std::vector<std::vector<MaskBox>> advance(const std::vector<FrameTime>& frames, const Keyframe& next);

  // Masks for the frames from the last keyframe (inclusive) to the end, extending every box
  // forward; resets the timeline.
  std::vector<std::vector<MaskBox>> finish(const std::vector<FrameTime>& frames);

 private:
  struct Velocity {
    double x = 0;
    double y = 0;
  };

  std::vector<std::pair<size_t, size_t>> match(const Keyframe& next, double steps) const;
  void add(std::vector<MaskBox>& out, const Box& box, double motionX, double motionY,
           MaskOrigin origin) const;

  TimelineConfig config_;
  int width_;
  int height_;
  std::optional<Keyframe> prev_;
  std::vector<std::optional<Velocity>> velocity_;  // per box of prev_, px per frame index
  double lastSteps_ = 0;                           // frames in the last interval
};

}  // namespace vc
