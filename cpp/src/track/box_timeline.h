#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
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
  double matchMaxDistance = 1.0;      // first pass: max |predicted centre - centre| / mean box side
  double expandFraction = 0.15;       // margin per side, fraction of the box width / height
  double motionExpandFraction = 0.5;  // extra margin per side, fraction of the displacement per interval
  double minBoxPx = 16.0;             // masks grow to at least this width and height
  double matchMaxDistanceLoose = 6.0; // second pass: max |dx| and |dy| / the largest side of the two boxes
};

// Turns detections on keyframes into masks for every frame (main spec §7.2): boxes matched across
// keyframes are interpolated, unmatched ones are extended, and every mask is expanded.
//
// Matching runs in two greedy, one-to-one passes over same-class pairs, cheapest first; both try the
// target staying put and keeping its velocity. The first pass takes pairs whose centres are at most
// matchMaxDistance mean box sides apart. The second pass, over what is left, takes pairs of similar
// size (each side within a factor of 3 of the other's) whose centres are at most
// matchMaxDistanceLoose times the largest side of the two boxes apart along each axis: a target that
// moves several box widths between keyframes. A loose match, and a first-pass match of a track that
// has no velocity yet (nothing to check the pairing against), is interpolated AND keeps the masks the
// track and the box would have had unmatched (over the whole interval), so a wrong pairing can add
// masks but never remove one. Any match gives the next interval a velocity.
//
// An unmatched track with a velocity masks the region it may have swept: on each frame the bounding
// box of where the target was last detected, where the velocity takes it, and as far back from the
// detection (it may have turned round), so a target that stopped or turned round stays covered. A box missed at one keyframe is kept for one more interval as a ghost
// track (moved along its velocity, masked as `extended`, swept from its last real detection) and can
// still be matched at the next keyframe; a second miss drops it. A track that is matched loosely
// is kept as a ghost as well, whatever it was matched with: the second pass pairs any similar box
// within reach, so a target missed while a look-alike (a false positive, a newly visible plate)
// turns up nearby must not lose its extrapolation. A ghost that is matched also gets
// the new box held in place over the whole interval (`extended`, next to the interpolated mask) and a
// second interpolation from the target's last real detection: its position was a guess, and a target
// that stopped or turned round after the missed keyframe must stay covered. The velocity of a match is
// measured from the target's last real detection. Feed keyframes in order;
// frames are identified by decode index.
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
  // (ghosts included) forward along its velocity as the region it sweeps; the shift stops growing
  // after one keyframe interval. Resets the timeline.
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

  struct Match {
    size_t track = 0;
    size_t next = 0;
    bool loose = false;  // found by the second pass
  };

  std::vector<Match> match(const Keyframe& next, double steps) const;
  // Masks `region`, grown by the margins of a box of `sized`'s width and height plus the motion margins.
  void add(std::vector<MaskBox>& out, const Box& region, const Box& sized, double motionX, double motionY,
           MaskOrigin origin) const;
  void add(std::vector<MaskBox>& out, const Box& box, double motionX, double motionY, MaskOrigin origin) const {
    add(out, box, box, motionX, motionY, origin);
  }
  // The swept region of an unmatched track `since` frames after the keyframe: its last real
  // detection, the box moved along its velocity, and the detection moved as far the other way;
  // motion margins for `span` frames of velocity.
  void sweep(std::vector<MaskBox>& out, const Track& track, double since, double span, MaskOrigin origin) const;

  TimelineConfig config_;
  int width_;
  int height_;
  std::optional<FrameTime> keyTime_;  // the last keyframe
  std::vector<Track> tracks_;         // its boxes, then its ghosts
  double lastSteps_ = 0;              // frames in the last interval
};

}  // namespace vc
