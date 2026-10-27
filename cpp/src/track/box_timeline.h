#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "detect/box.h"
#include "track/box_matching.h"

namespace vc {

enum class MaskOrigin { Detected, Interpolated, Held, Bridged, Backfilled };

// "detected" | "interpolated" | "held" | "bridged" | "backfilled" — the values written to masks.jsonl.
const char* toString(MaskOrigin origin);

// What a box on a linked keyframe is (KeyframeLinker, spec 2026-10-27 §2): a detection, or filled in where its
// track was missed (bridged) or before it was first detected (backfilled).
enum class LinkKind { Real, Bridged, Backfilled };

// A box's place in KeyframeLinker's tracks.
struct BoxLink {
  int64_t track = 0;  // the linker's track id, from 1
  LinkKind kind = LinkKind::Real;
  bool masked = true;  // the linker's confirmation: false for a low-score box it did not confirm
};

struct MaskBox {
  int cls = 0;
  float x1 = 0;
  float y1 = 0;
  float x2 = 0;
  float y2 = 0;
  MaskOrigin origin = MaskOrigin::Detected;
  int64_t track = 0;  // the track the mask is for (ids from 1); a track has at most one mask per frame
  int block = 0;      // mosaic block size in px; BoxTimeline leaves 0, vc_redact fills it in (spec §4)
  bool snap = true;   // mosaic edges snapped out to the block grid; vc_redact clears it for plates (spec §4)
};

struct FrameTime {
  int64_t index = 0;
  double ptsMs = 0;
};

struct Keyframe {
  FrameTime time;
  std::vector<Box> boxes;
  // With TimelineConfig::linked, one per box (unique track ids); otherwise empty.
  std::vector<BoxLink> links{};
};

struct TimelineConfig {
  double matchMaxDistance = 1.0;      // first pass: max |predicted centre - centre| / mean box side
  double expandFraction = 0.15;       // margin per side, fraction of the box width / height
  double motionExpandFraction = 0.5;  // extra margin per side, fraction of the displacement per interval
  double minBoxPx = 16.0;             // masks grow to at least this width and height
  double matchMaxDistanceLoose = 6.0; // second pass: max |dx| and |dy| / the largest side of the two boxes
  double shrinkPerFrame = 0.02;   // per frame a mask's width and height shrink by at most this fraction, (0, 1]
  double maxMaskAreaRatio = 4.0;  // a mask covers at most this many times the area of its box, >= 1
  // Per class index: a box scoring below its class's entry is masked only once its track is matched at
  // two consecutive keyframes (spec 2026-10-19 §3.4); a class without an entry needs no confirmation.
  std::vector<float> confirmScore{};
  // Whether a ghost that is missed again at the next keyframe is still masked, at the box it was held to, over
  // its whole interval (spec 2026-10-19 §3.7); off: that interval is not masked.
  bool holdLostGhosts = false;
  // Keyframes come from KeyframeLinker (spec 2026-10-27 §3.2): a track follows its BoxLink::track instead of being
  // matched, and is masked as BoxLink::masked says instead of by confirmScore.
  bool linked = false;
};

// What advance() and finish() return.
struct TimelineOutput {
  // The masks of each frame passed in, in the same order.
  std::vector<std::vector<MaskBox>> masks{};
  // One flag per box of the keyframe those frames start at, in Keyframe::boxes order: whether that box is
  // masked (false: a low-score box seen on that keyframe only). Empty for the first advance() call.
  std::vector<bool> confirmed{};
};

// Turns detections on keyframes into one mask per target and frame (spec 2026-10-19 §3). Boxes on
// consecutive keyframes are matched into tracks; a track has at most one mask per frame.
//
// Boxes are matched to the tracks with matchBoxes (box_matching.h); a ghost's anchor is its last real detection.
//
// A matched track is interpolated between its two keyframe boxes (centre and size, linearly in time). A
// track missed at the next keyframe is held at its box, moved along its velocity by at most half the
// box's width along x and half its height along y; it lives on for one keyframe as a ghost at the end of
// that hold. A ghost matched at the following keyframe is interpolated from where it was held to the new
// box; an unmatched ghost ends, and the interval after its last hold is not masked, unless holdLostGhosts
// is set: then that interval (and, for a ghost left after the last keyframe, the tail of the video) is
// masked at the box the hold ended at, which does not move on, with no motion margin and origin held, and
// the track ends after it. A box that starts a track is masked from its own keyframe on, never before. Every
// mask is grown by expandFraction of its box plus motionExpandFraction of the interval's displacement along
// each axis (for a hold, of its capped shift), to at least minBoxPx, and clipped to the frame.
//
// The mask's size is smoothed per track (spec §3.1), around the centre of the box it is for: its width and
// height grow to the box's at once but shrink by at most shrinkPerFrame of themselves per frame, never
// below the box's (not at all while the track is held), so the mask always contains the box and follows a
// moving target without trailing behind it. A mask larger than maxMaskAreaRatio times the box has both
// sides pulled the same fraction of their way towards the box's until it is exactly that large.
//
// A box scoring below confirmScore[cls] is masked only once its track is matched at the next keyframe,
// and then from its own keyframe on; a track that is not confirmed that way ends unmasked and leaves no
// ghost. Higher-scoring boxes are masked at once, and a confirmed track needs no further confirmation.
// TimelineOutput::confirmed says which boxes of the starting keyframe were masked.
//
// A track's velocity runs from its last real detection. Feed keyframes in order; frames are identified by
// decode index.
//
// With config.linked the keyframes come from KeyframeLinker: tracks follow the boxes' link ids instead of matching,
// a box is masked as its link says (no confirmScore), and frames between a bridged or backfilled box and its
// neighbours get that origin (backfilled before bridged before interpolated). Filled-in boxes never set a velocity.
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
  // detected on that keyframe is held there as above; ghosts end (with holdLostGhosts they are held at
  // their box over these frames first). Resets the timeline.
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
    bool confirmed = true;             // masked: a high score, or matched at two consecutive keyframes
    int64_t link = 0;                // linked: KeyframeLinker's track id
    LinkKind kind = LinkKind::Real;  // linked: what `box` is
    bool anchorReal = true;          // `anchor` is a real detection (false while the track has only filled-in boxes)
  };

  // A track's smoothed mask size (spec §3.1).
  struct Size {
    double w = 0;
    double h = 0;
  };

  // The smoothed size after `shown` for a box of size `raw`: each side grows to raw's at once and otherwise
  // shrinks by at most `shrink` of itself, never below raw's; while `held` it does not shrink at all.
  static Size smoothed(Size shown, Size raw, double shrink, bool held);
  // `size` (at least `raw` on each side) with both sides pulled the same fraction of their way towards raw's,
  // just enough to bring its area down to `ratio` times raw's; unchanged when it is no larger than that.
  static Size cappedArea(Size size, Size raw, double ratio);

  std::vector<MatchPair> match(const Keyframe& next, double steps) const;
  bool needsConfirmation(const Box& box) const;  // scores below its class's confirmScore
  // Linked: pairs each track with the box of `next` that has its link id.
  std::vector<MatchPair> matchLinks(const Keyframe& next) const;
  // The track box j of `next` starts (a new id).
  Track startTrack(const Keyframe& next, size_t j);
  // `old` continued by box j of `next`, `steps` frames later: a real box becomes the anchor and gives the velocity
  // from the last real one; a filled-in box changes neither.
  Track continueTrack(const Track& old, const Keyframe& next, size_t j, double steps) const;
  // How far a held track has moved `since` frames after its keyframe: along its velocity, at most half
  // its box's width along x and half its height along y.
  Offset heldShift(const Track& track, double since) const;
  // Appends the mask of `track` for `region`: the region grown by expandFraction of its size plus
  // motionExpandFraction of `motion` per axis and to at least minBoxPx (the raw box), its size smoothed
  // against the track's previous mask (shown_) around the raw box's centre, clipped to the frame; nothing
  // when the raw box lies outside the frame.
  void emit(std::vector<MaskBox>& out, const Track& track, const Box& region, Offset motion, MaskOrigin origin);
  // The masks of `track` held from its keyframe `from` over `frames` (which start there); the motion
  // margin is the shift reached `span` frames after the keyframe.
  void hold(std::vector<std::vector<MaskBox>>& masks, const std::vector<FrameTime>& frames, const FrameTime& from,
            const Track& track, double span);
  // The masks of the ghost `track` over `frames` (which start at its keyframe), for holdLostGhosts: its box, where
  // its hold ended, on every frame and no further shift, motion margin 0, all held (so none shrinks).
  void holdGhost(std::vector<std::vector<MaskBox>>& masks, const std::vector<FrameTime>& frames, const Track& track);

  TimelineConfig config_;
  int width_;
  int height_;
  std::optional<FrameTime> keyTime_;  // the last keyframe
  std::vector<Track> tracks_;         // its boxes in order, then its ghosts
  size_t keyBoxes_ = 0;               // number of boxes on the last keyframe
  int64_t nextId_ = 1;
  std::unordered_map<int64_t, Size> shown_;  // each live track's smoothed mask size
};

}  // namespace vc
