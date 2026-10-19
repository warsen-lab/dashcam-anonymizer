#include "track/box_timeline.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vc {
namespace {

constexpr int kFace = 0;
constexpr int kPlate = 1;

// Margins and minimum size as given, everything else at its default except the mask smoothing, which is off:
// with shrinkPerFrame 1 a mask may shrink to its box at once, so each mask is its own box.
TimelineConfig config(double expand, double motion, double minBox) {
  TimelineConfig c;
  c.expandFraction = expand;
  c.motionExpandFraction = motion;
  c.minBoxPx = minBox;
  c.shrinkPerFrame = 1.0;
  return c;
}

// No margins and no minimum size: each mask is exactly its (interpolated or held) box.
TimelineConfig exact() { return config(0.0, 0.0, 0.0); }

// exact(), with the mask smoothing and the area cap as given.
TimelineConfig smooth(double shrink, double ratio) {
  TimelineConfig c = exact();
  c.shrinkPerFrame = shrink;
  c.maxMaskAreaRatio = ratio;
  return c;
}

Box face(float x1, float y1, float x2, float y2) { return Box{kFace, x1, y1, x2, y2, 0.9f}; }

// Frames first..last-1 at 60 fps pts, or all at `frozenPts` when it is >= 0.
std::vector<FrameTime> frames(int64_t first, int64_t last, double frozenPts = -1) {
  std::vector<FrameTime> out;
  for (int64_t i = first; i < last; ++i) out.push_back({i, frozenPts >= 0 ? frozenPts : i * 1000.0 / 60.0});
  return out;
}

Keyframe key(int64_t index, std::vector<Box> boxes, double frozenPts = -1) {
  return Keyframe{{index, frozenPts >= 0 ? frozenPts : index * 1000.0 / 60.0}, std::move(boxes)};
}

void expectBox(const MaskBox& m, float x1, float y1, float x2, float y2, MaskOrigin origin) {
  EXPECT_NEAR(m.x1, x1, 1e-3);
  EXPECT_NEAR(m.y1, y1, 1e-3);
  EXPECT_NEAR(m.x2, x2, 1e-3);
  EXPECT_NEAR(m.y2, y2, 1e-3);
  EXPECT_EQ(m.origin, origin);
}

// Fraction of the box [x1, x2) x [y1, y2) inside the union of the masks, sampled on a 0.5 px grid.
double coverage(const std::vector<MaskBox>& masks, double x1, double y1, double x2, double y2) {
  int samples = 0;
  int inside = 0;
  for (double y = y1 + 0.25; y < y2; y += 0.5) {
    for (double x = x1 + 0.25; x < x2; x += 0.5) {
      ++samples;
      if (std::any_of(masks.begin(), masks.end(),
                      [=](const MaskBox& m) { return x >= m.x1 && x < m.x2 && y >= m.y1 && y < m.y2; })) {
        ++inside;
      }
    }
  }
  return samples == 0 ? 1.0 : static_cast<double>(inside) / samples;
}

double coverage(const std::vector<MaskBox>& masks, const Box& b) { return coverage(masks, b.x1, b.y1, b.x2, b.y2); }

size_t countOrigin(const std::vector<MaskBox>& masks, MaskOrigin origin) {
  return static_cast<size_t>(
      std::count_if(masks.begin(), masks.end(), [origin](const MaskBox& m) { return m.origin == origin; }));
}

// No frame carries two masks of one track (spec 2026-10-19 §3).
void expectOneMaskPerTrack(const std::vector<std::vector<MaskBox>>& masks) {
  for (size_t f = 0; f < masks.size(); ++f) {
    std::set<int64_t> seen;
    for (const MaskBox& m : masks[f]) {
      EXPECT_TRUE(seen.insert(m.track).second) << "track " << m.track << " twice on frame " << f;
    }
  }
}

// A target moving uniformly: its box on frame f.
struct Mover {
  int cls = kFace;
  double x0 = 0;
  double y0 = 0;
  double w = 0;
  double h = 0;
  double vx = 0;  // px per frame
  double vy = 0;

  Box at(int64_t f) const {
    const double x = x0 + vx * static_cast<double>(f);
    const double y = y0 + vy * static_cast<double>(f);
    return Box{cls, static_cast<float>(x), static_cast<float>(y), static_cast<float>(x + w), static_cast<float>(y + h),
               0.9f};
  }
};

constexpr int64_t kStep = 12;  // 60 fps video, 5 fps keyframes

// A perfect detector (every mover found on every keyframe) over `intervals` keyframe intervals, then
// finish() for one more interval. Returns the masks of every frame, by frame index.
std::vector<std::vector<MaskBox>> runPerfectDetector(BoxTimeline& timeline, const std::vector<Mover>& movers,
                                                     int64_t intervals) {
  const auto boxesAt = [&](int64_t f) {
    std::vector<Box> boxes;
    for (const Mover& m : movers) boxes.push_back(m.at(f));
    return boxes;
  };
  std::vector<std::vector<MaskBox>> all;
  timeline.advance({}, key(0, boxesAt(0)));
  for (int64_t k = 1; k <= intervals; ++k) {
    auto masks = timeline.advance(frames((k - 1) * kStep, k * kStep), key(k * kStep, boxesAt(k * kStep))).masks;
    for (auto& m : masks) all.push_back(std::move(m));
  }
  auto tail = timeline.finish(frames(intervals * kStep, (intervals + 1) * kStep)).masks;
  for (auto& m : tail) all.push_back(std::move(m));
  return all;
}

void expectFinite(const std::vector<std::vector<MaskBox>>& masks) {
  for (size_t k = 0; k < masks.size(); ++k) {
    for (const MaskBox& m : masks[k]) {
      EXPECT_TRUE(std::isfinite(m.x1) && std::isfinite(m.y1) && std::isfinite(m.x2) && std::isfinite(m.y2))
          << "frame offset " << k;
    }
  }
}

TEST(BoxTimeline, InterpolatesMatchedBoxes) {
  BoxTimeline timeline(exact(), 1000, 1000);
  EXPECT_TRUE(timeline.advance({}, key(0, {face(100, 100, 120, 120)})).masks.empty());
  const TimelineOutput out = timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  EXPECT_EQ(out.confirmed, std::vector<bool>{true});  // the box of keyframe 0 is masked
  ASSERT_EQ(out.masks.size(), 12u);
  for (const auto& m : out.masks) {
    ASSERT_EQ(m.size(), 1u);  // one target, one mask: no end boxes held next to the interpolation
    EXPECT_EQ(m[0].track, 1);
  }
  expectBox(out.masks[0][0], 100, 100, 120, 120, MaskOrigin::Detected);
  expectBox(out.masks[6][0], 106, 100, 126, 120, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, FallsBackToFrameIndexWhenTimestampsStall) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}, 0.0));
  const auto masks = timeline.advance(frames(0, 12, 0.0), key(12, {face(112, 100, 132, 120)}, 0.0)).masks;
  ASSERT_EQ(masks[6].size(), 1u);
  expectBox(masks[6][0], 106, 100, 126, 120, MaskOrigin::Interpolated);
}

// A track missed at the next keyframe is held at its box over the interval (no velocity yet: in place).
TEST(BoxTimeline, HoldsATrackMissedAtTheNextKeyframe) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {})).masks;
  ASSERT_EQ(masks.size(), 12u);
  for (size_t k = 0; k < masks.size(); ++k) {
    ASSERT_EQ(masks[k].size(), 1u);
    expectBox(masks[k][0], 100, 100, 120, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Held);
  }
}

// The frames before a target's first detection are not masked (spec §3.5).
TEST(BoxTimeline, NeverMasksABoxBeforeItsFirstKeyframe) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {}));
  const auto before = timeline.advance(frames(0, 12), key(12, {face(300, 300, 320, 320)})).masks;
  ASSERT_EQ(before.size(), 12u);
  for (const auto& m : before) EXPECT_TRUE(m.empty());
  const auto after = timeline.advance(frames(12, 24), key(24, {face(300, 300, 320, 320)})).masks;
  ASSERT_EQ(after[0].size(), 1u);
  expectBox(after[0][0], 300, 300, 320, 320, MaskOrigin::Detected);
}

TEST(BoxTimeline, NeverMatchesAcrossClasses) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {Box{kPlate, 100, 100, 120, 120, 0.9f}})).masks;
  ASSERT_EQ(masks[6].size(), 1u);  // the face, held; the plate starts at 12
  EXPECT_EQ(masks[6][0].cls, kFace);
  EXPECT_EQ(masks[6][0].origin, MaskOrigin::Held);
  const auto next = timeline.advance(frames(12, 24), key(24, {})).masks;
  ASSERT_EQ(next[0].size(), 1u);  // the plate, held; the face's ghost is not matched at 24 and masks nothing
  EXPECT_EQ(next[0][0].cls, kPlate);
  EXPECT_EQ(next[0][0].track, 2);
}

TEST(BoxTimeline, PredictsWithVelocityToMatchFastMovers) {
  // 18 px per interval, then 22 px: the first pass needs the prediction (22 / 20 > 1).
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(118, 100, 138, 120)}));
  const auto masks = timeline.advance(frames(12, 24), key(24, {face(140, 100, 160, 120)})).masks;
  ASSERT_EQ(masks[6].size(), 1u);  // frame 18
  expectBox(masks[6][0], 129, 100, 149, 120, MaskOrigin::Interpolated);
  EXPECT_EQ(masks[6][0].track, 1);

  // Without a velocity only the second pass pairs them, and the pair is interpolated alone all the same.
  BoxTimeline fresh(exact(), 1000, 1000);
  fresh.advance({}, key(12, {face(118, 100, 138, 120)}));
  const auto loose = fresh.advance(frames(12, 24), key(24, {face(140, 100, 160, 120)})).masks;
  ASSERT_EQ(loose[6].size(), 1u);
  expectBox(loose[6][0], 129, 100, 149, 120, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, ExpandsByFractionAndMotionAndEnforcesTheMinimumSize) {
  BoxTimeline timeline(config(0.15, 0.5, 16.0), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 110)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 110)})).masks;
  // x margin 0.15 * 20 + 0.5 * 12 = 9; y margin 1.5 gives 13 px, grown to 16 around y = 105
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 91, 97, 129, 113, MaskOrigin::Detected);
}

TEST(BoxTimeline, ClipsToTheFrame) {
  BoxTimeline timeline(config(0.0, 0.0, 16.0), 1000, 1000);
  timeline.advance({}, key(0, {face(0, 0, 4, 4)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(0, 0, 4, 4)})).masks;
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 0, 0, 10, 10, MaskOrigin::Detected);
}

// After the last keyframe a track is held, moving along its velocity by at most half its width (spec §3.3).
TEST(BoxTimeline, FinishHoldsAlongTheVelocityUpToHalfABoxSide) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));  // 1 px per frame
  const auto masks = timeline.finish(frames(12, 42)).masks;
  ASSERT_EQ(masks.size(), 30u);
  for (const auto& m : masks) {
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0].track, 1);
  }
  expectBox(masks[0][0], 112, 100, 132, 120, MaskOrigin::Detected);
  expectBox(masks[3][0], 115, 100, 135, 120, MaskOrigin::Held);   // frame 15: 3 px along
  expectBox(masks[10][0], 122, 100, 142, 120, MaskOrigin::Held);  // frame 22: 10 px, half the 20 px width
  expectBox(masks[29][0], 122, 100, 142, 120, MaskOrigin::Held);  // frame 41: still 10 px
}

TEST(BoxTimeline, RejectsCallsOutOfOrder) {
  BoxTimeline timeline(exact(), 1000, 1000);
  EXPECT_THROW(timeline.advance(frames(0, 3), key(3, {})), std::invalid_argument);
  timeline.advance({}, key(0, {}));
  EXPECT_THROW(timeline.advance(frames(1, 6), key(6, {})), std::invalid_argument);
  EXPECT_THROW(timeline.advance(frames(0, 6), key(5, {})), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{0.0, 0.1, 0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_STREQ(toString(MaskOrigin::Interpolated), "interpolated");
  EXPECT_STREQ(toString(MaskOrigin::Held), "held");
  EXPECT_STREQ(toString(MaskOrigin::Detected), "detected");
}

// Missed at 24, found again at 36: the ghost is matched and interpolated from where it was held (spec §3.3).
TEST(BoxTimeline, ReMatchesAGhostFromItsHeldPosition) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));  // 1 px per frame
  const auto gap = timeline.advance(frames(12, 24), key(24, {})).masks;
  ASSERT_EQ(gap[6].size(), 1u);
  expectBox(gap[6][0], 118, 100, 138, 120, MaskOrigin::Held);  // frame 18: 6 px along

  const auto bridged = timeline.advance(frames(24, 36), key(36, {face(136, 100, 156, 120)})).masks;
  for (const auto& m : bridged) {
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0].track, 1);
  }
  // held to 112 + min(12, 10) = 122 at frame 24, then interpolated to 136 at 36
  expectBox(bridged[0][0], 122, 100, 142, 120, MaskOrigin::Held);
  expectBox(bridged[6][0], 129, 100, 149, 120, MaskOrigin::Interpolated);
}

// Missed at 24 and again at 36: the track ended at 24, and frames 24 to 35 are not masked.
TEST(BoxTimeline, AGhostMissedAgainEndsUnmasked) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  const auto held = timeline.advance(frames(12, 24), key(24, {})).masks;
  for (const auto& m : held) EXPECT_EQ(m.size(), 1u);
  const auto ghost = timeline.advance(frames(24, 36), key(36, {})).masks;
  ASSERT_EQ(ghost.size(), 12u);
  for (const auto& m : ghost) EXPECT_TRUE(m.empty());
  const auto after = timeline.advance(frames(36, 48), key(48, {})).masks;
  for (const auto& m : after) EXPECT_TRUE(m.empty());
}

TEST(BoxTimeline, FinishDoesNotMaskAGhost) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  timeline.advance(frames(12, 24), key(24, {}));
  const auto tail = timeline.finish(frames(24, 30)).masks;
  ASSERT_EQ(tail.size(), 6u);
  for (const auto& m : tail) EXPECT_TRUE(m.empty());
}

TEST(BoxTimeline, ExpandsForMotionInAnyDirection) {
  BoxTimeline timeline(config(0.0, 0.5, 0.0), 1000, 1000);
  timeline.advance({}, key(0, {face(112, 112, 132, 132)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(100, 100, 120, 120)})).masks;
  ASSERT_EQ(masks[0].size(), 1u);
  // displacement 12 px in x and in y, moving left / up: 0.5 * 12 = 6 px on every side
  expectBox(masks[0][0], 106, 106, 138, 138, MaskOrigin::Detected);
}

TEST(BoxTimeline, HeldMasksFollowTheVelocityAndGrowByTheCappedShift) {
  BoxTimeline plain(exact(), 1000, 1000);
  plain.advance({}, key(0, {face(100, 100, 120, 120)}));
  plain.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  const auto masks = plain.advance(frames(12, 24), key(24, {})).masks;
  ASSERT_EQ(masks[6].size(), 1u);  // frame 18: 1 px per frame, 6 px along
  expectBox(masks[6][0], 118, 100, 138, 120, MaskOrigin::Held);

  BoxTimeline moving(config(0.0, 0.5, 0.0), 1000, 1000);
  moving.advance({}, key(0, {face(112, 112, 132, 132)}));
  moving.advance(frames(0, 12), key(12, {face(100, 100, 120, 120)}));  // v = (-1, -1)
  const auto gone = moving.advance(frames(12, 24), key(24, {})).masks;
  // The shift over the interval is 12 px, capped at 10 (half the side): 0.5 * 10 = 5 px of margin per side.
  ASSERT_EQ(gone[0].size(), 1u);
  expectBox(gone[0][0], 95, 95, 125, 125, MaskOrigin::Detected);
  ASSERT_EQ(gone[6].size(), 1u);  // frame 18: 6 px up and left
  expectBox(gone[6][0], 89, 89, 119, 119, MaskOrigin::Held);
}

TEST(BoxTimeline, FinishMarginIsTheCappedShift) {
  BoxTimeline timeline(config(0.0, 0.5, 0.0), 1000, 1000);
  timeline.advance({}, key(0, {face(112, 112, 132, 132)}));
  timeline.advance(frames(0, 12), key(12, {face(100, 100, 120, 120)}));  // v = (-1, -1)
  const auto masks = timeline.finish(frames(12, 40)).masks;
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 95, 95, 125, 125, MaskOrigin::Detected);  // 28 tail frames: shift capped at 10, margin 5
  ASSERT_EQ(masks[20].size(), 1u);                                  // frame 32: 20 px along, capped at 10
  expectBox(masks[20][0], 85, 85, 115, 115, MaskOrigin::Held);
}

// A held shift is capped per axis: half the box's width along x, half its height along y. For a 48 x 16 plate
// that is 24 px along x and 8 px along y (not half the width, 24, on both axes, and not half the height on both).
TEST(BoxTimeline, AHeldShiftIsCappedByHalfTheBoxSideAlongEachAxis) {
  // Vertically at 1 px a frame: held for 12 frames, the shift reaches the cap of 8 at frame 20 and stays there.
  BoxTimeline vertical(exact(), 1000, 1000);
  vertical.advance({}, key(0, {Box{kPlate, 500, 200, 548, 216, 0.9f}}));
  vertical.advance(frames(0, 12), key(12, {Box{kPlate, 500, 212, 548, 228, 0.9f}}));
  const auto up = vertical.advance(frames(12, 24), key(24, {})).masks;
  ASSERT_EQ(up.size(), 12u);
  ASSERT_EQ(up[4].size(), 1u);
  expectBox(up[4][0], 500, 216, 548, 232, MaskOrigin::Held);  // frame 16: 4 px along, below the cap
  for (const size_t k : {8u, 10u, 11u}) {
    SCOPED_TRACE("frame " + std::to_string(12 + k));
    ASSERT_EQ(up[k].size(), 1u);
    expectBox(up[k][0], 500, 220, 548, 236, MaskOrigin::Held);  // 212 + 8
  }

  // Horizontally at 3 px a frame: the shift reaches the cap of 24 at frame 20 and stays there.
  BoxTimeline horizontal(exact(), 1000, 1000);
  horizontal.advance({}, key(0, {Box{kPlate, 500, 200, 548, 216, 0.9f}}));
  horizontal.advance(frames(0, 12), key(12, {Box{kPlate, 536, 200, 584, 216, 0.9f}}));
  const auto along = horizontal.advance(frames(12, 24), key(24, {})).masks;
  ASSERT_EQ(along.size(), 12u);
  ASSERT_EQ(along[4].size(), 1u);
  expectBox(along[4][0], 548, 200, 596, 216, MaskOrigin::Held);  // frame 16: 12 px along, below the cap
  for (const size_t k : {8u, 10u, 11u}) {
    SCOPED_TRACE("frame " + std::to_string(12 + k));
    ASSERT_EQ(along[k].size(), 1u);
    expectBox(along[k][0], 560, 200, 608, 216, MaskOrigin::Held);  // 536 + 24
  }
}

TEST(BoxTimeline, MatchesGreedilyByCost) {
  // Both previous faces are candidates for the next one (costs 1.25 and 0.25, limit 2.0): the cheaper pair
  // wins and the other face is held.
  TimelineConfig c = exact();
  c.matchMaxDistance = 2.0;
  BoxTimeline timeline(c, 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120), face(130, 100, 150, 120)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(125, 100, 145, 120)})).masks;
  ASSERT_EQ(masks[6].size(), 2u);
  expectBox(masks[6][0], 100, 100, 120, 120, MaskOrigin::Held);
  EXPECT_EQ(masks[6][0].track, 1);
  expectBox(masks[6][1], 127.5f, 100, 147.5f, 120, MaskOrigin::Interpolated);
  EXPECT_EQ(masks[6][1].track, 2);
}

TEST(BoxTimeline, InterpolatesByPtsForUnevenTimestamps) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, Keyframe{{0, 0.0}, {face(100, 100, 130, 130)}});
  const std::vector<FrameTime> uneven = {{0, 0.0}, {1, 10.0}, {2, 20.0}, {3, 200.0}, {4, 210.0}};
  const auto masks = timeline.advance(uneven, Keyframe{{5, 270.0}, {face(127, 100, 157, 130)}}).masks;
  ASSERT_EQ(masks.size(), 5u);
  ASSERT_EQ(masks[3].size(), 1u);  // pts 200 of 270: 100 + 27 * 200 / 270
  expectBox(masks[3][0], 120, 100, 150, 130, MaskOrigin::Interpolated);

  // A repeated pts is not usable: positions fall back to the frame index.
  BoxTimeline stalled(exact(), 1000, 1000);
  stalled.advance({}, Keyframe{{0, 0.0}, {face(100, 100, 130, 130)}});
  const std::vector<FrameTime> repeated = {{0, 0.0}, {1, 10.0}, {2, 10.0}, {3, 30.0}};
  const auto byIndex = stalled.advance(repeated, Keyframe{{4, 100.0}, {face(124, 100, 154, 130)}}).masks;
  ASSERT_EQ(byIndex.size(), 4u);
  ASSERT_EQ(byIndex[1].size(), 1u);
  expectBox(byIndex[1][0], 106, 100, 136, 130, MaskOrigin::Interpolated);
  ASSERT_EQ(byIndex[2].size(), 1u);
  expectBox(byIndex[2][0], 112, 100, 142, 130, MaskOrigin::Interpolated);
  ASSERT_EQ(byIndex[3].size(), 1u);
  expectBox(byIndex[3][0], 118, 100, 148, 130, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, ClipsAtTheRightAndBottomOfANonSquareFrame) {
  BoxTimeline timeline(config(0.0, 0.0, 16.0), 640, 360);
  timeline.advance({}, key(0, {face(630, 350, 640, 360)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(630, 350, 640, 360)})).masks;
  // 10 px grown to 16 around the centre (635, 355) is 627..643 x 347..363, clipped to 640 x 360
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 627, 347, 640, 360, MaskOrigin::Detected);
  ASSERT_EQ(masks[3].size(), 1u);
  expectBox(masks[3][0], 627, 347, 640, 360, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, MatchesATargetThatStops) {
  // The face speeds up to 3 px per frame, then stops: the velocity predicts a centre 36 px away
  // (cost 1.8), but the box is exactly where it was (zero-velocity cost 0).
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(118, 100, 138, 120)}));
  timeline.advance(frames(12, 24), key(24, {face(154, 100, 174, 120)}));
  const auto masks = timeline.advance(frames(24, 36), key(36, {face(154, 100, 174, 120)})).masks;
  ASSERT_EQ(masks[6].size(), 1u);  // frame 30
  expectBox(masks[6][0], 154, 100, 174, 120, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, RejectsNonIncreasingFrames) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  std::vector<FrameTime> repeated = frames(0, 6);
  repeated[3] = repeated[2];
  EXPECT_THROW(timeline.advance(repeated, key(6, {})), std::invalid_argument);
  std::vector<FrameTime> swapped = frames(0, 6);
  std::swap(swapped[3], swapped[4]);
  EXPECT_THROW(timeline.advance(swapped, key(6, {})), std::invalid_argument);
  EXPECT_THROW(timeline.finish(repeated), std::invalid_argument);
  EXPECT_THROW(timeline.finish(swapped), std::invalid_argument);
  // The rejected calls left the timeline untouched.
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(100, 100, 120, 120)})).masks;
  EXPECT_EQ(masks.size(), 6u);
}

TEST(BoxTimeline, RejectsNonFiniteConfig) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double bad : {nan, inf, -inf}) {
    EXPECT_THROW(BoxTimeline(TimelineConfig{bad, 0.1, 0.1, 1}, 10, 10), std::invalid_argument);
    EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, bad, 0.1, 1}, 10, 10), std::invalid_argument);
    EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, bad, 1}, 10, 10), std::invalid_argument);
    EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, 0.1, bad}, 10, 10), std::invalid_argument);
    EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, 0.1, 1, bad}, 10, 10), std::invalid_argument);
  }
  EXPECT_THROW(BoxTimeline(TimelineConfig{-1.0, 0.1, 0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, -0.1, 0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, -0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, 0.1, -1}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, 0.1, 1, 0.0}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, 0.1, 1, -1.0}, 10, 10), std::invalid_argument);
}

// Each target keeps its track id from keyframe to keyframe whatever the order of the boxes; a new target gets
// the next id. Masks come in the order of the tracks on the keyframe they start at.
TEST(BoxTimeline, TracksKeepTheirIdsAndNewTargetsGetNewOnes) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120), face(500, 100, 520, 120)}));
  const auto first =
      timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120), face(500, 112, 520, 132)})).masks;
  ASSERT_EQ(first[6].size(), 2u);
  EXPECT_EQ(first[6][0].track, 1);
  EXPECT_EQ(first[6][1].track, 2);
  const auto second = timeline
                          .advance(frames(12, 24), key(24, {face(500, 124, 520, 144), face(800, 800, 820, 820),
                                                            face(124, 100, 144, 120)}))
                          .masks;
  ASSERT_EQ(second[6].size(), 2u);  // the face at 800 starts at 24
  EXPECT_EQ(second[6][0].track, 1);
  EXPECT_EQ(second[6][1].track, 2);
  const auto third = timeline.advance(frames(24, 36), key(36, {})).masks;
  ASSERT_EQ(third[0].size(), 3u);  // all three held, in the order of keyframe 24's boxes
  EXPECT_EQ(third[0][0].track, 2);
  EXPECT_EQ(third[0][1].track, 3);
  EXPECT_EQ(third[0][2].track, 1);
}

// `confirmed` has one flag per box of the keyframe the returned frames START at (the previous one), not of
// `next`; the first advance() has no frames and no flags. finish() resets the timeline: the next video starts
// at track 1 again.
TEST(BoxTimeline, ConfirmedFollowsThePreviousKeyframeAndFinishRestartsTheTrackIds) {
  BoxTimeline timeline(exact(), 1000, 1000);
  const TimelineOutput first = timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  EXPECT_TRUE(first.masks.empty());
  EXPECT_TRUE(first.confirmed.empty());

  // Keyframe 0 had one box, keyframe 12 has two.
  const TimelineOutput second =
      timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120), face(500, 500, 520, 520)}));
  EXPECT_EQ(second.confirmed, std::vector<bool>({true}));
  for (const auto& m : second.masks) {
    ASSERT_EQ(m.size(), 1u);  // the second box starts at keyframe 12
    EXPECT_EQ(m[0].track, 1);
  }

  // Keyframe 12 had two boxes, keyframe 24 has one.
  const TimelineOutput third = timeline.advance(frames(12, 24), key(24, {face(124, 100, 144, 120)}));
  EXPECT_EQ(third.confirmed, std::vector<bool>({true, true}));
  ASSERT_EQ(third.masks[6].size(), 2u);
  EXPECT_EQ(third.masks[6][0].track, 1);
  EXPECT_EQ(third.masks[6][1].track, 2);

  // Keyframe 24 had one box (the second target is a ghost there, unmasked).
  const TimelineOutput tail = timeline.finish(frames(24, 30));
  EXPECT_EQ(tail.confirmed, std::vector<bool>({true}));
  for (const auto& m : tail.masks) {
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0].track, 1);
  }

  // The timeline is fresh again: the first target of the next video is track 1, not 3.
  timeline.advance({}, key(0, {face(300, 300, 320, 320)}));
  const TimelineOutput next = timeline.advance(frames(0, 6), key(6, {face(300, 300, 320, 320)}));
  EXPECT_EQ(next.confirmed, std::vector<bool>({true}));
  for (const auto& m : next.masks) {
    ASSERT_EQ(m.size(), 1u);
    EXPECT_EQ(m[0].track, 1);
  }
}

// Detected at 12 (centre 122), missed at 24, found at 36 (centre (150, 118)): the velocity runs from the last
// real detection, (28, 8) / 24 frames, not from the ghost's held position.
TEST(BoxTimeline, AReMatchedGhostTakesItsVelocityFromItsLastDetection) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  timeline.advance(frames(12, 24), key(24, {}));
  timeline.advance(frames(24, 36), key(36, {face(140, 108, 160, 128)}));
  const auto masks = timeline.finish(frames(36, 42)).masks;
  ASSERT_EQ(masks[3].size(), 1u);  // frame 39: (7/6, 1/3) px per frame for 3 frames
  expectBox(masks[3][0], 143.5f, 109, 163.5f, 129, MaskOrigin::Held);
}

TEST(BoxTimeline, AGhostWithoutAVelocityIsHeldInPlaceAndReMatched) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto held = timeline.advance(frames(0, 12), key(12, {})).masks;  // no velocity yet: held where it was
  for (size_t k = 0; k < held.size(); ++k) {
    ASSERT_EQ(held[k].size(), 1u);
    expectBox(held[k][0], 100, 100, 120, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Held);
  }
  const auto bridged = timeline.advance(frames(12, 24), key(24, {face(118, 100, 138, 120)})).masks;
  ASSERT_EQ(bridged[0].size(), 1u);
  expectBox(bridged[0][0], 100, 100, 120, 120, MaskOrigin::Held);
  ASSERT_EQ(bridged[6].size(), 1u);
  expectBox(bridged[6][0], 109, 100, 129, 120, MaskOrigin::Interpolated);
  // From the detection at 0 to the one at 24: 18 px in 24 frames (from the ghost it would be 18 in 12).
  const auto tail = timeline.finish(frames(24, 31)).masks;
  ASSERT_EQ(tail.size(), 7u);
  expectBox(tail[0][0], 118, 100, 138, 120, MaskOrigin::Detected);
  expectBox(tail[6][0], 122.5f, 100, 142.5f, 120, MaskOrigin::Held);
}

TEST(BoxTimeline, RejectsNonFiniteKeyframeBoxes) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (const float bad : {nan, inf, -inf}) {
    const std::vector<Box> broken = {face(bad, 100, 120, 120), face(100, bad, 120, 120), face(100, 100, bad, 120),
                                     face(100, 100, 120, bad)};
    for (const Box& box : broken) {
      BoxTimeline first(exact(), 1000, 1000);  // as the very first keyframe
      EXPECT_THROW(first.advance({}, key(0, {box})), std::invalid_argument);
      // The rejected call left the timeline untouched.
      EXPECT_TRUE(first.advance({}, key(0, {face(100, 100, 120, 120)})).masks.empty());

      BoxTimeline later(exact(), 1000, 1000);  // and as a later one
      later.advance({}, key(0, {face(100, 100, 120, 120)}));
      EXPECT_THROW(later.advance(frames(0, 12), key(12, {face(100, 100, 120, 120), box})), std::invalid_argument);
      const auto masks = later.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)})).masks;
      ASSERT_EQ(masks.size(), 12u);
      ASSERT_EQ(masks[6].size(), 1u);
      expectBox(masks[6][0], 106, 100, 126, 120, MaskOrigin::Interpolated);
    }
  }
}

TEST(BoxTimeline, RejectsNonFiniteKeyframeTimestamps) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double bad : {nan, inf, -inf}) {
    BoxTimeline first(exact(), 1000, 1000);
    EXPECT_THROW(first.advance({}, Keyframe{{0, bad}, {face(100, 100, 120, 120)}}), std::invalid_argument);
    BoxTimeline later(exact(), 1000, 1000);
    later.advance({}, key(0, {face(100, 100, 120, 120)}));
    EXPECT_THROW(later.advance(frames(0, 12), Keyframe{{12, bad}, {face(112, 100, 132, 120)}}),
                 std::invalid_argument);
    EXPECT_EQ(later.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)})).masks.size(), 12u);
  }
}

TEST(BoxTimeline, FallsBackToTheIndexForAFrameWithoutAUsableTimestamp) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, Keyframe{{0, 0.0}, {face(100, 100, 130, 130)}});
  const std::vector<FrameTime> broken = {{0, 0.0}, {1, 10.0}, {2, 20.0}, {3, nan}, {4, 210.0}};
  const auto masks = timeline.advance(broken, Keyframe{{5, 270.0}, {face(127, 100, 157, 130)}}).masks;
  ASSERT_EQ(masks.size(), 5u);
  expectFinite(masks);
  // By index (pts alone would put frame 1 at 101.0): 100 + 27 * k / 5.
  ASSERT_EQ(masks[1].size(), 1u);
  expectBox(masks[1][0], 105.4f, 100, 135.4f, 130, MaskOrigin::Interpolated);
  ASSERT_EQ(masks[3].size(), 1u);
  expectBox(masks[3][0], 116.2f, 100, 146.2f, 130, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, FallsBackToTheIndexWhenAFrameLiesOutsideTheKeyframeTimestamps) {
  // Keyframes at index 0 (pts 100) and 5 (pts 270); x1 100 -> 127.
  const auto run = [](const std::vector<double>& pts) {
    BoxTimeline timeline(exact(), 1000, 1000);
    timeline.advance({}, Keyframe{{0, 100.0}, {face(100, 100, 130, 130)}});
    std::vector<FrameTime> list;
    for (size_t i = 0; i < pts.size(); ++i) list.push_back({static_cast<int64_t>(i), pts[i]});
    return timeline.advance(list, Keyframe{{5, 270.0}, {face(127, 100, 157, 130)}}).masks;
  };
  // All inside: by pts, frame 1 sits at (110 - 100) / 170 of the interval.
  auto masks = run({100, 110, 120, 130, 140});
  ASSERT_EQ(masks[1].size(), 1u);
  expectBox(masks[1][0], 100.0f + 27.0f * 10.0f / 170.0f, 100, 130.0f + 27.0f * 10.0f / 170.0f, 130,
            MaskOrigin::Interpolated);
  // A frame before the previous keyframe's pts, at the next keyframe's pts, or after it: by index
  // (100 + 27 * 1 / 5 = 105.4 for frame 1).
  for (const std::vector<double>& pts : {std::vector<double>{90, 110, 120, 130, 140},
                                         std::vector<double>{100, 110, 120, 130, 270},
                                         std::vector<double>{100, 110, 120, 130, 300}}) {
    masks = run(pts);
    ASSERT_EQ(masks.size(), 5u);
    ASSERT_EQ(masks[1].size(), 1u);
    expectBox(masks[1][0], 105.4f, 100, 135.4f, 130, MaskOrigin::Interpolated);
  }
}

TEST(BoxTimeline, NeverEmitsAMaskWithNonFiniteCoordinates) {
  // Finite coordinates whose width overflows a float: with no expansion fraction the margin would be
  // 0 * inf = NaN if computed in float. The target must stay masked, so every frame carries a finite mask.
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(-3e38f, 100, 3e38f, 120)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(-3e38f, 100, 3e38f, 120)})).masks;
  ASSERT_EQ(masks.size(), 6u);
  expectFinite(masks);
  for (size_t k = 0; k < masks.size(); ++k) {
    ASSERT_EQ(masks[k].size(), 1u) << "frame offset " << k;
    expectBox(masks[k][0], 0, 100, 1000, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Interpolated);
  }
  const auto tail = timeline.finish(frames(6, 12)).masks;
  ASSERT_EQ(tail.size(), 6u);
  expectFinite(tail);
  for (size_t k = 0; k < tail.size(); ++k) {
    ASSERT_EQ(tail[k].size(), 1u) << "frame offset " << k;
    expectBox(tail[k][0], 0, 100, 1000, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Held);
  }
}

// Interpolating between two float-extreme boxes must not overflow either: (q - p) alone is 5.9e38.
TEST(BoxTimeline, InterpolatesBoxesAtTheFloatLimitsWithoutOverflowing) {
  // The centre moves 5.9e38 (118 box sizes), so the boxes only match with a huge matching distance.
  BoxTimeline timeline(TimelineConfig{1000.0, 0.0, 0.0, 0.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(-3e38f, 100, -2.9e38f, 120)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(2.9e38f, 100, 3e38f, 120)})).masks;
  ASSERT_EQ(masks.size(), 6u);
  expectFinite(masks);
  // Halfway the box spans [-5e36, 5e36], which covers the whole frame width.
  ASSERT_EQ(masks[3].size(), 1u);
  expectBox(masks[3][0], 0, 100, 1000, 120, MaskOrigin::Interpolated);
}

// Plates and faces moving faster than one mean box side per keyframe interval (the first pass cannot match
// them) are matched by the second pass and covered on every frame up to the last keyframe, interpolated from
// the first interval on. After the last keyframe a track is only held half a box side further (spec §3.3), so
// the tail is not checked.
TEST(BoxTimeline, CoversFastMoversOnEveryFrame) {
  struct Shape {
    const char* name;
    int cls;
    double w;
    double h;
  };
  for (const Shape& shape : {Shape{"plate 48x16", kPlate, 48, 16}, Shape{"plate 24x8", kPlate, 24, 8},
                             Shape{"face 20x24", kFace, 20, 24}}) {
    for (const double widths : {1.6, 2.0, 3.0, 5.0}) {
      SCOPED_TRACE(std::string(shape.name) + ", " + std::to_string(widths) + " widths per interval");
      const Mover mover{shape.cls, 200, 500, shape.w, shape.h, widths * shape.w / kStep, 0};
      BoxTimeline timeline(TimelineConfig{}, 4096, 2160);
      const auto masks = runPerfectDetector(timeline, {mover}, 8);
      ASSERT_EQ(masks.size(), static_cast<size_t>(9 * kStep));
      expectOneMaskPerTrack(masks);
      int uncovered = 0;
      for (int64_t f = 0; f <= 8 * kStep; ++f) {
        if (coverage(masks[static_cast<size_t>(f)], mover.at(f)) < 0.8) ++uncovered;
      }
      EXPECT_EQ(uncovered, 0);
      for (int64_t f = 1; f < 8 * kStep; ++f) {
        if (f % kStep == 0) continue;  // keyframes carry the detection itself
        EXPECT_EQ(countOrigin(masks[static_cast<size_t>(f)], MaskOrigin::Interpolated), 1u) << "frame " << f;
      }
    }
  }
}

// Two plates on neighbouring rows driving towards each other at two box widths per interval: they pass each
// other between keyframes 4 and 5, and both stay covered up to the last keyframe, one mask each.
TEST(BoxTimeline, CoversTwoTargetsThatCrossEachOther) {
  const Mover right{kPlate, 100, 500, 48, 16, 8, 0};
  const Mover left{kPlate, 900, 508, 48, 16, -8, 0};
  BoxTimeline timeline(TimelineConfig{}, 1920, 1080);
  const auto masks = runPerfectDetector(timeline, {right, left}, 8);
  expectOneMaskPerTrack(masks);
  for (int64_t f = 0; f <= 8 * kStep; ++f) {
    const auto& m = masks[static_cast<size_t>(f)];
    EXPECT_GE(coverage(m, right.at(f)), 0.8) << "right-moving plate, frame " << f;
    EXPECT_GE(coverage(m, left.at(f)), 0.8) << "left-moving plate, frame " << f;
  }
}

// A plate moving vertically several of its (small) heights per interval: the second pass measures both axes in
// the larger side of the boxes, so 9 heights of a 24 x 8 plate are 3 sides.
TEST(BoxTimeline, CoversPlatesMovingVertically) {
  for (const double heights : {3.0, 6.0, 9.0}) {
    SCOPED_TRACE(std::to_string(heights) + " heights per interval");
    const Mover mover{kPlate, 500, 200, 24, 8, 0, heights * 8 / kStep};
    BoxTimeline timeline(TimelineConfig{}, 4096, 2160);
    const auto masks = runPerfectDetector(timeline, {mover}, 8);
    expectOneMaskPerTrack(masks);
    int uncovered = 0;
    for (int64_t f = 0; f <= 8 * kStep; ++f) {
      if (coverage(masks[static_cast<size_t>(f)], mover.at(f)) < 0.8) ++uncovered;
    }
    EXPECT_EQ(uncovered, 0);
    for (int64_t f = 1; f < 8 * kStep; ++f) {
      if (f % kStep == 0) continue;
      EXPECT_EQ(countOrigin(masks[static_cast<size_t>(f)], MaskOrigin::Interpolated), 1u) << "frame " << f;
    }
  }
}

// A box three widths away (beyond the first pass) is matched by the second and interpolated; nothing else is
// masked (spec §3.2: no end boxes held).
TEST(BoxTimeline, ALooseMatchInterpolatesAlone) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(160, 100, 180, 120)})).masks;
  ASSERT_EQ(masks.size(), 12u);
  for (size_t k = 0; k < masks.size(); ++k) {
    SCOPED_TRACE("frame " + std::to_string(k));
    ASSERT_EQ(masks[k].size(), 1u);
    const float x = 100 + 5.0f * static_cast<float>(k);  // 60 px over 12 frames
    expectBox(masks[k][0], x, 100, x + 20, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Interpolated);
  }
}

// The loose match seeds the velocity like a normal one: the next interval is matched by the first pass along the
// prediction, and the old track leaves no ghost behind.
TEST(BoxTimeline, ALooseMatchSeedsTheVelocity) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(160, 100, 180, 120)}));
  const auto masks = timeline.advance(frames(12, 24), key(24, {face(220, 100, 240, 120)})).masks;
  ASSERT_EQ(masks[6].size(), 1u);  // frame 18
  expectBox(masks[6][0], 190, 100, 210, 120, MaskOrigin::Interpolated);
  const auto tail = timeline.finish(frames(24, 30)).masks;
  ASSERT_EQ(tail[3].size(), 1u);  // frame 27: 5 px per frame for 3 frames is 15, capped at 10
  expectBox(tail[3][0], 230, 100, 250, 120, MaskOrigin::Held);
}

// The second pass gates each axis by the larger box size along it and pairs only similar sizes.
TEST(BoxTimeline, ALooseMatchNeedsCloseEnoughBoxesOfASimilarSize) {
  const auto interpolated = [](const TimelineConfig& c, const Box& from, const Box& to) {
    BoxTimeline timeline(c, 4000, 4000);
    timeline.advance({}, key(0, {from}));
    const auto masks = timeline.advance(frames(0, 12), key(12, {to})).masks;
    return countOrigin(masks[6], MaskOrigin::Interpolated);
  };
  // Both axes count in the larger side of the two boxes (20 here): 6 sides right and 3 down of a
  // 20 x 10 box is matched, and so is 6.5 heights down; 6.5 sides either way is not.
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 110), face(220, 160, 240, 170)), 1u);
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 110), face(100, 165, 120, 175)), 1u);
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 110), face(230, 100, 250, 110)), 0u);
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 110), face(100, 230, 120, 240)), 0u);
  // The gate is the configured value.
  TimelineConfig tight = exact();
  tight.matchMaxDistanceLoose = 2.0;
  EXPECT_EQ(interpolated(tight, face(100, 100, 120, 110), face(160, 100, 180, 110)), 0u);
  EXPECT_EQ(interpolated(tight, face(100, 100, 120, 110), face(140, 100, 160, 110)), 1u);
  // Sizes within a factor of 3 on each axis pair, beyond it they do not; classes never mix.
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 120), face(160, 100, 220, 160)), 1u);
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 120), face(160, 100, 221, 120)), 0u);
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 120), face(160, 100, 180, 161)), 0u);
  EXPECT_EQ(interpolated(exact(), face(100, 100, 120, 120), Box{kPlate, 160, 100, 180, 120, 0.9f}), 0u);
}

// The scene that used to stack masks: a plate missed at keyframe 36 while a look-alike turns up 100 px away (and
// stays), two plates crossing, and a face appearing late. Every track masks a frame at most once, and every mask
// is detected, interpolated or held (no `extended` any more).
TEST(BoxTimeline, EveryTrackHasAtMostOneMaskPerFrame) {
  const Mover plate{kPlate, 400, 500, 48, 16, 3, 0};
  const Mover right{kPlate, 100, 700, 48, 16, 8, 0};
  const Mover left{kPlate, 900, 708, 48, 16, -8, 0};
  const float lookAlikeX = plate.at(3 * kStep).x1 + 100;
  const Box lookAlike{kPlate, lookAlikeX, 520, lookAlikeX + 48, 536, 0.9f};
  BoxTimeline timeline(TimelineConfig{}, 1920, 1080);
  std::vector<std::vector<MaskBox>> masks;
  timeline.advance({}, key(0, {plate.at(0), right.at(0), left.at(0)}));
  for (int64_t k = 1; k <= 6; ++k) {
    std::vector<Box> boxes{right.at(k * kStep), left.at(k * kStep)};
    if (k != 3) boxes.push_back(plate.at(k * kStep));
    if (k >= 3) boxes.push_back(lookAlike);
    if (k >= 4) boxes.push_back(face(1000, 200, 1020, 224));
    for (auto& m : timeline.advance(frames((k - 1) * kStep, k * kStep), key(k * kStep, boxes)).masks) {
      masks.push_back(std::move(m));
    }
  }
  for (auto& m : timeline.finish(frames(6 * kStep, 7 * kStep)).masks) masks.push_back(std::move(m));
  ASSERT_EQ(masks.size(), static_cast<size_t>(7 * kStep));
  expectOneMaskPerTrack(masks);
  for (const auto& frame : masks) {
    for (const MaskBox& m : frame) {
      const std::string origin = toString(m.origin);
      EXPECT_TRUE(origin == "detected" || origin == "interpolated" || origin == "held") << origin;
      EXPECT_GE(m.track, 1);
    }
  }
}

// A mask grows at once (spec 2026-10-19 §3.1): the face doubles in 2 frames and frame 1's mask is its box.
TEST(BoxTimeline, AMaskGrowsAtOnce) {
  BoxTimeline timeline(smooth(0.02, 100.0), 1000, 1000);
  timeline.advance({}, key(0, {face(125, 125, 175, 175)}));
  const auto masks = timeline.advance(frames(0, 2), key(2, {face(100, 100, 200, 200)})).masks;
  ASSERT_EQ(masks[1].size(), 1u);
  expectBox(masks[1][0], 112.5f, 112.5f, 187.5f, 187.5f, MaskOrigin::Interpolated);
}

// ... and shrinks slowly: its width and height by at most shrinkPerFrame of themselves per frame, never below the
// box's, around the box's centre. The face halves in 2 frames (100 -> 75 -> 50 px) and then stays; with 10 % per
// frame the mask's side goes 100, 90, 81, 72.9, 65.61, 59.049, 53.1441, then 50 (0.9 x 53.1441 is less).
TEST(BoxTimeline, AMaskShrinksByAtMostItsShrinkFractionPerFrame) {
  BoxTimeline timeline(smooth(0.1, 100.0), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 200, 200)}));
  std::vector<std::vector<MaskBox>> masks = timeline.advance(frames(0, 2), key(2, {face(125, 125, 175, 175)})).masks;
  for (auto& m : timeline.advance(frames(2, 8), key(8, {face(125, 125, 175, 175)})).masks) masks.push_back(std::move(m));
  ASSERT_EQ(masks.size(), 8u);
  const double sides[] = {100, 90, 81, 72.9, 65.61, 59.049, 53.1441, 50};
  for (size_t k = 0; k < masks.size(); ++k) {
    SCOPED_TRACE("frame " + std::to_string(k));
    ASSERT_EQ(masks[k].size(), 1u);
    const auto half = static_cast<float>(sides[k] / 2);
    expectBox(masks[k][0], 150 - half, 150 - half, 150 + half, 150 + half,
              k == 0 || k == 2 ? MaskOrigin::Detected : MaskOrigin::Interpolated);
  }
}

// A lost target's mask keeps its smoothed size while it is held, and moves with the held box. The face shrinks to
// 50 px while moving 2 px a frame (masks of 100, 90 and 81 px), then is missed at keyframe 6: frames 3 to 5 keep
// 81 px around the held centre.
TEST(BoxTimeline, AHeldMaskKeepsItsSmoothedSizeAtTheHeldPosition) {
  BoxTimeline timeline(smooth(0.1, 100.0), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 200, 200)}));
  std::vector<std::vector<MaskBox>> masks = timeline.advance(frames(0, 2), key(2, {face(129, 125, 179, 175)})).masks;
  for (auto& m : timeline.advance(frames(2, 6), key(6, {})).masks) masks.push_back(std::move(m));
  ASSERT_EQ(masks.size(), 6u);
  ASSERT_EQ(masks[1].size(), 1u);
  expectBox(masks[1][0], 107, 105, 197, 195, MaskOrigin::Interpolated);  // 90 px around (152, 150)
  ASSERT_EQ(masks[2].size(), 1u);
  expectBox(masks[2][0], 113.5f, 109.5f, 194.5f, 190.5f, MaskOrigin::Detected);  // 81 px around (154, 150)
  for (size_t k = 3; k < masks.size(); ++k) {
    SCOPED_TRACE("frame " + std::to_string(k));
    ASSERT_EQ(masks[k].size(), 1u);
    const float cx = 154 + 2.0f * static_cast<float>(k - 2);  // held along the velocity
    expectBox(masks[k][0], cx - 40.5f, 109.5f, cx + 40.5f, 190.5f, MaskOrigin::Held);
  }
}

// Some single mask contains the whole box.
bool covers(const std::vector<MaskBox>& masks, float x1, float y1, float x2, float y2) {
  return std::any_of(masks.begin(), masks.end(), [=](const MaskBox& m) {
    return m.x1 <= x1 + 1e-3f && m.y1 <= y1 + 1e-3f && m.x2 >= x2 - 1e-3f && m.y2 >= y2 - 1e-3f;
  });
}

// The smoothing is of the size, around the box's own centre (spec §3.1): a fast target's mask follows it instead
// of trailing behind. Its width and height never exceed the larger of the box's and the previous mask's, and its
// centre is the box's.
TEST(BoxTimeline, AFastTargetIsNotSmearedAlongItsMotion) {
  // A 20 px face at 5 px a frame: every mask is the box itself, nothing is left behind.
  BoxTimeline steady(smooth(0.02, 4.0), 1000, 1000);
  steady.advance({}, key(0, {face(100, 100, 120, 120)}));
  std::vector<std::vector<MaskBox>> masks = steady.advance(frames(0, 12), key(12, {face(160, 100, 180, 120)})).masks;
  for (auto& m : steady.advance(frames(12, 24), key(24, {face(220, 100, 240, 120)})).masks) masks.push_back(std::move(m));
  ASSERT_EQ(masks.size(), 24u);
  for (size_t k = 0; k < masks.size(); ++k) {
    SCOPED_TRACE("frame " + std::to_string(k));
    ASSERT_EQ(masks[k].size(), 1u);
    const float x = 100 + 5.0f * static_cast<float>(k);
    expectBox(masks[k][0], x, 100, x + 20, 120, k % 12 == 0 ? MaskOrigin::Detected : MaskOrigin::Interpolated);
  }

  // A face shrinking from 100 to 50 px while it moves 175 px in 12 frames: the mask's side shrinks 2 % a frame,
  // centred on the box wherever the box is.
  BoxTimeline shrinking(smooth(0.02, 4.0), 1000, 1000);
  shrinking.advance({}, key(0, {face(100, 100, 200, 200)}));
  const auto moving = shrinking.advance(frames(0, 12), key(12, {face(300, 125, 350, 175)})).masks;
  ASSERT_EQ(moving.size(), 12u);
  double side = 0;
  for (size_t k = 0; k < moving.size(); ++k) {
    SCOPED_TRACE("frame " + std::to_string(k));
    ASSERT_EQ(moving[k].size(), 1u);
    const MaskBox& m = moving[k][0];
    const double t = static_cast<double>(k) / 12;
    const double cx = 150 + 175 * t;  // the interpolated box's centre (its y stays 150)
    const double box = 100 - 50 * t;  // and its width and height
    const double previous = side;
    side = k == 0 ? box : std::max(box, 0.98 * side);
    EXPECT_NEAR((m.x1 + m.x2) / 2, cx, 1e-3);
    EXPECT_NEAR((m.y1 + m.y2) / 2, 150, 1e-3);
    EXPECT_NEAR(m.x2 - m.x1, side, 1e-3);
    EXPECT_NEAR(m.y2 - m.y1, side, 1e-3);
    EXPECT_LE(m.x2 - m.x1, std::max(box, previous) + 1e-3);
    EXPECT_LE(m.y2 - m.y1, std::max(box, previous) + 1e-3);
    EXPECT_TRUE(covers(moving[k], static_cast<float>(cx - box / 2), static_cast<float>(150 - box / 2),
                       static_cast<float>(cx + box / 2), static_cast<float>(150 + box / 2)));
  }
}

// A mask covers at most max_mask_area_ratio times its box: beyond that its width and height are pulled the same
// fraction of their way towards the box's. Frame 1's box is 120..180 (60 px, 3600 px^2); the side may only shrink
// from 100 to 98 (9604 px^2, 2.67 times), so it is pulled towards 60 until the mask is 2 x 3600 = 7200 px^2:
// sqrt(7200) px square around the box's centre.
TEST(BoxTimeline, CapsTheMaskAreaAtTheConfiguredMultipleOfItsBox) {
  BoxTimeline timeline(smooth(0.02, 2.0), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 200, 200)}));
  const auto masks = timeline.advance(frames(0, 2), key(2, {face(140, 140, 160, 160)})).masks;
  ASSERT_EQ(masks[1].size(), 1u);
  const double half = std::sqrt(7200.0) / 2;
  const MaskBox& m = masks[1][0];
  EXPECT_NEAR(m.x1, 150 - half, 1e-3);
  EXPECT_NEAR(m.x2, 150 + half, 1e-3);
  EXPECT_NEAR(m.y1, 150 - half, 1e-3);
  EXPECT_NEAR(m.y2, 150 + half, 1e-3);
}

// Frame 1's box is 145..255 x 100..120 (110 x 20 = 2200 px^2); the width may only shrink from 200 to 196 (3920 px^2),
// the height is the box's. The width is pulled towards 110 until the mask is 1.5 x 2200 = 3300 px^2: 165 x 20 around
// the box's centre.
TEST(BoxTimeline, TheAreaCapPullsBothSidesTowardsTheBox) {
  BoxTimeline timeline(smooth(0.02, 1.5), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 300, 120)}));
  const auto masks = timeline.advance(frames(0, 2), key(2, {face(190, 100, 210, 120)})).masks;
  ASSERT_EQ(masks[1].size(), 1u);
  expectBox(masks[1][0], 117.5f, 100, 282.5f, 120, MaskOrigin::Interpolated);
}

// A track whose box lies wholly outside the frame forgets its mask size: when it comes back, its mask is its box
// (40 px), not a size shrunk slowly from the 83 px it had when it left. The face shrinks from 100 to 40 px while it
// moves right out of the 1000 px frame (frames 10 and 11 are outside, then keyframe 12 and the first 6 frames after
// it) and back in (frame 19: x from 983.3).
TEST(BoxTimeline, ForgetsTheSizeOfATargetThatLeavesTheFrame) {
  BoxTimeline timeline(smooth(0.02, 100.0), 1000, 1000);
  timeline.advance({}, key(0, {face(600, 100, 700, 200)}));
  std::vector<std::vector<MaskBox>> masks =
      timeline.advance(frames(0, 12), key(12, {face(1100, 130, 1140, 170)})).masks;
  for (auto& m : timeline.advance(frames(12, 24), key(24, {face(900, 130, 940, 170)})).masks) {
    masks.push_back(std::move(m));
  }
  ASSERT_EQ(masks.size(), 24u);
  ASSERT_EQ(masks[9].size(), 1u);  // frame 9: the box still overlaps the frame, 55 px in a mask of 83.4 px
  EXPECT_NEAR(masks[9][0].y2 - masks[9][0].y1, 100 * std::pow(0.98, 9), 1e-3);
  for (size_t k = 10; k < 19; ++k) EXPECT_TRUE(masks[k].empty()) << "frame " << k;
  ASSERT_EQ(masks[19].size(), 1u);
  expectBox(masks[19][0], 983.3333f, 130, 1000, 170, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, RejectsAnInvalidShrinkOrAreaRatio) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double bad : {0.0, -0.1, 1.5, nan, inf}) {
    EXPECT_THROW(BoxTimeline(smooth(bad, 4.0), 10, 10), std::invalid_argument) << bad;
  }
  for (const double bad : {0.99, 0.0, -1.0, nan, inf}) {
    EXPECT_THROW(BoxTimeline(smooth(0.02, bad), 10, 10), std::invalid_argument) << bad;
  }
  EXPECT_NO_THROW(BoxTimeline(smooth(1.0, 1.0), 10, 10));
  EXPECT_DOUBLE_EQ(TimelineConfig{}.shrinkPerFrame, 0.02);
  EXPECT_DOUBLE_EQ(TimelineConfig{}.maxMaskAreaRatio, 4.0);
}

}  // namespace
}  // namespace vc
