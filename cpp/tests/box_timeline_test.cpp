#include "track/box_timeline.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vc {
namespace {

constexpr int kFace = 0;
constexpr int kPlate = 1;

TimelineConfig exact() { return TimelineConfig{1.0, 0.0, 0.0, 0.0}; }

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

// Some single mask contains the whole box.
bool covers(const std::vector<MaskBox>& masks, float x1, float y1, float x2, float y2) {
  return std::any_of(masks.begin(), masks.end(), [=](const MaskBox& m) {
    return m.x1 <= x1 + 1e-3f && m.y1 <= y1 + 1e-3f && m.x2 >= x2 - 1e-3f && m.y2 >= y2 - 1e-3f;
  });
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
  EXPECT_TRUE(timeline.advance({}, key(0, {face(100, 100, 120, 120)})).empty());
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  ASSERT_EQ(masks.size(), 12u);
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 100, 100, 120, 120, MaskOrigin::Detected);
  ASSERT_EQ(masks[6].size(), 1u);
  expectBox(masks[6][0], 106, 100, 126, 120, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, FallsBackToFrameIndexWhenTimestampsStall) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}, 0.0));
  const auto masks = timeline.advance(frames(0, 12, 0.0), key(12, {face(112, 100, 132, 120)}, 0.0));
  ASSERT_EQ(masks[6].size(), 1u);
  expectBox(masks[6][0], 106, 100, 126, 120, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, ExtendsDisappearingAndAppearingBoxesOverTheInterval) {
  BoxTimeline gone(exact(), 1000, 1000);
  gone.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto a = gone.advance(frames(0, 12), key(12, {}));
  for (size_t k = 0; k < a.size(); ++k) {
    ASSERT_EQ(a[k].size(), 1u);
    expectBox(a[k][0], 100, 100, 120, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Extended);
  }
  BoxTimeline appears(exact(), 1000, 1000);
  appears.advance({}, key(0, {}));
  const auto b = appears.advance(frames(0, 12), key(12, {face(300, 300, 320, 320)}));
  for (const auto& m : b) {
    ASSERT_EQ(m.size(), 1u);
    expectBox(m[0], 300, 300, 320, 320, MaskOrigin::Extended);
  }
}

TEST(BoxTimeline, NeverMatchesAcrossClasses) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {Box{kPlate, 100, 100, 120, 120, 0.9f}}));
  ASSERT_EQ(masks[6].size(), 2u);
  EXPECT_EQ(masks[6][0].origin, MaskOrigin::Extended);
  EXPECT_EQ(masks[6][1].origin, MaskOrigin::Extended);
}

TEST(BoxTimeline, PredictsWithVelocityToMatchFastMovers) {
  // 18 px per interval, then 22 px: unmatched without the prediction (22 / 20 > 1).
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(118, 100, 138, 120)}));
  const auto masks = timeline.advance(frames(12, 24), key(24, {face(140, 100, 160, 120)}));
  ASSERT_EQ(masks[6].size(), 1u);  // frame 18
  expectBox(masks[6][0], 129, 100, 149, 120, MaskOrigin::Interpolated);

  BoxTimeline fresh(exact(), 1000, 1000);
  fresh.advance({}, key(12, {face(118, 100, 138, 120)}));
  const auto unpredicted = fresh.advance(frames(12, 24), key(24, {face(140, 100, 160, 120)}));
  EXPECT_EQ(unpredicted[6].size(), 2u);
}

TEST(BoxTimeline, ExpandsByFractionAndMotionAndEnforcesTheMinimumSize) {
  BoxTimeline timeline(TimelineConfig{1.0, 0.15, 0.5, 16.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 110)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 110)}));
  // x margin 0.15 * 20 + 0.5 * 12 = 9; y margin 1.5 gives 13 px, grown to 16 around y = 105
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 91, 97, 129, 113, MaskOrigin::Detected);
}

TEST(BoxTimeline, ClipsToTheFrame) {
  BoxTimeline timeline(TimelineConfig{1.0, 0.0, 0.0, 16.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(0, 0, 4, 4)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(0, 0, 4, 4)}));
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 0, 0, 10, 10, MaskOrigin::Detected);
}

TEST(BoxTimeline, FinishExtendsForwardWithVelocity) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  const auto masks = timeline.finish(frames(12, 18));
  ASSERT_EQ(masks.size(), 6u);
  ASSERT_EQ(masks[0].size(), 1u);
  ASSERT_EQ(masks[3].size(), 1u);
  expectBox(masks[0][0], 112, 100, 132, 120, MaskOrigin::Detected);
  expectBox(masks[3][0], 115, 100, 135, 120, MaskOrigin::Extended);  // frame 15, 1 px per frame
}

TEST(BoxTimeline, RejectsCallsOutOfOrder) {
  BoxTimeline timeline(exact(), 1000, 1000);
  EXPECT_THROW(timeline.advance(frames(0, 3), key(3, {})), std::invalid_argument);
  timeline.advance({}, key(0, {}));
  EXPECT_THROW(timeline.advance(frames(1, 6), key(6, {})), std::invalid_argument);
  EXPECT_THROW(timeline.advance(frames(0, 6), key(5, {})), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{0.0, 0.1, 0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_STREQ(toString(MaskOrigin::Interpolated), "interpolated");
}

TEST(BoxTimeline, BridgesOneMissedKeyframeWithAGhostTrack) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  const auto gap = timeline.advance(frames(12, 24), key(24, {}));
  ASSERT_EQ(gap[6].size(), 1u);  // frame 18
  expectBox(gap[6][0], 118, 100, 138, 120, MaskOrigin::Extended);

  const auto bridged = timeline.advance(frames(24, 36), key(36, {face(136, 100, 156, 120)}));
  // The ghost is matched, not duplicated: its interpolation plus the next box held in place.
  for (const auto& m : bridged) EXPECT_EQ(m.size(), 2u);
  ASSERT_EQ(bridged[0].size(), 2u);  // frame 24: the ghost's predicted position
  expectBox(bridged[0][0], 124, 100, 144, 120, MaskOrigin::Extended);
  expectBox(bridged[0][1], 136, 100, 156, 120, MaskOrigin::Extended);
  ASSERT_EQ(bridged[6].size(), 2u);  // frame 30
  expectBox(bridged[6][0], 130, 100, 150, 120, MaskOrigin::Interpolated);
  expectBox(bridged[6][1], 136, 100, 156, 120, MaskOrigin::Extended);
}

TEST(BoxTimeline, CoversATargetThatStopsAfterAMissedKeyframe) {
  // Moving at 1 px per frame, missed at index 24, but really stopped at x1 = 112 since index 12. The
  // ghost (predicted at 124) is matched to the box found at 112 (centre distance 12, cost 0.6) and its
  // interpolation 124 -> 112 leaves the real box uncovered until the interval's end, so the next box
  // is also held in place over the whole interval.
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  timeline.advance(frames(12, 24), key(24, {}));
  const auto masks = timeline.advance(frames(24, 36), key(36, {face(112, 100, 132, 120)}));
  ASSERT_EQ(masks.size(), 12u);
  for (size_t k = 0; k < masks.size(); ++k) {
    EXPECT_TRUE(covers(masks[k], 112, 100, 132, 120)) << "frame " << 24 + k;
  }
}

TEST(BoxTimeline, DropsAGhostAfterTwoMissedKeyframes) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  timeline.advance(frames(12, 24), key(24, {}));
  const auto ghost = timeline.advance(frames(24, 36), key(36, {}));
  for (const auto& m : ghost) EXPECT_EQ(m.size(), 1u);  // the ghost still covers its last interval
  ASSERT_EQ(ghost[6].size(), 1u);
  expectBox(ghost[6][0], 130, 100, 150, 120, MaskOrigin::Extended);

  const auto after = timeline.advance(frames(36, 48), key(48, {}));
  ASSERT_EQ(after.size(), 12u);
  for (const auto& m : after) EXPECT_TRUE(m.empty());
}

TEST(BoxTimeline, FinishExtendsAGhostTrack) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  timeline.advance(frames(12, 24), key(24, {}));
  const auto masks = timeline.finish(frames(24, 30));
  ASSERT_EQ(masks.size(), 6u);
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 124, 100, 144, 120, MaskOrigin::Extended);  // not a detection
  ASSERT_EQ(masks[3].size(), 1u);
  expectBox(masks[3][0], 127, 100, 147, 120, MaskOrigin::Extended);
}

TEST(BoxTimeline, ExpandsForMotionInAnyDirection) {
  BoxTimeline timeline(TimelineConfig{1.0, 0.0, 0.5, 0.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(112, 112, 132, 132)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(100, 100, 120, 120)}));
  ASSERT_EQ(masks[0].size(), 1u);
  // displacement 12 px in x and in y, moving left / up: 0.5 * 12 = 6 px on every side
  expectBox(masks[0][0], 106, 106, 138, 138, MaskOrigin::Detected);
}

TEST(BoxTimeline, ExtrapolatesADisappearingBoxAlongItsVelocity) {
  BoxTimeline plain(exact(), 1000, 1000);
  plain.advance({}, key(0, {face(100, 100, 120, 120)}));
  plain.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  const auto masks = plain.advance(frames(12, 24), key(24, {}));
  ASSERT_EQ(masks[6].size(), 1u);  // frame 18, 1 px per frame
  expectBox(masks[6][0], 118, 100, 138, 120, MaskOrigin::Extended);

  BoxTimeline moving(TimelineConfig{1.0, 0.0, 0.5, 0.0}, 1000, 1000);
  moving.advance({}, key(0, {face(100, 100, 120, 120)}));
  moving.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  const auto grown = moving.advance(frames(12, 24), key(24, {}));
  ASSERT_EQ(grown[6].size(), 1u);  // |v| * interval = 12 px, half of it per side
  expectBox(grown[6][0], 112, 100, 144, 120, MaskOrigin::Extended);
}

TEST(BoxTimeline, MatchesGreedilyByCost) {
  // Both previous faces are candidates for the next one (costs 1.25 and 0.25, limit 2.0): the
  // cheaper pair wins and the other face is extended.
  BoxTimeline timeline(TimelineConfig{2.0, 0.0, 0.0, 0.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120), face(130, 100, 150, 120)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(125, 100, 145, 120)}));
  ASSERT_EQ(masks[6].size(), 2u);
  const auto count = [&](MaskOrigin origin) {
    return std::count_if(masks[6].begin(), masks[6].end(), [origin](const MaskBox& m) { return m.origin == origin; });
  };
  EXPECT_EQ(count(MaskOrigin::Interpolated), 1);
  EXPECT_EQ(count(MaskOrigin::Extended), 1);
  for (const MaskBox& m : masks[6]) {
    if (m.origin == MaskOrigin::Interpolated) expectBox(m, 127.5f, 100, 147.5f, 120, MaskOrigin::Interpolated);
    if (m.origin == MaskOrigin::Extended) expectBox(m, 100, 100, 120, 120, MaskOrigin::Extended);
  }
}

TEST(BoxTimeline, InterpolatesByPtsForUnevenTimestamps) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, Keyframe{{0, 0.0}, {face(100, 100, 130, 130)}});
  const std::vector<FrameTime> uneven = {{0, 0.0}, {1, 10.0}, {2, 20.0}, {3, 200.0}, {4, 210.0}};
  const auto masks = timeline.advance(uneven, Keyframe{{5, 270.0}, {face(127, 100, 157, 130)}});
  ASSERT_EQ(masks.size(), 5u);
  ASSERT_EQ(masks[3].size(), 1u);  // pts 200 of 270: 100 + 27 * 200 / 270
  expectBox(masks[3][0], 120, 100, 150, 130, MaskOrigin::Interpolated);

  // A repeated pts is not usable: positions fall back to the frame index.
  BoxTimeline stalled(exact(), 1000, 1000);
  stalled.advance({}, Keyframe{{0, 0.0}, {face(100, 100, 130, 130)}});
  const std::vector<FrameTime> repeated = {{0, 0.0}, {1, 10.0}, {2, 10.0}, {3, 30.0}};
  const auto byIndex = stalled.advance(repeated, Keyframe{{4, 100.0}, {face(124, 100, 154, 130)}});
  ASSERT_EQ(byIndex.size(), 4u);
  ASSERT_EQ(byIndex[1].size(), 1u);
  expectBox(byIndex[1][0], 106, 100, 136, 130, MaskOrigin::Interpolated);
  ASSERT_EQ(byIndex[2].size(), 1u);
  expectBox(byIndex[2][0], 112, 100, 142, 130, MaskOrigin::Interpolated);
  ASSERT_EQ(byIndex[3].size(), 1u);
  expectBox(byIndex[3][0], 118, 100, 148, 130, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, ClipsAtTheRightAndBottomOfANonSquareFrame) {
  BoxTimeline timeline(TimelineConfig{1.0, 0.0, 0.0, 16.0}, 640, 360);
  timeline.advance({}, key(0, {face(630, 350, 640, 360)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(630, 350, 640, 360)}));
  // 10 px grown to 16 around the centre (635, 355) is 627..643 x 347..363, clipped to 640 x 360
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 627, 347, 640, 360, MaskOrigin::Detected);
  ASSERT_EQ(masks[3].size(), 1u);
  expectBox(masks[3][0], 627, 347, 640, 360, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, FinishCapsExtrapolationAtOneInterval) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  const auto masks = timeline.finish(frames(12, 42));
  ASSERT_EQ(masks.size(), 30u);
  ASSERT_EQ(masks[11].size(), 1u);  // frame 23: still inside the first interval
  expectBox(masks[11][0], 123, 100, 143, 120, MaskOrigin::Extended);
  ASSERT_EQ(masks[25].size(), 1u);  // frame 37: 25 frames past the keyframe, capped at 12
  expectBox(masks[25][0], 124, 100, 144, 120, MaskOrigin::Extended);
  ASSERT_EQ(masks[29].size(), 1u);
  expectBox(masks[29][0], 124, 100, 144, 120, MaskOrigin::Extended);
}

TEST(BoxTimeline, MatchesATargetThatStops) {
  // The face speeds up to 3 px per frame, then stops: the velocity predicts a centre 36 px away
  // (cost 1.8), but the box is exactly where it was (zero-velocity cost 0).
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(118, 100, 138, 120)}));
  timeline.advance(frames(12, 24), key(24, {face(154, 100, 174, 120)}));
  const auto masks = timeline.advance(frames(24, 36), key(36, {face(154, 100, 174, 120)}));
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
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(100, 100, 120, 120)}));
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
  }
  EXPECT_THROW(BoxTimeline(TimelineConfig{-1.0, 0.1, 0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, -0.1, 0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, -0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, 0.1, -1}, 10, 10), std::invalid_argument);
}

TEST(BoxTimeline, ExtensionMarginUsesAbsoluteVelocity) {
  BoxTimeline timeline(TimelineConfig{1.0, 0.0, 0.5, 0.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(112, 112, 132, 132)}));
  timeline.advance(frames(0, 12), key(12, {face(100, 100, 120, 120)}));  // v = (-1, -1)
  const auto gone = timeline.advance(frames(12, 24), key(24, {}));
  ASSERT_EQ(gone[6].size(), 1u);  // frame 18: shifted -6, margin 0.5 * 12 per side
  expectBox(gone[6][0], 88, 88, 120, 120, MaskOrigin::Extended);
}

TEST(BoxTimeline, FinishMarginUsesAbsoluteVelocityAndCap) {
  BoxTimeline timeline(TimelineConfig{1.0, 0.0, 0.5, 0.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(112, 112, 132, 132)}));
  timeline.advance(frames(0, 12), key(12, {face(100, 100, 120, 120)}));  // v = (-1, -1)
  const auto masks = timeline.finish(frames(12, 40));
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 94, 94, 126, 126, MaskOrigin::Detected);  // margin 6 at the keyframe frame
  ASSERT_EQ(masks[20].size(), 1u);  // frame 32: shift capped at -12, margin 6
  expectBox(masks[20][0], 82, 82, 114, 114, MaskOrigin::Extended);
}

TEST(BoxTimeline, OneTrackMatchesAtMostOneBox) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(104, 100, 124, 120), face(108, 100, 128, 120)}));
  ASSERT_EQ(masks[6].size(), 2u);
  const auto interpolated = std::count_if(masks[6].begin(), masks[6].end(),
                                          [](const MaskBox& m) { return m.origin == MaskOrigin::Interpolated; });
  EXPECT_EQ(interpolated, 1);
}

TEST(BoxTimeline, MatchedGhostVelocityIsRecomputed) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));   // v = 1
  timeline.advance(frames(12, 24), key(24, {}));                          // ghost at 124
  timeline.advance(frames(24, 36), key(36, {face(140, 108, 160, 128)}));  // found again
  const auto masks = timeline.finish(frames(36, 42));
  ASSERT_EQ(masks[3].size(), 1u);
  // The velocity runs from the last detection (centre (122, 110) at index 12) to the new one (centre
  // (150, 118) at index 36): (28, 8) / 24 = (7/6, 1/3) px per frame, so frame 39 is 3.5 px right and
  // 1 px down of the box. (Measured from the ghost it would be (4/3, 2/3): 4 px and 2 px.)
  expectBox(masks[3][0], 143.5f, 109, 163.5f, 129, MaskOrigin::Extended);
}

TEST(BoxTimeline, UnmatchedGhostIsExtendedAtItsKeyFrame) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  timeline.advance(frames(12, 24), key(24, {}));
  const auto ghost = timeline.advance(frames(24, 36), key(36, {}));
  ASSERT_EQ(ghost[0].size(), 1u);
  expectBox(ghost[0][0], 124, 100, 144, 120, MaskOrigin::Extended);
}

TEST(BoxTimeline, MatchedGhostVelocityIsMeasuredFromItsLastDetection) {
  // Detected at x1 = 100 (index 0), missed at index 12, found at x1 = 118 at index 24. The ghost has
  // no velocity yet and stays at 100, so a velocity taken from the ghost (18 px over 12 frames =
  // 1.5 px/frame) would overshoot; from the detection it is 18 px over 24 frames = 0.75 px/frame.
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {}));
  timeline.advance(frames(12, 24), key(24, {face(118, 100, 138, 120)}));
  const auto masks = timeline.finish(frames(24, 31));
  ASSERT_EQ(masks.size(), 7u);
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 118, 100, 138, 120, MaskOrigin::Detected);
  ASSERT_EQ(masks[6].size(), 1u);  // frame 30: 118 + 0.75 * 6
  expectBox(masks[6][0], 122.5f, 100, 142.5f, 120, MaskOrigin::Extended);
}

TEST(BoxTimeline, AGhostWithoutVelocityStaysInPlace) {
  BoxTimeline gone(exact(), 1000, 1000);
  gone.advance({}, key(0, {face(100, 100, 120, 120)}));
  gone.advance(frames(0, 12), key(12, {}));  // no velocity yet: the ghost sits where the box was
  const auto ghost = gone.advance(frames(12, 24), key(24, {}));
  ASSERT_EQ(ghost.size(), 12u);
  for (const auto& m : ghost) {
    ASSERT_EQ(m.size(), 1u);
    expectBox(m[0], 100, 100, 120, 120, MaskOrigin::Extended);
  }

  BoxTimeline ending(exact(), 1000, 1000);
  ending.advance({}, key(0, {face(100, 100, 120, 120)}));
  ending.advance(frames(0, 12), key(12, {}));
  const auto tail = ending.finish(frames(12, 18));
  ASSERT_EQ(tail.size(), 6u);
  for (const auto& m : tail) {
    ASSERT_EQ(m.size(), 1u);
    expectBox(m[0], 100, 100, 120, 120, MaskOrigin::Extended);
  }
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
      EXPECT_TRUE(first.advance({}, key(0, {face(100, 100, 120, 120)})).empty());

      BoxTimeline later(exact(), 1000, 1000);  // and as a later one
      later.advance({}, key(0, {face(100, 100, 120, 120)}));
      EXPECT_THROW(later.advance(frames(0, 12), key(12, {face(100, 100, 120, 120), box})), std::invalid_argument);
      const auto masks = later.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
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
    EXPECT_EQ(later.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)})).size(), 12u);
  }
}

TEST(BoxTimeline, FallsBackToTheIndexForAFrameWithoutAUsableTimestamp) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, Keyframe{{0, 0.0}, {face(100, 100, 130, 130)}});
  const std::vector<FrameTime> broken = {{0, 0.0}, {1, 10.0}, {2, 20.0}, {3, nan}, {4, 210.0}};
  const auto masks = timeline.advance(broken, Keyframe{{5, 270.0}, {face(127, 100, 157, 130)}});
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
    return timeline.advance(list, Keyframe{{5, 270.0}, {face(127, 100, 157, 130)}});
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
  // Finite coordinates whose width overflows a float: with no expansion fraction the margin is
  // 0 * inf = NaN, and a NaN mask would pass an `x2 <= x1` test.
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(-3e38f, 100, 3e38f, 120)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(-3e38f, 100, 3e38f, 120)}));
  ASSERT_EQ(masks.size(), 6u);
  expectFinite(masks);
  expectFinite(timeline.finish(frames(6, 12)));
}

}  // namespace
}  // namespace vc
