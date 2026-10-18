#include "track/box_timeline.h"

#include <gtest/gtest.h>

#include <algorithm>
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
  for (const auto& m : bridged) EXPECT_EQ(m.size(), 1u);  // the ghost is matched, not duplicated
  ASSERT_EQ(bridged[0].size(), 1u);  // frame 24: the ghost's predicted position
  expectBox(bridged[0][0], 124, 100, 144, 120, MaskOrigin::Extended);
  ASSERT_EQ(bridged[6].size(), 1u);  // frame 30
  expectBox(bridged[6][0], 130, 100, 150, 120, MaskOrigin::Interpolated);
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

}  // namespace
}  // namespace vc
