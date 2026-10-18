#include "track/box_timeline.h"

#include <gtest/gtest.h>

#include <stdexcept>
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

}  // namespace
}  // namespace vc
