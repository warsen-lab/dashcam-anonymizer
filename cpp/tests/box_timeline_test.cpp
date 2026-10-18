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

// A perfect detector (every mover found on every keyframe) over `intervals` keyframe intervals,
// then finish() for one more interval. Returns the masks of every frame, by frame index.
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
    auto masks = timeline.advance(frames((k - 1) * kStep, k * kStep), key(k * kStep, boxesAt(k * kStep)));
    for (auto& m : masks) all.push_back(std::move(m));
  }
  auto tail = timeline.finish(frames(intervals * kStep, (intervals + 1) * kStep));
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
  EXPECT_TRUE(timeline.advance({}, key(0, {face(100, 100, 120, 120)})).empty());
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  ASSERT_EQ(masks.size(), 12u);
  ASSERT_EQ(masks[0].size(), 3u);  // a first pairing: the interpolation, then both end boxes held
  expectBox(masks[0][0], 100, 100, 120, 120, MaskOrigin::Detected);
  ASSERT_EQ(masks[6].size(), 3u);
  expectBox(masks[6][0], 106, 100, 126, 120, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, FallsBackToFrameIndexWhenTimestampsStall) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}, 0.0));
  const auto masks = timeline.advance(frames(0, 12, 0.0), key(12, {face(112, 100, 132, 120)}, 0.0));
  ASSERT_EQ(masks[6].size(), 3u);  // first pairing: interpolation + both end boxes
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

  // Without a velocity only the second pass pairs them: interpolated, both endpoints held as well.
  BoxTimeline fresh(exact(), 1000, 1000);
  fresh.advance({}, key(12, {face(118, 100, 138, 120)}));
  const auto unpredicted = fresh.advance(frames(12, 24), key(24, {face(140, 100, 160, 120)}));
  EXPECT_EQ(unpredicted[6].size(), 3u);
}

TEST(BoxTimeline, ExpandsByFractionAndMotionAndEnforcesTheMinimumSize) {
  BoxTimeline timeline(TimelineConfig{1.0, 0.15, 0.5, 16.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 110)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 110)}));
  // x margin 0.15 * 20 + 0.5 * 12 = 9; y margin 1.5 gives 13 px, grown to 16 around y = 105
  ASSERT_EQ(masks[0].size(), 3u);  // first pairing: interpolation + both end boxes
  expectBox(masks[0][0], 91, 97, 129, 113, MaskOrigin::Detected);
}

TEST(BoxTimeline, ClipsToTheFrame) {
  BoxTimeline timeline(TimelineConfig{1.0, 0.0, 0.0, 16.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(0, 0, 4, 4)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(0, 0, 4, 4)}));
  ASSERT_EQ(masks[0].size(), 3u);  // first pairing: interpolation + both (identical) end boxes
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
  expectBox(masks[3][0], 109, 100, 135, 120, MaskOrigin::Extended);  // frame 15, swept 3 px either way
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
  ASSERT_EQ(gap[6].size(), 1u);  // frame 18: swept 6 px either way of the detection at 112
  expectBox(gap[6][0], 106, 100, 138, 120, MaskOrigin::Extended);

  const auto bridged = timeline.advance(frames(24, 36), key(36, {face(136, 100, 156, 120)}));
  // The ghost is matched, not duplicated: its interpolation, the next box held in place and the
  // interpolation from its last detection (112 at index 12; here the same path).
  for (const auto& m : bridged) EXPECT_EQ(m.size(), 3u);
  ASSERT_EQ(bridged[0].size(), 3u);  // frame 24: the ghost's predicted position
  expectBox(bridged[0][0], 124, 100, 144, 120, MaskOrigin::Extended);
  expectBox(bridged[0][1], 136, 100, 156, 120, MaskOrigin::Extended);
  expectBox(bridged[0][2], 124, 100, 144, 120, MaskOrigin::Interpolated);
  ASSERT_EQ(bridged[6].size(), 3u);  // frame 30
  expectBox(bridged[6][0], 130, 100, 150, 120, MaskOrigin::Interpolated);
  expectBox(bridged[6][1], 136, 100, 156, 120, MaskOrigin::Extended);
  expectBox(bridged[6][2], 130, 100, 150, 120, MaskOrigin::Interpolated);
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
  ASSERT_EQ(ghost[6].size(), 1u);  // 18 frames either way of the last detection (112, index 12)
  expectBox(ghost[6][0], 94, 100, 150, 120, MaskOrigin::Extended);

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
  ASSERT_EQ(masks[0].size(), 1u);  // the ghost at 124: 12 frames either way of the detection at 112
  expectBox(masks[0][0], 100, 100, 144, 120, MaskOrigin::Extended);  // not a detection
  ASSERT_EQ(masks[3].size(), 1u);
  expectBox(masks[3][0], 97, 100, 147, 120, MaskOrigin::Extended);
}

TEST(BoxTimeline, ExpandsForMotionInAnyDirection) {
  BoxTimeline timeline(TimelineConfig{1.0, 0.0, 0.5, 0.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(112, 112, 132, 132)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(100, 100, 120, 120)}));
  ASSERT_EQ(masks[0].size(), 3u);  // first pairing: interpolation + both end boxes
  // displacement 12 px in x and in y, moving left / up: 0.5 * 12 = 6 px on every side
  expectBox(masks[0][0], 106, 106, 138, 138, MaskOrigin::Detected);
}

TEST(BoxTimeline, ExtrapolatesADisappearingBoxAlongItsVelocity) {
  BoxTimeline plain(exact(), 1000, 1000);
  plain.advance({}, key(0, {face(100, 100, 120, 120)}));
  plain.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  const auto masks = plain.advance(frames(12, 24), key(24, {}));
  ASSERT_EQ(masks[6].size(), 1u);  // frame 18, 1 px per frame, swept 6 px either way of 112
  expectBox(masks[6][0], 106, 100, 138, 120, MaskOrigin::Extended);

  BoxTimeline moving(TimelineConfig{1.0, 0.0, 0.5, 0.0}, 1000, 1000);
  moving.advance({}, key(0, {face(100, 100, 120, 120)}));
  moving.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  const auto grown = moving.advance(frames(12, 24), key(24, {}));
  ASSERT_EQ(grown[6].size(), 1u);  // swept 106..138, |v| * interval = 12 px, half of it per side
  expectBox(grown[6][0], 100, 100, 144, 120, MaskOrigin::Extended);
}

TEST(BoxTimeline, MatchesGreedilyByCost) {
  // Both previous faces are candidates for the next one (costs 1.25 and 0.25, limit 2.0): the
  // cheaper pair wins (a first pairing: interpolated, both end boxes held) and the other face is
  // extended.
  BoxTimeline timeline(TimelineConfig{2.0, 0.0, 0.0, 0.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120), face(130, 100, 150, 120)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(125, 100, 145, 120)}));
  ASSERT_EQ(masks[6].size(), 4u);
  EXPECT_EQ(countOrigin(masks[6], MaskOrigin::Interpolated), 1u);
  expectBox(masks[6][0], 100, 100, 120, 120, MaskOrigin::Extended);  // the face left unmatched
  expectBox(masks[6][1], 127.5f, 100, 147.5f, 120, MaskOrigin::Interpolated);
  expectBox(masks[6][2], 130, 100, 150, 120, MaskOrigin::Extended);
  expectBox(masks[6][3], 125, 100, 145, 120, MaskOrigin::Extended);
}

TEST(BoxTimeline, InterpolatesByPtsForUnevenTimestamps) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, Keyframe{{0, 0.0}, {face(100, 100, 130, 130)}});
  const std::vector<FrameTime> uneven = {{0, 0.0}, {1, 10.0}, {2, 20.0}, {3, 200.0}, {4, 210.0}};
  const auto masks = timeline.advance(uneven, Keyframe{{5, 270.0}, {face(127, 100, 157, 130)}});
  ASSERT_EQ(masks.size(), 5u);
  ASSERT_EQ(masks[3].size(), 3u);  // pts 200 of 270: 100 + 27 * 200 / 270 (+ both end boxes)
  expectBox(masks[3][0], 120, 100, 150, 130, MaskOrigin::Interpolated);

  // A repeated pts is not usable: positions fall back to the frame index.
  BoxTimeline stalled(exact(), 1000, 1000);
  stalled.advance({}, Keyframe{{0, 0.0}, {face(100, 100, 130, 130)}});
  const std::vector<FrameTime> repeated = {{0, 0.0}, {1, 10.0}, {2, 10.0}, {3, 30.0}};
  const auto byIndex = stalled.advance(repeated, Keyframe{{4, 100.0}, {face(124, 100, 154, 130)}});
  ASSERT_EQ(byIndex.size(), 4u);
  ASSERT_EQ(byIndex[1].size(), 3u);
  expectBox(byIndex[1][0], 106, 100, 136, 130, MaskOrigin::Interpolated);
  ASSERT_EQ(byIndex[2].size(), 3u);
  expectBox(byIndex[2][0], 112, 100, 142, 130, MaskOrigin::Interpolated);
  ASSERT_EQ(byIndex[3].size(), 3u);
  expectBox(byIndex[3][0], 118, 100, 148, 130, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, ClipsAtTheRightAndBottomOfANonSquareFrame) {
  BoxTimeline timeline(TimelineConfig{1.0, 0.0, 0.0, 16.0}, 640, 360);
  timeline.advance({}, key(0, {face(630, 350, 640, 360)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(630, 350, 640, 360)}));
  // 10 px grown to 16 around the centre (635, 355) is 627..643 x 347..363, clipped to 640 x 360
  ASSERT_EQ(masks[0].size(), 3u);  // first pairing: interpolation + both end boxes
  expectBox(masks[0][0], 627, 347, 640, 360, MaskOrigin::Detected);
  ASSERT_EQ(masks[3].size(), 3u);
  expectBox(masks[3][0], 627, 347, 640, 360, MaskOrigin::Interpolated);
}

TEST(BoxTimeline, FinishCapsExtrapolationAtOneInterval) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  const auto masks = timeline.finish(frames(12, 42));
  ASSERT_EQ(masks.size(), 30u);
  ASSERT_EQ(masks[11].size(), 1u);  // frame 23: still inside the first interval (11 px either way)
  expectBox(masks[11][0], 101, 100, 143, 120, MaskOrigin::Extended);
  ASSERT_EQ(masks[25].size(), 1u);  // frame 37: 25 frames past the keyframe, capped at 12
  expectBox(masks[25][0], 100, 100, 144, 120, MaskOrigin::Extended);
  ASSERT_EQ(masks[29].size(), 1u);
  expectBox(masks[29][0], 100, 100, 144, 120, MaskOrigin::Extended);
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
    EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, 0.1, 1, bad}, 10, 10), std::invalid_argument);
  }
  EXPECT_THROW(BoxTimeline(TimelineConfig{-1.0, 0.1, 0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, -0.1, 0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, -0.1, 1}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, 0.1, -1}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, 0.1, 1, 0.0}, 10, 10), std::invalid_argument);
  EXPECT_THROW(BoxTimeline(TimelineConfig{1.0, 0.1, 0.1, 1, -1.0}, 10, 10), std::invalid_argument);
}

TEST(BoxTimeline, ExtensionMarginUsesAbsoluteVelocity) {
  BoxTimeline timeline(TimelineConfig{1.0, 0.0, 0.5, 0.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(112, 112, 132, 132)}));
  timeline.advance(frames(0, 12), key(12, {face(100, 100, 120, 120)}));  // v = (-1, -1)
  const auto gone = timeline.advance(frames(12, 24), key(24, {}));
  ASSERT_EQ(gone[6].size(), 1u);  // frame 18: swept 94..126 (6 px either way), margin 0.5 * 12 per side
  expectBox(gone[6][0], 88, 88, 132, 132, MaskOrigin::Extended);
}

TEST(BoxTimeline, FinishMarginUsesAbsoluteVelocityAndCap) {
  BoxTimeline timeline(TimelineConfig{1.0, 0.0, 0.5, 0.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(112, 112, 132, 132)}));
  timeline.advance(frames(0, 12), key(12, {face(100, 100, 120, 120)}));  // v = (-1, -1)
  const auto masks = timeline.finish(frames(12, 40));
  ASSERT_EQ(masks[0].size(), 1u);
  expectBox(masks[0][0], 94, 94, 126, 126, MaskOrigin::Detected);  // margin 6 at the keyframe frame
  ASSERT_EQ(masks[20].size(), 1u);  // frame 32: swept 88..132 (shift capped at 12 either way), margin 6
  expectBox(masks[20][0], 82, 82, 138, 138, MaskOrigin::Extended);
}

TEST(BoxTimeline, OneTrackMatchesAtMostOneBox) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(104, 100, 124, 120), face(108, 100, 128, 120)}));
  ASSERT_EQ(masks[6].size(), 4u);  // the pairing (interpolation + both end boxes) and the other box
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
  // 1 px down of the box, and as far the other way (swept). (Measured from the ghost it would be
  // (4/3, 2/3): 4 px and 2 px.)
  expectBox(masks[3][0], 136.5f, 107, 163.5f, 129, MaskOrigin::Extended);
}

TEST(BoxTimeline, UnmatchedGhostIsExtendedAtItsKeyFrame) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  timeline.advance(frames(12, 24), key(24, {}));
  const auto ghost = timeline.advance(frames(24, 36), key(36, {}));
  ASSERT_EQ(ghost[0].size(), 1u);  // the ghost at 124: 12 frames either way of the detection at 112
  expectBox(ghost[0][0], 100, 100, 144, 120, MaskOrigin::Extended);
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
  ASSERT_EQ(masks[6].size(), 1u);  // frame 30: swept 0.75 * 6 px either way of 118
  expectBox(masks[6][0], 113.5f, 100, 142.5f, 120, MaskOrigin::Extended);
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
      ASSERT_EQ(masks[6].size(), 3u);  // first pairing: interpolation + both end boxes
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
  // By index (pts alone would put frame 1 at 101.0): 100 + 27 * k / 5 (+ both end boxes).
  ASSERT_EQ(masks[1].size(), 3u);
  expectBox(masks[1][0], 105.4f, 100, 135.4f, 130, MaskOrigin::Interpolated);
  ASSERT_EQ(masks[3].size(), 3u);
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
  ASSERT_EQ(masks[1].size(), 3u);  // first pairing: interpolation + both end boxes
  expectBox(masks[1][0], 100.0f + 27.0f * 10.0f / 170.0f, 100, 130.0f + 27.0f * 10.0f / 170.0f, 130,
            MaskOrigin::Interpolated);
  // A frame before the previous keyframe's pts, at the next keyframe's pts, or after it: by index
  // (100 + 27 * 1 / 5 = 105.4 for frame 1).
  for (const std::vector<double>& pts : {std::vector<double>{90, 110, 120, 130, 140},
                                         std::vector<double>{100, 110, 120, 130, 270},
                                         std::vector<double>{100, 110, 120, 130, 300}}) {
    masks = run(pts);
    ASSERT_EQ(masks.size(), 5u);
    ASSERT_EQ(masks[1].size(), 3u);
    expectBox(masks[1][0], 105.4f, 100, 135.4f, 130, MaskOrigin::Interpolated);
  }
}

TEST(BoxTimeline, NeverEmitsAMaskWithNonFiniteCoordinates) {
  // Finite coordinates whose width overflows a float: with no expansion fraction the margin is
  // 0 * inf = NaN, and a NaN mask would pass an `x2 <= x1` test. The target must stay masked,
  // not silently dropped, so every frame has to carry a (finite) mask.
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(-3e38f, 100, 3e38f, 120)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(-3e38f, 100, 3e38f, 120)}));
  ASSERT_EQ(masks.size(), 6u);
  expectFinite(masks);
  for (size_t k = 0; k < masks.size(); ++k) {
    ASSERT_EQ(masks[k].size(), 3u) << "frame offset " << k;  // first pairing: + both end boxes
    expectBox(masks[k][0], 0, 100, 1000, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Interpolated);
    expectBox(masks[k][1], 0, 100, 1000, 120, MaskOrigin::Extended);
    expectBox(masks[k][2], 0, 100, 1000, 120, MaskOrigin::Extended);
  }
  const auto tail = timeline.finish(frames(6, 12));
  ASSERT_EQ(tail.size(), 6u);
  expectFinite(tail);
  for (size_t k = 0; k < tail.size(); ++k) {
    ASSERT_EQ(tail[k].size(), 1u) << "frame offset " << k;
    expectBox(tail[k][0], 0, 100, 1000, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Extended);
  }
}

// Interpolating between two float-extreme boxes must not overflow either: (q - p) alone is 5.9e38.
TEST(BoxTimeline, InterpolatesBoxesAtTheFloatLimitsWithoutOverflowing) {
  // The centre moves 5.9e38 (118 box sizes), so the boxes only match with a huge matching distance.
  BoxTimeline timeline(TimelineConfig{1000.0, 0.0, 0.0, 0.0}, 1000, 1000);
  timeline.advance({}, key(0, {face(-3e38f, 100, -2.9e38f, 120)}));
  const auto masks = timeline.advance(frames(0, 6), key(6, {face(2.9e38f, 100, 3e38f, 120)}));
  ASSERT_EQ(masks.size(), 6u);
  expectFinite(masks);
  // Halfway the box spans [-5e36, 5e36], which covers the whole frame width.
  ASSERT_EQ(masks[3].size(), 1u);
  expectBox(masks[3][0], 0, 100, 1000, 120, MaskOrigin::Interpolated);
}

// Plates and faces moving faster than one mean box side per keyframe interval (the first pass cannot
// match them) are matched by the second pass and covered on every frame: interpolated from the
// second interval on (the first interval has no velocity yet), extrapolated after the last keyframe.
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
      int uncovered = 0;
      for (size_t f = 0; f < masks.size(); ++f) {
        if (coverage(masks[f], mover.at(static_cast<int64_t>(f))) < 0.8) ++uncovered;
      }
      EXPECT_EQ(uncovered, 0);
      for (int64_t f = kStep; f < 8 * kStep; ++f) {
        if (f % kStep == 0) continue;  // keyframes carry the detection itself
        EXPECT_GE(countOrigin(masks[static_cast<size_t>(f)], MaskOrigin::Interpolated), 1u) << "frame " << f;
      }
    }
  }
}

// Two plates on neighbouring rows driving towards each other at two box widths per interval: they
// pass each other between keyframes 4 and 5, and both stay covered on every frame.
TEST(BoxTimeline, CoversTwoTargetsThatCrossEachOther) {
  const Mover right{kPlate, 100, 500, 48, 16, 8, 0};
  const Mover left{kPlate, 900, 508, 48, 16, -8, 0};
  BoxTimeline timeline(TimelineConfig{}, 1920, 1080);
  const auto masks = runPerfectDetector(timeline, {right, left}, 8);
  for (size_t f = 0; f < masks.size(); ++f) {
    const auto frame = static_cast<int64_t>(f);
    EXPECT_GE(coverage(masks[f], right.at(frame)), 0.8) << "right-moving plate, frame " << f;
    EXPECT_GE(coverage(masks[f], left.at(frame)), 0.8) << "left-moving plate, frame " << f;
  }
}

// A box three widths away (beyond the first pass): interpolated, with both endpoints held over the
// whole interval, so that a wrong pairing never covers less than leaving both unmatched would.
TEST(BoxTimeline, ALooseMatchInterpolatesAndHoldsBothEndpoints) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto masks = timeline.advance(frames(0, 12), key(12, {face(160, 100, 180, 120)}));
  ASSERT_EQ(masks.size(), 12u);
  for (size_t k = 0; k < masks.size(); ++k) {
    SCOPED_TRACE("frame " + std::to_string(k));
    ASSERT_EQ(masks[k].size(), 3u);
    const float x = 100 + 5.0f * static_cast<float>(k);  // 60 px over 12 frames
    expectBox(masks[k][0], x, 100, x + 20, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Interpolated);
    expectBox(masks[k][1], 100, 100, 120, 120, MaskOrigin::Extended);
    expectBox(masks[k][2], 160, 100, 180, 120, MaskOrigin::Extended);
  }
}

// The loose match seeds the velocity like a normal one: the next interval is matched by the first
// pass along the prediction, one interpolated mask per frame.
TEST(BoxTimeline, ALooseMatchSeedsTheVelocity) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(160, 100, 180, 120)}));
  const auto masks = timeline.advance(frames(12, 24), key(24, {face(220, 100, 240, 120)}));
  ASSERT_EQ(masks[6].size(), 1u);  // frame 18
  expectBox(masks[6][0], 190, 100, 210, 120, MaskOrigin::Interpolated);
  const auto tail = timeline.finish(frames(24, 30));
  ASSERT_EQ(tail[3].size(), 1u);  // frame 27: 5 px per frame from the keyframe, 15 px either way of 220
  expectBox(tail[3][0], 205, 100, 255, 120, MaskOrigin::Extended);
}

// The second pass gates each axis by the larger box size along it and pairs only similar sizes.
TEST(BoxTimeline, ALooseMatchNeedsCloseEnoughBoxesOfASimilarSize) {
  const auto interpolated = [](const TimelineConfig& config, const Box& from, const Box& to) {
    BoxTimeline timeline(config, 4000, 4000);
    timeline.advance({}, key(0, {from}));
    const auto masks = timeline.advance(frames(0, 12), key(12, {to}));
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

// A track that is matched loosely (perhaps to the wrong box) still gets the masks it would have had
// unmatched: here the target was missed and kept moving at 1 px per frame, while another face
// appeared 2.5 widths ahead.
TEST(BoxTimeline, ALooseMatchKeepsTheExtrapolationOfATrackWithAVelocity) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));  // v = 1 px per frame
  const auto masks = timeline.advance(frames(12, 24), key(24, {face(162, 100, 182, 120)}));
  for (size_t k = 0; k < masks.size(); ++k) {
    const float x = 112 + static_cast<float>(k);
    EXPECT_TRUE(covers(masks[k], x, 100, x + 20, 120)) << "frame " << 12 + k;
  }
  EXPECT_EQ(countOrigin(masks[6], MaskOrigin::Interpolated), 1u);
}

// An unmatched track with a velocity masks the region it may have swept since its last detection:
// as far along its velocity as it would have got, and as far back (it may have turned round), so a
// target that stops or turns round right after its last detection stays covered.
TEST(BoxTimeline, AnUnmatchedTrackMasksTheRegionItSweeps) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));  // v = 1 px per frame
  const auto missed = timeline.advance(frames(12, 24), key(24, {}));     // stopped at 112, then missed
  for (size_t k = 0; k < missed.size(); ++k) {
    ASSERT_EQ(missed[k].size(), 1u);
    const float reach = static_cast<float>(k);
    expectBox(missed[k][0], 112 - reach, 100, 132 + reach, 120, k == 0 ? MaskOrigin::Detected : MaskOrigin::Extended);
  }

  // The same at the end of the video.
  BoxTimeline ending(exact(), 1000, 1000);
  ending.advance({}, key(0, {face(112, 112, 132, 132)}));
  ending.advance(frames(0, 12), key(12, {face(100, 100, 120, 120)}));  // v = (-1, -1)
  const auto tail = ending.finish(frames(12, 18));
  ASSERT_EQ(tail[5].size(), 1u);  // frame 17: 5 px either way
  expectBox(tail[5][0], 95, 95, 125, 125, MaskOrigin::Extended);

  // A ghost reaches back as far from the last detection (112 at index 12) as it reaches forward.
  BoxTimeline twice(exact(), 1000, 1000);
  twice.advance({}, key(0, {face(100, 100, 120, 120)}));
  twice.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  twice.advance(frames(12, 24), key(24, {}));
  const auto ghost = twice.advance(frames(24, 36), key(36, {}));
  ASSERT_EQ(ghost[6].size(), 1u);  // frame 30: 18 frames after the detection
  expectBox(ghost[6][0], 94, 100, 150, 120, MaskOrigin::Extended);

  // With the default margins a target that stops after its last detection is covered on every frame.
  BoxTimeline defaults(TimelineConfig{}, 1920, 1080);
  defaults.advance({}, key(0, {Box{kPlate, 400, 500, 448, 516, 0.9f}}));
  defaults.advance(frames(0, 12), key(12, {Box{kPlate, 500, 500, 548, 516, 0.9f}}));  // 100 px per interval
  const auto stopped = defaults.advance(frames(12, 24), key(24, {}));
  for (size_t k = 0; k < stopped.size(); ++k) {
    EXPECT_GE(coverage(stopped[k], 500, 500, 548, 516), 0.999) << "frame " << 12 + k;
  }
  const auto stoppedAtTheEnd = defaults.finish(frames(24, 36));
  for (size_t k = 0; k < stoppedAtTheEnd.size(); ++k) {
    EXPECT_GE(coverage(stoppedAtTheEnd[k], 500, 500, 548, 516), 0.999) << "frame " << 24 + k;
  }
}

// A plate moving vertically several of its (small) heights per interval: the second pass measures
// both axes in the larger side of the boxes, so 9 heights of a 24 x 8 plate are 3 sides.
TEST(BoxTimeline, CoversPlatesMovingVertically) {
  for (const double heights : {3.0, 6.0, 9.0}) {
    SCOPED_TRACE(std::to_string(heights) + " heights per interval");
    const Mover mover{kPlate, 500, 200, 24, 8, 0, heights * 8 / kStep};
    BoxTimeline timeline(TimelineConfig{}, 4096, 2160);
    const auto masks = runPerfectDetector(timeline, {mover}, 8);
    int uncovered = 0;
    for (size_t f = 0; f < masks.size(); ++f) {
      if (coverage(masks[f], mover.at(static_cast<int64_t>(f))) < 0.8) ++uncovered;
    }
    EXPECT_EQ(uncovered, 0);
    for (int64_t f = kStep; f < 8 * kStep; ++f) {
      if (f % kStep == 0) continue;
      EXPECT_GE(countOrigin(masks[static_cast<size_t>(f)], MaskOrigin::Interpolated), 1u) << "frame " << f;
    }
  }
}

// Two faces one width apart, driving through each other within their first interval: with no
// velocity yet the first pass pairs each with the other's new box (centres 1 side apart, while their
// own are 2 sides away). A first pairing keeps both end boxes over the interval, so both stay covered
// by the positions of the masks alone (no margins), and with the default margins.
TEST(BoxTimeline, CoversTwoNearbyTargetsPairedTheWrongWayRound) {
  const Mover right{kFace, 400, 100, 20, 20, 40.0 / kStep, 0};
  const Mover left{kFace, 420, 100, 20, 20, -40.0 / kStep, 0};
  for (const TimelineConfig& config : {exact(), TimelineConfig{}}) {
    SCOPED_TRACE(config.expandFraction == 0 ? "no margins" : "default margins");
    BoxTimeline timeline(config, 1920, 1080);
    const auto masks = runPerfectDetector(timeline, {right, left}, 3);
    for (size_t f = 0; f < masks.size(); ++f) {
      const auto frame = static_cast<int64_t>(f);
      EXPECT_GE(coverage(masks[f], right.at(frame)), 0.8) << "right-moving face, frame " << f;
      EXPECT_GE(coverage(masks[f], left.at(frame)), 0.8) << "left-moving face, frame " << f;
    }
  }
}

// A first pairing (no velocity yet) is interpolated and holds both end boxes; once there is a
// velocity, a first-pass match is interpolated alone.
TEST(BoxTimeline, AFirstPairingHoldsBothEndBoxes) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  const auto first = timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));
  ASSERT_EQ(first[6].size(), 3u);  // frame 6
  expectBox(first[6][0], 106, 100, 126, 120, MaskOrigin::Interpolated);
  expectBox(first[6][1], 100, 100, 120, 120, MaskOrigin::Extended);
  expectBox(first[6][2], 112, 100, 132, 120, MaskOrigin::Extended);
  const auto second = timeline.advance(frames(12, 24), key(24, {face(124, 100, 144, 120)}));
  ASSERT_EQ(second[6].size(), 1u);  // frame 18
  expectBox(second[6][0], 118, 100, 138, 120, MaskOrigin::Interpolated);
}

// A target that turns round after keyframe 12, is missed at 24 and found again at 36: the ghost's
// guessed position (it kept going) is wrong, the interpolation from its last real detection is right.
TEST(BoxTimeline, AMatchedGhostIsAlsoInterpolatedFromItsLastDetection) {
  BoxTimeline timeline(exact(), 1000, 1000);
  timeline.advance({}, key(0, {face(100, 100, 120, 120)}));
  timeline.advance(frames(0, 12), key(12, {face(112, 100, 132, 120)}));  // v = 1 px per frame
  timeline.advance(frames(12, 24), key(24, {}));  // really back at 100 by now; the ghost guesses 124
  const auto masks = timeline.advance(frames(24, 36), key(36, {face(88, 100, 108, 120)}));
  ASSERT_EQ(masks.size(), 12u);
  for (size_t k = 0; k < masks.size(); ++k) {
    const float x = 100 - static_cast<float>(k);
    EXPECT_TRUE(covers(masks[k], x, 100, x + 20, 120)) << "frame " << 24 + k;
  }
  // From 112 at index 12 to 88 at index 36: frame 30 is 18 of 24 frames along.
  ASSERT_FALSE(masks[6].empty());
  expectBox(masks[6].back(), 94, 100, 114, 120, MaskOrigin::Interpolated);

  // The reviewer's probe: a 40 x 14 plate at 1 to 6 px per frame turning round after keyframe 12,
  // missed at 24, found at 36. Every frame from the re-detection back to the missed keyframe is covered.
  for (const double v : {1.0, 2.0, 3.0, 4.0, 6.0}) {
    SCOPED_TRACE("v = " + std::to_string(v) + " px per frame");
    const auto x = [v](int64_t f) { return 600 + v * static_cast<double>(f <= kStep ? f : 2 * kStep - f); };
    const auto plate = [&](int64_t f) {
      return Box{kPlate, static_cast<float>(x(f)), 500, static_cast<float>(x(f) + 40), 514, 0.9f};
    };
    BoxTimeline probe(TimelineConfig{}, 1920, 1080);
    probe.advance({}, key(0, {plate(0)}));
    probe.advance(frames(0, kStep), key(kStep, {plate(kStep)}));
    probe.advance(frames(kStep, 2 * kStep), key(2 * kStep, {}));
    const auto found = probe.advance(frames(2 * kStep, 3 * kStep), key(3 * kStep, {plate(3 * kStep)}));
    for (size_t k = 0; k < found.size(); ++k) {
      EXPECT_GE(coverage(found[k], plate(2 * kStep + static_cast<int64_t>(k))), 0.8) << "frame " << 2 * kStep + k;
    }
  }
}

// The reviewer's probe: a 40 x 14 plate at v px per frame that stops or turns round right after
// keyframe 12, is missed at 24 and found again at 36 (or missed at 36 too). The masks of the missed
// interval are made before the re-detection exists, so only the swept region can cover it.
TEST(BoxTimeline, CoversATargetThatStopsOrTurnsRoundWhileMissed) {
  struct Case {
    bool turns;
    bool missedTwice;
  };
  for (const Case c : {Case{false, false}, Case{true, false}, Case{true, true}}) {
    for (const double v : {1.0, 2.0, 3.0, 4.0, 6.0}) {
      SCOPED_TRACE(std::string(c.turns ? "turns round" : "stops") + (c.missedTwice ? ", missed twice" : "") +
                   ", v = " + std::to_string(v) + " px per frame");
      const auto x = [&](int64_t f) {
        if (f <= kStep) return 600 + v * static_cast<double>(f);
        return 600 + v * static_cast<double>(c.turns ? 2 * kStep - f : kStep);
      };
      const auto plate = [&](int64_t f) {
        return Box{kPlate, static_cast<float>(x(f)), 500, static_cast<float>(x(f) + 40), 514, 0.9f};
      };
      BoxTimeline timeline(TimelineConfig{}, 1920, 1080);
      timeline.advance({}, key(0, {plate(0)}));
      std::vector<std::vector<MaskBox>> masks = timeline.advance(frames(0, kStep), key(kStep, {plate(kStep)}));
      for (auto& m : timeline.advance(frames(kStep, 2 * kStep), key(2 * kStep, {}))) masks.push_back(std::move(m));
      const std::vector<Box> last = c.missedTwice ? std::vector<Box>{} : std::vector<Box>{plate(3 * kStep)};
      for (auto& m : timeline.advance(frames(2 * kStep, 3 * kStep), key(3 * kStep, last))) masks.push_back(std::move(m));
      ASSERT_EQ(masks.size(), static_cast<size_t>(3 * kStep));
      for (size_t f = 0; f < masks.size(); ++f) {
        EXPECT_GE(coverage(masks[f], plate(static_cast<int64_t>(f))), 0.8) << "frame " << f;
      }
    }
  }
}

}  // namespace
}  // namespace vc
