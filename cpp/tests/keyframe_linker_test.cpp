#include "track/keyframe_linker.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vc {
namespace {

constexpr int kFace = 0;

// Faces scoring below 0.5 need confirming (and plates below 0.15), as in BoxTimeline's tests.
LinkerConfig linker(int lookahead, int backfill) { return LinkerConfig{MatchParams{1.0, 6.0, {0.5f, 0.15f}}, lookahead, backfill}; }

Box face(float x1, float x2, float score = 0.9f) { return Box{kFace, x1, 100, x2, 180, score}; }

// Keyframe n at frame 6 n, 60 fps.
Keyframe key(int64_t n, std::vector<Box> boxes) { return Keyframe{{6 * n, 6 * n * 1000.0 / 60.0}, std::move(boxes)}; }

// Every keyframe that comes out, in order: keyframes[n] is keyframe n, then finish().
std::vector<Keyframe> run(const LinkerConfig& c, const std::vector<std::vector<Box>>& keyframes) {
  KeyframeLinker l(c);
  std::vector<Keyframe> out;
  for (size_t n = 0; n < keyframes.size(); ++n) {
    for (Keyframe& k : l.push(key(static_cast<int64_t>(n), keyframes[n]))) out.push_back(std::move(k));
  }
  for (Keyframe& k : l.finish()) out.push_back(std::move(k));
  return out;
}

TEST(KeyframeLinker, ReturnsEachKeyframeLookaheadKeyframesLater) {
  KeyframeLinker l(linker(3, 0));
  std::vector<size_t> counts;
  for (int n = 0; n < 6; ++n) counts.push_back(l.push(key(n, {face(100, 180)})).size());
  EXPECT_EQ(counts, (std::vector<size_t>{0, 0, 0, 1, 1, 1}));
  const std::vector<Keyframe> rest = l.finish();
  ASSERT_EQ(rest.size(), 3u);
  EXPECT_EQ(rest.front().time.index, 18);
}

TEST(KeyframeLinker, ATargetSeenOnEveryKeyframeIsOneMaskedTrack) {
  const std::vector<Keyframe> out = run(linker(3, 0), std::vector<std::vector<Box>>(6, {face(100, 180)}));
  ASSERT_EQ(out.size(), 6u);
  for (size_t n = 0; n < out.size(); ++n) {
    EXPECT_EQ(out[n].time.index, static_cast<int64_t>(6 * n));
    ASSERT_EQ(out[n].links.size(), 1u);
    EXPECT_EQ(out[n].links[0].track, 1);
    EXPECT_EQ(out[n].links[0].kind, LinkKind::Real);
    EXPECT_TRUE(out[n].links[0].masked);
  }
}

TEST(KeyframeLinker, BridgesUpToLookaheadMissedKeyframes) {
  // missed on keyframes 1-3 (W = 3), found again on keyframe 4, 40 px to the right
  const std::vector<Keyframe> out = run(linker(3, 0), {{face(100, 180)}, {}, {}, {}, {face(140, 220)}});
  for (int n = 1; n <= 3; ++n) {
    ASSERT_EQ(out[n].boxes.size(), 1u) << n;
    EXPECT_EQ(out[n].links[0].kind, LinkKind::Bridged);
    EXPECT_EQ(out[n].links[0].track, 1);
    EXPECT_TRUE(out[n].links[0].masked);
    EXPECT_NEAR(out[n].boxes[0].x1, 100 + 10 * n, 1e-3);
  }
  EXPECT_EQ(out[4].links[0].track, 1);
}

TEST(KeyframeLinker, DoesNotBridgeMoreThanLookahead) {
  const std::vector<Keyframe> out = run(linker(3, 0), {{face(100, 180)}, {}, {}, {}, {}, {face(140, 220)}});
  for (int n = 1; n <= 4; ++n) EXPECT_TRUE(out[n].boxes.empty()) << n;
  EXPECT_EQ(out[5].links[0].track, 2);
}

TEST(KeyframeLinker, AFoundAgainTrackIsPredictedAlongItsVelocity) {
  // low scores keep the loose pass out: only the velocity (10 px per frame) brings keyframe 4 within reach
  const float weak = 0.3f;
  const std::vector<Keyframe> out =
      run(linker(3, 0), {{face(0, 80, weak)}, {face(60, 140, weak)}, {}, {}, {face(240, 320, weak)}});
  EXPECT_EQ(out[4].links[0].track, out[0].links[0].track);
  ASSERT_EQ(out[2].boxes.size(), 1u);
  EXPECT_NEAR(out[2].boxes[0].x1, 120, 1e-3);  // frame 12, a third of the way from frame 6 to frame 24
}

TEST(KeyframeLinker, AnUnconfirmedTrackIsNeitherBridgedNorMasked) {
  const float weak = 0.3f;
  const std::vector<Keyframe> out = run(linker(3, 2), {{face(100, 180, weak)}, {}, {face(100, 180, weak)}});
  EXPECT_TRUE(out[1].boxes.empty());
  EXPECT_FALSE(out[0].links[0].masked);
  EXPECT_FALSE(out[2].links[0].masked);
  EXPECT_NE(out[2].links[0].track, out[0].links[0].track);
}

TEST(KeyframeLinker, ALowScoreTrackSeenOnTwoConsecutiveKeyframesIsMaskedOnBoth) {
  const float weak = 0.3f;
  const std::vector<Keyframe> out = run(linker(3, 0), {{}, {face(100, 180, weak)}, {face(100, 180, weak)}});
  EXPECT_TRUE(out[1].links[0].masked);
  EXPECT_TRUE(out[2].links[0].masked);
}

TEST(KeyframeLinker, BackfillsAHighScoreTrackAlongItsVelocity) {
  // first seen on keyframe 3 (frame 18), then 12 px further right on keyframe 4: 2 px per frame
  const std::vector<Keyframe> out = run(linker(3, 2), {{}, {}, {}, {face(100, 180)}, {face(112, 192)}});
  EXPECT_TRUE(out[0].boxes.empty());
  ASSERT_EQ(out[1].boxes.size(), 1u);
  ASSERT_EQ(out[2].boxes.size(), 1u);
  EXPECT_EQ(out[2].links[0].kind, LinkKind::Backfilled);
  EXPECT_EQ(out[2].links[0].track, out[3].links[0].track);
  EXPECT_NEAR(out[2].boxes[0].x1, 88, 1e-3);  // 6 frames earlier
  EXPECT_NEAR(out[1].boxes[0].x1, 76, 1e-3);  // 12 frames earlier
}

TEST(KeyframeLinker, ABackfilledBoxMovesAtMostHalfASidePerInterval) {
  // 20 px per frame: 120 px back on keyframe 2 is capped at 40 (half of 80), 240 px on keyframe 1 at 80
  const std::vector<Keyframe> out = run(linker(3, 2), {{}, {}, {}, {face(400, 480)}, {face(520, 600)}});
  EXPECT_NEAR(out[2].boxes[0].x1, 360, 1e-3);
  EXPECT_NEAR(out[1].boxes[0].x1, 320, 1e-3);
}

TEST(KeyframeLinker, BackfillsALowScoreTrackOnceConfirmed) {
  const float weak = 0.3f;
  const std::vector<Keyframe> out =
      run(linker(3, 2), {{}, {}, {}, {face(100, 180, weak)}, {face(100, 180, weak)}});
  EXPECT_TRUE(out[0].boxes.empty());
  ASSERT_EQ(out[1].boxes.size(), 1u);
  EXPECT_EQ(out[1].links[0].kind, LinkKind::Backfilled);
  EXPECT_NEAR(out[1].boxes[0].x1, 100, 1e-3);  // standing still
  EXPECT_TRUE(out[3].links[0].masked);
}

TEST(KeyframeLinker, DoesNotBackfillBeforeTheFirstKeyframe) {
  const std::vector<Keyframe> out = run(linker(3, 2), {{}, {face(100, 180)}});
  ASSERT_EQ(out[0].boxes.size(), 1u);
  EXPECT_EQ(out[0].links[0].kind, LinkKind::Backfilled);
}

TEST(KeyframeLinker, FinishStartsAfresh) {
  KeyframeLinker l(linker(2, 0));
  l.push(key(0, {face(100, 180)}));
  l.push(key(1, {face(500, 580)}));
  EXPECT_EQ(l.finish().size(), 2u);
  l.push(key(0, {face(100, 180)}));
  const std::vector<Keyframe> out = l.finish();
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].links[0].track, 1);
}

TEST(KeyframeLinker, RejectsBadSettingsAndKeyframes) {
  EXPECT_THROW(KeyframeLinker{linker(0, 0)}, std::invalid_argument);
  EXPECT_THROW(KeyframeLinker{linker(3, 3)}, std::invalid_argument);
  EXPECT_THROW(KeyframeLinker{linker(3, -1)}, std::invalid_argument);
  KeyframeLinker l(linker(2, 0));
  l.push(key(1, {}));
  EXPECT_THROW(l.push(key(1, {})), std::invalid_argument);
  EXPECT_THROW(l.push(key(2, {Box{kFace, NAN, 0, 1, 1, 0.9f}})), std::invalid_argument);
  EXPECT_NO_THROW(l.push(key(2, {})));
}

}  // namespace
}  // namespace vc
