#include "track/track_masker.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace vc {
namespace {

constexpr int kFace = 0;

// No margins and no smoothing (each mask is its box); faces below 0.5 and plates below 0.15 need confirming.
TrackMaskerConfig masker(int lookahead, int backfill) {
  TrackMaskerConfig c;
  c.timeline.expandFraction = 0;
  c.timeline.motionExpandFraction = 0;
  c.timeline.minBoxPx = 0;
  c.timeline.shrinkPerFrame = 1.0;
  c.timeline.confirmScore = {0.5f, 0.15f};
  c.lookahead = lookahead;
  c.backfill = backfill;
  return c;
}

Box face(float x1, float x2, float score = 0.9f) { return Box{kFace, x1, 100, x2, 180, score}; }

struct Result {
  std::vector<std::vector<MaskBox>> masks;
  std::vector<std::vector<bool>> confirmed;
};

FrameTime at(int64_t i) { return {i, i * 1000.0 / 60.0}; }

// keyframes[n] on frame 6 n, then `tail` frames after the last keyframe; with a TrackMasker, or with a bare
// BoxTimeline when `bare`.
Result run(const TrackMaskerConfig& c, const std::vector<std::vector<Box>>& keyframes, int tail = 6, bool bare = false) {
  TrackMasker m(c, 1920, 1080);
  BoxTimeline t(c.timeline, 1920, 1080);
  bool started = false;
  Result r;
  const auto take = [&r](std::vector<std::vector<MaskBox>> masks, std::vector<std::vector<bool>> confirmed) {
    for (auto& x : masks) r.masks.push_back(std::move(x));
    for (auto& x : confirmed) r.confirmed.push_back(std::move(x));
  };
  std::vector<FrameTime> interval;
  for (size_t n = 0; n < keyframes.size(); ++n) {
    const auto index = static_cast<int64_t>(6 * n);
    const Keyframe k{at(index), keyframes[n]};
    if (bare) {
      TimelineOutput o = t.advance(interval, k);
      take(std::move(o.masks), started ? std::vector<std::vector<bool>>{o.confirmed} : std::vector<std::vector<bool>>{});
      started = true;
    } else {
      TrackingOutput o = m.advance(interval, k);
      take(std::move(o.masks), std::move(o.confirmed));
    }
    interval.clear();
    const int length = n + 1 < keyframes.size() ? 6 : tail;
    for (int64_t i = index; i < index + length; ++i) interval.push_back(at(i));
  }
  if (bare) {
    TimelineOutput o = t.finish(interval);
    take(std::move(o.masks), {o.confirmed});
  } else {
    TrackingOutput o = m.finish(interval);
    take(std::move(o.masks), std::move(o.confirmed));
  }
  return r;
}

TEST(TrackMasker, WithoutLookaheadItIsBoxTimeline) {
  const std::vector<std::vector<Box>> script{
      {face(100, 180)}, {face(130, 210), face(900, 980, 0.3f)}, {}, {}, {face(200, 280)}, {face(600, 680, 0.3f)}, {}};
  const Result masked = run(masker(0, 0), script);
  const Result bare = run(masker(0, 0), script, 6, true);
  ASSERT_EQ(masked.masks.size(), bare.masks.size());
  for (size_t f = 0; f < bare.masks.size(); ++f) {
    ASSERT_EQ(masked.masks[f].size(), bare.masks[f].size()) << f;
    for (size_t k = 0; k < bare.masks[f].size(); ++k) {
      const MaskBox& a = masked.masks[f][k];
      const MaskBox& b = bare.masks[f][k];
      EXPECT_EQ(a.x1, b.x1);
      EXPECT_EQ(a.x2, b.x2);
      EXPECT_EQ(a.origin, b.origin);
      EXPECT_EQ(a.track, b.track);
    }
  }
  EXPECT_EQ(masked.confirmed, bare.confirmed);
}

TEST(TrackMasker, WithLookaheadEveryFrameGetsItsMasksOnceInOrder) {
  const Result r = run(masker(3, 0), std::vector<std::vector<Box>>(10, {face(100, 180)}), 4);
  ASSERT_EQ(r.masks.size(), 58u);  // 9 intervals of 6 and a tail of 4
  for (const auto& m : r.masks) EXPECT_EQ(m.size(), 1u);
  EXPECT_EQ(r.confirmed.size(), 10u);
}

TEST(TrackMasker, BridgesTheGapBoxTimelineLeaves) {
  const std::vector<std::vector<Box>> script{{face(100, 180)}, {}, {}, {face(100, 180)}};
  const Result before = run(masker(0, 0), script);
  for (int f = 6; f < 18; ++f) EXPECT_TRUE(before.masks[f].empty()) << f;
  const Result after = run(masker(3, 0), script);
  for (int f = 0; f < 24; ++f) ASSERT_EQ(after.masks[f].size(), 1u) << f;
  EXPECT_EQ(after.masks[0][0].origin, MaskOrigin::Detected);
  for (int f = 1; f < 18; ++f) EXPECT_EQ(after.masks[f][0].origin, MaskOrigin::Bridged) << f;
  EXPECT_EQ(after.masks[18][0].origin, MaskOrigin::Detected);
}

TEST(TrackMasker, BackfillsBeforeTheFirstDetection) {
  const Result r = run(masker(3, 2), {{}, {}, {}, {face(100, 180)}});
  for (int f = 0; f < 6; ++f) EXPECT_TRUE(r.masks[f].empty()) << f;
  for (int f = 6; f < 18; ++f) {
    ASSERT_EQ(r.masks[f].size(), 1u) << f;
    EXPECT_EQ(r.masks[f][0].origin, MaskOrigin::Backfilled) << f;
  }
  EXPECT_EQ(r.masks[18][0].origin, MaskOrigin::Detected);
}

TEST(TrackMasker, ConfirmedListsTheRealBoxesOfEachKeyframe) {
  const Result r = run(masker(2, 1), {{face(100, 180), face(900, 980, 0.3f)}, {face(100, 180)}, {}, {face(500, 580, 0.3f)}});
  EXPECT_EQ(r.confirmed, (std::vector<std::vector<bool>>{{true, false}, {true}, {}, {false}}));
}

TEST(TrackMasker, RejectsALookaheadOrBackfillOutOfRange) {
  EXPECT_THROW(TrackMasker(masker(0, 1), 1920, 1080), std::invalid_argument);
  EXPECT_THROW(TrackMasker(masker(3, 3), 1920, 1080), std::invalid_argument);
  EXPECT_THROW(TrackMasker(masker(-1, 0), 1920, 1080), std::invalid_argument);
}

}  // namespace
}  // namespace vc
