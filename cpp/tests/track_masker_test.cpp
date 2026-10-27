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
Result run(const TrackMaskerConfig& c, const std::vector<std::vector<Box>>& keyframes, int tail = 6, bool bare = false,
           TrackMasker* reuse = nullptr) {
  TrackMasker own(c, 1920, 1080);
  TrackMasker& m = reuse ? *reuse : own;
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

std::vector<FrameTime> frames(int64_t from, int64_t to) {
  std::vector<FrameTime> f;
  for (int64_t i = from; i < to; ++i) f.push_back(at(i));
  return f;
}

// What a sequence of calls handed back, summed up.
struct Tally {
  std::vector<std::vector<MaskBox>> masks;
  size_t confirmed = 0;
  void add(TrackingOutput o) {
    for (auto& m : o.masks) masks.push_back(std::move(m));
    confirmed += o.confirmed.size();
  }
};

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

TEST(TrackMasker, WithLookaheadEachMaskFollowsItsFramesInterpolatedPosition) {
  // The face on keyframe n (frame 6 n) is at x1 = 100 + 6 n^2, so the steps grow (6, 18, 30, ...): on a straight line any
  // interval paired with the wrong keyframes would interpolate to the same x1, here it does not. Only the frames before
  // the last keyframe are checked; past it the masker holds and extrapolates, which this test is not about.
  const auto x1 = [](int n) { return 100.0 + 6.0 * n * n; };
  std::vector<std::vector<Box>> script;
  for (int n = 0; n < 10; ++n) {
    script.push_back({face(static_cast<float>(x1(n)), static_cast<float>(x1(n) + 80))});
  }
  for (const auto& config : {masker(3, 0), masker(2, 1), masker(0, 0)}) {
    const Result r = run(config, script, 4);
    ASSERT_EQ(r.masks.size(), 58u);
    for (int f = 0; f < 54; ++f) {  // frame 54 is the last keyframe
      ASSERT_EQ(r.masks[f].size(), 1u) << f;
      const int n = f / 6;
      const double expected = x1(n) + (x1(n + 1) - x1(n)) * (f - 6 * n) / 6.0;
      EXPECT_NEAR(r.masks[f][0].x1, expected, 1e-3) << "lookahead " << config.lookahead << " frame " << f;
    }
  }
}

TEST(TrackMasker, FinishWithoutKeyframesReturnsNothing) {
  for (const int lookahead : {0, 3}) {
    TrackMasker m(masker(lookahead, 0), 1920, 1080);
    const TrackingOutput o = m.finish({});
    EXPECT_TRUE(o.masks.empty()) << lookahead;
    EXPECT_TRUE(o.confirmed.empty()) << lookahead;
  }
}

TEST(TrackMasker, FewerKeyframesThanTheLookaheadStillMaskEveryFrame) {
  const Result r = run(masker(3, 0), {{face(100, 180)}}, 6);
  ASSERT_EQ(r.masks.size(), 6u);
  for (size_t f = 0; f < r.masks.size(); ++f) EXPECT_EQ(r.masks[f].size(), 1u) << f;
  EXPECT_EQ(r.confirmed.size(), 1u);
}

TEST(TrackMasker, LinkedAdvanceRejectsBadFramesAndStaysUsable) {
  struct Case {
    const char* name;
    std::vector<FrameTime> frames;
  };
  const std::vector<Case> cases{
      {"not starting at the previous keyframe", frames(1, 6)},
      {"no frames", {}},
      {"not strictly increasing", {at(6), at(8), at(7)}},
      {"repeating an index", {at(6), at(6), at(7)}},
      {"reaching the next keyframe", frames(6, 13)},
  };
  for (const Case& c : cases) {
    TrackMasker m(masker(3, 0), 1920, 1080);
    Tally t;
    t.add(m.advance({}, Keyframe{at(0), {face(100, 180)}}));
    t.add(m.advance(frames(0, 6), Keyframe{at(6), {face(100, 180)}}));
    // frames of the interval 6..12 go with the keyframe at 12.
    EXPECT_THROW(m.advance(c.frames, Keyframe{at(12), {face(100, 180)}}), std::invalid_argument) << c.name;
    t.add(m.advance(frames(6, 12), Keyframe{at(12), {face(100, 180)}}));
    t.add(m.advance(frames(12, 18), Keyframe{at(18), {face(100, 180)}}));
    t.add(m.finish(frames(18, 24)));
    ASSERT_EQ(t.masks.size(), 24u) << c.name;
    for (const auto& x : t.masks) EXPECT_EQ(x.size(), 1u) << c.name;
    EXPECT_EQ(t.confirmed, 4u) << c.name;
  }
}

TEST(TrackMasker, LinkedAdvanceRejectsFramesBeforeTheFirstKeyframeAndStaysUsable) {
  TrackMasker m(masker(3, 0), 1920, 1080);
  Tally t;
  EXPECT_THROW(m.advance(frames(0, 3), Keyframe{at(6), {face(100, 180)}}), std::invalid_argument);
  t.add(m.advance({}, Keyframe{at(0), {face(100, 180)}}));
  t.add(m.advance(frames(0, 6), Keyframe{at(6), {face(100, 180)}}));
  t.add(m.finish(frames(6, 12)));
  EXPECT_EQ(t.masks.size(), 12u);
  EXPECT_EQ(t.confirmed, 2u);
}

TEST(TrackMasker, LinkedFinishRejectsBadFramesAndStaysUsable) {
  struct Case {
    const char* name;
    std::vector<FrameTime> frames;
  };
  const std::vector<Case> cases{
      {"not starting at the last keyframe", frames(7, 12)},
      {"not strictly increasing", {at(6), at(8), at(7)}},
  };
  for (const Case& c : cases) {
    TrackMasker m(masker(3, 0), 1920, 1080);
    Tally t;
    t.add(m.advance({}, Keyframe{at(0), {face(100, 180)}}));
    t.add(m.advance(frames(0, 6), Keyframe{at(6), {face(100, 180)}}));
    EXPECT_THROW(m.finish(c.frames), std::invalid_argument) << c.name;
    t.add(m.finish(frames(6, 12)));
    EXPECT_EQ(t.masks.size(), 12u) << c.name;
    EXPECT_EQ(t.confirmed, 2u) << c.name;
  }
  TrackMasker m(masker(3, 0), 1920, 1080);
  EXPECT_THROW(m.finish(frames(0, 3)), std::invalid_argument);  // no keyframe yet
}

TEST(TrackMasker, CanBeReusedAfterFinish) {
  const std::vector<std::vector<Box>> first{{face(100, 180)}, {face(130, 210)}, {}, {face(200, 280)}, {face(220, 300)}};
  const std::vector<std::vector<Box>> second{{face(400, 480)}, {face(430, 510)}, {face(460, 540)}};
  const TrackMaskerConfig c = masker(3, 0);
  TrackMasker reused(c, 1920, 1080);
  (void)run(c, first, 6, false, &reused);
  const Result again = run(c, second, 5, false, &reused);
  const Result fresh = run(c, second, 5);
  ASSERT_EQ(again.masks.size(), fresh.masks.size());
  ASSERT_EQ(again.masks.size(), 17u);
  for (size_t f = 0; f < fresh.masks.size(); ++f) {
    ASSERT_EQ(again.masks[f].size(), fresh.masks[f].size()) << f;
    for (size_t k = 0; k < fresh.masks[f].size(); ++k) EXPECT_EQ(again.masks[f][k].x1, fresh.masks[f][k].x1) << f;
  }
  EXPECT_EQ(again.confirmed, fresh.confirmed);
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
