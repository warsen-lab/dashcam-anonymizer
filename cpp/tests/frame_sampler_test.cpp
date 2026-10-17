#include "sampler/frame_sampler.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace vc {
namespace {

double ptsAt60Fps(int index) { return index * 1000.0 / 60.0; }

struct Kept {
  int index;
  SampleReason reason;
};

// Offers frames [0, count) at 60 fps with the given per-frame scores.
template <typename Motion, typename DiffToKept>
std::vector<Kept> offerFrames(FrameSampler& sampler, int count, Motion motion,
                              DiffToKept diffToKept) {
  std::vector<Kept> kept;
  for (int i = 0; i < count; ++i) {
    const SampleReason reason = sampler.offer(ptsAt60Fps(i), motion(i), diffToKept(i));
    if (reason != SampleReason::None) kept.push_back({i, reason});
  }
  return kept;
}

std::vector<int> indicesOf(const std::vector<Kept>& kept) {
  std::vector<int> indices;
  for (const Kept& k : kept) indices.push_back(k.index);
  return indices;
}

double still(int) { return 0.0; }
double different(int) { return 1.0; }

TEST(FrameSampler, RejectsBaseFpsBelowFive) {
  SamplerConfig config;
  config.baseFps = 4.9;
  EXPECT_THROW(FrameSampler{config}, std::invalid_argument);
}

TEST(FrameSampler, RejectsMaxFpsBelowBaseFps) {
  SamplerConfig config;
  config.baseFps = 10.0;
  config.maxFps = 8.0;
  EXPECT_THROW(FrameSampler{config}, std::invalid_argument);
}

TEST(FrameSampler, RejectsEmptyClip) {
  SamplerConfig config;
  config.clip = ClipWindow{0.0, 0.0};
  EXPECT_THROW(FrameSampler{config}, std::invalid_argument);
}

TEST(FrameSampler, KeepsEveryTwelfthFrameAt60FpsWhenStill) {
  FrameSampler sampler(SamplerConfig{});

  const std::vector<Kept> kept = offerFrames(sampler, 120, still, different);

  EXPECT_EQ(indicesOf(kept), (std::vector<int>{0, 12, 24, 36, 48, 60, 72, 84, 96, 108}));
  for (const Kept& k : kept) EXPECT_EQ(k.reason, SampleReason::Base);
}

TEST(FrameSampler, SwitchesToMaxFpsWhileMotionIsHigh) {
  FrameSampler sampler(SamplerConfig{});  // base 5 fps, max 15 fps, threshold 0.05

  const std::vector<Kept> kept =
      offerFrames(sampler, 120, [](int i) { return i < 60 ? 0.0 : 0.5; }, different);

  std::vector<int> expected = {0, 12, 24, 36, 48, 60};
  for (int i = 64; i < 120; i += 4) expected.push_back(i);
  EXPECT_EQ(indicesOf(kept), expected);
  EXPECT_EQ(kept[5].reason, SampleReason::Base);    // frame 60 was due at the base rate
  EXPECT_EQ(kept[6].reason, SampleReason::Motion);  // frame 64 only due because of motion
}

TEST(FrameSampler, DedupSkipsFramesSimilarToLastKept) {
  SamplerConfig config;
  config.dedupThreshold = 0.01;
  FrameSampler sampler(config);

  const std::vector<Kept> kept =
      offerFrames(sampler, 30, still, [](int i) { return i < 24 ? 0.001 : 0.5; });

  EXPECT_EQ(indicesOf(kept), (std::vector<int>{0, 24}));
}

TEST(FrameSampler, ClipKeepsEveryFrameInsideWindowOnly) {
  SamplerConfig config;
  config.clip = ClipWindow{500.0, 100.0};
  FrameSampler sampler(config);

  const std::vector<Kept> kept = offerFrames(sampler, 60, still, different);

  EXPECT_EQ(indicesOf(kept), (std::vector<int>{30, 31, 32, 33, 34, 35}));
  for (const Kept& k : kept) EXPECT_EQ(k.reason, SampleReason::Clip);
}

TEST(FrameSampler, ReasonNamesMatchCsvValues) {
  EXPECT_STREQ(toString(SampleReason::None), "none");
  EXPECT_STREQ(toString(SampleReason::Base), "base");
  EXPECT_STREQ(toString(SampleReason::Motion), "motion");
  EXPECT_STREQ(toString(SampleReason::Clip), "clip");
}

}  // namespace
}  // namespace vc
