#include "vc_redact/detect_rate.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/vc_error.h"

namespace vc {
namespace {

struct StrideCase {
  double videoFps;
  double detectFps;
  int64_t stride;
};

// stride = max(1, floor(video fps / detect_fps + 0.5)) for the recommended rates (spec 2026-10-19 §2) on
// common frame rates, including the 59.94 fps of the dashcam segments.
TEST(DetectRate, StrideFollowsTheVideoFrameRate) {
  const double ntsc60 = 60000.0 / 1001.0;
  const double film = 24000.0 / 1001.0;
  const std::vector<StrideCase> cases = {
      {60, 5, 12},     {60, 10, 6},     {60, 15, 4},     {60, 30, 2},     {60, 60, 1},
      {ntsc60, 5, 12}, {ntsc60, 10, 6}, {ntsc60, 15, 4}, {ntsc60, 30, 2}, {ntsc60, 60, 1},
      {30, 5, 6},      {30, 10, 3},     {30, 15, 2},     {30, 30, 1},
      {25, 5, 5},      {25, 10, 3},     {25, 15, 2},     {25, 25, 1},     {25, 30, 1},
      {film, 5, 5},    {film, 10, 2},   {film, 15, 2},   {film, 30, 1},
  };
  for (const StrideCase& c : cases) {
    const DetectSchedule s = detectSchedule(c.videoFps, c.detectFps);
    EXPECT_EQ(s.stride, c.stride) << c.videoFps << " fps video, detect_fps " << c.detectFps;
    EXPECT_DOUBLE_EQ(s.videoFps, c.videoFps);
    EXPECT_DOUBLE_EQ(s.actualFps, c.videoFps / static_cast<double>(c.stride));
  }
}

// The spec's example: 10 on a 25 fps video is every third frame, 8.3 detections a second.
TEST(DetectRate, TenOnATwentyFiveFpsVideoIsEveryThirdFrame) {
  const DetectSchedule s = detectSchedule(25, 10);
  EXPECT_EQ(s.stride, 3);
  EXPECT_NEAR(s.actualFps, 8.333, 1e-3);
  EXPECT_FALSE(s.aboveVideoRate);
}

// A request above the video frame rate detects on every frame and says so; exactly the video rate is not above it.
TEST(DetectRate, ARequestAboveTheVideoRateDetectsEveryFrame) {
  const DetectSchedule above = detectSchedule(25, 30);
  EXPECT_EQ(above.stride, 1);
  EXPECT_DOUBLE_EQ(above.actualFps, 25);
  EXPECT_TRUE(above.aboveVideoRate);
  const DetectSchedule equal = detectSchedule(30, 30);
  EXPECT_EQ(equal.stride, 1);
  EXPECT_FALSE(equal.aboveVideoRate);
  EXPECT_TRUE(detectSchedule(60000.0 / 1001.0, 60).aboveVideoRate);
}

TEST(DetectRate, AcceptsFiveToSixtyAndNothingElse) {
  EXPECT_EQ(detectSchedule(60, 5).stride, 12);
  EXPECT_EQ(detectSchedule(60, 60).stride, 1);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double bad : {4.999, 60.001, 0.0, -10.0, nan, inf}) {
    EXPECT_THROW(detectSchedule(60, bad), std::invalid_argument) << bad;
  }
  for (const double bad : {0.0, -30.0, nan, inf, 2e6}) {
    EXPECT_THROW(detectSchedule(bad, 10), std::invalid_argument) << bad;
    EXPECT_THROW(everyFrameSchedule(bad), std::invalid_argument) << bad;
  }
}

TEST(DetectRate, EveryFrameScheduleHasStrideOne) {
  const DetectSchedule s = everyFrameSchedule(29.97);
  EXPECT_EQ(s.stride, 1);
  EXPECT_DOUBLE_EQ(s.actualFps, 29.97);
  EXPECT_FALSE(s.aboveVideoRate);
}

TEST(DetectRate, KeyframesAreEveryStrideFramesByIndex) {
  const DetectSchedule s = detectSchedule(60, 10);
  std::vector<int64_t> keys;
  for (int64_t i = 0; i < 20; ++i) {
    if (onSchedule(s, i)) keys.push_back(i);
  }
  EXPECT_EQ(keys, (std::vector<int64_t>{0, 6, 12, 18}));
}

TEST(DetectRate, TheFrameRateFallsBackFromTheAverageToTheBaseRate) {
  EXPECT_DOUBLE_EQ(streamFrameRate(AVRational{60000, 1001}, AVRational{30, 1}, "a.mp4"), 60000.0 / 1001.0);
  EXPECT_DOUBLE_EQ(streamFrameRate(AVRational{0, 1}, AVRational{25, 1}, "a.mp4"), 25.0);
  EXPECT_DOUBLE_EQ(streamFrameRate(AVRational{25, 0}, AVRational{30000, 1001}, "a.mp4"), 30000.0 / 1001.0);
  try {
    streamFrameRate(AVRational{0, 1}, AVRational{0, 0}, "clip.mp4");
    ADD_FAILURE() << "accepted a stream without a frame rate";
  } catch (const VcError& e) {
    EXPECT_NE(std::string(e.what()).find("clip.mp4"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("frame rate"), std::string::npos) << e.what();
  }
}

}  // namespace
}  // namespace vc
