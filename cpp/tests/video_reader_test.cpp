#include "io/video_reader.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

#include "common/vc_error.h"
#include "sampler/luma_thumb.h"
#include "test_media.h"

extern "C" {
#include <libavutil/frame.h>
}

namespace vc {
namespace {

TEST(VideoReader, ReadsEveryFrameWithTimestamps) {
  VideoReader reader(test::basicVideo().string(), HwAccel::None);
  EXPECT_EQ(reader.info().width, 640);
  EXPECT_EQ(reader.info().height, 360);
  EXPECT_NEAR(reader.info().fps, 30.0, 0.01);
  EXPECT_NEAR(reader.info().durationMs, 2000.0, 1.0);
  EXPECT_EQ(reader.info().codecName, "h264");

  int64_t count = 0;
  while (const VideoFrame* frame = reader.next()) {
    EXPECT_EQ(frame->index, count);
    EXPECT_NEAR(frame->ptsMs, static_cast<double>(count) * 1000.0 / 30.0, 1.0);
    const LumaPlane luma = frame->luma();
    EXPECT_EQ(luma.width, 640);
    EXPECT_EQ(luma.height, 360);
    EXPECT_GE(luma.stride, 640);
    ++count;
  }

  EXPECT_EQ(count, 60);
  EXPECT_EQ(reader.decodeErrors(), 0);
  EXPECT_FALSE(reader.usingHardware());
  EXPECT_EQ(reader.next(), nullptr);  // stays at end of stream
}

TEST(VideoReader, ConvertsTenBitVideoToEightBitNv12) {
  VideoReader reader(test::tenBitVideo().string(), HwAccel::None);

  int64_t count = 0;
  while (const VideoFrame* frame = reader.next()) {
    EXPECT_EQ(frame->av->format, AV_PIX_FMT_NV12);
    ++count;
  }

  EXPECT_EQ(count, 30);
}

TEST(VideoReader, ThrowsForMissingFile) {
  EXPECT_THROW(VideoReader((test::tmpDir() / "missing.mp4").string(), HwAccel::None), VcError);
}

TEST(VideoReader, ThrowsForFileWithoutVideo) {
  EXPECT_THROW(VideoReader(test::audioOnlyFile().string(), HwAccel::None), VcError);
}

TEST(VideoReader, VideoToolboxMatchesSoftwareDecode) {
  std::unique_ptr<VideoReader> hardware;
  try {
    hardware = std::make_unique<VideoReader>(test::basicVideo().string(), HwAccel::VideoToolbox);
  } catch (const VcError& e) {
    GTEST_SKIP() << e.what();
  }
  VideoReader software(test::basicVideo().string(), HwAccel::None);

  int64_t count = 0;
  while (const VideoFrame* sw = software.next()) {
    const LumaThumb expected = makeLumaThumb(sw->luma());
    const VideoFrame* hw = hardware->next();
    ASSERT_NE(hw, nullptr);
    EXPECT_EQ(hw->av->format, AV_PIX_FMT_NV12);
    EXPECT_LT(meanAbsDiff(makeLumaThumb(hw->luma()), expected), 0.01);
    ++count;
  }

  EXPECT_EQ(hardware->next(), nullptr);
  EXPECT_EQ(count, 60);
  EXPECT_TRUE(hardware->usingHardware());
}

TEST(VideoReader, ParsesHwAccelNames) {
  EXPECT_EQ(parseHwAccel("none"), HwAccel::None);
  EXPECT_EQ(parseHwAccel("videotoolbox"), HwAccel::VideoToolbox);
  EXPECT_EQ(parseHwAccel("auto"), HwAccel::Auto);
  EXPECT_THROW(parseHwAccel("cuda"), std::invalid_argument);
}

}  // namespace
}  // namespace vc
