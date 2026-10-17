#include "io/video_reader.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

TEST(VideoReader, SkipsCorruptPacketsAndCountsThem) {
  // Overwrite the length prefix of the first NAL unit (right after the "mdat" fourcc) so the
  // decoder rejects the first access unit.
  std::vector<char> bytes;
  {
    std::ifstream in(test::basicVideo(), std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  const std::string fourcc = "mdat";
  const auto mdat = std::search(bytes.begin(), bytes.end(), fourcc.begin(), fourcc.end());
  ASSERT_NE(mdat, bytes.end());
  const std::size_t prefix = static_cast<std::size_t>(mdat - bytes.begin()) + fourcc.size();
  ASSERT_LT(prefix + 4, bytes.size());
  std::fill_n(bytes.begin() + static_cast<std::ptrdiff_t>(prefix), 4, static_cast<char>(0xFF));
  const std::filesystem::path corrupt = test::tmpDir() / "basic_corrupt_first_nal.mp4";
  std::ofstream(corrupt, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));

  VideoReader reader(corrupt.string(), HwAccel::None);
  int64_t count = 0;
  EXPECT_NO_THROW({
    while (reader.next() != nullptr) ++count;
  });

  EXPECT_GE(reader.decodeErrors(), 1);
  EXPECT_GE(count, 1);
  EXPECT_EQ(reader.next(), nullptr);
}

TEST(VideoReader, ParsesHwAccelNames) {
  EXPECT_EQ(parseHwAccel("none"), HwAccel::None);
  EXPECT_EQ(parseHwAccel("videotoolbox"), HwAccel::VideoToolbox);
  EXPECT_EQ(parseHwAccel("auto"), HwAccel::Auto);
  EXPECT_THROW(parseHwAccel("cuda"), std::invalid_argument);
}

}  // namespace
}  // namespace vc
