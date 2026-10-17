#include "io/jpeg_writer.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "io/video_reader.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

std::vector<unsigned char> readBytes(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

double lumaMean(const LumaPlane& luma) {
  double sum = 0;
  for (int y = 0; y < luma.height; ++y) {
    const uint8_t* row = luma.data + static_cast<std::ptrdiff_t>(y) * luma.stride;
    for (int x = 0; x < luma.width; ++x) sum += row[x];
  }
  return sum / (static_cast<double>(luma.width) * luma.height);
}

TEST(JpegWriter, WritesJpegThatDecodesToSameSize) {
  VideoReader reader(test::basicVideo().string(), HwAccel::None);
  const VideoFrame* frame = reader.next();
  ASSERT_NE(frame, nullptr);
  const fs::path path = test::tmpDir() / "jpeg_writer_basic.jpg";

  JpegWriter().write(*frame->av, path);

  const std::vector<unsigned char> bytes = readBytes(path);
  ASSERT_GT(bytes.size(), 4u);
  EXPECT_EQ(bytes[0], 0xFF);
  EXPECT_EQ(bytes[1], 0xD8);  // SOI marker
  EXPECT_EQ(bytes[bytes.size() - 2], 0xFF);
  EXPECT_EQ(bytes[bytes.size() - 1], 0xD9);  // EOI marker
  VideoReader decoded(path.string(), HwAccel::None);
  EXPECT_EQ(decoded.info().width, 640);
  EXPECT_EQ(decoded.info().height, 360);
  EXPECT_NE(decoded.next(), nullptr);
}

TEST(JpegWriter, LowerQscaleGivesLargerFile) {
  VideoReader reader(test::basicVideo().string(), HwAccel::None);
  const VideoFrame* frame = reader.next();
  ASSERT_NE(frame, nullptr);
  const fs::path best = test::tmpDir() / "jpeg_writer_q2.jpg";
  const fs::path worse = test::tmpDir() / "jpeg_writer_q20.jpg";

  JpegWriter(2).write(*frame->av, best);
  JpegWriter(20).write(*frame->av, worse);

  EXPECT_GT(fs::file_size(best), 2 * fs::file_size(worse));
}

TEST(JpegWriter, WritesNv12FramesFromTenBitVideo) {
  VideoReader reader(test::tenBitVideo().string(), HwAccel::None);
  const VideoFrame* frame = reader.next();
  ASSERT_NE(frame, nullptr);
  const fs::path path = test::tmpDir() / "jpeg_writer_ten_bit.jpg";

  JpegWriter().write(*frame->av, path);

  VideoReader decoded(path.string(), HwAccel::None);
  EXPECT_EQ(decoded.info().width, 640);
  EXPECT_EQ(decoded.info().height, 360);
}

TEST(JpegWriter, ConvertsLimitedRangeLumaToFullRange) {
  VideoReader reader(test::basicVideo().string(), HwAccel::None);
  const VideoFrame* frame = reader.next();
  ASSERT_NE(frame, nullptr);
  const double sourceMean = lumaMean(frame->luma());
  const fs::path path = test::tmpDir() / "jpeg_writer_range.jpg";

  JpegWriter().write(*frame->av, path);

  VideoReader decoded(path.string(), HwAccel::None);
  const VideoFrame* jpegFrame = decoded.next();
  ASSERT_NE(jpegFrame, nullptr);
  // The H.264 source is limited range (16..235); the JPEG is full range (0..255).
  EXPECT_NEAR(lumaMean(jpegFrame->luma()), (sourceMean - 16.0) * 255.0 / 219.0, 1.5);
}

TEST(JpegWriter, ThrowsVcErrorForUnwritablePath) {
  VideoReader reader(test::basicVideo().string(), HwAccel::None);
  const VideoFrame* frame = reader.next();
  ASSERT_NE(frame, nullptr);

  EXPECT_THROW(JpegWriter().write(*frame->av, test::tmpDir() / "no_such_dir" / "frame.jpg"), VcError);
}

TEST(JpegWriter, ReusesOneWriterAcrossFrames) {
  VideoReader reader(test::basicVideo().string(), HwAccel::None);
  JpegWriter writer;
  // The first frames include non-key frames, so the writer is reused across picture types.
  for (int i = 0; i < 3; ++i) {
    const VideoFrame* frame = reader.next();
    ASSERT_NE(frame, nullptr);
    const fs::path path = test::tmpDir() / ("jpeg_writer_reuse_" + std::to_string(i) + ".jpg");

    writer.write(*frame->av, path);

    const std::vector<unsigned char> bytes = readBytes(path);
    ASSERT_GT(bytes.size(), 4u);
    EXPECT_EQ(bytes[0], 0xFF);
    EXPECT_EQ(bytes[1], 0xD8);
    EXPECT_EQ(bytes[bytes.size() - 2], 0xFF);
    EXPECT_EQ(bytes[bytes.size() - 1], 0xD9);
  }
}

TEST(JpegWriter, RejectsQscaleOutsideRange) {
  EXPECT_THROW(JpegWriter(1), std::invalid_argument);
  EXPECT_THROW(JpegWriter(32), std::invalid_argument);
}

}  // namespace
}  // namespace vc
