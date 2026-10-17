#include "io/jpeg_writer.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "io/video_reader.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

std::vector<unsigned char> readBytes(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
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

TEST(JpegWriter, RejectsQscaleOutsideRange) {
  EXPECT_THROW(JpegWriter(1), std::invalid_argument);
  EXPECT_THROW(JpegWriter(32), std::invalid_argument);
}

}  // namespace
}  // namespace vc
