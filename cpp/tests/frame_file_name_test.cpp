#include "io/frame_file_name.h"

#include <gtest/gtest.h>

namespace vc {
namespace {

TEST(FrameFileName, PadsTheFrameNumberToSixDigits) {
  EXPECT_EQ(frameFileName(0), "000000.jpg");
  EXPECT_EQ(frameFileName(7), "000007.jpg");
  EXPECT_EQ(frameFileName(123456), "123456.jpg");
}

TEST(FrameFileName, NeverTruncatesALongFrameNumber) {
  EXPECT_EQ(frameFileName(1234567), "1234567.jpg");
  EXPECT_EQ(frameFileName(4'000'000'000LL), "4000000000.jpg");
}

}  // namespace
}  // namespace vc
