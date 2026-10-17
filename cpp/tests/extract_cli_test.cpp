#include "vc_extract/extract_cli.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace vc {
namespace {

using Args = std::vector<std::string>;

TEST(ExtractCli, AppliesDefaults) {
  const Args args = {"--input", "trip.mp4", "--out", "frames"};

  const ExtractOptions options = parseExtractArgs(args);

  EXPECT_EQ(options.input, "trip.mp4");
  EXPECT_EQ(options.outRoot, "frames");
  EXPECT_DOUBLE_EQ(options.sampler.baseFps, 5.0);
  EXPECT_DOUBLE_EQ(options.sampler.maxFps, 15.0);
  EXPECT_DOUBLE_EQ(options.sampler.motionThreshold, 0.05);
  EXPECT_FALSE(options.sampler.dedupThreshold.has_value());
  EXPECT_FALSE(options.sampler.clip.has_value());
  EXPECT_EQ(options.hwAccel, HwAccel::Auto);
  EXPECT_EQ(options.jpegQscale, 2);
  EXPECT_FALSE(options.overwrite);
}

TEST(ExtractCli, ParsesEveryOption) {
  const Args args = {"--input", "a.mp4",  "--out",     "o",    "--fps",
                     "10",      "--max-fps", "30",     "--motion-threshold",
                     "0.1",     "--dedup", "0.01",     "--clip", "1.5,2",
                     "--hwaccel", "none",  "--jpeg-q", "5",    "--overwrite"};

  const ExtractOptions options = parseExtractArgs(args);

  EXPECT_DOUBLE_EQ(options.sampler.baseFps, 10.0);
  EXPECT_DOUBLE_EQ(options.sampler.maxFps, 30.0);
  EXPECT_DOUBLE_EQ(options.sampler.motionThreshold, 0.1);
  ASSERT_TRUE(options.sampler.dedupThreshold.has_value());
  EXPECT_DOUBLE_EQ(*options.sampler.dedupThreshold, 0.01);
  ASSERT_TRUE(options.sampler.clip.has_value());
  EXPECT_DOUBLE_EQ(options.sampler.clip->startMs, 1500.0);
  EXPECT_DOUBLE_EQ(options.sampler.clip->durationMs, 2000.0);
  EXPECT_EQ(options.hwAccel, HwAccel::None);
  EXPECT_EQ(options.jpegQscale, 5);
  EXPECT_TRUE(options.overwrite);
}

TEST(ExtractCli, RejectsInvalidArguments) {
  const std::vector<Args> invalid = {
      {"--out", "o"},                                           // missing --input
      {"--input", "a.mp4"},                                     // missing --out
      {"--input", "a.mp4", "--out", "o", "--fps", "4"},         // below 5 fps
      {"--input", "a.mp4", "--out", "o", "--fps", "5x"},        // not a number
      {"--input", "a.mp4", "--out", "o", "--max-fps", "4"},     // below base fps
      {"--input", "a.mp4", "--out", "o", "--clip", "1.5"},      // no duration
      {"--input", "a.mp4", "--out", "o", "--clip", "1,0"},      // empty window
      {"--input", "a.mp4", "--out", "o", "--hwaccel", "cuda"},  // unsupported here
      {"--input", "a.mp4", "--out", "o", "--jpeg-q", "1"},      // out of range
      {"--input", "a.mp4", "--out", "o", "--bogus", "1"},       // unknown option
      {"--input"},                                              // missing value
      {"--input", "a.mp4", "--out", "--overwrite"},             // option where a value belongs
      {"--input", "a.mp4", "--out", "o", "--jpeg-q", "32"},     // above the range
      {"--input", "a.mp4", "--out", "o", "--jpeg-q", "x"},      // not an integer
      {"--input", "a.mp4", "--out", "o", "--clip", "x,1"},      // start is not a number
      {"--input", "a.mp4", "--out", "o", "--clip", "-1,2"},     // negative start
      {"--input", "a.mp4", "--out", "o", "--fps", "nan"},       // not finite
      {"--input", "a.mp4", "--out", "o", "--motion-threshold", "-1"},  // negative
  };
  for (const Args& args : invalid) {
    EXPECT_THROW(parseExtractArgs(args), UsageError) << ::testing::PrintToString(args);
  }
}

TEST(ExtractCli, ReportsOptionUsedAsValue) {
  const Args args = {"--input", "a.mp4", "--out", "--overwrite"};

  try {
    parseExtractArgs(args);
    FAIL() << "expected UsageError";
  } catch (const UsageError& e) {
    EXPECT_STREQ(e.what(), "--out needs a value, got option '--overwrite'");
  }
}

TEST(ExtractCli, AcceptsNegativeNumbersAsValuesUntilValidation) {
  // A single leading dash is a value, so the range check (not the parser) rejects it.
  const Args args = {"--input", "a.mp4", "--out", "o", "--motion-threshold", "-1"};

  try {
    parseExtractArgs(args);
    FAIL() << "expected UsageError";
  } catch (const UsageError& e) {
    EXPECT_STREQ(e.what(), "motion threshold must be >= 0");
  }
}

TEST(ExtractCli, UsageMentionsEveryOption) {
  const std::string usage = extractUsage();
  for (const char* option : {"--input", "--out", "--fps", "--max-fps", "--motion-threshold",
                             "--dedup", "--clip", "--hwaccel", "--jpeg-q", "--overwrite"}) {
    EXPECT_NE(usage.find(option), std::string::npos) << option;
  }
}

}  // namespace
}  // namespace vc
