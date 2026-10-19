#include "vc_redact/redact_cli.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "test_media.h"

namespace vc {
namespace {

using Args = std::vector<std::string>;

TEST(RedactCli, ParsesAModelRun) {
  const RedactOptions o = parseRedactArgs(
      {"--input", "in.mp4", "--out", "out", "--model", "data/models/r1", "--ep", "cpu", "--encoder", "software"});
  EXPECT_EQ(o.input, "in.mp4");
  EXPECT_EQ(o.outRoot, "out");
  ASSERT_TRUE(o.modelDir.has_value());
  EXPECT_EQ(*o.modelDir, "data/models/r1");
  EXPECT_FALSE(o.fakeDetections.has_value());
  EXPECT_EQ(o.provider, ExecutionProvider::Cpu);
  EXPECT_FALSE(o.preferHardwareEncoder);
  EXPECT_EQ(o.hwAccel, HwAccel::None);
  EXPECT_FALSE(o.detectEveryFrame);
  EXPECT_FALSE(o.overwrite);
}

TEST(RedactCli, AppliesDefaults) {
  const RedactOptions o = parseRedactArgs({"--input", "in.mp4", "--out", "out", "--model", "m"});
  EXPECT_EQ(o.provider, ExecutionProvider::CoreML);
  EXPECT_EQ(o.hwAccel, HwAccel::None);
  EXPECT_TRUE(o.preferHardwareEncoder);
  EXPECT_DOUBLE_EQ(o.config.detectFps, RedactConfig{}.detectFps);
  EXPECT_DOUBLE_EQ(o.config.maxDetectFps, RedactConfig{}.maxDetectFps);
}

TEST(RedactCli, ParsesHardwareAndFakeDetectionOptions) {
  const RedactOptions o = parseRedactArgs({"--input", "in.mp4", "--out", "out", "--fake-detections", "f.jsonl",
                                           "--hwaccel", "videotoolbox", "--encoder", "auto", "--ep", "coreml"});
  ASSERT_TRUE(o.fakeDetections.has_value());
  EXPECT_EQ(*o.fakeDetections, "f.jsonl");
  EXPECT_FALSE(o.modelDir.has_value());
  EXPECT_EQ(o.hwAccel, HwAccel::VideoToolbox);
  EXPECT_TRUE(o.preferHardwareEncoder);
  EXPECT_EQ(o.provider, ExecutionProvider::CoreML);
}

TEST(RedactCli, DetectFpsOverridesAndRaisesTheMaximum) {
  const RedactOptions o =
      parseRedactArgs({"--detect-fps", "30", "--input", "a.mp4", "--out", "o", "--fake-detections", "f.jsonl"});
  EXPECT_DOUBLE_EQ(o.config.detectFps, 30);
  EXPECT_DOUBLE_EQ(o.config.maxDetectFps, 30);
}

TEST(RedactCli, DetectFpsKeepsAHigherMaximum) {
  const RedactOptions o =
      parseRedactArgs({"--detect-fps", "6", "--input", "a.mp4", "--out", "o", "--fake-detections", "f.jsonl"});
  EXPECT_DOUBLE_EQ(o.config.detectFps, 6);
  EXPECT_DOUBLE_EQ(o.config.maxDetectFps, RedactConfig{}.maxDetectFps);
}

TEST(RedactCli, ReadsTheConfigFile) {
  const std::filesystem::path cfg = test::tmpDir() / "redact_cli.json";
  std::ofstream(cfg) << R"({"detect_fps": 10, "expand_fraction": 0.3})";
  const RedactOptions o = parseRedactArgs(
      {"--input", "a.mp4", "--out", "o", "--fake-detections", "f.jsonl", "--config", cfg.string(), "--overwrite",
       "--detect-every-frame"});
  EXPECT_DOUBLE_EQ(o.config.detectFps, 10);
  EXPECT_DOUBLE_EQ(o.config.timeline.expandFraction, 0.3);
  EXPECT_TRUE(o.overwrite);
  EXPECT_TRUE(o.detectEveryFrame);
}

TEST(RedactCli, DetectFpsOverridesTheConfigFile) {
  const std::filesystem::path cfg = test::tmpDir() / "redact_cli_fps.json";
  std::ofstream(cfg) << R"({"detect_fps": 10, "max_detect_fps": 12})";
  const RedactOptions o = parseRedactArgs(
      {"--input", "a.mp4", "--out", "o", "--fake-detections", "f.jsonl", "--config", cfg.string(), "--detect-fps", "20"});
  EXPECT_DOUBLE_EQ(o.config.detectFps, 20);
  EXPECT_DOUBLE_EQ(o.config.maxDetectFps, 20);
}

TEST(RedactCli, RejectsBadCommandLines) {
  const std::filesystem::path bad = test::tmpDir() / "redact_cli_bad.json";
  std::ofstream(bad) << R"({"detect_fps": 1})";
  const std::vector<Args> cases = {
      {"--out", "o", "--model", "m"},
      {"--input", "a.mp4", "--model", "m"},
      {"--input", "a.mp4", "--out", "o"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--fake-detections", "f"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--ep", "cuda"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--encoder", "nvenc"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--hwaccel", "cuda"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--detect-fps", "fast"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--detect-fps", "2"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--detect-fps", "10x"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--config", bad.string()},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--config", "missing.json"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "--bogus"},
      {"--input", "a.mp4", "--out", "o", "--model", "m", "stray"},
      {"--input", "--out", "o", "--model", "m"},
      {"--input"},
      {},
  };
  for (const Args& args : cases) EXPECT_THROW(parseRedactArgs(args), UsageError) << ::testing::PrintToString(args);
}

TEST(RedactCli, RejectsNonFiniteDetectFps) {
  for (const char* value : {"nan", "inf", "-inf"}) {
    EXPECT_THROW(parseRedactArgs({"--input", "a.mp4", "--out", "o", "--model", "m", "--detect-fps", value}),
                 UsageError)
        << value;
  }
}

TEST(RedactCli, UsageMentionsEveryOption) {
  const std::string usage = redactUsage();
  for (const char* option : {"--input", "--out", "--model", "--fake-detections", "--config", "--detect-fps",
                             "--detect-every-frame", "--ep", "--hwaccel", "--encoder", "--overwrite"}) {
    EXPECT_NE(usage.find(option), std::string::npos) << option;
  }
}

}  // namespace
}  // namespace vc
