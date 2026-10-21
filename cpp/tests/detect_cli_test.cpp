#include "vc_detect/detect_cli.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "common/vc_error.h"

namespace vc {
namespace {

using Args = std::vector<std::string>;

const Args kRequired = {"--model", "m", "--frames", "f", "--out", "o.jsonl"};

Args with(Args extra) {
  Args args = kRequired;
  args.insert(args.end(), extra.begin(), extra.end());
  return args;
}

void expectUsageError(const Args& args, const std::string& needle) {
  try {
    parseDetectArgs(args);
    ADD_FAILURE() << "accepted: " << (args.empty() ? "(nothing)" : args.back());
  } catch (const UsageError& e) {
    EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << "'" << needle << "' not in: " << e.what();
  }
}

TEST(DetectCli, ParsesTheRequiredOptionsAndDefaults) {
  const DetectImagesOptions o = parseDetectArgs(kRequired);
  EXPECT_EQ(o.modelDir, "m");
  EXPECT_EQ(o.framesDir, "f");
  EXPECT_EQ(o.out, "o.jsonl");
  EXPECT_EQ(o.provider, ExecutionProvider::Cpu);
  EXPECT_FLOAT_EQ(o.decode.scoreThreshold, DecodeParams{}.scoreThreshold);
  EXPECT_FLOAT_EQ(o.decode.nmsIou, DecodeParams{}.nmsIou);
}

TEST(DetectCli, ParsesEpAndScore) {
  const DetectImagesOptions o = parseDetectArgs(with({"--ep", "coreml", "--score", "0.4"}));
  EXPECT_EQ(o.provider, ExecutionProvider::CoreML);
  EXPECT_FLOAT_EQ(o.decode.scoreThreshold, 0.4f);
  EXPECT_FLOAT_EQ(parseDetectArgs(with({"--score", "0"})).decode.scoreThreshold, 0.0f);
  EXPECT_FLOAT_EQ(parseDetectArgs(with({"--score", "1"})).decode.scoreThreshold, 1.0f);
}

TEST(DetectCli, RequiresModelFramesAndOut) {
  expectUsageError({}, "required");
  expectUsageError({"--model", "m", "--frames", "f"}, "required");
  expectUsageError({"--model", "m", "--out", "o"}, "required");
  expectUsageError({"--frames", "f", "--out", "o"}, "required");
}

TEST(DetectCli, RejectsUnknownAndMisplacedArguments) {
  expectUsageError(with({"--nms-iou", "0.5"}), "unknown option '--nms-iou'");
  expectUsageError(with({"stray"}), "unexpected argument 'stray'");
}

TEST(DetectCli, RejectsAMissingValue) {
  expectUsageError({"--model", "m", "--frames", "f", "--out"}, "--out needs a value");
  expectUsageError(with({"--score"}), "--score needs a value");
}

// An option name where a value belongs is a forgotten value, not a path named "--frames".
TEST(DetectCli, RejectsAValueThatLooksLikeAnOption) {
  expectUsageError({"--model", "--frames", "f", "--out", "o"}, "--model needs a value, got option '--frames'");
  expectUsageError({"--model", "m", "--frames", "--out", "o"}, "--frames needs a value, got option '--out'");
  expectUsageError({"--model", "m", "--frames", "f", "--out", "--ep"}, "--out needs a value, got option '--ep'");
  expectUsageError(with({"--ep", "--score", "0.5"}), "--ep needs a value, got option '--score'");
  expectUsageError(with({"--score", "--ep", "cpu"}), "--score needs a value, got option '--ep'");
}

TEST(DetectCli, RejectsAScoreOutsideTheUnitInterval) {
  for (const char* bad : {"-0.1", "1.5", "2", "nan", "inf", "-inf", "abc", "", "0.5x", "1e999"}) {
    expectUsageError(with({"--score", bad}), "--score expects a number in [0, 1]");
  }
}

TEST(DetectCli, RejectsAnUnknownExecutionProvider) {
  expectUsageError(with({"--ep", "rocm"}), "unknown execution provider 'rocm'");
}

TEST(DetectCli, UsageMentionsEveryOptionAndTheFixedNmsIou) {
  const std::string usage = detectUsage();
  for (const char* option : {"--model", "--frames", "--out", "--ep", "--score", "NMS IoU"}) {
    EXPECT_NE(usage.find(option), std::string::npos) << option;
  }
}

}  // namespace
}  // namespace vc
