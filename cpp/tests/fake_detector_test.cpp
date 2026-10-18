#include "detect/fake_detector.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

fs::path writeFile(const std::string& name, const std::string& text) {
  const fs::path path = test::tmpDir() / name;
  std::ofstream(path) << text;
  return path;
}

VideoFrame frameNumber(int64_t index) {
  VideoFrame frame;
  frame.index = index;
  return frame;
}

TEST(FakeDetector, ReplaysBoxesByFrame) {
  const fs::path path = writeFile(
      "fake_ok.jsonl",
      R"({"video":"v.mp4","frame":0,"pts_ms":0,"cls":"face","x1":1,"y1":2,"x2":3,"y2":4,"score":0.9,"src":"t"})"
      "\n\n"
      R"({"video":"v.mp4","frame":2,"pts_ms":66.7,"cls":"plate","x1":10,"y1":20,"x2":30,"y2":40,"score":0.5,"src":"t"})"
      "\n"
      R"({"video":"v.mp4","frame":2,"pts_ms":66.7,"cls":"face","x1":5,"y1":5,"x2":9,"y2":9,"score":0.7,"src":"t"})"
      "\n");
  FakeDetector detector(path);
  EXPECT_EQ(detector.source(), "fake");
  EXPECT_EQ(detector.classes(), (std::vector<std::string>{"face", "plate"}));
  const std::vector<Box> first = detector.detect(frameNumber(0));
  ASSERT_EQ(first.size(), 1u);
  EXPECT_EQ(first[0].cls, 0);
  EXPECT_FLOAT_EQ(first[0].x2, 3);
  EXPECT_FLOAT_EQ(first[0].score, 0.9f);
  EXPECT_TRUE(detector.detect(frameNumber(1)).empty());
  const std::vector<Box> third = detector.detect(frameNumber(2));
  ASSERT_EQ(third.size(), 2u);
  EXPECT_EQ(third[0].cls, 1);
  EXPECT_EQ(detector.calls(), (std::vector<int64_t>{0, 1, 2}));
}

TEST(FakeDetector, RejectsUnknownClassesAndBadLines) {
  const fs::path unknown = writeFile(
      "fake_unknown.jsonl",
      R"({"frame":0,"cls":"car","x1":1,"y1":2,"x2":3,"y2":4,"score":0.9})" "\n");
  EXPECT_THROW(FakeDetector{unknown}, VcError);
  const fs::path broken = writeFile(
      "fake_broken.jsonl",
      R"({"frame":0,"cls":"face","x1":1,"y1":2,"x2":3,"y2":4,"score":0.9})" "\n{oops\n");
  try {
    FakeDetector detector(broken);
    FAIL() << "expected VcError";
  } catch (const VcError& e) {
    EXPECT_NE(std::string(e.what()).find("line 2"), std::string::npos) << e.what();
  }
  EXPECT_THROW(FakeDetector{test::tmpDir() / "missing.jsonl"}, VcError);
}

TEST(FakeDetector, FailsOnRequestedFrames) {
  const fs::path path = writeFile("fake_fail.jsonl", R"({"frame":3,"fail":true})" "\n");
  FakeDetector detector(path);
  EXPECT_TRUE(detector.detect(frameNumber(2)).empty());
  EXPECT_THROW(detector.detect(frameNumber(3)), VcError);
}

}  // namespace
}  // namespace vc
