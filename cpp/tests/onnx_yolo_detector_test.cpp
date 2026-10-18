#include "detect/onnx_yolo_detector.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

fs::path constModel() { return fs::path(VC_TEST_DATA_DIR) / "const_yolo"; }

VideoFrame view(const AVFrame* frame) {
  VideoFrame v;
  v.av = frame;
  return v;
}

std::string readText(const fs::path& path) {
  std::ifstream in(path);
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

// A copy of the constant model whose model.json has `from` replaced by `to`.
fs::path variant(const std::string& name, const std::string& from, const std::string& to) {
  const fs::path dir = test::tmpDir() / name;
  fs::create_directories(dir);
  fs::copy_file(constModel() / "model.onnx", dir / "model.onnx", fs::copy_options::overwrite_existing);
  std::string json = readText(constModel() / "model.json");
  json.replace(json.find(from), from.size(), to);
  std::ofstream(dir / "model.json") << json;
  return dir;
}

void expectConstBoxes(const std::vector<Box>& boxes, float tolerance) {
  ASSERT_EQ(boxes.size(), 2u);
  // input 64x64, frame 128x128 -> r = 0.5
  EXPECT_EQ(boxes[0].cls, 0);
  EXPECT_NEAR(boxes[0].score, 0.9, tolerance);
  EXPECT_NEAR(boxes[0].x1, 24, tolerance * 100);
  EXPECT_NEAR(boxes[0].y1, 24, tolerance * 100);
  EXPECT_NEAR(boxes[0].x2, 40, tolerance * 100);
  EXPECT_NEAR(boxes[0].y2, 40, tolerance * 100);
  EXPECT_EQ(boxes[1].cls, 1);
  EXPECT_NEAR(boxes[1].x1, 64, tolerance * 100);
  EXPECT_NEAR(boxes[1].y1, 72, tolerance * 100);
  EXPECT_NEAR(boxes[1].x2, 96, tolerance * 100);
  EXPECT_NEAR(boxes[1].y2, 88, tolerance * 100);
}

TEST(OnnxYoloDetector, DecodesTheConstantModelOnCpu) {
  OnnxYoloDetector detector(constModel(), ExecutionProvider::Cpu, DecodeParams{}, 128, 128);
  EXPECT_EQ(detector.backend(), "CPU");
  EXPECT_EQ(detector.source(), "student");
  EXPECT_EQ(detector.classes(), (std::vector<std::string>{"face", "plate"}));
  const auto frame = test::makeFrame(128, 128, AV_PIX_FMT_YUV420P, 128, 128, 128);
  expectConstBoxes(detector.detect(view(frame.get())), 1e-5f);
}

TEST(OnnxYoloDetector, CoreMlGivesTheSameBoxes) {
  OnnxYoloDetector detector(constModel(), ExecutionProvider::CoreML, DecodeParams{}, 128, 128);
  EXPECT_TRUE(detector.backend() == "CoreML" || detector.backend() == "CPU") << detector.backend();
  const auto frame = test::makeFrame(128, 128, AV_PIX_FMT_YUV420P, 128, 128, 128);
  expectConstBoxes(detector.detect(view(frame.get())), 1e-3f);
}

TEST(OnnxYoloDetector, RejectsAFrameOfAnotherSize) {
  OnnxYoloDetector detector(constModel(), ExecutionProvider::Cpu, DecodeParams{}, 128, 128);
  const auto frame = test::makeFrame(64, 64, AV_PIX_FMT_YUV420P, 128, 128, 128);
  EXPECT_THROW(detector.detect(view(frame.get())), VcError);
}

TEST(OnnxYoloDetector, RejectsModelJsonThatDisagreesWithTheOnnx) {
  const fs::path wrongSize = variant("const_yolo_size", R"("width": 64)", R"("width": 128)");
  EXPECT_THROW(OnnxYoloDetector(wrongSize, ExecutionProvider::Cpu, DecodeParams{}, 128, 128), VcError);
  const fs::path wrongClasses =
      variant("const_yolo_classes", R"(["face", "plate"])", R"(["face", "plate", "car"])");
  EXPECT_THROW(OnnxYoloDetector(wrongClasses, ExecutionProvider::Cpu, DecodeParams{}, 128, 128), VcError);
}

TEST(OnnxYoloDetector, ParsesExecutionProviders) {
  EXPECT_EQ(parseExecutionProvider("coreml"), ExecutionProvider::CoreML);
  EXPECT_EQ(parseExecutionProvider("cpu"), ExecutionProvider::Cpu);
  EXPECT_THROW(parseExecutionProvider("cuda"), std::invalid_argument);
}

}  // namespace
}  // namespace vc
