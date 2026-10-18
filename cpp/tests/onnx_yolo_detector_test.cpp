#include "detect/onnx_yolo_detector.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
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

// Sets (or, with nullptr, unsets) an environment variable for the lifetime of the object.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) old_ = old;
    if (value != nullptr) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }
  ~ScopedEnv() {
    if (old_) {
      setenv(name_.c_str(), old_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  std::string name_;
  std::optional<std::string> old_;
};

// A fresh VC_CACHE under the test directory, so that the CoreML tests never write to ~/.cache.
fs::path freshCacheRoot(const std::string& name) {
  const fs::path root = test::tmpDir() / name;
  fs::remove_all(root);
  return root;
}

// Every file under `dir` with its last write time.
std::map<std::string, fs::file_time_type> filesIn(const fs::path& dir) {
  std::map<std::string, fs::file_time_type> files;
  for (const auto& entry : fs::recursive_directory_iterator(dir)) {
    if (entry.is_regular_file()) files[entry.path().string()] = entry.last_write_time();
  }
  return files;
}

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

// Constructs a detector on `dir` (CPU) and expects a VcError whose message contains `needle`.
void expectConstructionFails(const fs::path& dir, const std::string& needle) {
  try {
    OnnxYoloDetector detector(dir, ExecutionProvider::Cpu, DecodeParams{}, 128, 128);
    ADD_FAILURE() << "expected a VcError mentioning '" << needle << "' for " << dir;
  } catch (const VcError& e) {
    EXPECT_NE(std::string(e.what()).find(needle), std::string::npos)
        << "'" << needle << "' not in: " << e.what();
  }
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
  const ScopedEnv cache("VC_CACHE", freshCacheRoot("coreml_cache_boxes").c_str());
  OnnxYoloDetector detector(constModel(), ExecutionProvider::CoreML, DecodeParams{}, 128, 128);
  EXPECT_TRUE(detector.backend() == "CoreML" || detector.backend() == "CPU") << detector.backend();
  RecordProperty("backend", detector.backend());
  std::cout << "[ backend  ] " << detector.backend() << std::endl;
  const auto frame = test::makeFrame(128, 128, AV_PIX_FMT_YUV420P, 128, 128, 128);
  expectConstBoxes(detector.detect(view(frame.get())), 1e-3f);
}

// CoreML compiles the model when the session is created; the compiled model is kept in
// $VC_CACHE/coreml-cache/<sha256 of model.onnx> (VC_CACHE defaults to ~/.cache/video-compliance),
// so a changed model gets a new directory and an unchanged one is not compiled again.
TEST(OnnxYoloDetector, KeysTheCoreMlCacheByTheModelContent) {
  const fs::path root = freshCacheRoot("coreml_cache_keys");
  const ScopedEnv cache("VC_CACHE", root.c_str());
  const std::optional<fs::path> dir = coreMlCacheDirectory(constModel() / "model.onnx");
  ASSERT_TRUE(dir.has_value());
  // `shasum -a 256 tests/data/const_yolo/model.onnx`
  const std::string sha = "cf8f5bf0e958cf75b422ac5b6991c6c4c6f50c389e389eb5ec69262076b35d51";
  EXPECT_EQ(*dir, root / "coreml-cache" / sha);

  // The same bytes elsewhere share the directory; other bytes get another one.
  const fs::path copy = test::tmpDir() / "coreml_cache_copy.onnx";
  fs::copy_file(constModel() / "model.onnx", copy, fs::copy_options::overwrite_existing);
  EXPECT_EQ(coreMlCacheDirectory(copy), dir);
  const fs::path other = test::tmpDir() / "coreml_cache_other.onnx";
  fs::copy_file(fs::path(VC_TEST_DATA_DIR) / "const_yolo_fp16" / "model.onnx", other,
                fs::copy_options::overwrite_existing);
  EXPECT_NE(coreMlCacheDirectory(other), dir);

  // Without VC_CACHE it lives under ~/.cache/video-compliance; without a home directory there is none.
  const ScopedEnv home("HOME", "/nonexistent-home-for-test");
  const ScopedEnv noCache("VC_CACHE", nullptr);
  EXPECT_EQ(coreMlCacheDirectory(copy),
            fs::path("/nonexistent-home-for-test") / ".cache" / "video-compliance" / "coreml-cache" / sha);
  const ScopedEnv noHome("HOME", nullptr);
  EXPECT_FALSE(coreMlCacheDirectory(copy).has_value());
}

TEST(OnnxYoloDetector, ReusesTheCompiledCoreMlModel) {
  const fs::path root = freshCacheRoot("coreml_cache_reuse");
  const ScopedEnv cache("VC_CACHE", root.c_str());
  const std::optional<fs::path> dir = coreMlCacheDirectory(constModel() / "model.onnx");
  ASSERT_TRUE(dir.has_value());
  {
    OnnxYoloDetector first(constModel(), ExecutionProvider::CoreML, DecodeParams{}, 128, 128);
    if (first.backend() != "CoreML") GTEST_SKIP() << "CoreML is not available here";
  }
  ASSERT_TRUE(fs::is_directory(*dir));
  const auto compiled = filesIn(*dir);
  ASSERT_FALSE(compiled.empty());
  OnnxYoloDetector second(constModel(), ExecutionProvider::CoreML, DecodeParams{}, 128, 128);
  EXPECT_EQ(second.backend(), "CoreML");
  EXPECT_EQ(filesIn(*dir), compiled);  // nothing compiled or written again
  const auto frame = test::makeFrame(128, 128, AV_PIX_FMT_YUV420P, 128, 128, 128);
  expectConstBoxes(second.detect(view(frame.get())), 1e-3f);
}

TEST(OnnxYoloDetector, RejectsAFrameOfAnotherSize) {
  OnnxYoloDetector detector(constModel(), ExecutionProvider::Cpu, DecodeParams{}, 128, 128);
  const auto frame = test::makeFrame(64, 64, AV_PIX_FMT_YUV420P, 128, 128, 128);
  EXPECT_THROW(detector.detect(view(frame.get())), VcError);
}

TEST(OnnxYoloDetector, RejectsModelJsonThatDisagreesWithTheOnnx) {
  expectConstructionFails(variant("const_yolo_width", R"("width": 64)", R"("width": 128)"), "input shape");
  expectConstructionFails(variant("const_yolo_height", R"("height": 64)", R"("height": 128)"), "input shape");
  expectConstructionFails(
      variant("const_yolo_classes", R"(["face", "plate"])", R"(["face", "plate", "car"])"), "output shape");
}

TEST(OnnxYoloDetector, RejectsANonFloatModel) {
  // Same graph, but output0 is FLOAT16: GetTensorData<float> would read it as garbage boxes.
  expectConstructionFails(fs::path(VC_TEST_DATA_DIR) / "const_yolo_fp16", "float32");
}

TEST(OnnxYoloDetector, ParsesExecutionProviders) {
  EXPECT_EQ(parseExecutionProvider("coreml"), ExecutionProvider::CoreML);
  EXPECT_EQ(parseExecutionProvider("cpu"), ExecutionProvider::Cpu);
  EXPECT_THROW(parseExecutionProvider("cuda"), std::invalid_argument);
}

}  // namespace
}  // namespace vc
