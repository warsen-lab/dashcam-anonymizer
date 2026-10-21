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

// onnxruntime's macOS build ships Microsoft telemetry that uploads usage events over the system
// network stack; a detector must leave it switched off (the link is metered and the data is the
// user's). The environment variable is what the library reads when its environment is created.
TEST(OnnxYoloDetector, SwitchesOnnxRuntimeTelemetryOff) {
  OnnxYoloDetector detector(constModel(), ExecutionProvider::Cpu, DecodeParams{}, 128, 128);
  const char* value = std::getenv("ORT_DISABLE_TELEMETRY");
  ASSERT_NE(value, nullptr);
  EXPECT_STREQ(value, "1");
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

// Empties every regular file of the compiled model in `dir`, the way a compile that was killed
// half-way leaves it; returns how many it emptied.
size_t truncateCompiledModel(const fs::path& dir) {
  size_t emptied = 0;
  for (const auto& entry : fs::recursive_directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    if (entry.path().string().find("compiled_model.mlmodelc") == std::string::npos) continue;
    std::ofstream(entry.path(), std::ios::trunc);
    ++emptied;
  }
  return emptied;
}

// Concurrent first runs race on the shared cache ("Error copying compiled model to cache path"), and a
// compile that died half-way leaves a model that cannot be loaded. Neither may cost the run CoreML:
// the session is created once more without the cache (and the directory, which another process may
// be writing, is left alone) before the CPU takes over. One stderr line names what happened.
TEST(OnnxYoloDetector, RetriesCoreMlWithoutTheCacheWhenItsCompiledModelIsCorrupt) {
  const fs::path root = freshCacheRoot("coreml_cache_corrupt");
  const ScopedEnv cache("VC_CACHE", root.c_str());
  const std::optional<fs::path> dir = coreMlCacheDirectory(constModel() / "model.onnx");
  ASSERT_TRUE(dir.has_value());
  {
    OnnxYoloDetector first(constModel(), ExecutionProvider::CoreML, DecodeParams{}, 128, 128);
    if (first.backend() != "CoreML") GTEST_SKIP() << "CoreML is not available here";
  }
  ASSERT_GT(truncateCompiledModel(*dir), 0u);
  const auto corrupt = filesIn(*dir);

  testing::internal::CaptureStderr();
  OnnxYoloDetector second(constModel(), ExecutionProvider::CoreML, DecodeParams{}, 128, 128);
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_EQ(second.backend(), "CoreML") << log;
  EXPECT_NE(log.find("without the cache"), std::string::npos) << log;
  EXPECT_EQ(log.find("using CPU"), std::string::npos) << log;
  EXPECT_EQ(filesIn(*dir), corrupt);  // the shared directory is as it was: not repaired, not deleted
  const auto frame = test::makeFrame(128, 128, AV_PIX_FMT_YUV420P, 128, 128, 128);
  expectConstBoxes(second.detect(view(frame.get())), 1e-3f);
}

// A cache directory CoreML cannot write to (a read-only volume) costs it the cache, not the backend
// (onnxruntime copes with that by itself; this keeps it so).
TEST(OnnxYoloDetector, UsesCoreMlWhenItsCacheCannotBeWritten) {
  const fs::path root = freshCacheRoot("coreml_cache_readonly");
  const ScopedEnv cache("VC_CACHE", root.c_str());
  const std::optional<fs::path> dir = coreMlCacheDirectory(constModel() / "model.onnx");
  ASSERT_TRUE(dir.has_value());
  fs::create_directories(*dir);
  fs::permissions(*dir, fs::perms::owner_read | fs::perms::owner_exec, fs::perm_options::replace);
  struct Restore {
    fs::path dir;
    ~Restore() { fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace); }
  } restore{*dir};
  {
    std::ofstream probe(*dir / "probe");
    if (probe.is_open()) GTEST_SKIP() << "the directory is writable despite its mode (running as root?)";
  }

  testing::internal::CaptureStderr();
  OnnxYoloDetector detector(constModel(), ExecutionProvider::CoreML, DecodeParams{}, 128, 128);
  const std::string log = testing::internal::GetCapturedStderr();
  if (log.find("CoreML execution provider unavailable") != std::string::npos) GTEST_SKIP() << "CoreML is not available here";
  EXPECT_EQ(detector.backend(), "CoreML") << log;
  EXPECT_EQ(log.find("using CPU"), std::string::npos) << log;
  EXPECT_TRUE(fs::is_empty(*dir));
  const auto frame = test::makeFrame(128, 128, AV_PIX_FMT_YUV420P, 128, 128, 128);
  expectConstBoxes(detector.detect(view(frame.get())), 1e-3f);
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

TEST(OnnxYoloDetector, RejectsAFaceModel) {
  // The YuNet model directory belongs to --face-model, not --model.
  expectConstructionFails(fs::path(VC_TEST_DATA_DIR) / "const_yunet", "yolov8");
}

TEST(OnnxYoloDetector, RejectsANonFloatModel) {
  // Same graph, but output0 is FLOAT16: GetTensorData<float> would read it as garbage boxes.
  expectConstructionFails(fs::path(VC_TEST_DATA_DIR) / "const_yolo_fp16", "float32");
}

TEST(OnnxYoloDetector, ParsesExecutionProviders) {
  EXPECT_EQ(parseExecutionProvider("coreml"), ExecutionProvider::CoreML);
  EXPECT_EQ(parseExecutionProvider("cpu"), ExecutionProvider::Cpu);
  EXPECT_EQ(parseExecutionProvider("cuda"), ExecutionProvider::Cuda);
  EXPECT_THROW(parseExecutionProvider("rocm"), std::invalid_argument);
}

// ort_intra_threads > 0 sets the session's intra-op thread count; the boxes stay the same.
TEST(OnnxYoloDetector, RunsWithAFixedIntraOpThreadCount) {
  OnnxYoloDetector detector(constModel(), ExecutionProvider::Cpu, DecodeParams{}, 128, 128, SessionSettings{1, true});
  EXPECT_EQ(detector.backend(), "CPU");
  const auto frame = test::makeFrame(128, 128, AV_PIX_FMT_YUV420P, 128, 128, 128);
  expectConstBoxes(detector.detect(view(frame.get())), 1e-5f);
}

// An explicit GPU provider is unavailable on this build: one line on stderr, then the CPU.
TEST(OnnxYoloDetector, AnUnavailableGpuProviderFallsBackToTheCpu) {
  for (const ExecutionProvider provider : {ExecutionProvider::TensorRT, ExecutionProvider::Cuda}) {
    testing::internal::CaptureStderr();
    OnnxYoloDetector detector(constModel(), provider, DecodeParams{}, 128, 128);
    const std::string log = testing::internal::GetCapturedStderr();
    if (detector.backend() != "CPU") GTEST_SKIP() << "this build runs " << detector.backend();
    EXPECT_NE(log.find("vc: " + providerLabel(provider) + " execution provider unavailable ("), std::string::npos) << log;
    EXPECT_NE(log.find("); using CPU"), std::string::npos) << log;
    const auto frame = test::makeFrame(128, 128, AV_PIX_FMT_YUV420P, 128, 128, 128);
    expectConstBoxes(detector.detect(view(frame.get())), 1e-5f);
  }
}

// Auto is CoreML then the CPU on macOS: the backend is one of the two, never a GPU provider.
TEST(OnnxYoloDetector, AutoUsesCoreMlOrTheCpuOnMacOs) {
  if (buildPlatform() != Platform::MacOS) GTEST_SKIP() << "macOS only";
  const ScopedEnv cache("VC_CACHE", freshCacheRoot("coreml_cache_auto").c_str());
  OnnxYoloDetector detector(constModel(), ExecutionProvider::Auto, DecodeParams{}, 128, 128);
  EXPECT_TRUE(detector.backend() == "CoreML" || detector.backend() == "CPU") << detector.backend();
  const auto frame = test::makeFrame(128, 128, AV_PIX_FMT_YUV420P, 128, 128, 128);
  expectConstBoxes(detector.detect(view(frame.get())), 1e-3f);
}

}  // namespace
}  // namespace vc
