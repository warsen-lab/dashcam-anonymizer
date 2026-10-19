#include "detect/yunet_detector.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/vc_error.h"
#include "detect/onnx_yolo_detector.h"
#include "test_media.h"

extern "C" {
#include <libavutil/mem.h>
#include <libavutil/sha.h>
}

namespace vc {
namespace {
namespace fs = std::filesystem;

fs::path constModel() { return fs::path(VC_TEST_DATA_DIR) / "const_yunet"; }

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

VideoFrame view(const AVFrame* frame) {
  VideoFrame v;
  v.av = frame;
  return v;
}

using Bgr = std::array<uint8_t, 3>;

// A BGR24 frame filled with `fill`, with the pixel rectangle [x1, x2) x [y1, y2) set to `rect`.
av::FramePtr bgrFrame(int width, int height, Bgr fill, int x1 = 0, int y1 = 0, int x2 = 0, int y2 = 0, Bgr rect = {}) {
  av::FramePtr frame = av::allocFrame();
  frame->format = AV_PIX_FMT_BGR24;
  frame->width = width;
  frame->height = height;
  checkAv(av_frame_get_buffer(frame.get(), 0), "allocate BGR frame");
  for (int y = 0; y < height; ++y) {
    uint8_t* row = frame->data[0] + static_cast<ptrdiff_t>(y) * frame->linesize[0];
    for (int x = 0; x < width; ++x) {
      const bool inside = x >= x1 && x < x2 && y >= y1 && y < y2;
      for (int c = 0; c < 3; ++c) row[3 * x + c] = inside ? rect[c] : fill[c];
    }
  }
  return frame;
}

constexpr Bgr kCyan = {255, 255, 0};  // B = G = 255: the constant model scores a full cell 1
constexpr Bgr kRed = {0, 0, 255};

void expectBox(const Box& b, float x1, float y1, float x2, float y2, float score, float tolerance = 1e-4f) {
  EXPECT_EQ(b.cls, 0);
  EXPECT_NEAR(b.x1, x1, tolerance);
  EXPECT_NEAR(b.y1, y1, tolerance);
  EXPECT_NEAR(b.x2, x2, tolerance);
  EXPECT_NEAR(b.y2, y2, tolerance);
  EXPECT_NEAR(b.score, score, tolerance);
}

TEST(YuNetDetector, FeedsBgrUnscaledAtTheFramesOwnResolution) {
  YuNetDetector detector(constModel(), ExecutionProvider::Cpu, YuNetParams{}, 64, 64);
  EXPECT_EQ(detector.backend(), "CPU");
  EXPECT_EQ(detector.source(), "yunet");
  EXPECT_EQ(detector.classes(), (std::vector<std::string>{"face"}));
  // A cyan cell on red: fed as RGB, or scaled to 0-1, nothing would reach the threshold.
  const auto frame = bgrFrame(64, 64, kRed, 32, 0, 64, 32, kCyan);
  const std::vector<Box> boxes = detector.detect(view(frame.get()));
  ASSERT_EQ(boxes.size(), 1u);
  expectBox(boxes[0], 32, 0, 64, 32, 1.0f);
}

TEST(YuNetDetector, PadsWithZerosAndClipsToTheFrame) {
  // 40x40 pads to 64x64: the cells right of and below the image hold 8 of their 32 columns / rows.
  YuNetDetector detector(constModel(), ExecutionProvider::Cpu, YuNetParams{0.2f, 0.3f, 5000}, 40, 40);
  const auto frame = bgrFrame(40, 40, kCyan);
  const std::vector<Box> boxes = detector.detect(view(frame.get()));
  ASSERT_EQ(boxes.size(), 3u);
  expectBox(boxes[0], 0, 0, 32, 32, 1.0f);
  expectBox(boxes[1], 32, 0, 40, 32, 0.25f);  // cls = obj = 8 / 32
  expectBox(boxes[2], 0, 32, 32, 40, 0.25f);
}

TEST(YuNetDetector, ConvertsYuvFramesToBgr) {
  YuNetDetector detector(constModel(), ExecutionProvider::Cpu, YuNetParams{}, 64, 64);
  // Mid grey, limited range: about 130 in every BGR channel, so each cell scores about 130 / 255.
  const auto frame = test::makeFrame(64, 64, AV_PIX_FMT_YUV420P, 128, 128, 128);
  const std::vector<Box> boxes = detector.detect(view(frame.get()));
  ASSERT_EQ(boxes.size(), 4u);
  for (const Box& b : boxes) EXPECT_NEAR(b.score, 130.0f / 255.0f, 0.02f);
}

TEST(YuNetDetector, RebuildsTheSessionWhenTheFrameSizeChanges) {
  YuNetDetector detector(constModel(), ExecutionProvider::Cpu, YuNetParams{}, 64, 64);
  const auto first = bgrFrame(64, 64, kRed, 32, 0, 64, 32, kCyan);
  const auto wide = bgrFrame(100, 30, kRed, 64, 0, 96, 30, kCyan);  // pads to 128x32
  const std::vector<Box> wideBoxes = detector.detect(view(wide.get()));
  ASSERT_EQ(wideBoxes.size(), 1u);
  expectBox(wideBoxes[0], 64, 0, 96, 30, 30.0f / 32.0f);  // 30 of the cell's 32 rows are cyan
  const std::vector<Box> again = detector.detect(view(first.get()));
  ASSERT_EQ(again.size(), 1u);
  expectBox(again[0], 32, 0, 64, 32, 1.0f);
}

TEST(YuNetDetector, CoreMlGivesTheSameBoxesAndCompilesOncePerPaddedSize) {
  const fs::path root = freshCacheRoot("yunet_coreml_cache");
  const ScopedEnv cache("VC_CACHE", root.c_str());
  YuNetDetector detector(constModel(), ExecutionProvider::CoreML, YuNetParams{}, 64, 64);
  EXPECT_TRUE(detector.backend() == "CoreML" || detector.backend() == "CPU") << detector.backend();
  std::cout << "[ backend  ] " << detector.backend() << std::endl;
  const auto frame = bgrFrame(64, 64, kRed, 32, 0, 64, 32, kCyan);
  std::vector<Box> boxes = detector.detect(view(frame.get()));
  ASSERT_EQ(boxes.size(), 1u);
  expectBox(boxes[0], 32, 0, 64, 32, 1.0f, 1e-3f);
  const auto wide = bgrFrame(100, 30, kRed, 64, 0, 96, 30, kCyan);
  boxes = detector.detect(view(wide.get()));
  ASSERT_EQ(boxes.size(), 1u);
  expectBox(boxes[0], 64, 0, 96, 30, 30.0f / 32.0f, 1e-3f);
  if (detector.backend() != "CoreML") GTEST_SKIP() << "CoreML is not available here";
  const std::optional<fs::path> square = yunetCoreMlCacheDirectory(constModel() / "model.onnx", 64, 64);
  const std::optional<fs::path> wider = yunetCoreMlCacheDirectory(constModel() / "model.onnx", 128, 32);
  ASSERT_TRUE(square && wider);
  EXPECT_NE(*square, *wider);
  EXPECT_TRUE(fs::is_directory(*square) && !fs::is_empty(*square));
  EXPECT_TRUE(fs::is_directory(*wider) && !fs::is_empty(*wider));
}

// The model compiled for one input size must never be loaded for another: the cache directory is
// keyed by the model's content and the padded size.
TEST(YuNetDetector, KeysTheCoreMlCacheByTheModelAndThePaddedSize) {
  const fs::path root = freshCacheRoot("yunet_coreml_keys");
  const ScopedEnv cache("VC_CACHE", root.c_str());
  const fs::path onnx = constModel() / "model.onnx";
  const std::optional<fs::path> base = coreMlCacheDirectory(onnx);
  ASSERT_TRUE(base.has_value());
  EXPECT_EQ(yunetCoreMlCacheDirectory(onnx, 1920, 1088), *base / "1920x1088");
  EXPECT_EQ(yunetCoreMlCacheDirectory(onnx, 3840, 2176), *base / "3840x2176");
}

TEST(YuNetDetector, RejectsAStaticModelOfAnotherSize) {
  const fs::path staticModel = fs::path(VC_TEST_DATA_DIR) / "const_yunet_static";  // fixed 64x64 input
  {
    YuNetDetector fits(staticModel, ExecutionProvider::Cpu, YuNetParams{}, 64, 50);  // pads to 64x64
    const auto frame = bgrFrame(64, 50, kRed, 32, 0, 64, 32, kCyan);
    EXPECT_EQ(fits.detect(view(frame.get())).size(), 1u);
  }
  try {
    YuNetDetector tooBig(staticModel, ExecutionProvider::Cpu, YuNetParams{}, 100, 64);
    ADD_FAILURE() << "accepted a 128x64 input on a fixed 64x64 model";
  } catch (const VcError& e) {
    EXPECT_NE(std::string(e.what()).find("vc_train yunet"), std::string::npos) << e.what();
  }
}

// A copy of const_yunet in a directory of its own, so that a test can change its model.onnx.
fs::path copiedConstModel(const std::string& name) {
  const fs::path dir = test::tmpDir() / name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  fs::copy_file(constModel() / "model.json", dir / "model.json");
  fs::copy_file(constModel() / "model.onnx", dir / "model.onnx");
  return dir;
}

// A re-open that fails must leave the detector as it was: the frames of the old size still run on the
// old session, with the old names and buffer.
TEST(YuNetDetector, AFailedReopenLeavesTheDetectorAsItWas) {
  const fs::path dir = copiedConstModel("yunet_failed_reopen");
  YuNetDetector detector(dir, ExecutionProvider::Cpu, YuNetParams{}, 100, 30);  // pads to 128x32
  const auto wide = bgrFrame(100, 30, kRed, 64, 0, 96, 30, kCyan);
  std::vector<Box> boxes = detector.detect(view(wide.get()));
  ASSERT_EQ(boxes.size(), 1u);
  expectBox(boxes[0], 64, 0, 96, 30, 30.0f / 32.0f);

  // The model file is replaced by one fixed at 64x64: a 100x64 frame (128x64) opens a session on it,
  // but the model check refuses it.
  fs::copy_file(fs::path(VC_TEST_DATA_DIR) / "const_yunet_static" / "model.onnx", dir / "model.onnx",
                fs::copy_options::overwrite_existing);
  const auto taller = bgrFrame(100, 64, kRed, 64, 0, 96, 32, kCyan);
  EXPECT_THROW(detector.detect(view(taller.get())), VcError);
  EXPECT_THROW(detector.detect(view(taller.get())), VcError);  // again: it is refused again, not run on stale state

  // The next frame of the old size runs as before (a session on the fixed model would refuse a 128x32 input).
  boxes = detector.detect(view(wide.get()));
  ASSERT_EQ(boxes.size(), 1u);
  expectBox(boxes[0], 64, 0, 96, 30, 30.0f / 32.0f);
}

// A frame size whose input buffer cannot exist is a VcError (not a bare std::length_error), and the
// detector keeps working at its own size afterwards.
TEST(YuNetDetector, AFrameSizeNoBufferCanHoldIsRefusedWithAVcError) {
  YuNetDetector detector(constModel(), ExecutionProvider::Cpu, YuNetParams{}, 64, 64);
  av::FramePtr huge = av::allocFrame();  // only its size is read before the buffer fails
  huge->format = AV_PIX_FMT_BGR24;
  huge->width = 1 << 30;
  huge->height = 1 << 30;
  EXPECT_THROW(detector.detect(view(huge.get())), VcError);
  const auto frame = bgrFrame(64, 64, kRed, 32, 0, 64, 32, kCyan);
  const std::vector<Box> boxes = detector.detect(view(frame.get()));
  ASSERT_EQ(boxes.size(), 1u);
  expectBox(boxes[0], 32, 0, 64, 32, 1.0f);
}

TEST(YuNetDetector, RejectsAStudentModelAndMissingFiles) {
  try {
    YuNetDetector detector(fs::path(VC_TEST_DATA_DIR) / "const_yolo", ExecutionProvider::Cpu, YuNetParams{}, 64, 64);
    ADD_FAILURE() << "accepted a yolov8 model";
  } catch (const VcError& e) {
    EXPECT_NE(std::string(e.what()).find("yunet"), std::string::npos) << e.what();
  }
  EXPECT_THROW(YuNetDetector(test::tmpDir() / "yunet_missing_model", ExecutionProvider::Cpu, YuNetParams{}, 64, 64),
               VcError);
}

// --- The OpenCV Zoo model on the synthetic image of cpp/tests/data/make_yunet_expected.py ---------

std::optional<fs::path> cachedYuNet() {
  const char* cache = std::getenv("VC_CACHE");
  const char* home = std::getenv("HOME");
  const fs::path root = cache != nullptr && *cache != '\0' ? fs::path(cache)
                        : home != nullptr                   ? fs::path(home) / ".cache" / "video-compliance"
                                                            : fs::path();
  const fs::path model = root / "models" / "face_detection_yunet_2023mar.onnx";
  if (root.empty() || !fs::is_regular_file(model)) return std::nullopt;
  return model;
}

nlohmann::json expectation() {
  std::ifstream in(fs::path(VC_TEST_DATA_DIR) / "yunet_synthetic_expected.json");
  return nlohmann::json::parse(in);
}

std::string sha256Hex(const uint8_t* data, size_t size) {
  const std::unique_ptr<AVSHA, decltype(&av_free)> sha(av_sha_alloc(), &av_free);
  av_sha_init(sha.get(), 256);
  av_sha_update(sha.get(), data, size);
  std::array<uint8_t, 32> digest{};
  av_sha_final(sha.get(), digest.data());
  std::ostringstream hex;
  for (const uint8_t byte : digest) hex << "0123456789abcdef"[byte >> 4] << "0123456789abcdef"[byte & 0xF];
  return hex.str();
}

// The generator's image: B = 60 + 80x/W, G = 90 + 60y/H, R = 70, then every ellipse filled in order.
av::FramePtr syntheticFaces(const nlohmann::json& spec) {
  const int width = spec.at("width").get<int>();
  const int height = spec.at("height").get<int>();
  std::vector<uint8_t> bgr(static_cast<size_t>(width) * height * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uint8_t* p = &bgr[(static_cast<size_t>(y) * width + x) * 3];
      p[0] = static_cast<uint8_t>(60 + x * 80 / width);
      p[1] = static_cast<uint8_t>(90 + y * 60 / height);
      p[2] = 70;
    }
  }
  for (const auto& e : spec.at("ellipses")) {
    const int64_t cx = e[0], cy = e[1], rx = e[2], ry = e[3];
    for (int64_t y = 0; y < height; ++y) {
      for (int64_t x = 0; x < width; ++x) {
        if ((x - cx) * (x - cx) * ry * ry + (y - cy) * (y - cy) * rx * rx > rx * rx * ry * ry) continue;
        uint8_t* p = &bgr[(static_cast<size_t>(y) * width + static_cast<size_t>(x)) * 3];
        for (int c = 0; c < 3; ++c) p[c] = static_cast<uint8_t>(e[4 + c].get<int>());
      }
    }
  }
  if (sha256Hex(bgr.data(), bgr.size()) != spec.at("image_sha256").get<std::string>()) return nullptr;
  av::FramePtr frame = bgrFrame(width, height, {0, 0, 0});
  for (int y = 0; y < height; ++y) {
    std::copy_n(&bgr[static_cast<size_t>(y) * width * 3], width * 3,
                frame->data[0] + static_cast<ptrdiff_t>(y) * frame->linesize[0]);
  }
  return frame;
}

// A model directory with the cached OpenCV Zoo file as model.onnx. Its input is fixed at 640x640,
// the size of the synthetic image, so no conversion is needed here.
fs::path zooModelDir(const fs::path& source) {
  const fs::path dir = test::tmpDir() / "yunet_zoo";
  fs::create_directories(dir);
  fs::copy_file(source, dir / "model.onnx", fs::copy_options::overwrite_existing);
  fs::copy_file(constModel() / "model.json", dir / "model.json", fs::copy_options::overwrite_existing);
  return dir;
}

// The boxes with a score >= minScore must be cv2's, in cv2's order (highest score first).
void expectTheCv2Faces(const fs::path& source, ExecutionProvider provider, float minScore, float pxTolerance,
                       float scoreTolerance) {
  const nlohmann::json spec = expectation();
  const av::FramePtr frame = syntheticFaces(spec);
  ASSERT_NE(frame, nullptr) << "the C++ image differs from make_yunet_expected.py's (image_sha256)";
  const YuNetParams params{spec.at("score_threshold").get<float>(), spec.at("nms_iou").get<float>(),
                           spec.at("top_k").get<int>()};
  YuNetDetector detector(zooModelDir(source), provider, params, frame->width, frame->height);
  std::cout << "[ backend  ] " << detector.backend() << std::endl;
  std::vector<Box> boxes = detector.detect(view(frame.get()));
  std::vector<nlohmann::json> faces;
  for (const auto& f : spec.at("faces")) {
    if (f.at("score").get<float>() >= minScore) faces.push_back(f);
  }
  std::erase_if(boxes, [&](const Box& b) { return b.score < minScore; });
  ASSERT_GE(faces.size(), minScore > 0.1f ? 3u : 5u);
  ASSERT_EQ(boxes.size(), faces.size());
  for (size_t i = 0; i < boxes.size(); ++i) {
    SCOPED_TRACE("face " + std::to_string(i));
    EXPECT_NEAR(boxes[i].x1, faces[i].at("x1").get<float>(), pxTolerance);
    EXPECT_NEAR(boxes[i].y1, faces[i].at("y1").get<float>(), pxTolerance);
    EXPECT_NEAR(boxes[i].x2, faces[i].at("x2").get<float>(), pxTolerance);
    EXPECT_NEAR(boxes[i].y2, faces[i].at("y2").get<float>(), pxTolerance);
    EXPECT_NEAR(boxes[i].score, faces[i].at("score").get<float>(), scoreTolerance);
  }
}

TEST(YuNetDetector, FindsWhatCv2FindsOnTheSyntheticImage) {
  const std::optional<fs::path> source = cachedYuNet();
  if (!source) GTEST_SKIP() << "the cached face_detection_yunet_2023mar.onnx is missing";
  expectTheCv2Faces(*source, ExecutionProvider::Cpu, 0.0f, 0.01f, 1e-4f);
}

TEST(YuNetDetector, CoreMlFindsTheConfidentFacesCv2Finds) {
  const std::optional<fs::path> source = cachedYuNet();  // before VC_CACHE moves to the fresh compile cache
  if (!source) GTEST_SKIP() << "the cached face_detection_yunet_2023mar.onnx is missing";
  const ScopedEnv cache("VC_CACHE", freshCacheRoot("yunet_coreml_zoo").c_str());
  expectTheCv2Faces(*source, ExecutionProvider::CoreML, 0.3f, 1.0f, 0.02f);
}

}  // namespace
}  // namespace vc
