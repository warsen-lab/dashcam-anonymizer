#include "detect/yunet_decode.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "common/vc_error.h"

namespace vc {
namespace {

constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

// Raw YuNet outputs for a padWidth x padHeight input: every score 0 unless set().
class Outputs {
 public:
  Outputs(int padWidth, int padHeight) : padWidth_(padWidth), padHeight_(padHeight) {
    for (size_t i = 0; i < kYuNetStrides.size(); ++i) {
      const size_t n = static_cast<size_t>(cols(i)) * (padHeight / kYuNetStrides[i]);
      cls_[i].assign(n, 0.0f);
      obj_[i].assign(n, 0.0f);
      bbox_[i].assign(4 * n, 0.0f);
    }
  }

  // The anchor at (row, col) of `stride`.
  void set(int stride, int row, int col, float cls, float obj, std::array<float, 4> bbox) {
    const size_t i = level(stride);
    const size_t idx = static_cast<size_t>(row) * cols(i) + col;
    cls_[i][idx] = cls;
    obj_[i][idx] = obj;
    for (size_t k = 0; k < 4; ++k) bbox_[i][4 * idx + k] = bbox[k];
  }
  float& cls(int stride, size_t idx) { return cls_[level(stride)][idx]; }
  float& obj(int stride, size_t idx) { return obj_[level(stride)][idx]; }

  std::vector<Box> decode(int frameWidth, int frameHeight, const YuNetParams& params) const {
    std::array<YuNetLevel, 3> levels{};
    for (size_t i = 0; i < 3; ++i) levels[i] = YuNetLevel{cls_[i].data(), obj_[i].data(), bbox_[i].data()};
    return decodeYuNet(levels, padWidth_, padHeight_, frameWidth, frameHeight, params);
  }
  std::vector<Box> decode(const YuNetParams& params = YuNetParams{}) const {
    return decode(padWidth_, padHeight_, params);
  }

 private:
  static size_t level(int stride) { return stride == 8 ? 0 : stride == 16 ? 1 : 2; }
  int cols(size_t i) const { return padWidth_ / kYuNetStrides[i]; }

  int padWidth_;
  int padHeight_;
  std::array<std::vector<float>, 3> cls_;
  std::array<std::vector<float>, 3> obj_;
  std::array<std::vector<float>, 3> bbox_;
};

// The raw bbox values that decode to the box (x1, y1, w, h) at anchor (row, col) of `stride`.
std::array<float, 4> bboxFor(int stride, int row, int col, double x1, double y1, double w, double h) {
  return {static_cast<float>((x1 + w / 2) / stride - col), static_cast<float>((y1 + h / 2) / stride - row),
          static_cast<float>(std::log(w / stride)), static_cast<float>(std::log(h / stride))};
}

YuNetParams params(float score, float nms = 0.3f, int topK = 5000) { return YuNetParams{score, nms, topK}; }

void expectBox(const Box& b, float x1, float y1, float x2, float y2, float tolerance = 1e-4f) {
  EXPECT_EQ(b.cls, 0);
  EXPECT_NEAR(b.x1, x1, tolerance);
  EXPECT_NEAR(b.y1, y1, tolerance);
  EXPECT_NEAR(b.x2, x2, tolerance);
  EXPECT_NEAR(b.y2, y2, tolerance);
}

TEST(YuNetDecode, ScoresTheRootOfClsTimesObjAndMapsTheAnchorToPixels) {
  // Stride 16 on a 64x32 input: 4 columns, 2 rows; the anchor at row 1, column 2.
  Outputs out(64, 32);
  out.set(16, 1, 2, 0.81f, 0.25f, {0.5f, 0.25f, std::log(2.0f), 0.0f});
  const std::vector<Box> boxes = out.decode();
  ASSERT_EQ(boxes.size(), 1u);
  EXPECT_NEAR(boxes[0].score, 0.45f, 1e-6f);
  // cx = (2 + 0.5) * 16 = 40, cy = (1 + 0.25) * 16 = 20, w = 2 * 16, h = 16.
  expectBox(boxes[0], 24, 12, 56, 28);
}

TEST(YuNetDecode, ReadsEveryStrideInRowMajorOrder) {
  Outputs out(64, 32);
  out.set(8, 3, 7, 1.0f, 0.64f, {0.5f, 0.5f, 0, 0});
  out.set(32, 0, 1, 1.0f, 0.36f, {0.5f, 0.5f, 0, 0});
  const std::vector<Box> boxes = out.decode(params(0.1f));
  ASSERT_EQ(boxes.size(), 2u);
  expectBox(boxes[0], 56, 24, 64, 32);
  EXPECT_NEAR(boxes[0].score, 0.8f, 1e-6f);
  expectBox(boxes[1], 32, 0, 64, 32);
  EXPECT_NEAR(boxes[1].score, 0.6f, 1e-6f);
}

TEST(YuNetDecode, ClampsClsAndObjToZeroOneBeforeTheRoot) {
  Outputs out(64, 32);
  out.set(32, 0, 0, 1.5f, 1.2f, {0.5f, 0.5f, 0, 0});
  out.set(32, 0, 1, -0.5f, 1.0f, {0.5f, 0.5f, 0, 0});
  const std::vector<Box> boxes = out.decode(params(0.0f));
  ASSERT_FALSE(boxes.empty());
  EXPECT_EQ(boxes[0].score, 1.0f);
  for (size_t i = 1; i < boxes.size(); ++i) EXPECT_EQ(boxes[i].score, 0.0f);
}

TEST(YuNetDecode, KeepsAScoreEqualToTheThreshold) {
  Outputs out(64, 32);
  out.set(32, 0, 0, 0.25f, 0.25f, {0.5f, 0.5f, 0, 0});  // sqrt(0.0625) = 0.25 exactly
  EXPECT_EQ(out.decode(params(0.25f)).size(), 1u);
  EXPECT_TRUE(out.decode(params(0.2500001f)).empty());
}

TEST(YuNetDecode, SuppressesOnlyAnOverlapAboveTheThreshold) {
  // A is 20x10, B is 10x10 inside it: IoU exactly 0.5.
  Outputs out(64, 32);
  out.set(8, 0, 0, 1.0f, 0.81f, bboxFor(8, 0, 0, 0, 0, 20, 10));
  out.set(8, 0, 1, 1.0f, 0.64f, bboxFor(8, 0, 1, 0, 0, 10, 10));
  EXPECT_EQ(out.decode(params(0.3f, 0.5f)).size(), 2u);
  EXPECT_EQ(out.decode(params(0.3f, 0.49f)).size(), 1u);
}

// OpenCV measures the NMS overlap on the boxes truncated to integer rectangles (cv::Rect2i).
TEST(YuNetDecode, MeasuresTheOverlapOnTruncatedIntegerRectanglesLikeOpenCv) {
  // A = (0, 0, 10, 10); B = (5.9, 0, 10.9, 10): IoU 0.244 in floats, but (5, 0, 10, 10) truncated: IoU 1/3 > 0.3.
  Outputs out(64, 32);
  out.set(8, 0, 0, 1.0f, 0.81f, bboxFor(8, 0, 0, 0, 0, 10, 10));
  out.set(8, 0, 1, 1.0f, 0.64f, bboxFor(8, 0, 1, 5.9, 0, 10.9, 10));
  const std::vector<Box> boxes = out.decode(params(0.3f, 0.3f));
  ASSERT_EQ(boxes.size(), 1u);
  EXPECT_NEAR(boxes[0].score, 0.9f, 1e-6f);
}

TEST(YuNetDecode, RunsHighestScoreFirstAndKeepsTheTopK) {
  Outputs out(64, 32);
  out.set(8, 0, 0, 1.0f, 0.36f, {0.5f, 0.5f, 0, 0});
  out.set(8, 2, 4, 1.0f, 0.81f, {0.5f, 0.5f, 0, 0});
  out.set(8, 1, 6, 1.0f, 0.64f, {0.5f, 0.5f, 0, 0});
  const std::vector<Box> all = out.decode();
  ASSERT_EQ(all.size(), 3u);
  EXPECT_NEAR(all[0].score, 0.9f, 1e-6f);
  EXPECT_NEAR(all[1].score, 0.8f, 1e-6f);
  EXPECT_NEAR(all[2].score, 0.6f, 1e-6f);
  const std::vector<Box> top = out.decode(params(0.3f, 0.3f, 2));
  ASSERT_EQ(top.size(), 2u);
  EXPECT_NEAR(top[1].score, 0.8f, 1e-6f);
}

TEST(YuNetDecode, ClipsBoxesToTheFrameNotThePaddedInput) {
  Outputs out(64, 32);
  out.set(32, 0, 1, 1.0f, 1.0f, bboxFor(32, 0, 1, -6, -4, 80, 40));
  const std::vector<Box> boxes = out.decode(50, 30, YuNetParams{});
  ASSERT_EQ(boxes.size(), 1u);
  expectBox(boxes[0], 0, 0, 50, 30);
}

// A broken model must stop the run, not lose a mask: a NaN / inf score would fail every threshold
// test and vanish.
TEST(YuNetDecode, ANonFiniteScoreAnywhereThrows) {
  for (const float value : {kNan, kInf, -kInf}) {
    Outputs cls(64, 32);
    cls.cls(16, 5) = value;
    EXPECT_THROW(cls.decode(), VcError) << value;
    Outputs obj(64, 32);
    obj.obj(8, 30) = value;
    EXPECT_THROW(obj.decode(), VcError) << value;
  }
}

TEST(YuNetDecode, AKeptBoxWithNonFiniteCoordinatesThrows) {
  const std::vector<std::array<float, 4>> broken = {
      {kNan, 0.5f, 0, 0}, {0.5f, kInf, 0, 0}, {0.5f, 0.5f, kInf, 0}, {0.5f, 0.5f, 0, 1000.0f}};  // exp overflows
  for (const auto& bbox : broken) {
    Outputs out(64, 32);
    out.set(32, 0, 0, 1.0f, 1.0f, bbox);
    try {
      out.decode();
      ADD_FAILURE() << "no throw for " << bbox[0] << " " << bbox[1] << " " << bbox[2] << " " << bbox[3];
    } catch (const VcError& e) {
      EXPECT_NE(std::string(e.what()).find("non-finite box"), std::string::npos) << e.what();
    }
  }
}

TEST(YuNetDecode, ADroppedAnchorMayHaveAnyCoordinates) {
  Outputs out(64, 32);
  out.set(32, 0, 0, 0.1f, 0.1f, {kNan, kInf, 1000.0f, -kInf});
  EXPECT_TRUE(out.decode().empty());
}

// Absurd but finite boxes must not hit undefined behaviour when they are truncated to integers.
TEST(YuNetDecode, SurvivesHugeFiniteBoxes) {
  Outputs out(64, 32);
  out.set(32, 0, 0, 1.0f, 1.0f, {1e30f, 0.5f, 80.0f, 80.0f});
  out.set(32, 0, 1, 1.0f, 0.81f, {-1e30f, 0.5f, 85.0f, 0.0f});
  const std::vector<Box> boxes = out.decode();
  EXPECT_EQ(boxes.size(), 2u);
}

}  // namespace
}  // namespace vc
