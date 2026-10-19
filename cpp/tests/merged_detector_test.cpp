#include "detect/merged_detector.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/vc_error.h"

namespace vc {
namespace {

constexpr int kFace = 0;
constexpr int kPlate = 1;

SourcedBox sourced(int cls, float x1, float y1, float x2, float y2, float score, const std::string& src) {
  return SourcedBox{Box{cls, x1, y1, x2, y2, score}, src};
}

void expectSourced(const SourcedBox& b, int cls, float x1, float y1, float x2, float y2, float score,
                   const std::string& src) {
  EXPECT_EQ(b.box.cls, cls);
  EXPECT_FLOAT_EQ(b.box.x1, x1);
  EXPECT_FLOAT_EQ(b.box.y1, y1);
  EXPECT_FLOAT_EQ(b.box.x2, x2);
  EXPECT_FLOAT_EQ(b.box.y2, y2);
  EXPECT_FLOAT_EQ(b.box.score, score);
  EXPECT_EQ(b.src, src);
}

TEST(DetectionMerge, OverlappingSameClassBoxesBecomeTheirUnionWithBothSources) {
  // IoU = 72 / 128 = 0.56
  const std::vector<SourcedBox> merged = mergeDetections({sourced(kFace, 0, 0, 10, 10, 0.2f, "student")},
                                                         {sourced(kFace, 2, -1, 12, 9, 0.7f, "yunet")}, 0.5f);
  ASSERT_EQ(merged.size(), 1u);
  expectSourced(merged[0], kFace, 0, -1, 12, 10, 0.7f, "student+yunet");
}

TEST(DetectionMerge, AnIouOfExactlyTheThresholdMerges) {
  // B (10x10) inside A (20x10): IoU 0.5.
  const std::vector<SourcedBox> merged = mergeDetections({sourced(kFace, 0, 0, 20, 10, 0.9f, "student")},
                                                         {sourced(kFace, 0, 0, 10, 10, 0.4f, "yunet")}, 0.5f);
  ASSERT_EQ(merged.size(), 1u);
  expectSourced(merged[0], kFace, 0, 0, 20, 10, 0.9f, "student+yunet");
}

TEST(DetectionMerge, KeepsEverythingElseFirstDetectorFirst) {
  const std::vector<SourcedBox> first = {
      sourced(kPlate, 100, 100, 140, 110, 0.6f, "student"),  // a plate where the face model sees a face
      sourced(kFace, 0, 0, 10, 10, 0.5f, "student"),
  };
  const std::vector<SourcedBox> second = {
      sourced(kFace, 100, 100, 140, 110, 0.8f, "yunet"),  // same box, other class: both stay
      sourced(kFace, 4, 0, 14, 10, 0.9f, "yunet"),        // IoU 6 / 14 = 0.43 with the student face: both stay
  };
  const std::vector<SourcedBox> merged = mergeDetections(first, second, 0.5f);
  ASSERT_EQ(merged.size(), 4u);
  expectSourced(merged[0], kPlate, 100, 100, 140, 110, 0.6f, "student");
  expectSourced(merged[1], kFace, 0, 0, 10, 10, 0.5f, "student");
  expectSourced(merged[2], kFace, 100, 100, 140, 110, 0.8f, "yunet");
  expectSourced(merged[3], kFace, 4, 0, 14, 10, 0.9f, "yunet");
}

TEST(DetectionMerge, PairsOneToOneHighestIouFirst) {
  // Both student faces overlap the one YuNet face; the better overlap merges, the other stays alone.
  const std::vector<SourcedBox> first = {
      sourced(kFace, 0, 0, 10, 10, 0.5f, "student"),  // IoU 0.82 with the YuNet box
      sourced(kFace, 1, 0, 11, 10, 0.4f, "student"),  // IoU 1.0
  };
  const std::vector<SourcedBox> merged = mergeDetections(first, {sourced(kFace, 1, 0, 11, 10, 0.6f, "yunet")}, 0.5f);
  ASSERT_EQ(merged.size(), 2u);
  expectSourced(merged[0], kFace, 0, 0, 10, 10, 0.5f, "student");
  expectSourced(merged[1], kFace, 1, 0, 11, 10, 0.6f, "student+yunet");
}

// Recall first: every box either detector found is inside some merged box.
TEST(DetectionMerge, CoversEveryInputBox) {
  std::vector<SourcedBox> first;
  std::vector<SourcedBox> second;
  for (int i = 0; i < 12; ++i) {
    const auto f = static_cast<float>(i);
    first.push_back(sourced(i % 2, 13 * f, 7 * f, 13 * f + 20 + f, 7 * f + 15, 0.1f * f, "student"));
    second.push_back(sourced(0, 13 * f + (i % 3), 7 * f - 1, 13 * f + 22, 7 * f + 14 + f, 0.05f * f, "yunet"));
  }
  const std::vector<SourcedBox> merged = mergeDetections(first, second, 0.5f);
  for (const std::vector<SourcedBox>* input : {&first, &second}) {
    for (const SourcedBox& in : *input) {
      const bool covered = std::any_of(merged.begin(), merged.end(), [&](const SourcedBox& m) {
        return m.box.cls == in.box.cls && m.box.x1 <= in.box.x1 && m.box.y1 <= in.box.y1 && m.box.x2 >= in.box.x2 &&
               m.box.y2 >= in.box.y2 && m.box.score >= in.box.score;
      });
      EXPECT_TRUE(covered) << in.src << " box at " << in.box.x1 << "," << in.box.y1;
    }
  }
}

// Replays fixed boxes; counts calls.
class FixedDetector : public Detector {
 public:
  FixedDetector(std::vector<std::string> classes, std::vector<Box> boxes, std::string source, std::string backend)
      : classes_(std::move(classes)), boxes_(std::move(boxes)), source_(std::move(source)), backend_(std::move(backend)) {}
  std::vector<Box> detect(const VideoFrame&) override {
    ++calls;
    return boxes_;
  }
  const std::vector<std::string>& classes() const override { return classes_; }
  std::string source() const override { return source_; }
  std::string backend() const override { return backend_; }
  int calls = 0;

 private:
  std::vector<std::string> classes_;
  std::vector<Box> boxes_;
  std::string source_;
  std::string backend_;
};

TEST(MergedDetector, RunsBothAndMapsTheSecondDetectorsClassesByName) {
  auto student = std::make_unique<FixedDetector>(std::vector<std::string>{"plate", "face"},
                                                 std::vector<Box>{Box{1, 0, 0, 10, 10, 0.2f}}, "student", "CoreML");
  auto yunet = std::make_unique<FixedDetector>(std::vector<std::string>{"face"},
                                               std::vector<Box>{Box{0, 1, 1, 11, 11, 0.8f}, Box{0, 50, 50, 60, 60, 0.6f}},
                                               "yunet", "CPU");
  FixedDetector* s = student.get();
  FixedDetector* y = yunet.get();
  MergedDetector merged(std::move(student), std::move(yunet));
  EXPECT_EQ(merged.classes(), (std::vector<std::string>{"plate", "face"}));
  EXPECT_EQ(merged.source(), "student+yunet");
  EXPECT_EQ(merged.description(), "student/CoreML + yunet/CPU");

  const std::vector<SourcedBox> boxes = merged.detectSourced(VideoFrame{});
  EXPECT_EQ(s->calls, 1);
  EXPECT_EQ(y->calls, 1);
  ASSERT_EQ(boxes.size(), 2u);
  expectSourced(boxes[0], 1, 0, 0, 11, 11, 0.8f, "student+yunet");  // "face" is class 1 of the student
  expectSourced(boxes[1], 1, 50, 50, 60, 60, 0.6f, "yunet");
  EXPECT_EQ(merged.detect(VideoFrame{}).size(), 2u);
}

TEST(MergedDetector, RefusesAClassTheFirstDetectorDoesNotHave) {
  auto student = std::make_unique<FixedDetector>(std::vector<std::string>{"plate"}, std::vector<Box>{}, "student", "");
  auto yunet = std::make_unique<FixedDetector>(std::vector<std::string>{"face"}, std::vector<Box>{}, "yunet", "");
  try {
    MergedDetector merged(std::move(student), std::move(yunet));
    ADD_FAILURE() << "accepted a face model next to a plate-only model";
  } catch (const VcError& e) {
    EXPECT_NE(std::string(e.what()).find("face"), std::string::npos) << e.what();
  }
}

TEST(Detector, DescribesItselfAsSourceAndBackend) {
  FixedDetector withBackend({"face"}, {}, "student", "CoreML");
  FixedDetector without({"face"}, {}, "fake", "");
  EXPECT_EQ(withBackend.description(), "student/CoreML");
  EXPECT_EQ(without.description(), "fake");
  FixedDetector one({"face"}, {Box{0, 1, 2, 3, 4, 0.5f}}, "student", "");
  const std::vector<SourcedBox> boxes = one.detectSourced(VideoFrame{});
  ASSERT_EQ(boxes.size(), 1u);
  EXPECT_EQ(boxes[0].src, "student");
}

}  // namespace
}  // namespace vc
