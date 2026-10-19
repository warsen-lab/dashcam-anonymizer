#include "detect/merged_detector.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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

// Two detectors that call arrive() before they sleep meet here: arrive() returns once the other one has
// arrived too (true), or after `patience` (false). With both on the frame at the same time they always meet,
// however slow the machine is; one after the other they never do.
class Rendezvous {
 public:
  bool arrive() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++arrived_;
    cv_.notify_all();
    return cv_.wait_for(lock, std::chrono::seconds(5), [this] { return arrived_ >= 2; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int arrived_ = 0;
};

// Meets the other detector at `meeting` (if there is one), sleeps, then returns fixed boxes or throws
// VcError("<source> failed"); `finished` is set when the sleep is over.
class DelayedDetector : public Detector {
 public:
  DelayedDetector(std::vector<std::string> classes, std::vector<Box> boxes, std::string source, int delayMs,
                  bool fails = false, Rendezvous* meeting = nullptr)
      : classes_(std::move(classes)), boxes_(std::move(boxes)), source_(std::move(source)), delayMs_(delayMs),
        fails_(fails), meeting_(meeting) {}
  std::vector<Box> detect(const VideoFrame&) override {
    if (meeting_ != nullptr) metPeer = meeting_->arrive();
    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
    finished = true;
    if (fails_) throw VcError(source_ + " failed");
    return boxes_;
  }
  const std::vector<std::string>& classes() const override { return classes_; }
  std::string source() const override { return source_; }
  std::atomic<bool> finished{false};
  std::atomic<bool> metPeer{false};

 private:
  std::vector<std::string> classes_;
  std::vector<Box> boxes_;
  std::string source_;
  int delayMs_;
  bool fails_;
  Rendezvous* meeting_;
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

// Both detectors run at once (spec 2026-10-19 §2). Proven by a rendezvous, not by a clock: each detector waits
// (up to 5 s) for the other to be inside detect() too, which can only happen when they overlap, however loaded
// the machine is. One after the other the first would wait out the 5 s and the test would fail.
TEST(MergedDetector, RunsBothDetectorsAtTheSameTime) {
  Rendezvous meeting;
  auto student = std::make_unique<DelayedDetector>(std::vector<std::string>{"face", "plate"},
                                                   std::vector<Box>{Box{1, 0, 0, 10, 10, 0.5f}}, "student", 300,
                                                   false, &meeting);
  auto yunet = std::make_unique<DelayedDetector>(std::vector<std::string>{"face"},
                                                 std::vector<Box>{Box{0, 50, 50, 60, 60, 0.6f}}, "yunet", 600, false,
                                                 &meeting);
  DelayedDetector* s = student.get();
  DelayedDetector* y = yunet.get();
  MergedDetector merged(std::move(student), std::move(yunet));
  const auto started = std::chrono::steady_clock::now();
  const std::vector<SourcedBox> boxes = merged.detectSourced(VideoFrame{});
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  EXPECT_TRUE(s->metPeer) << "the student never saw the face model running";
  EXPECT_TRUE(y->metPeer) << "the face model never saw the student running";
  EXPECT_EQ(boxes.size(), 2u);
  // Each detector's busy time covers at least its own sleep and lies inside the call that measured it.
  EXPECT_GE(wall, 0.6);
  EXPECT_GE(merged.firstBusySec(), 0.3);
  EXPECT_GE(merged.secondBusySec(), 0.6);
  EXPECT_LE(merged.firstBusySec(), wall);
  EXPECT_LE(merged.secondBusySec(), wall);
}

// Either detector failing fails the call with its own error, and the other one has finished by then: nothing
// keeps running on the frame after the call returns.
TEST(MergedDetector, EitherDetectorThrowingFailsTheCall) {
  for (const bool firstFails : {true, false}) {
    auto student = std::make_unique<DelayedDetector>(std::vector<std::string>{"face"}, std::vector<Box>{}, "student",
                                                     firstFails ? 0 : 200, firstFails);
    auto yunet = std::make_unique<DelayedDetector>(std::vector<std::string>{"face"}, std::vector<Box>{}, "yunet",
                                                   firstFails ? 200 : 0, !firstFails);
    DelayedDetector* s = student.get();
    DelayedDetector* y = yunet.get();
    MergedDetector merged(std::move(student), std::move(yunet));
    try {
      merged.detectSourced(VideoFrame{});
      ADD_FAILURE() << "no exception";
    } catch (const VcError& e) {
      EXPECT_EQ(std::string(e.what()), firstFails ? "student failed" : "yunet failed");
    }
    EXPECT_TRUE(s->finished);
    EXPECT_TRUE(y->finished);
  }
}

// When both fail, the first detector's error is the one that comes out.
TEST(MergedDetector, TheFirstDetectorsErrorWinsWhenBothFail) {
  auto student = std::make_unique<DelayedDetector>(std::vector<std::string>{"face"}, std::vector<Box>{}, "student", 0, true);
  auto yunet = std::make_unique<DelayedDetector>(std::vector<std::string>{"face"}, std::vector<Box>{}, "yunet", 0, true);
  MergedDetector merged(std::move(student), std::move(yunet));
  try {
    merged.detectSourced(VideoFrame{});
    ADD_FAILURE() << "no exception";
  } catch (const VcError& e) {
    EXPECT_EQ(std::string(e.what()), "student failed");
  }
}

// The parallel merge is the serial one: both detectors' boxes, the second's classes mapped by name, through
// mergeDetections, in the same order.
TEST(MergedDetector, MergesExactlyAsTheSerialMerge) {
  std::vector<Box> studentBoxes;
  std::vector<Box> yunetBoxes;
  for (int i = 0; i < 12; ++i) {
    const auto f = static_cast<float>(i);
    studentBoxes.push_back(Box{i % 2, 13 * f, 7 * f, 13 * f + 20 + f, 7 * f + 15, 0.1f * f});
    yunetBoxes.push_back(Box{0, 13 * f + static_cast<float>(i % 3), 7 * f - 1, 13 * f + 22, 7 * f + 14 + f, 0.05f * f});
  }
  // The student's classes are {"plate", "face"}: YuNet's "face" (its class 0) is the student's class 1.
  MergedDetector merged(
      std::make_unique<FixedDetector>(std::vector<std::string>{"plate", "face"}, studentBoxes, "student", ""),
      std::make_unique<FixedDetector>(std::vector<std::string>{"face"}, yunetBoxes, "yunet", ""));
  std::vector<SourcedBox> first;
  std::vector<SourcedBox> second;
  for (const Box& b : studentBoxes) first.push_back(SourcedBox{b, "student"});
  for (Box b : yunetBoxes) {
    b.cls = 1;
    second.push_back(SourcedBox{b, "yunet"});
  }
  const std::vector<SourcedBox> expected = mergeDetections(first, second, kMergeIou);
  const std::vector<SourcedBox> actual = merged.detectSourced(VideoFrame{});
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    const Box& e = expected[i].box;
    expectSourced(actual[i], e.cls, e.x1, e.y1, e.x2, e.y2, e.score, expected[i].src);
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
