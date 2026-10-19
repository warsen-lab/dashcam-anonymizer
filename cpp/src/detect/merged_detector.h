#pragma once

#include <memory>
#include <string>
#include <vector>

#include "detect/detector.h"

namespace vc {

// IoU at which a box of one detector and a same-class box of the other are taken for one target.
inline constexpr float kMergeIou = 0.5f;

// Merges the boxes two detectors found on one frame. Same-class pairs (one box of each list) with
// IoU >= iouThreshold are paired greedily one-to-one, highest IoU first (ties: `first`'s order, then
// `second`'s); a pair becomes the union of the two boxes with the higher score and src
// "<first src>+<second src>". Every other box is kept as it is. The result lists `first`'s boxes in
// their order (a merged pair in its first box's place), then `second`'s unpaired boxes in theirs.
// Recall first: every input box lies inside an output box of its class with at least its score.
std::vector<SourcedBox> mergeDetections(const std::vector<SourcedBox>& first, const std::vector<SourcedBox>& second,
                                        float iouThreshold);

// Runs two detectors on every frame and merges their boxes (mergeDetections). The classes are the
// first detector's; the second's boxes are mapped onto them by class name. detect() times both.
class MergedDetector : public Detector {
 public:
  // Throws VcError when a class of `second` is not one of `first`'s classes.
  MergedDetector(std::unique_ptr<Detector> first, std::unique_ptr<Detector> second, float iouThreshold = kMergeIou);

  std::vector<Box> detect(const VideoFrame& frame) override;
  std::vector<SourcedBox> detectSourced(const VideoFrame& frame) override;
  const std::vector<std::string>& classes() const override { return first_->classes(); }
  // "student+yunet"
  std::string source() const override;
  // "student/CoreML + yunet/CPU"
  std::string description() const override;

 private:
  std::unique_ptr<Detector> first_;
  std::unique_ptr<Detector> second_;
  float iouThreshold_;
  std::vector<int> secondClassMap_;  // second's class index -> first's
};

}  // namespace vc
