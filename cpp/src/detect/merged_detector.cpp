#include "detect/merged_detector.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <future>
#include <optional>
#include <string>
#include <utility>

#include "common/vc_error.h"

namespace vc {
namespace {

struct Pair {
  size_t first = 0;
  size_t second = 0;
  float iou = 0;
};

Box unionOf(const Box& a, const Box& b) {
  Box u = a;
  u.x1 = std::min(a.x1, b.x1);
  u.y1 = std::min(a.y1, b.y1);
  u.x2 = std::max(a.x2, b.x2);
  u.y2 = std::max(a.y2, b.y2);
  u.score = std::max(a.score, b.score);
  return u;
}

// A detector's boxes and how long it took to find them.
struct Timed {
  std::vector<SourcedBox> boxes{};
  double sec = 0;
};

Timed timedDetect(Detector& detector, const VideoFrame& frame) {
  const auto started = std::chrono::steady_clock::now();
  Timed out;
  out.boxes = detector.detectSourced(frame);
  out.sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return out;
}

}  // namespace

std::vector<SourcedBox> mergeDetections(const std::vector<SourcedBox>& first, const std::vector<SourcedBox>& second,
                                        float iouThreshold) {
  std::vector<Pair> pairs;
  for (size_t i = 0; i < first.size(); ++i) {
    for (size_t j = 0; j < second.size(); ++j) {
      if (first[i].box.cls != second[j].box.cls) continue;
      const float overlap = iou(first[i].box, second[j].box);
      if (overlap >= iouThreshold) pairs.push_back(Pair{i, j, overlap});
    }
  }
  // Highest IoU first; the pairs were listed in (first, second) order, which breaks ties.
  std::stable_sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.iou > b.iou; });
  std::vector<std::optional<size_t>> partner(first.size());
  std::vector<bool> taken(second.size(), false);
  for (const Pair& p : pairs) {
    if (partner[p.first] || taken[p.second]) continue;
    partner[p.first] = p.second;
    taken[p.second] = true;
  }
  std::vector<SourcedBox> out;
  out.reserve(first.size() + second.size());
  for (size_t i = 0; i < first.size(); ++i) {
    if (!partner[i]) {
      out.push_back(first[i]);
      continue;
    }
    const SourcedBox& other = second[*partner[i]];
    out.push_back(SourcedBox{unionOf(first[i].box, other.box), first[i].src + "+" + other.src});
  }
  for (size_t j = 0; j < second.size(); ++j) {
    if (!taken[j]) out.push_back(second[j]);
  }
  return out;
}

MergedDetector::MergedDetector(std::unique_ptr<Detector> first, std::unique_ptr<Detector> second, float iouThreshold)
    : first_(std::move(first)), second_(std::move(second)), iouThreshold_(iouThreshold) {
  const std::vector<std::string>& mine = first_->classes();
  for (const std::string& name : second_->classes()) {
    const auto it = std::find(mine.begin(), mine.end(), name);
    if (it == mine.end()) {
      throw VcError("the " + second_->source() + " detector's class '" + name + "' is not a class of the " +
                    first_->source() + " detector");
    }
    secondClassMap_.push_back(static_cast<int>(it - mine.begin()));
  }
}

std::vector<SourcedBox> MergedDetector::detectSourced(const VideoFrame& frame) {
  // The second detector starts on a thread of its own, then the first runs here. The future is always
  // waited for, also when the first detector throws, so nothing is still reading the frame afterwards.
  std::future<Timed> pending =
      std::async(std::launch::async, [this, &frame] { return timedDetect(*second_, frame); });
  Timed a;
  std::exception_ptr firstError;
  try {
    a = timedDetect(*first_, frame);
  } catch (...) {
    firstError = std::current_exception();
  }
  Timed b;
  std::exception_ptr secondError;
  try {
    b = pending.get();
  } catch (...) {
    secondError = std::current_exception();
  }
  if (firstError) std::rethrow_exception(firstError);
  if (secondError) std::rethrow_exception(secondError);
  firstBusySec_ += a.sec;
  secondBusySec_ += b.sec;
  for (SourcedBox& s : b.boxes) {
    if (s.box.cls < 0 || static_cast<size_t>(s.box.cls) >= secondClassMap_.size()) {
      throw VcError("the " + second_->source() + " detector returned class index " + std::to_string(s.box.cls));
    }
    s.box.cls = secondClassMap_[static_cast<size_t>(s.box.cls)];
  }
  return mergeDetections(a.boxes, b.boxes, iouThreshold_);
}

std::vector<Box> MergedDetector::detect(const VideoFrame& frame) {
  std::vector<Box> boxes;
  for (const SourcedBox& s : detectSourced(frame)) boxes.push_back(s.box);
  return boxes;
}

std::string MergedDetector::source() const { return first_->source() + "+" + second_->source(); }

std::string MergedDetector::description() const { return first_->description() + " + " + second_->description(); }

}  // namespace vc
