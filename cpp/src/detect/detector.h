#pragma once

#include <string>
#include <vector>

#include "detect/box.h"
#include "io/video_reader.h"

namespace vc {

// A detected box and the `src` written for it to detections.jsonl: the detector that found it, or
// "<first>+<second>" when two detectors found the same target (MergedDetector).
struct SourcedBox {
  Box box;
  std::string src;
};

// Finds faces and plates in one decoded frame (spec §3.3).
class Detector {
 public:
  virtual ~Detector() = default;

  // Boxes in frame pixels; Box::cls indexes classes(). Throws VcError.
  virtual std::vector<Box> detect(const VideoFrame& frame) = 0;
  virtual const std::vector<std::string>& classes() const = 0;
  // The `src` written to detections.jsonl ("student", "yunet", "fake").
  virtual std::string source() const = 0;
  // Execution backend for reports ("CoreML", "CPU"); empty when not applicable. "CoreML" means the
  // CoreML execution provider was registered for the session; unsupported nodes may still run on
  // the CPU.
  virtual std::string backend() const { return {}; }

  // detect() with the source of every box: source() for each, unless the detector merges several.
  virtual std::vector<SourcedBox> detectSourced(const VideoFrame& frame) {
    std::vector<SourcedBox> out;
    const std::string src = source();
    for (const Box& b : detect(frame)) out.push_back(SourcedBox{b, src});
    return out;
  }
  // The `detector` of the run's report: "<source>/<backend>" ("student/CoreML"), or source() alone
  // when there is no backend ("fake"); a merged detector names both ("student/CoreML + yunet/CPU").
  virtual std::string description() const {
    const std::string b = backend();
    return b.empty() ? source() : source() + "/" + b;
  }
};

}  // namespace vc
