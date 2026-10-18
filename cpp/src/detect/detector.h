#pragma once

#include <string>
#include <vector>

#include "detect/box.h"
#include "io/video_reader.h"

namespace vc {

// Finds faces and plates in one decoded frame (spec §3.3).
class Detector {
 public:
  virtual ~Detector() = default;

  // Boxes in frame pixels; Box::cls indexes classes(). Throws VcError.
  virtual std::vector<Box> detect(const VideoFrame& frame) = 0;
  virtual const std::vector<std::string>& classes() const = 0;
  // The `src` written to detections.jsonl ("student", "fake").
  virtual std::string source() const = 0;
  // Execution backend for reports ("CoreML", "CPU"); empty when not applicable.
  virtual std::string backend() const { return {}; }
};

}  // namespace vc
