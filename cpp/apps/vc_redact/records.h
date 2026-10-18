#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "detect/box.h"
#include "track/box_timeline.h"

namespace vc {

// One detections.jsonl line (spec §3.2), newline included. Throws VcError when box.cls is not an
// index into `classes`.
std::string detectionRecord(const std::string& video, int64_t frame, double ptsMs, const Box& box,
                            const std::vector<std::string>& classes, const std::string& src);

// One masks.jsonl line for a frame (boxes may be empty), newline included.
std::string maskRecord(const std::string& video, int64_t frame, const std::vector<MaskBox>& boxes,
                       const std::vector<std::string>& classes);

}  // namespace vc
