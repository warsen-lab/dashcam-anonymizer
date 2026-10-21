#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "detect/box.h"
#include "track/box_timeline.h"

namespace vc {

// One detections.jsonl line (spec §3.2), newline included; with `confirmed` a last field saying whether
// the box was masked (false: a low-score box seen on one keyframe only, spec 2026-10-19 §3.4). Throws
// VcError when box.cls is not an index into `classes`, or when ptsMs, a box coordinate or the score is
// not finite (or overflows when rounded). Invalid UTF-8 in `video` or `src` is written as U+FFFD.
std::string detectionRecord(const std::string& video, int64_t frame, double ptsMs, const Box& box,
                            const std::vector<std::string>& classes, const std::string& src,
                            std::optional<bool> confirmed = std::nullopt);

// One masks.jsonl line for a frame (boxes may be empty), newline included; each box is
// {cls, x1, y1, x2, y2, track, origin, block, snap}. Throws VcError as detectionRecord does for class indices and
// for non-finite coordinates.
std::string maskRecord(const std::string& video, int64_t frame, const std::vector<MaskBox>& boxes,
                       const std::vector<std::string>& classes);

}  // namespace vc
