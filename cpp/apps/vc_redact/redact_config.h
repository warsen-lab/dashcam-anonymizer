#pragma once

#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

#include "detect/yolo_decode.h"
#include "mask/mosaic.h"
#include "track/box_timeline.h"

namespace vc {

// configs/redact.json (proxy-validation spec §5.2); the defaults equal the repository file.
struct RedactConfig {
  double detectFps = 5.0;       // base keyframe rate, >= 5
  double maxDetectFps = 15.0;   // keyframe rate while the motion score is high
  double motionThreshold = 0.05;
  DecodeParams decode;
  TimelineConfig timeline;
  MosaicConfig mosaic;
};

// Keys may be omitted; unknown keys, wrong types and out-of-range values throw VcError("<origin>: …").
// mosaic_min_block_px and mosaic_blocks_per_long_side must be JSON integers in [1, 4096]; every
// other value must be a finite number.
RedactConfig parseRedactConfig(const std::string& text, const std::string& origin);
RedactConfig readRedactConfig(const std::filesystem::path& path);
void validateRedactConfig(const RedactConfig& config, const std::string& origin);
nlohmann::ordered_json redactConfigToJson(const RedactConfig& config);

}  // namespace vc
