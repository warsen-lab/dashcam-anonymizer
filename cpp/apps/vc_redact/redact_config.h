#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "detect/yolo_decode.h"
#include "detect/yunet_decode.h"
#include "mask/mosaic.h"
#include "track/box_timeline.h"

namespace vc {

// configs/redact.json (proxy-validation spec §5.2, updated by spec 2026-10-19); the defaults equal the
// repository file. DecodeParams and TimelineConfig keep their own defaults, which vc_detect and the unit
// tests use. The mask and confirmation settings are the 2026-10-19 calibration's
// (docs/reports/stable-masks-2026-10-19.md).
//
// `face` is the YuNet face model's (vc_redact --face-model; unused without it). Its score threshold
// 0.3 is the lower bound of the teachers' low-confidence face band (configs/labeling.json
// face_low_score [0.3, 0.5]) and its NMS IoU the YuNet teacher's 0.3, both chosen before looking at
// any held-out video, not tuned on it.
struct RedactConfig {
  double detectFps = 10.0;  // keyframes per second of video, in [5, 60]; spaced by frame index (detect_rate.h)
  DecodeParams decode{.scoreThreshold = 0.05f};
  TimelineConfig timeline{.expandFraction = 0.2, .motionExpandFraction = 0.25, .shrinkPerFrame = 0.2};
  MosaicConfig mosaic;
  YuNetParams face{.scoreThreshold = 0.3f, .nmsIou = 0.3f};
  // Boxes scoring below these are masked only once their track is matched at two consecutive keyframes
  // (spec 2026-10-19 §3.4), by class whatever the detector; a merged box goes by its (higher) merged score.
  float faceConfirmScore = 0.5f;
  float plateConfirmScore = 0.2f;
};

// Keys may be omitted; unknown keys, the removed max_detect_fps and motion_threshold, wrong types and
// out-of-range values throw VcError("<origin>: …"). detect_fps is in [5, 60]; shrink_per_frame is in (0, 1],
// max_mask_area_ratio >= 1; mosaic_min_block_px and mosaic_blocks_per_long_side must be JSON integers in
// [1, 4096]; every other value must be a finite number. face_score_threshold, face_confirm_score and
// plate_confirm_score are in [0, 1], face_nms_iou in (0, 1].
RedactConfig parseRedactConfig(const std::string& text, const std::string& origin);
RedactConfig readRedactConfig(const std::filesystem::path& path);
void validateRedactConfig(const RedactConfig& config, const std::string& origin);
nlohmann::ordered_json redactConfigToJson(const RedactConfig& config);

// config.timeline with confirmScore set for a detector with these classes: "face" -> faceConfirmScore,
// "plate" -> plateConfirmScore, any other class 0 (needs no confirmation).
TimelineConfig timelineFor(const RedactConfig& config, const std::vector<std::string>& classes);

}  // namespace vc
