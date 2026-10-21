#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "detect/yolo_decode.h"
#include "detect/yunet_decode.h"
#include "io/rate_control.h"
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
  // A face box over largeFaceAreaFraction of the frame (its own area, before any margin) scoring below
  // largeFaceMinScore is dropped before it reaches the timeline (spec 2026-10-19 §3.6): giant YuNet false
  // positives on road and buildings. largeFaceMinScore 0 switches the rule off.
  double largeFaceAreaFraction = 0.05;
  float largeFaceMinScore = 0.6f;
  // Linux deployment (spec 2026-10-20 §3, §5). The output is always H.264. rateControl "quality" (the default) encodes
  // at constant quality, a file as large as the picture needs; "bitrate" encodes at the source's bit rate, times
  // outputBitrateFactorHevc, in [1, 4], for a source in another codec (HEVC, say): an H.264 source keeps its bit rate.
  // outputBitrateFactorHevc is only used with "bitrate".
  double outputBitrateFactorHevc = 1.5;
  RateControl rateControl = RateControl::Quality;
  // h264_nvenc's preset, "p1" (fastest) to "p7" (best); only used when NVENC encodes.
  std::string nvencPreset = "p4";
  // The TensorRT execution provider in FP16 (used once TensorRT runs: stage 1c of the Linux plan).
  bool trtFp16 = true;
  // onnxruntime's intra-op CPU threads per session, in [0, 1024]: 0 keeps onnxruntime's default, more sets it
  // (smaller when several vc_redact processes share a machine). Applies on every platform.
  int ortIntraThreads = 0;
};

// The values nvenc_preset takes.
inline constexpr std::array<std::string_view, 7> kNvencPresets = {"p1", "p2", "p3", "p4", "p5", "p6", "p7"};

// Keys may be omitted; unknown keys, the removed max_detect_fps and motion_threshold, wrong types and
// out-of-range values throw VcError("<origin>: …"). detect_fps is in [5, 60]; shrink_per_frame is in (0, 1],
// max_mask_area_ratio >= 1; mosaic_min_block_px and mosaic_blocks_per_long_side must be JSON integers in
// [1, 4096], ort_intra_threads a JSON integer in [0, 1024]; hold_lost_ghosts and trt_fp16 must be JSON booleans,
// nvenc_preset one of the strings "p1" to "p7", rate_control the string "quality" or "bitrate"; every other value must be a finite number.
// face_score_threshold, face_confirm_score, plate_confirm_score and large_face_min_score are in [0, 1],
// face_nms_iou and large_face_area_fraction in (0, 1], output_bitrate_factor_hevc in [1, 4].
RedactConfig parseRedactConfig(const std::string& text, const std::string& origin);
RedactConfig readRedactConfig(const std::filesystem::path& path);
void validateRedactConfig(const RedactConfig& config, const std::string& origin);
nlohmann::ordered_json redactConfigToJson(const RedactConfig& config);

// config.timeline with confirmScore set for a detector with these classes: "face" -> faceConfirmScore,
// "plate" -> plateConfirmScore, any other class 0 (needs no confirmation).
TimelineConfig timelineFor(const RedactConfig& config, const std::vector<std::string>& classes);

}  // namespace vc
