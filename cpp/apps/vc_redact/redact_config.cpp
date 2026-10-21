#include "vc_redact/redact_config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string_view>

#include "common/vc_error.h"
#include "vc_redact/detect_rate.h"

namespace vc {
namespace {

constexpr std::array<std::string_view, 28> kKeys = {
    "detect_fps",          "score_threshold",        "nms_iou",
    "match_max_distance",  "match_max_distance_loose", "expand_fraction",
    "motion_expand_fraction", "min_box_px",          "shrink_per_frame",
    "max_mask_area_ratio", "mosaic_min_block_px",    "mosaic_blocks_per_long_side",
    "face_score_threshold", "face_nms_iou",       "face_confirm_score",
    "plate_confirm_score", "large_face_area_fraction", "large_face_min_score",
    "hold_lost_ghosts",    "output_bitrate_factor_hevc", "rate_control",
    "nvenc_preset",        "trt_fp16",            "ort_intra_threads",
    "verify_plate_below",  "verify_face_unless_both_above", "verify_threshold_face",
    "verify_threshold_plate"};

// Keys of older configs that no longer exist (spec 2026-10-19 §2): refused with a message that says so.
constexpr std::array<std::string_view, 2> kRemovedKeys = {"max_detect_fps", "motion_threshold"};

// Mosaic block settings are small positive integers; anything larger is a typo, not a block size.
constexpr int kMaxMosaicSetting = 4096;
// More onnxruntime threads than this is a typo, not a thread count.
constexpr int kMaxOrtIntraThreads = 1024;
constexpr double kMinBitrateFactor = 1.0;
constexpr double kMaxBitrateFactor = 4.0;

// Float settings are written with this many decimals: 0.15f must read 0.15, not 0.15000000596….
constexpr double kJsonFloatScale = 1e6;

constexpr std::string_view kQualityName = "quality";
constexpr std::string_view kBitrateName = "bitrate";

// "rate_control must be ..." for a value that is neither name; `got` is what it was.
std::string rateControlRule(const std::string& origin, const std::string& got) {
  return origin + ": rate_control must be \"quality\" or \"bitrate\", got '" + got + "'";
}

// A number as it reads in a message: 5, not 5.000000.
std::string plainNumber(double v) {
  std::ostringstream out;
  out << v;
  return out.str();
}

// A float setting as JSON, rounded to kJsonFloatScale steps per unit.
double jsonFloat(float v) {
  return std::round(static_cast<double>(v) * kJsonFloatScale) / kJsonFloatScale;
}

// The setting as an int: `key` must be a JSON integer in [low, high] (a fraction or an out-of-range number must
// not be truncated or overflow into a different value); absent: `fallback`.
int integerSetting(const nlohmann::json& j, const char* key, int fallback, int low, int high,
                   const std::string& origin) {
  const auto it = j.find(key);
  if (it == j.end()) return fallback;
  const std::string rule =
      origin + ": " + key + " must be an integer in [" + std::to_string(low) + ", " + std::to_string(high) + "]";
  if (!it->is_number_integer()) throw VcError(rule);
  const double value = it->get<double>();  // exact for every value that can pass the range test
  if (!(value >= low && value <= high)) throw VcError(rule);
  return static_cast<int>(value);
}

// The setting as a double: `key` must be a JSON number (absent: `fallback`); any other type is refused naming the key.
double numberSetting(const nlohmann::json& j, const char* key, double fallback, const std::string& origin) {
  const auto it = j.find(key);
  if (it == j.end()) return fallback;
  if (!it->is_number()) throw VcError(origin + ": " + key + " must be a number");
  return it->get<double>();
}

// numberSetting() as a float; a finite number beyond the float range saturates (and then fails the range check)
// instead of overflowing in the conversion.
float floatSetting(const nlohmann::json& j, const char* key, float fallback, const std::string& origin) {
  const double v = numberSetting(j, key, static_cast<double>(fallback), origin);
  constexpr double kMax = std::numeric_limits<float>::max();
  return static_cast<float>(std::isfinite(v) ? std::clamp(v, -kMax, kMax) : v);
}

// The setting as a bool: `key` must be a JSON boolean (not 0 / 1, not a string); absent: `fallback`.
bool boolSetting(const nlohmann::json& j, const char* key, bool fallback, const std::string& origin) {
  const auto it = j.find(key);
  if (it == j.end()) return fallback;
  if (!it->is_boolean()) throw VcError(origin + ": " + key + " must be true or false");
  return it->get<bool>();
}

// The setting as an optional float: absent or null: nullopt; otherwise it must be a number.
std::optional<float> optionalFloatSetting(const nlohmann::json& j, const char* key, const std::string& origin) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return std::nullopt;
  return floatSetting(j, key, 0.0f, origin);
}

// The setting as a string: `key` must be a JSON string; absent: `fallback`.
std::string stringSetting(const nlohmann::json& j, const char* key, const std::string& fallback,
                          const std::string& origin) {
  const auto it = j.find(key);
  if (it == j.end()) return fallback;
  if (!it->is_string()) throw VcError(origin + ": " + key + " must be a string");
  return it->get<std::string>();
}

// The rate_control setting: the string "quality" or "bitrate" (absent: `fallback`).
RateControl rateControlSetting(const nlohmann::json& j, RateControl fallback, const std::string& origin) {
  const auto it = j.find("rate_control");
  if (it == j.end()) return fallback;
  if (!it->is_string()) throw VcError(origin + ": rate_control must be a string");
  const std::string name = it->get<std::string>();
  if (name == kQualityName) return RateControl::Quality;
  if (name == kBitrateName) return RateControl::Bitrate;
  throw VcError(rateControlRule(origin, name));
}

}  // namespace

void validateRedactConfig(const RedactConfig& c, const std::string& origin) {
  const auto require = [&](bool ok, const std::string& what) {
    if (!ok) throw VcError(origin + ": " + what);
  };
  // NaN fails the range tests below, but +inf passes some of them: every double is checked here.
  require(std::isfinite(c.detectFps), "detect_fps must be finite");
  require(std::isfinite(c.decode.scoreThreshold), "score_threshold must be finite");
  require(std::isfinite(c.decode.nmsIou), "nms_iou must be finite");
  require(std::isfinite(c.timeline.matchMaxDistance), "match_max_distance must be finite");
  require(std::isfinite(c.timeline.matchMaxDistanceLoose), "match_max_distance_loose must be finite");
  require(std::isfinite(c.timeline.expandFraction), "expand_fraction must be finite");
  require(std::isfinite(c.timeline.motionExpandFraction), "motion_expand_fraction must be finite");
  require(std::isfinite(c.timeline.minBoxPx), "min_box_px must be finite");
  require(std::isfinite(c.timeline.shrinkPerFrame), "shrink_per_frame must be finite");
  require(std::isfinite(c.timeline.maxMaskAreaRatio), "max_mask_area_ratio must be finite");
  require(std::isfinite(c.face.scoreThreshold), "face_score_threshold must be finite");
  require(std::isfinite(c.face.nmsIou), "face_nms_iou must be finite");
  require(std::isfinite(c.faceConfirmScore), "face_confirm_score must be finite");
  require(std::isfinite(c.plateConfirmScore), "plate_confirm_score must be finite");
  require(std::isfinite(c.largeFaceAreaFraction), "large_face_area_fraction must be finite");
  require(std::isfinite(c.largeFaceMinScore), "large_face_min_score must be finite");
  require(std::isfinite(c.outputBitrateFactorHevc), "output_bitrate_factor_hevc must be finite");
  require(c.detectFps >= kMinDetectFps && c.detectFps <= kMaxDetectFps,
          "detect_fps must be in [" + plainNumber(kMinDetectFps) + ", " + plainNumber(kMaxDetectFps) + "]");
  require(c.decode.scoreThreshold >= 0 && c.decode.scoreThreshold <= 1, "score_threshold must be in [0, 1]");
  require(c.decode.nmsIou > 0 && c.decode.nmsIou <= 1, "nms_iou must be in (0, 1]");
  require(c.timeline.matchMaxDistance > 0, "match_max_distance must be > 0");
  require(c.timeline.matchMaxDistanceLoose > 0, "match_max_distance_loose must be > 0");
  require(c.timeline.expandFraction >= 0, "expand_fraction must be >= 0");
  require(c.timeline.motionExpandFraction >= 0, "motion_expand_fraction must be >= 0");
  require(c.timeline.minBoxPx >= 0, "min_box_px must be >= 0");
  require(c.timeline.shrinkPerFrame > 0 && c.timeline.shrinkPerFrame <= 1, "shrink_per_frame must be in (0, 1]");
  require(c.timeline.maxMaskAreaRatio >= 1, "max_mask_area_ratio must be >= 1");
  require(c.mosaic.minBlockPx >= 1 && c.mosaic.minBlockPx <= kMaxMosaicSetting,
          "mosaic_min_block_px must be in [1, 4096]");
  require(c.mosaic.blocksPerLongSide >= 1 && c.mosaic.blocksPerLongSide <= kMaxMosaicSetting,
          "mosaic_blocks_per_long_side must be in [1, 4096]");
  require(c.face.scoreThreshold >= 0 && c.face.scoreThreshold <= 1, "face_score_threshold must be in [0, 1]");
  require(c.face.nmsIou > 0 && c.face.nmsIou <= 1, "face_nms_iou must be in (0, 1]");
  require(c.faceConfirmScore >= 0 && c.faceConfirmScore <= 1, "face_confirm_score must be in [0, 1]");
  require(c.plateConfirmScore >= 0 && c.plateConfirmScore <= 1, "plate_confirm_score must be in [0, 1]");
  require(c.largeFaceAreaFraction > 0 && c.largeFaceAreaFraction <= 1, "large_face_area_fraction must be in (0, 1]");
  require(c.largeFaceMinScore >= 0 && c.largeFaceMinScore <= 1, "large_face_min_score must be in [0, 1]");
  require(c.verifyPlateBelow >= 0 && c.verifyPlateBelow <= 1, "verify_plate_below must be in [0, 1]");
  require(c.verifyFaceUnlessBothAbove >= 0 && c.verifyFaceUnlessBothAbove <= 1,
          "verify_face_unless_both_above must be in [0, 1]");
  require(!c.verifyThresholdFace || (*c.verifyThresholdFace >= 0 && *c.verifyThresholdFace <= 1),
          "verify_threshold_face must be in [0, 1]");
  require(!c.verifyThresholdPlate || (*c.verifyThresholdPlate >= 0 && *c.verifyThresholdPlate <= 1),
          "verify_threshold_plate must be in [0, 1]");
  require(c.face.topK >= 1, "the face model's top_k must be >= 1");
  require(c.outputBitrateFactorHevc >= kMinBitrateFactor && c.outputBitrateFactorHevc <= kMaxBitrateFactor,
          "output_bitrate_factor_hevc must be in [1, 4]");
  require(c.rateControl == RateControl::Quality || c.rateControl == RateControl::Bitrate,
          "rate_control must be \"quality\" or \"bitrate\", got '" + std::to_string(static_cast<int>(c.rateControl)) +
              "'");
  require(std::find(kNvencPresets.begin(), kNvencPresets.end(), c.nvencPreset) != kNvencPresets.end(),
          "nvenc_preset must be one of p1 ... p7, got '" + c.nvencPreset + "'");
  require(c.ortIntraThreads >= 0 && c.ortIntraThreads <= kMaxOrtIntraThreads, "ort_intra_threads must be in [0, 1024]");
}

RedactConfig parseRedactConfig(const std::string& text, const std::string& origin) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(text);
  } catch (const nlohmann::json::exception& e) {
    throw VcError(origin + ": " + e.what());
  }
  if (!j.is_object()) throw VcError(origin + ": the root must be a JSON object");
  for (const auto& item : j.items()) {
    if (std::find(kRemovedKeys.begin(), kRemovedKeys.end(), item.key()) != kRemovedKeys.end()) {
      throw VcError(origin + ": " + item.key() +
                    " was removed: keyframes now come every round(video fps / detect_fps) frames, whatever the motion "
                    "(delete the key)");
    }
    if (std::find(kKeys.begin(), kKeys.end(), item.key()) == kKeys.end()) {
      throw VcError(origin + ": unknown key '" + item.key() + "'");
    }
  }
  RedactConfig c;
  try {
    c.detectFps = j.value("detect_fps", c.detectFps);
    c.decode.scoreThreshold = j.value("score_threshold", c.decode.scoreThreshold);
    c.decode.nmsIou = j.value("nms_iou", c.decode.nmsIou);
    c.timeline.matchMaxDistance = j.value("match_max_distance", c.timeline.matchMaxDistance);
    c.timeline.matchMaxDistanceLoose = j.value("match_max_distance_loose", c.timeline.matchMaxDistanceLoose);
    c.timeline.expandFraction = j.value("expand_fraction", c.timeline.expandFraction);
    c.timeline.motionExpandFraction = j.value("motion_expand_fraction", c.timeline.motionExpandFraction);
    c.timeline.minBoxPx = j.value("min_box_px", c.timeline.minBoxPx);
    c.timeline.shrinkPerFrame = j.value("shrink_per_frame", c.timeline.shrinkPerFrame);
    c.timeline.maxMaskAreaRatio = j.value("max_mask_area_ratio", c.timeline.maxMaskAreaRatio);
    c.face.scoreThreshold = j.value("face_score_threshold", c.face.scoreThreshold);
    c.face.nmsIou = j.value("face_nms_iou", c.face.nmsIou);
    c.faceConfirmScore = j.value("face_confirm_score", c.faceConfirmScore);
    c.plateConfirmScore = j.value("plate_confirm_score", c.plateConfirmScore);
  } catch (const nlohmann::json::exception& e) {
    throw VcError(origin + ": " + e.what());
  }
  c.largeFaceAreaFraction = numberSetting(j, "large_face_area_fraction", c.largeFaceAreaFraction, origin);
  c.largeFaceMinScore = floatSetting(j, "large_face_min_score", c.largeFaceMinScore, origin);
  c.verifyPlateBelow = floatSetting(j, "verify_plate_below", c.verifyPlateBelow, origin);
  c.verifyFaceUnlessBothAbove = floatSetting(j, "verify_face_unless_both_above", c.verifyFaceUnlessBothAbove, origin);
  c.verifyThresholdFace = optionalFloatSetting(j, "verify_threshold_face", origin);
  c.verifyThresholdPlate = optionalFloatSetting(j, "verify_threshold_plate", origin);
  c.timeline.holdLostGhosts = boolSetting(j, "hold_lost_ghosts", c.timeline.holdLostGhosts, origin);
  c.mosaic.minBlockPx = integerSetting(j, "mosaic_min_block_px", c.mosaic.minBlockPx, 1, kMaxMosaicSetting, origin);
  c.mosaic.blocksPerLongSide =
      integerSetting(j, "mosaic_blocks_per_long_side", c.mosaic.blocksPerLongSide, 1, kMaxMosaicSetting, origin);
  c.outputBitrateFactorHevc = numberSetting(j, "output_bitrate_factor_hevc", c.outputBitrateFactorHevc, origin);
  c.rateControl = rateControlSetting(j, c.rateControl, origin);
  c.nvencPreset = stringSetting(j, "nvenc_preset", c.nvencPreset, origin);
  c.trtFp16 = boolSetting(j, "trt_fp16", c.trtFp16, origin);
  c.ortIntraThreads = integerSetting(j, "ort_intra_threads", c.ortIntraThreads, 0, kMaxOrtIntraThreads, origin);
  validateRedactConfig(c, origin);
  return c;
}

RedactConfig readRedactConfig(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw VcError("cannot read " + path.string());
  std::ostringstream text;
  text << in.rdbuf();
  return parseRedactConfig(text.str(), path.string());
}

nlohmann::ordered_json redactConfigToJson(const RedactConfig& c) {
  nlohmann::ordered_json j;
  j["detect_fps"] = c.detectFps;
  j["score_threshold"] = jsonFloat(c.decode.scoreThreshold);
  j["nms_iou"] = jsonFloat(c.decode.nmsIou);
  j["match_max_distance"] = c.timeline.matchMaxDistance;
  j["match_max_distance_loose"] = c.timeline.matchMaxDistanceLoose;
  j["expand_fraction"] = c.timeline.expandFraction;
  j["motion_expand_fraction"] = c.timeline.motionExpandFraction;
  j["min_box_px"] = c.timeline.minBoxPx;
  j["shrink_per_frame"] = c.timeline.shrinkPerFrame;
  j["max_mask_area_ratio"] = c.timeline.maxMaskAreaRatio;
  j["mosaic_min_block_px"] = c.mosaic.minBlockPx;
  j["mosaic_blocks_per_long_side"] = c.mosaic.blocksPerLongSide;
  j["face_score_threshold"] = jsonFloat(c.face.scoreThreshold);
  j["face_nms_iou"] = jsonFloat(c.face.nmsIou);
  j["face_confirm_score"] = jsonFloat(c.faceConfirmScore);
  j["plate_confirm_score"] = jsonFloat(c.plateConfirmScore);
  j["large_face_area_fraction"] = c.largeFaceAreaFraction;
  j["large_face_min_score"] = jsonFloat(c.largeFaceMinScore);
  j["verify_plate_below"] = jsonFloat(c.verifyPlateBelow);
  j["verify_face_unless_both_above"] = jsonFloat(c.verifyFaceUnlessBothAbove);
  j["verify_threshold_face"] =
      c.verifyThresholdFace ? nlohmann::ordered_json(jsonFloat(*c.verifyThresholdFace)) : nlohmann::ordered_json(nullptr);
  j["verify_threshold_plate"] =
      c.verifyThresholdPlate ? nlohmann::ordered_json(jsonFloat(*c.verifyThresholdPlate)) : nlohmann::ordered_json(nullptr);
  j["hold_lost_ghosts"] = c.timeline.holdLostGhosts;
  j["output_bitrate_factor_hevc"] = c.outputBitrateFactorHevc;
  j["rate_control"] = std::string(c.rateControl == RateControl::Quality ? kQualityName : kBitrateName);
  j["nvenc_preset"] = c.nvencPreset;
  j["trt_fp16"] = c.trtFp16;
  j["ort_intra_threads"] = c.ortIntraThreads;
  return j;
}

TimelineConfig timelineFor(const RedactConfig& config, const std::vector<std::string>& classes) {
  TimelineConfig t = config.timeline;
  t.confirmScore.clear();
  for (const std::string& name : classes) {
    t.confirmScore.push_back(name == "face"    ? config.faceConfirmScore
                             : name == "plate" ? config.plateConfirmScore
                                               : 0.0f);
  }
  return t;
}

}  // namespace vc
