#include "vc_redact/redact_config.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <sstream>
#include <string_view>

#include "common/vc_error.h"
#include "sampler/frame_sampler.h"

namespace vc {
namespace {

constexpr std::array<std::string_view, 11> kKeys = {
    "detect_fps",         "max_detect_fps",  "motion_threshold",       "score_threshold",
    "nms_iou",            "match_max_distance", "expand_fraction",     "motion_expand_fraction",
    "min_box_px",         "mosaic_min_block_px", "mosaic_blocks_per_long_side"};

// Mosaic block settings are small positive integers; anything larger is a typo, not a block size.
constexpr double kMaxMosaicSetting = 4096;

// The setting as an int: `key` must be a JSON integer in [1, kMaxMosaicSetting] (a fraction or an
// out-of-range number must not be truncated or overflow into a different value); absent: `fallback`.
int mosaicSetting(const nlohmann::json& j, const char* key, int fallback, const std::string& origin) {
  const auto it = j.find(key);
  if (it == j.end()) return fallback;
  const std::string rule = origin + ": " + key + " must be an integer in [1, 4096]";
  if (!it->is_number_integer()) throw VcError(rule);
  const double value = it->get<double>();  // exact for every value that can pass the range test
  if (!(value >= 1 && value <= kMaxMosaicSetting)) throw VcError(rule);
  return static_cast<int>(value);
}

}  // namespace

void validateRedactConfig(const RedactConfig& c, const std::string& origin) {
  const auto require = [&](bool ok, const std::string& what) {
    if (!ok) throw VcError(origin + ": " + what);
  };
  require(c.detectFps >= kMinBaseFps, "detect_fps must be >= 5");
  require(c.maxDetectFps >= c.detectFps, "max_detect_fps must be >= detect_fps");
  require(c.motionThreshold >= 0 && c.motionThreshold <= 1, "motion_threshold must be in [0, 1]");
  require(c.decode.scoreThreshold >= 0 && c.decode.scoreThreshold <= 1, "score_threshold must be in [0, 1]");
  require(c.decode.nmsIou > 0 && c.decode.nmsIou <= 1, "nms_iou must be in (0, 1]");
  require(c.timeline.matchMaxDistance > 0, "match_max_distance must be > 0");
  require(c.timeline.expandFraction >= 0, "expand_fraction must be >= 0");
  require(c.timeline.motionExpandFraction >= 0, "motion_expand_fraction must be >= 0");
  require(c.timeline.minBoxPx >= 0, "min_box_px must be >= 0");
  require(c.mosaic.minBlockPx >= 1 && c.mosaic.minBlockPx <= kMaxMosaicSetting,
          "mosaic_min_block_px must be in [1, 4096]");
  require(c.mosaic.blocksPerLongSide >= 1 && c.mosaic.blocksPerLongSide <= kMaxMosaicSetting,
          "mosaic_blocks_per_long_side must be in [1, 4096]");
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
    if (std::find(kKeys.begin(), kKeys.end(), item.key()) == kKeys.end()) {
      throw VcError(origin + ": unknown key '" + item.key() + "'");
    }
  }
  RedactConfig c;
  try {
    c.detectFps = j.value("detect_fps", c.detectFps);
    c.maxDetectFps = j.value("max_detect_fps", c.maxDetectFps);
    c.motionThreshold = j.value("motion_threshold", c.motionThreshold);
    c.decode.scoreThreshold = j.value("score_threshold", c.decode.scoreThreshold);
    c.decode.nmsIou = j.value("nms_iou", c.decode.nmsIou);
    c.timeline.matchMaxDistance = j.value("match_max_distance", c.timeline.matchMaxDistance);
    c.timeline.expandFraction = j.value("expand_fraction", c.timeline.expandFraction);
    c.timeline.motionExpandFraction = j.value("motion_expand_fraction", c.timeline.motionExpandFraction);
    c.timeline.minBoxPx = j.value("min_box_px", c.timeline.minBoxPx);
  } catch (const nlohmann::json::exception& e) {
    throw VcError(origin + ": " + e.what());
  }
  c.mosaic.minBlockPx = mosaicSetting(j, "mosaic_min_block_px", c.mosaic.minBlockPx, origin);
  c.mosaic.blocksPerLongSide = mosaicSetting(j, "mosaic_blocks_per_long_side", c.mosaic.blocksPerLongSide, origin);
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
  j["max_detect_fps"] = c.maxDetectFps;
  j["motion_threshold"] = c.motionThreshold;
  j["score_threshold"] = c.decode.scoreThreshold;
  j["nms_iou"] = c.decode.nmsIou;
  j["match_max_distance"] = c.timeline.matchMaxDistance;
  j["expand_fraction"] = c.timeline.expandFraction;
  j["motion_expand_fraction"] = c.timeline.motionExpandFraction;
  j["min_box_px"] = c.timeline.minBoxPx;
  j["mosaic_min_block_px"] = c.mosaic.minBlockPx;
  j["mosaic_blocks_per_long_side"] = c.mosaic.blocksPerLongSide;
  return j;
}

}  // namespace vc
