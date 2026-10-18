#include "vc_redact/redact_config.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/vc_error.h"

namespace vc {
namespace {

// The keys of configs/redact.json (and of redactConfigToJson).
const std::set<std::string> kAllKeys = {
    "detect_fps",         "max_detect_fps",       "motion_threshold",    "score_threshold",
    "nms_iou",            "match_max_distance",   "expand_fraction",     "motion_expand_fraction",
    "min_box_px",         "mosaic_min_block_px",  "mosaic_blocks_per_long_side"};

// Field-by-field comparison (RedactConfig has no operator==).
void expectSameConfig(const RedactConfig& c, const RedactConfig& d) {
  EXPECT_DOUBLE_EQ(c.detectFps, d.detectFps);
  EXPECT_DOUBLE_EQ(c.maxDetectFps, d.maxDetectFps);
  EXPECT_DOUBLE_EQ(c.motionThreshold, d.motionThreshold);
  EXPECT_FLOAT_EQ(c.decode.scoreThreshold, d.decode.scoreThreshold);
  EXPECT_FLOAT_EQ(c.decode.nmsIou, d.decode.nmsIou);
  EXPECT_DOUBLE_EQ(c.timeline.matchMaxDistance, d.timeline.matchMaxDistance);
  EXPECT_DOUBLE_EQ(c.timeline.expandFraction, d.timeline.expandFraction);
  EXPECT_DOUBLE_EQ(c.timeline.motionExpandFraction, d.timeline.motionExpandFraction);
  EXPECT_DOUBLE_EQ(c.timeline.minBoxPx, d.timeline.minBoxPx);
  EXPECT_EQ(c.mosaic.minBlockPx, d.mosaic.minBlockPx);
  EXPECT_EQ(c.mosaic.blocksPerLongSide, d.mosaic.blocksPerLongSide);
}

void expectDefaults(const RedactConfig& c) { expectSameConfig(c, RedactConfig{}); }

// A config that differs from the defaults in every field. Values are exact in binary or have at
// most six decimals, so a JSON round trip reproduces them.
RedactConfig nonDefaultConfig() {
  RedactConfig c;
  c.detectFps = 10.0;
  c.maxDetectFps = 30.0;
  c.motionThreshold = 0.125;
  c.decode.scoreThreshold = 0.25f;
  c.decode.nmsIou = 0.375f;
  c.timeline.matchMaxDistance = 1.5;
  c.timeline.expandFraction = 0.2;
  c.timeline.motionExpandFraction = 0.75;
  c.timeline.minBoxPx = 24.0;
  c.mosaic.minBlockPx = 12;
  c.mosaic.blocksPerLongSide = 6;
  return c;
}

// Parsing `text` must throw a VcError that starts with the origin "cfg: ".
void expectRejected(const std::string& text) {
  try {
    parseRedactConfig(text, "cfg");
    ADD_FAILURE() << "accepted " << text;
  } catch (const VcError& e) {
    EXPECT_EQ(std::string(e.what()).rfind("cfg: ", 0), 0u) << e.what();
  }
}

TEST(RedactConfig, RepositoryFileMatchesTheDefaults) {
  const std::filesystem::path file = std::filesystem::path(VC_REPO_DIR) / "configs" / "redact.json";
  expectDefaults(readRedactConfig(file));
  std::ifstream in(file);
  const nlohmann::json repo = nlohmann::json::parse(in);
  std::set<std::string> keys;
  for (const auto& item : repo.items()) keys.insert(item.key());
  EXPECT_EQ(keys, kAllKeys);
}

TEST(RedactConfig, EmptyObjectGivesDefaultsAndKeysOverride) {
  expectDefaults(parseRedactConfig("{}", "t"));
  const RedactConfig c = parseRedactConfig(R"({"detect_fps": 10, "max_detect_fps": 30, "expand_fraction": 0.2})", "t");
  EXPECT_DOUBLE_EQ(c.detectFps, 10);
  EXPECT_DOUBLE_EQ(c.maxDetectFps, 30);
  EXPECT_DOUBLE_EQ(c.timeline.expandFraction, 0.2);
}

TEST(RedactConfig, RejectsBadInput) {
  const std::vector<std::string> bad = {
      R"({"detect_fps": 4})",          R"({"detect_fps": 20, "max_detect_fps": 10})",
      R"({"score_threshold": 1.5})",   R"({"nms_iou": 0})",
      R"({"match_max_distance": 0})",  R"({"expand_fraction": -0.1})",
      R"({"mosaic_min_block_px": 0})", R"({"unknown_key": 1})",
      R"({"detect_fps": "5"})",        "[]",
      "{",
      // The mosaic settings are JSON integers in [1, 4096]: no fractions, no overflow, no bounds missed.
      R"({"mosaic_min_block_px": 16.5})",           R"({"mosaic_min_block_px": 1e10})",
      R"({"mosaic_min_block_px": 10000000000})",    R"({"mosaic_min_block_px": 4097})",
      // 2^32 + 1 would be 1 after a truncation to 32 bits.
      R"({"mosaic_min_block_px": 4294967297})",     R"({"mosaic_blocks_per_long_side": 4294967297})",
      R"({"mosaic_min_block_px": 18446744073709551617})",
      R"({"mosaic_min_block_px": -3})",             R"({"mosaic_min_block_px": 16.0})",
      R"({"mosaic_min_block_px": null})",           R"({"mosaic_min_block_px": true})",
      R"({"mosaic_blocks_per_long_side": 16.5})",   R"({"mosaic_blocks_per_long_side": 1e10})",
      R"({"mosaic_blocks_per_long_side": 10000000000})", R"({"mosaic_blocks_per_long_side": 4097})",
      R"({"mosaic_blocks_per_long_side": 0})",      R"({"mosaic_blocks_per_long_side": "8"})",
      // Numbers that overflow to infinity while parsing.
      R"({"detect_fps": 1e400})",                   R"({"expand_fraction": 1e400})",
  };
  for (const std::string& text : bad) expectRejected(text);
}

TEST(RedactConfig, RejectsOutOfRangeValuesAndNulls) {
  const std::vector<std::string> bad = {
      R"({"motion_threshold": 1.5})",  R"({"motion_threshold": -0.1})",
      R"({"nms_iou": 1.5})",           R"({"min_box_px": -1})",
      R"({"detect_fps": null})",       R"({"max_detect_fps": null})",
      R"({"motion_threshold": null})", R"({"score_threshold": null})",
      R"({"nms_iou": null})",          R"({"match_max_distance": null})",
      R"({"expand_fraction": null})",  R"({"motion_expand_fraction": null})",
      R"({"min_box_px": null})",       R"({"mosaic_blocks_per_long_side": null})",
  };
  for (const std::string& text : bad) expectRejected(text);
}

TEST(RedactConfig, AcceptsTheMosaicSettingBounds) {
  const RedactConfig low = parseRedactConfig(R"({"mosaic_min_block_px": 1, "mosaic_blocks_per_long_side": 1})", "t");
  EXPECT_EQ(low.mosaic.minBlockPx, 1);
  EXPECT_EQ(low.mosaic.blocksPerLongSide, 1);
  const RedactConfig high =
      parseRedactConfig(R"({"mosaic_min_block_px": 4096, "mosaic_blocks_per_long_side": 4096})", "t");
  EXPECT_EQ(high.mosaic.minBlockPx, 4096);
  EXPECT_EQ(high.mosaic.blocksPerLongSide, 4096);
}

TEST(RedactConfig, ValidateAppliesTheSameMosaicBounds) {
  RedactConfig c;
  c.mosaic.minBlockPx = 4097;
  EXPECT_THROW(validateRedactConfig(c, "cli"), VcError);
  c = RedactConfig{};
  c.mosaic.blocksPerLongSide = 4097;
  EXPECT_THROW(validateRedactConfig(c, "cli"), VcError);
  EXPECT_NO_THROW(validateRedactConfig(RedactConfig{}, "cli"));
}

// Every double (and float) field, set to each non-finite value, must be refused by validation.
struct NumericField {
  const char* key;
  std::function<void(RedactConfig&, double)> set;
};

TEST(RedactConfig, ValidateRefusesEveryNonFiniteDoubleField) {
  const std::vector<NumericField> fields = {
      {"detect_fps", [](RedactConfig& c, double v) { c.detectFps = v; }},
      {"max_detect_fps", [](RedactConfig& c, double v) { c.maxDetectFps = v; }},
      {"motion_threshold", [](RedactConfig& c, double v) { c.motionThreshold = v; }},
      {"score_threshold", [](RedactConfig& c, double v) { c.decode.scoreThreshold = static_cast<float>(v); }},
      {"nms_iou", [](RedactConfig& c, double v) { c.decode.nmsIou = static_cast<float>(v); }},
      {"match_max_distance", [](RedactConfig& c, double v) { c.timeline.matchMaxDistance = v; }},
      {"expand_fraction", [](RedactConfig& c, double v) { c.timeline.expandFraction = v; }},
      {"motion_expand_fraction", [](RedactConfig& c, double v) { c.timeline.motionExpandFraction = v; }},
      {"min_box_px", [](RedactConfig& c, double v) { c.timeline.minBoxPx = v; }},
  };
  const std::vector<double> nonFinite = {std::numeric_limits<double>::quiet_NaN(),
                                         std::numeric_limits<double>::infinity(),
                                         -std::numeric_limits<double>::infinity()};
  for (const NumericField& field : fields) {
    for (const double value : nonFinite) {
      SCOPED_TRACE(std::string(field.key) + " = " + std::to_string(value));
      RedactConfig c;
      field.set(c, value);
      try {
        validateRedactConfig(c, "cli");
        ADD_FAILURE() << "accepted";
      } catch (const VcError& e) {
        EXPECT_EQ(std::string(e.what()).rfind("cli: ", 0), 0u) << e.what();
      }
    }
  }
}

TEST(RedactConfig, DetectFpsMessageFormatsTheMinimum) {
  try {
    parseRedactConfig(R"({"detect_fps": 4})", "cfg");
    ADD_FAILURE() << "accepted detect_fps 4";
  } catch (const VcError& e) {
    EXPECT_EQ(std::string(e.what()), "cfg: detect_fps must be >= 5");
  }
}

TEST(RedactConfig, SerialisesEveryKey) {
  const nlohmann::ordered_json j = redactConfigToJson(RedactConfig{});
  EXPECT_EQ(j.size(), 11u);
  EXPECT_DOUBLE_EQ(j.at("detect_fps").get<double>(), 5.0);
  EXPECT_EQ(j.at("mosaic_blocks_per_long_side").get<int>(), 8);
}

TEST(RedactConfig, FloatSettingsAreWrittenWithSixDecimals) {
  const nlohmann::ordered_json j = redactConfigToJson(RedactConfig{});
  EXPECT_EQ(j.at("score_threshold").dump(), "0.15");
  EXPECT_EQ(j.at("nms_iou").dump(), "0.5");
}

TEST(RedactConfig, DumpParsesBackToTheSameConfig) {
  const RedactConfig c = nonDefaultConfig();
  const std::string text = redactConfigToJson(c).dump();
  expectSameConfig(parseRedactConfig(text, "round"), c);
}

}  // namespace
}  // namespace vc
