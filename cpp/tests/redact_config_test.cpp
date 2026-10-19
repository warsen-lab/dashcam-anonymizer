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
    "detect_fps",          "score_threshold",        "nms_iou",
    "match_max_distance",  "match_max_distance_loose", "expand_fraction",
    "motion_expand_fraction", "min_box_px",          "shrink_per_frame",
    "max_mask_area_ratio", "mosaic_min_block_px",    "mosaic_blocks_per_long_side",
    "face_score_threshold", "face_nms_iou"};

// Field-by-field comparison (RedactConfig has no operator==).
void expectSameConfig(const RedactConfig& c, const RedactConfig& d) {
  EXPECT_DOUBLE_EQ(c.detectFps, d.detectFps);
  EXPECT_FLOAT_EQ(c.decode.scoreThreshold, d.decode.scoreThreshold);
  EXPECT_FLOAT_EQ(c.decode.nmsIou, d.decode.nmsIou);
  EXPECT_DOUBLE_EQ(c.timeline.matchMaxDistance, d.timeline.matchMaxDistance);
  EXPECT_DOUBLE_EQ(c.timeline.matchMaxDistanceLoose, d.timeline.matchMaxDistanceLoose);
  EXPECT_DOUBLE_EQ(c.timeline.expandFraction, d.timeline.expandFraction);
  EXPECT_DOUBLE_EQ(c.timeline.motionExpandFraction, d.timeline.motionExpandFraction);
  EXPECT_DOUBLE_EQ(c.timeline.minBoxPx, d.timeline.minBoxPx);
  EXPECT_DOUBLE_EQ(c.timeline.shrinkPerFrame, d.timeline.shrinkPerFrame);
  EXPECT_DOUBLE_EQ(c.timeline.maxMaskAreaRatio, d.timeline.maxMaskAreaRatio);
  EXPECT_EQ(c.mosaic.minBlockPx, d.mosaic.minBlockPx);
  EXPECT_EQ(c.mosaic.blocksPerLongSide, d.mosaic.blocksPerLongSide);
  EXPECT_FLOAT_EQ(c.face.scoreThreshold, d.face.scoreThreshold);
  EXPECT_FLOAT_EQ(c.face.nmsIou, d.face.nmsIou);
  EXPECT_EQ(c.face.topK, d.face.topK);
}

void expectDefaults(const RedactConfig& c) { expectSameConfig(c, RedactConfig{}); }

// A config that differs from the defaults in every field. Values are exact in binary or have at
// most six decimals, so a JSON round trip reproduces them.
RedactConfig nonDefaultConfig() {
  RedactConfig c;
  c.detectFps = 12.0;
  c.decode.scoreThreshold = 0.25f;
  c.decode.nmsIou = 0.375f;
  c.timeline.matchMaxDistance = 1.5;
  c.timeline.matchMaxDistanceLoose = 4.5;
  c.timeline.expandFraction = 0.2;
  c.timeline.motionExpandFraction = 0.75;
  c.timeline.minBoxPx = 24.0;
  c.timeline.shrinkPerFrame = 0.125;
  c.timeline.maxMaskAreaRatio = 2.5;
  c.mosaic.minBlockPx = 12;
  c.mosaic.blocksPerLongSide = 6;
  c.face.scoreThreshold = 0.4375f;
  c.face.nmsIou = 0.25f;
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

// vc_redact's defaults are the ones the 2026-10-18 proxy validation chose (10 fps, score 0.05, expand
// 0.3); the lower-level structs keep their own, which vc_detect and the unit tests use.
TEST(RedactConfig, DefaultsAreTheValidatedSettings) {
  const RedactConfig c;
  EXPECT_DOUBLE_EQ(c.detectFps, 10.0);
  EXPECT_FLOAT_EQ(c.decode.scoreThreshold, 0.05f);
  EXPECT_DOUBLE_EQ(c.timeline.expandFraction, 0.3);
  EXPECT_FLOAT_EQ(DecodeParams{}.scoreThreshold, 0.15f);
  EXPECT_DOUBLE_EQ(TimelineConfig{}.expandFraction, 0.15);
}

// The face model's defaults were chosen before looking at any held-out video: the score is the lower
// bound of the teachers' low-confidence face band (configs/labeling.json face_low_score [0.3, 0.5]),
// the NMS IoU the one the YuNet teacher uses; top_k is cv2.FaceDetectorYN's 5000.
TEST(RedactConfig, FaceModelDefaultsAreChosenAPriori) {
  const RedactConfig c;
  EXPECT_FLOAT_EQ(c.face.scoreThreshold, 0.3f);
  EXPECT_FLOAT_EQ(c.face.nmsIou, 0.3f);
  EXPECT_EQ(c.face.topK, 5000);
}

TEST(RedactConfig, ReadsTheFaceModelSettings) {
  const RedactConfig c = parseRedactConfig(R"({"face_score_threshold": 0.5, "face_nms_iou": 0.4})", "t");
  EXPECT_FLOAT_EQ(c.face.scoreThreshold, 0.5f);
  EXPECT_FLOAT_EQ(c.face.nmsIou, 0.4f);
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
  const RedactConfig c = parseRedactConfig(R"({"detect_fps": 15, "expand_fraction": 0.2})", "t");
  EXPECT_DOUBLE_EQ(c.detectFps, 15);
  EXPECT_DOUBLE_EQ(c.timeline.expandFraction, 0.2);
}

TEST(RedactConfig, ReadsTheLooseMatchingDistance) {
  EXPECT_DOUBLE_EQ(RedactConfig{}.timeline.matchMaxDistanceLoose, 6.0);
  EXPECT_DOUBLE_EQ(parseRedactConfig(R"({"match_max_distance_loose": 3.5})", "t").timeline.matchMaxDistanceLoose, 3.5);
}

// Mask smoothing (spec 2026-10-19 §3.1): a mask's width and height shrink by at most 2 % per frame, and a mask
// covers at most 4 times its box. The values are calibrated in the last task of the plan.
TEST(RedactConfig, ReadsTheMaskSmoothingSettings) {
  EXPECT_DOUBLE_EQ(RedactConfig{}.timeline.shrinkPerFrame, 0.02);
  EXPECT_DOUBLE_EQ(RedactConfig{}.timeline.maxMaskAreaRatio, 4.0);
  const RedactConfig c = parseRedactConfig(R"({"shrink_per_frame": 0.05, "max_mask_area_ratio": 3})", "t");
  EXPECT_DOUBLE_EQ(c.timeline.shrinkPerFrame, 0.05);
  EXPECT_DOUBLE_EQ(c.timeline.maxMaskAreaRatio, 3.0);
  EXPECT_NO_THROW(parseRedactConfig(R"({"shrink_per_frame": 1, "max_mask_area_ratio": 1})", "t"));
}

TEST(RedactConfig, RejectsBadInput) {
  const std::vector<std::string> bad = {
      R"({"detect_fps": 4})",          R"({"detect_fps": 61})",          R"({"detect_fps": 4.99})",
      R"({"score_threshold": 1.5})",   R"({"nms_iou": 0})",
      R"({"match_max_distance": 0})",  R"({"expand_fraction": -0.1})",
      R"({"match_max_distance_loose": 0})", R"({"match_max_distance_loose": -1})",
      R"({"match_max_distance_loose": "6"})", R"({"match_max_distance_loose": 1e400})",
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
      R"({"face_score_threshold": 1e400})",
      // The face model's settings: a score in [0, 1] and an IoU in (0, 1].
      R"({"face_score_threshold": 1.5})",           R"({"face_score_threshold": -0.1})",
      R"({"face_score_threshold": "0.3"})",         R"({"face_nms_iou": 0})",
      R"({"face_nms_iou": 1.5})",                   R"({"face_nms_iou": -0.3})",
      // Mask smoothing: a shrink fraction in (0, 1] and an area ratio of at least 1.
      R"({"shrink_per_frame": 0})",                 R"({"shrink_per_frame": 1.5})",
      R"({"shrink_per_frame": "0.02"})",            R"({"max_mask_area_ratio": 0.5})",
      R"({"max_mask_area_ratio": 1e400})",
  };
  for (const std::string& text : bad) expectRejected(text);
}

TEST(RedactConfig, RejectsOutOfRangeValuesAndNulls) {
  const std::vector<std::string> bad = {
      R"({"nms_iou": 1.5})",           R"({"min_box_px": -1})",
      R"({"detect_fps": null})",       R"({"score_threshold": null})",
      R"({"nms_iou": null})",          R"({"match_max_distance": null})",
      R"({"match_max_distance_loose": null})",
      R"({"expand_fraction": null})",  R"({"motion_expand_fraction": null})",
      R"({"min_box_px": null})",       R"({"mosaic_blocks_per_long_side": null})",
      R"({"face_score_threshold": null})", R"({"face_nms_iou": null})",
      R"({"shrink_per_frame": null})",     R"({"max_mask_area_ratio": null})",
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
      {"score_threshold", [](RedactConfig& c, double v) { c.decode.scoreThreshold = static_cast<float>(v); }},
      {"nms_iou", [](RedactConfig& c, double v) { c.decode.nmsIou = static_cast<float>(v); }},
      {"match_max_distance", [](RedactConfig& c, double v) { c.timeline.matchMaxDistance = v; }},
      {"match_max_distance_loose", [](RedactConfig& c, double v) { c.timeline.matchMaxDistanceLoose = v; }},
      {"expand_fraction", [](RedactConfig& c, double v) { c.timeline.expandFraction = v; }},
      {"motion_expand_fraction", [](RedactConfig& c, double v) { c.timeline.motionExpandFraction = v; }},
      {"min_box_px", [](RedactConfig& c, double v) { c.timeline.minBoxPx = v; }},
      {"shrink_per_frame", [](RedactConfig& c, double v) { c.timeline.shrinkPerFrame = v; }},
      {"max_mask_area_ratio", [](RedactConfig& c, double v) { c.timeline.maxMaskAreaRatio = v; }},
      {"face_score_threshold", [](RedactConfig& c, double v) { c.face.scoreThreshold = static_cast<float>(v); }},
      {"face_nms_iou", [](RedactConfig& c, double v) { c.face.nmsIou = static_cast<float>(v); }},
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

TEST(RedactConfig, DetectFpsMessageGivesTheRange) {
  for (const char* text : {R"({"detect_fps": 4})", R"({"detect_fps": 61})"}) {
    try {
      parseRedactConfig(text, "cfg");
      ADD_FAILURE() << "accepted " << text;
    } catch (const VcError& e) {
      EXPECT_EQ(std::string(e.what()), "cfg: detect_fps must be in [5, 60]");
    }
  }
}

TEST(RedactConfig, AcceptsTheDetectFpsBounds) {
  EXPECT_DOUBLE_EQ(parseRedactConfig(R"({"detect_fps": 5})", "t").detectFps, 5);
  EXPECT_DOUBLE_EQ(parseRedactConfig(R"({"detect_fps": 60})", "t").detectFps, 60);
}

// The motion-triggered rate is gone (spec 2026-10-19 §2): an old config that still sets it is refused with a message
// that says the key was removed, not that it is an unknown typo.
TEST(RedactConfig, RejectsTheRemovedKeysByName) {
  for (const std::string key : {"max_detect_fps", "motion_threshold"}) {
    try {
      parseRedactConfig("{\"" + key + "\": 15}", "cfg");
      ADD_FAILURE() << "accepted " << key;
    } catch (const VcError& e) {
      EXPECT_EQ(std::string(e.what()).rfind("cfg: " + key + " was removed", 0), 0u) << e.what();
    }
  }
}

TEST(RedactConfig, SerialisesEveryKey) {
  const nlohmann::ordered_json j = redactConfigToJson(RedactConfig{});
  EXPECT_EQ(j.size(), 14u);
  EXPECT_DOUBLE_EQ(j.at("match_max_distance_loose").get<double>(), 6.0);
  EXPECT_DOUBLE_EQ(j.at("detect_fps").get<double>(), 10.0);
  EXPECT_EQ(j.at("mosaic_blocks_per_long_side").get<int>(), 8);
}

TEST(RedactConfig, FloatSettingsAreWrittenWithSixDecimals) {
  const nlohmann::ordered_json j = redactConfigToJson(RedactConfig{});
  EXPECT_EQ(j.at("score_threshold").dump(), "0.05");
  EXPECT_EQ(j.at("nms_iou").dump(), "0.5");
  EXPECT_EQ(j.at("face_score_threshold").dump(), "0.3");
  EXPECT_EQ(j.at("face_nms_iou").dump(), "0.3");
}

TEST(RedactConfig, DumpParsesBackToTheSameConfig) {
  const RedactConfig c = nonDefaultConfig();
  const std::string text = redactConfigToJson(c).dump();
  expectSameConfig(parseRedactConfig(text, "round"), c);
}

}  // namespace
}  // namespace vc
