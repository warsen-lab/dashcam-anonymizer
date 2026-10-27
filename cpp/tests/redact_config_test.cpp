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
    "face_score_threshold", "face_nms_iou",       "face_confirm_score",
    "plate_confirm_score", "large_face_area_fraction", "large_face_min_score",
    "hold_lost_ghosts",    "output_bitrate_factor_hevc", "rate_control",
    "nvenc_preset",        "trt_fp16",            "ort_intra_threads",
    "verify_plate_below",  "verify_face_unless_both_above", "verify_threshold_face",
    "verify_threshold_plate", "plate_min_width_px", "track_lookahead_keyframes",
    "track_backfill_keyframes"};

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
  EXPECT_FLOAT_EQ(c.faceConfirmScore, d.faceConfirmScore);
  EXPECT_FLOAT_EQ(c.plateConfirmScore, d.plateConfirmScore);
  EXPECT_DOUBLE_EQ(c.largeFaceAreaFraction, d.largeFaceAreaFraction);
  EXPECT_FLOAT_EQ(c.largeFaceMinScore, d.largeFaceMinScore);
  EXPECT_DOUBLE_EQ(c.plateMinWidthPx, d.plateMinWidthPx);
  EXPECT_FLOAT_EQ(c.verifyPlateBelow, d.verifyPlateBelow);
  EXPECT_FLOAT_EQ(c.verifyFaceUnlessBothAbove, d.verifyFaceUnlessBothAbove);
  EXPECT_EQ(c.verifyThresholdFace, d.verifyThresholdFace);
  EXPECT_EQ(c.verifyThresholdPlate, d.verifyThresholdPlate);
  EXPECT_EQ(c.timeline.holdLostGhosts, d.timeline.holdLostGhosts);
  EXPECT_EQ(c.trackLookaheadKeyframes, d.trackLookaheadKeyframes);
  EXPECT_EQ(c.trackBackfillKeyframes, d.trackBackfillKeyframes);
  EXPECT_DOUBLE_EQ(c.outputBitrateFactorHevc, d.outputBitrateFactorHevc);
  EXPECT_EQ(c.rateControl, d.rateControl);
  EXPECT_EQ(c.nvencPreset, d.nvencPreset);
  EXPECT_EQ(c.trtFp16, d.trtFp16);
  EXPECT_EQ(c.ortIntraThreads, d.ortIntraThreads);
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
  c.faceConfirmScore = 0.375f;
  c.plateConfirmScore = 0.0625f;
  c.largeFaceAreaFraction = 0.25;
  c.largeFaceMinScore = 0.75f;
  c.plateMinWidthPx = 20.5;
  c.verifyPlateBelow = 0.375f;
  c.verifyFaceUnlessBothAbove = 0.625f;
  c.verifyThresholdFace = 0.25f;
  c.verifyThresholdPlate = 0.125f;
  c.timeline.holdLostGhosts = true;
  c.trackLookaheadKeyframes = 9;
  c.trackBackfillKeyframes = 4;
  c.outputBitrateFactorHevc = 2.25;
  c.rateControl = RateControl::Bitrate;
  c.nvencPreset = "p1";
  c.trtFp16 = false;
  c.ortIntraThreads = 3;
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

// vc_redact's defaults: 10 fps and score 0.05 from the 2026-10-18 proxy validation, the mask and confirmation
// settings from the 2026-10-19 calibration (docs/reports/stable-masks-2026-10-19.md); the lower-level structs keep
// their own, which vc_detect and the unit tests use.
TEST(RedactConfig, DefaultsAreTheValidatedSettings) {
  const RedactConfig c;
  EXPECT_DOUBLE_EQ(c.detectFps, 10.0);
  EXPECT_FLOAT_EQ(c.decode.scoreThreshold, 0.05f);
  EXPECT_DOUBLE_EQ(c.timeline.expandFraction, 0.2);
  EXPECT_DOUBLE_EQ(c.timeline.motionExpandFraction, 0.25);
  EXPECT_DOUBLE_EQ(c.timeline.shrinkPerFrame, 0.2);
  EXPECT_FLOAT_EQ(c.faceConfirmScore, 0.5f);
  EXPECT_FLOAT_EQ(c.plateConfirmScore, 0.2f);
  EXPECT_DOUBLE_EQ(c.largeFaceAreaFraction, 0.05);
  EXPECT_FLOAT_EQ(c.largeFaceMinScore, 0.6f);
  EXPECT_DOUBLE_EQ(c.plateMinWidthPx, 16.0);
  EXPECT_FALSE(c.timeline.holdLostGhosts);
  EXPECT_EQ(c.trackLookaheadKeyframes, 5);
  EXPECT_EQ(c.trackBackfillKeyframes, 2);
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

// Mask smoothing (spec 2026-10-19 §3.1): a mask's width and height shrink by at most 20 % per frame, and a mask
// covers at most 4 times its box. The shrink rate is the 2026-10-19 calibration's.
TEST(RedactConfig, ReadsTheMaskSmoothingSettings) {
  EXPECT_DOUBLE_EQ(RedactConfig{}.timeline.shrinkPerFrame, 0.2);
  EXPECT_DOUBLE_EQ(RedactConfig{}.timeline.maxMaskAreaRatio, 4.0);
  const RedactConfig c = parseRedactConfig(R"({"shrink_per_frame": 0.05, "max_mask_area_ratio": 3})", "t");
  EXPECT_DOUBLE_EQ(c.timeline.shrinkPerFrame, 0.05);
  EXPECT_DOUBLE_EQ(c.timeline.maxMaskAreaRatio, 3.0);
  EXPECT_NO_THROW(parseRedactConfig(R"({"shrink_per_frame": 1, "max_mask_area_ratio": 1})", "t"));
}

// Low-score confirmation (spec 2026-10-19 §3.4): faces below 0.5 and plates below 0.2 are masked only once seen on
// two consecutive keyframes. The values are the 2026-10-19 calibration's.
TEST(RedactConfig, ReadsTheConfirmScores) {
  EXPECT_FLOAT_EQ(RedactConfig{}.faceConfirmScore, 0.5f);
  EXPECT_FLOAT_EQ(RedactConfig{}.plateConfirmScore, 0.2f);
  const RedactConfig c = parseRedactConfig(R"({"face_confirm_score": 0.4, "plate_confirm_score": 0})", "t");
  EXPECT_FLOAT_EQ(c.faceConfirmScore, 0.4f);
  EXPECT_FLOAT_EQ(c.plateConfirmScore, 0.0f);
}

// The thresholds go to the timeline by the detector's class names, in its class order; other classes need none.
TEST(RedactConfig, ConfirmScoresFollowTheDetectorsClassNames) {
  RedactConfig c;
  c.faceConfirmScore = 0.4f;
  c.plateConfirmScore = 0.2f;
  c.timeline.expandFraction = 0.25;
  EXPECT_EQ(timelineFor(c, {"plate", "face"}).confirmScore, (std::vector<float>{0.2f, 0.4f}));
  EXPECT_EQ(timelineFor(c, {"face", "plate", "sign"}).confirmScore, (std::vector<float>{0.4f, 0.2f, 0.0f}));
  EXPECT_DOUBLE_EQ(timelineFor(c, {"face"}).expandFraction, 0.25);
}

// Large faces need a higher score (spec 2026-10-19 §3.6): a face box over 5 % of the frame scoring below 0.6 is
// dropped. The area fraction is in (0, 1], the score in [0, 1] (0 switches the rule off).
TEST(RedactConfig, ReadsTheLargeFaceRule) {
  EXPECT_DOUBLE_EQ(RedactConfig{}.largeFaceAreaFraction, 0.05);
  EXPECT_FLOAT_EQ(RedactConfig{}.largeFaceMinScore, 0.6f);
  const RedactConfig c = parseRedactConfig(R"({"large_face_area_fraction": 0.1, "large_face_min_score": 0.7})", "t");
  EXPECT_DOUBLE_EQ(c.largeFaceAreaFraction, 0.1);
  EXPECT_FLOAT_EQ(c.largeFaceMinScore, 0.7f);
  const RedactConfig edges = parseRedactConfig(R"({"large_face_area_fraction": 1, "large_face_min_score": 0})", "t");
  EXPECT_DOUBLE_EQ(edges.largeFaceAreaFraction, 1.0);
  EXPECT_FLOAT_EQ(edges.largeFaceMinScore, 0.0f);
  EXPECT_FLOAT_EQ(parseRedactConfig(R"({"large_face_min_score": 1})", "t").largeFaceMinScore, 1.0f);
}

// Plates narrower than plate_min_width_px are not masked (2026-10-22: too small to read); 0 is off.
TEST(RedactConfig, ReadsThePlateMinimumWidth) {
  EXPECT_DOUBLE_EQ(parseRedactConfig(R"({"plate_min_width_px": 24})", "t").plateMinWidthPx, 24.0);
  EXPECT_DOUBLE_EQ(parseRedactConfig(R"({"plate_min_width_px": 0})", "t").plateMinWidthPx, 0.0);
  for (const std::string value : {"-1", R"("24")", "null", "true"}) {
    const std::string text = R"({"plate_min_width_px": )" + value + "}";
    SCOPED_TRACE(text);
    try {
      parseRedactConfig(text, "cfg");
      ADD_FAILURE() << "accepted";
    } catch (const VcError& e) {
      EXPECT_NE(std::string(e.what()).find("plate_min_width_px"), std::string::npos) << e.what();
    }
  }
  RedactConfig c;
  c.plateMinWidthPx = std::numeric_limits<double>::infinity();
  EXPECT_THROW(validateRedactConfig(c, "cli"), VcError);
}

// Track-level masking (spec 2026-10-27): lookahead in [0, 30], backfill in [0, lookahead - 1].
TEST(RedactConfig, ReadsTheTrackLookahead) {
  const RedactConfig c = parseRedactConfig(R"({"track_lookahead_keyframes": 8, "track_backfill_keyframes": 7})", "t");
  EXPECT_EQ(c.trackLookaheadKeyframes, 8);
  EXPECT_EQ(c.trackBackfillKeyframes, 7);
  const RedactConfig off = parseRedactConfig(R"({"track_lookahead_keyframes": 0, "track_backfill_keyframes": 0})", "t");
  EXPECT_EQ(off.trackLookaheadKeyframes, 0);
  for (const std::string text :
       {R"({"track_lookahead_keyframes": 31})", R"({"track_lookahead_keyframes": -1})",
        R"({"track_lookahead_keyframes": 2.5})", R"({"track_backfill_keyframes": "2"})",
        R"({"track_backfill_keyframes": 5})", R"({"track_lookahead_keyframes": 0})"}) {
    SCOPED_TRACE(text);
    try {
      parseRedactConfig(text, "cfg");
      ADD_FAILURE() << "accepted";
    } catch (const VcError& e) {
      const std::string message = e.what();
      EXPECT_EQ(message.rfind("cfg: ", 0), 0u) << message;
      EXPECT_NE(message.find("track_"), std::string::npos) << message;
    }
  }
  const TrackMaskerConfig m = trackMaskerFor(c, {"face", "plate"});
  EXPECT_EQ(m.lookahead, 8);
  EXPECT_EQ(m.backfill, 7);
  EXPECT_EQ(m.timeline.confirmScore, (std::vector<float>{0.5f, 0.2f}));
}

// Each refusal starts with the origin and names the key.
TEST(RedactConfig, RefusesBadLargeFaceSettingsNamingTheKey) {
  const std::vector<std::pair<std::string, std::string>> bad = {
      {"large_face_area_fraction", "0"},     {"large_face_area_fraction", "1.5"},
      {"large_face_area_fraction", "-0.1"},  {"large_face_area_fraction", R"("0.05")"},
      {"large_face_area_fraction", "null"},  {"large_face_area_fraction", "true"},
      {"large_face_min_score", "-0.1"},      {"large_face_min_score", "1.5"},
      {"large_face_min_score", R"("0.6")"},  {"large_face_min_score", "null"},
      {"large_face_min_score", "false"},
  };
  for (const auto& [key, value] : bad) {
    const std::string text = "{\"" + key + "\": " + value + "}";
    SCOPED_TRACE(text);
    try {
      parseRedactConfig(text, "cfg");
      ADD_FAILURE() << "accepted";
    } catch (const VcError& e) {
      const std::string message = e.what();
      EXPECT_EQ(message.rfind("cfg: ", 0), 0u) << message;
      EXPECT_NE(message.find(key), std::string::npos) << message;
    }
  }
  RedactConfig c;
  c.largeFaceAreaFraction = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(validateRedactConfig(c, "cli"), VcError);
  c = RedactConfig{};
  c.largeFaceMinScore = std::numeric_limits<float>::infinity();
  EXPECT_THROW(validateRedactConfig(c, "cli"), VcError);
}

// hold_lost_ghosts (spec 2026-10-19 §3.7) is a JSON boolean, off by default, and reaches the timeline config.
TEST(RedactConfig, ReadsHoldLostGhostsAsABoolean) {
  EXPECT_FALSE(RedactConfig{}.timeline.holdLostGhosts);
  EXPECT_TRUE(parseRedactConfig(R"({"hold_lost_ghosts": true})", "t").timeline.holdLostGhosts);
  EXPECT_FALSE(parseRedactConfig(R"({"hold_lost_ghosts": false})", "t").timeline.holdLostGhosts);
  for (const char* text : {R"({"hold_lost_ghosts": 1})", R"({"hold_lost_ghosts": 0})",
                           R"({"hold_lost_ghosts": "true"})", R"({"hold_lost_ghosts": null})"}) {
    SCOPED_TRACE(text);
    expectRejected(text);
  }
  RedactConfig c;
  c.timeline.holdLostGhosts = true;
  EXPECT_TRUE(timelineFor(c, {"face", "plate"}).holdLostGhosts);
  EXPECT_FALSE(timelineFor(RedactConfig{}, {"face", "plate"}).holdLostGhosts);
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
      // The confirmation thresholds are scores in [0, 1].
      R"({"face_confirm_score": 1.5})",             R"({"plate_confirm_score": -0.1})",
      R"({"face_confirm_score": "0.5"})",
      // The large-face rule: numbers that overflow while parsing, and the range edges.
      R"({"large_face_area_fraction": 1e400})",     R"({"large_face_min_score": 1e400})",
      R"({"large_face_area_fraction": 1.0000001})", R"({"large_face_min_score": 1.0000001})",
      // The Linux settings (spec 2026-10-20): a bitrate factor in [1, 4], a preset p1-p7, a boolean, a thread count.
      R"({"output_bitrate_factor_hevc": 0.99})",    R"({"output_bitrate_factor_hevc": 4.01})",
      R"({"output_bitrate_factor_hevc": "1.5"})",   R"({"output_bitrate_factor_hevc": 1e400})",
      R"({"output_bitrate_factor_hevc": null})",
      R"({"rate_control": "crf"})",                 R"({"rate_control": "Quality"})",
      R"({"rate_control": ""})",                    R"({"rate_control": 1})",
      R"({"rate_control": null})",                  R"({"rate_control": true})",
      R"({"nvenc_preset": "p0"})",                  R"({"nvenc_preset": "p8"})",
      R"({"nvenc_preset": "P4"})",                  R"({"nvenc_preset": "fast"})",
      R"({"nvenc_preset": 4})",                     R"({"nvenc_preset": null})",
      R"({"trt_fp16": 1})",                         R"({"trt_fp16": "true"})",
      R"({"trt_fp16": null})",
      R"({"ort_intra_threads": -1})",               R"({"ort_intra_threads": 1025})",
      R"({"ort_intra_threads": 2.5})",              R"({"ort_intra_threads": "4"})",
      R"({"ort_intra_threads": 4294967297})",       R"({"ort_intra_threads": null})",
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
      R"({"face_confirm_score": null})",   R"({"plate_confirm_score": null})",
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
      {"face_confirm_score", [](RedactConfig& c, double v) { c.faceConfirmScore = static_cast<float>(v); }},
      {"plate_confirm_score", [](RedactConfig& c, double v) { c.plateConfirmScore = static_cast<float>(v); }},
      {"large_face_area_fraction", [](RedactConfig& c, double v) { c.largeFaceAreaFraction = v; }},
      {"large_face_min_score", [](RedactConfig& c, double v) { c.largeFaceMinScore = static_cast<float>(v); }},
      {"output_bitrate_factor_hevc", [](RedactConfig& c, double v) { c.outputBitrateFactorHevc = v; }},
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
  EXPECT_EQ(j.size(), 31u);
  EXPECT_DOUBLE_EQ(j.at("match_max_distance_loose").get<double>(), 6.0);
  EXPECT_DOUBLE_EQ(j.at("detect_fps").get<double>(), 10.0);
  EXPECT_EQ(j.at("mosaic_blocks_per_long_side").get<int>(), 8);
  EXPECT_DOUBLE_EQ(j.at("large_face_area_fraction").get<double>(), 0.05);
  ASSERT_TRUE(j.at("hold_lost_ghosts").is_boolean());
  EXPECT_FALSE(j.at("hold_lost_ghosts").get<bool>());
  std::set<std::string> keys;
  for (const auto& item : j.items()) keys.insert(item.key());
  EXPECT_EQ(keys, kAllKeys);
}

TEST(RedactConfig, FloatSettingsAreWrittenWithSixDecimals) {
  const nlohmann::ordered_json j = redactConfigToJson(RedactConfig{});
  EXPECT_EQ(j.at("score_threshold").dump(), "0.05");
  EXPECT_EQ(j.at("nms_iou").dump(), "0.5");
  EXPECT_EQ(j.at("face_score_threshold").dump(), "0.3");
  EXPECT_EQ(j.at("face_nms_iou").dump(), "0.3");
  EXPECT_EQ(j.at("face_confirm_score").dump(), "0.5");
  EXPECT_EQ(j.at("plate_confirm_score").dump(), "0.2");
  EXPECT_EQ(j.at("large_face_area_fraction").dump(), "0.05");
  EXPECT_EQ(j.at("large_face_min_score").dump(), "0.6");
  EXPECT_EQ(j.at("hold_lost_ghosts").dump(), "false");
}

// spec 2026-10-20 §3, §5: an HEVC source's H.264 gets 1.5 times its bitrate, NVENC runs preset p4, TensorRT in FP16,
// and onnxruntime keeps its own thread count.
TEST(RedactConfig, LinuxSettingsDefaults) {
  const RedactConfig c;
  EXPECT_DOUBLE_EQ(c.outputBitrateFactorHevc, 1.5);
  EXPECT_EQ(c.rateControl, RateControl::Quality);
  EXPECT_EQ(c.nvencPreset, "p4");
  EXPECT_TRUE(c.trtFp16);
  EXPECT_EQ(c.ortIntraThreads, 0);
  const nlohmann::ordered_json j = redactConfigToJson(c);
  EXPECT_EQ(j.at("output_bitrate_factor_hevc").dump(), "1.5");
  EXPECT_EQ(j.at("rate_control").dump(), "\"quality\"");
  EXPECT_EQ(j.at("nvenc_preset").dump(), "\"p4\"");
  EXPECT_EQ(j.at("trt_fp16").dump(), "true");
  EXPECT_EQ(j.at("ort_intra_threads").dump(), "0");
}

TEST(RedactConfig, ReadsTheLinuxSettings) {
  const RedactConfig c = parseRedactConfig(
      R"({"output_bitrate_factor_hevc": 2, "nvenc_preset": "p7", "trt_fp16": false, "ort_intra_threads": 2})", "t");
  EXPECT_DOUBLE_EQ(c.outputBitrateFactorHevc, 2.0);
  EXPECT_EQ(c.nvencPreset, "p7");
  EXPECT_FALSE(c.trtFp16);
  EXPECT_EQ(c.ortIntraThreads, 2);
  for (const char* preset : {"p1", "p2", "p3", "p4", "p5", "p6", "p7"}) {
    EXPECT_EQ(parseRedactConfig(std::string(R"({"nvenc_preset": ")") + preset + "\"}", "t").nvencPreset, preset);
  }
  EXPECT_DOUBLE_EQ(parseRedactConfig(R"({"output_bitrate_factor_hevc": 1})", "t").outputBitrateFactorHevc, 1.0);
  EXPECT_DOUBLE_EQ(parseRedactConfig(R"({"output_bitrate_factor_hevc": 4})", "t").outputBitrateFactorHevc, 4.0);
  EXPECT_EQ(parseRedactConfig(R"({"ort_intra_threads": 1024})", "t").ortIntraThreads, 1024);
}

// The messages name the key and the allowed values.
TEST(RedactConfig, LinuxSettingMessagesGiveTheRule) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {R"({"output_bitrate_factor_hevc": 5})", "cfg: output_bitrate_factor_hevc must be in [1, 4]"},
      {R"({"rate_control": "crf"})", "cfg: rate_control must be \"quality\" or \"bitrate\", got 'crf'"},
      {R"({"rate_control": 1})", "cfg: rate_control must be a string"},
      {R"({"rate_control": null})", "cfg: rate_control must be a string"},
      {R"({"nvenc_preset": "p9"})", "cfg: nvenc_preset must be one of p1 ... p7, got 'p9'"},
      {R"({"nvenc_preset": 4})", "cfg: nvenc_preset must be a string"},
      {R"({"trt_fp16": 1})", "cfg: trt_fp16 must be true or false"},
      {R"({"ort_intra_threads": -1})", "cfg: ort_intra_threads must be an integer in [0, 1024]"},
      {R"({"mosaic_min_block_px": 0})", "cfg: mosaic_min_block_px must be an integer in [1, 4096]"},
  };
  for (const auto& [text, message] : cases) {
    try {
      parseRedactConfig(text, "cfg");
      ADD_FAILURE() << "accepted " << text;
    } catch (const VcError& e) {
      EXPECT_EQ(std::string(e.what()), message);
    }
  }
}

TEST(RedactConfig, ReadsAndWritesTheRateControl) {
  EXPECT_EQ(parseRedactConfig(R"({"rate_control": "quality"})", "t").rateControl, RateControl::Quality);
  EXPECT_EQ(parseRedactConfig(R"({"rate_control": "bitrate"})", "t").rateControl, RateControl::Bitrate);
  EXPECT_EQ(parseRedactConfig("{}", "t").rateControl, RateControl::Quality);
  RedactConfig c;
  c.rateControl = RateControl::Bitrate;
  EXPECT_EQ(redactConfigToJson(c).at("rate_control").dump(), "\"bitrate\"");
}

TEST(RedactConfig, ValidateChecksTheLinuxSettings) {
  RedactConfig c;
  c.rateControl = static_cast<RateControl>(7);
  EXPECT_THROW(validateRedactConfig(c, "cli"), VcError);
  c = RedactConfig{};
  c.nvencPreset = "p0";
  EXPECT_THROW(validateRedactConfig(c, "cli"), VcError);
  c = RedactConfig{};
  c.ortIntraThreads = 1025;
  EXPECT_THROW(validateRedactConfig(c, "cli"), VcError);
  c = RedactConfig{};
  c.outputBitrateFactorHevc = 0.5;
  EXPECT_THROW(validateRedactConfig(c, "cli"), VcError);
}

TEST(RedactConfig, DumpParsesBackToTheSameConfig) {
  const RedactConfig c = nonDefaultConfig();
  const std::string text = redactConfigToJson(c).dump();
  expectSameConfig(parseRedactConfig(text, "round"), c);
}

TEST(RedactConfig, VerifySettingsDefaultToTheSpec) {
  const RedactConfig c;
  EXPECT_FLOAT_EQ(c.verifyPlateBelow, 0.5f);
  EXPECT_FLOAT_EQ(c.verifyFaceUnlessBothAbove, 0.5f);
  EXPECT_FALSE(c.verifyThresholdFace.has_value());
  EXPECT_FALSE(c.verifyThresholdPlate.has_value());
}

TEST(RedactConfig, ReadsAndWritesTheVerifySettings) {
  const RedactConfig c = parseRedactConfig(
      R"({"verify_plate_below": 0.4, "verify_face_unless_both_above": 0.6, "verify_threshold_face": 0.25,
          "verify_threshold_plate": null})",
      "t");
  EXPECT_FLOAT_EQ(c.verifyPlateBelow, 0.4f);
  EXPECT_FLOAT_EQ(c.verifyFaceUnlessBothAbove, 0.6f);
  ASSERT_TRUE(c.verifyThresholdFace.has_value());
  EXPECT_FLOAT_EQ(*c.verifyThresholdFace, 0.25f);
  EXPECT_FALSE(c.verifyThresholdPlate.has_value());
  const auto j = redactConfigToJson(c);
  EXPECT_DOUBLE_EQ(j.at("verify_threshold_face").get<double>(), 0.25);
  EXPECT_TRUE(j.at("verify_threshold_plate").is_null());
}

TEST(RedactConfig, RefusesVerifySettingsOutOfRange) {
  expectRejected(R"({"verify_plate_below": 1.5})");
  expectRejected(R"({"verify_face_unless_both_above": -0.1})");
  expectRejected(R"({"verify_threshold_face": 2})");
  expectRejected(R"({"verify_threshold_plate": "0.5"})");
}

}  // namespace
}  // namespace vc
