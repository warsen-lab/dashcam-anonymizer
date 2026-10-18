#include "vc_redact/redact_config.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "common/vc_error.h"

namespace vc {
namespace {

void expectDefaults(const RedactConfig& c) {
  const RedactConfig d;
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

TEST(RedactConfig, RepositoryFileMatchesTheDefaults) {
  expectDefaults(readRedactConfig(std::filesystem::path(VC_REPO_DIR) / "configs" / "redact.json"));
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
      R"({"mosaic_min_block_px": -3})",             R"({"mosaic_min_block_px": 16.0})",
      R"({"mosaic_min_block_px": null})",           R"({"mosaic_min_block_px": true})",
      R"({"mosaic_blocks_per_long_side": 16.5})",   R"({"mosaic_blocks_per_long_side": 1e10})",
      R"({"mosaic_blocks_per_long_side": 10000000000})", R"({"mosaic_blocks_per_long_side": 4097})",
      R"({"mosaic_blocks_per_long_side": 0})",      R"({"mosaic_blocks_per_long_side": "8"})",
  };
  for (const std::string& text : bad) {
    try {
      parseRedactConfig(text, "cfg");
      ADD_FAILURE() << "accepted " << text;
    } catch (const VcError& e) {
      EXPECT_EQ(std::string(e.what()).rfind("cfg: ", 0), 0u) << e.what();
    }
  }
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

TEST(RedactConfig, SerialisesEveryKey) {
  const nlohmann::ordered_json j = redactConfigToJson(RedactConfig{});
  EXPECT_EQ(j.size(), 11u);
  EXPECT_DOUBLE_EQ(j.at("detect_fps").get<double>(), 5.0);
  EXPECT_EQ(j.at("mosaic_blocks_per_long_side").get<int>(), 8);
}

}  // namespace
}  // namespace vc
