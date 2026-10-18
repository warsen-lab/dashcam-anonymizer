#include "detect/model_info.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "common/vc_error.h"

namespace vc {
namespace {

const std::string kValid =
    R"({"name":"m","classes":["face","plate"],"input":{"width":1920,"height":1088,"color":"RGB",)"
    R"("scale":0.00392156862745098,"letterbox":"top_left","pad_value":114},"output_format":"yolov8",)"
    R"("train_manifest_sha256":"","metrics":{"val_map50":0.5}})";

std::string replaced(std::string text, const std::string& from, const std::string& to) {
  text.replace(text.find(from), from.size(), to);
  return text;
}

TEST(ModelInfo, ParsesTheVcTrainContract) {
  const ModelInfo m = parseModelInfo(kValid, "model.json");
  EXPECT_EQ(m.name, "m");
  EXPECT_EQ(m.classes, (std::vector<std::string>{"face", "plate"}));
  EXPECT_EQ(m.inputWidth, 1920);
  EXPECT_EQ(m.inputHeight, 1088);
}

TEST(ModelInfo, RejectsUnsupportedValues) {
  const std::vector<std::string> bad = {
      replaced(kValid, R"("RGB")", R"("BGR")"),
      replaced(kValid, R"("top_left")", R"("center")"),
      replaced(kValid, R"("yolov8")", R"("yolov5")"),
      replaced(kValid, "1088", "1080"),
      replaced(kValid, R"("pad_value":114)", R"("pad_value":0)"),
      replaced(kValid, "0.00392156862745098", "1.0"),
      replaced(kValid, R"(["face","plate"])", "[]"),
      R"({"classes":["face"],"output_format":"yolov8"})",
      "{",
      "[]",
  };
  for (const std::string& text : bad) {
    try {
      parseModelInfo(text, "model.json");
      ADD_FAILURE() << "accepted " << text;
    } catch (const VcError& e) {
      EXPECT_EQ(std::string(e.what()).rfind("model.json: ", 0), 0u) << e.what();
    }
  }
}

}  // namespace
}  // namespace vc
