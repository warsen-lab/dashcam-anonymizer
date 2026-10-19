#include "detect/model_info.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "test_media.h"

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
      replaced(kValid, R"("height":1088)", R"("height":1088.5)"),
      replaced(kValid, R"("pad_value":114)", R"("pad_value":114.0)"),
      replaced(kValid, R"("width":1920)", R"("width":"1920")"),
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

const std::string kYuNet =
    R"({"name":"yunet-2023mar","classes":["face"],"input":{"color":"BGR","scale":1.0,"letterbox":"top_left",)"
    R"("pad_value":0,"pad_multiple":32},"output_format":"yunet","source_sha256":"8f23"})";

TEST(ModelInfo, ParsesTheVcTrainYuNetContract) {
  const ModelInfo m = parseModelInfo(kYuNet, "model.json");
  EXPECT_EQ(m.name, "yunet-2023mar");
  EXPECT_EQ(m.outputFormat, OutputFormat::YuNet);
  EXPECT_EQ(m.classes, (std::vector<std::string>{"face"}));
  EXPECT_EQ(m.padMultiple, 32);
  EXPECT_EQ(m.inputWidth, 0);  // the input follows the frame
  EXPECT_EQ(m.inputHeight, 0);
  EXPECT_EQ(parseModelInfo(kValid, "model.json").outputFormat, OutputFormat::Yolov8);
  EXPECT_EQ(parseModelInfo(replaced(kYuNet, R"("scale":1.0)", R"("scale":1)"), "model.json").padMultiple, 32);
}

TEST(ModelInfo, RejectsUnsupportedYuNetValues) {
  const std::vector<std::string> bad = {
      replaced(kYuNet, R"("BGR")", R"("RGB")"),
      replaced(kYuNet, R"("scale":1.0)", R"("scale":0.00392156862745098)"),
      replaced(kYuNet, R"("top_left")", R"("center")"),
      replaced(kYuNet, R"("pad_value":0)", R"("pad_value":114)"),
      replaced(kYuNet, R"("pad_value":0)", R"("pad_value":0.0)"),
      replaced(kYuNet, R"("pad_multiple":32)", R"("pad_multiple":16)"),
      replaced(kYuNet, R"("pad_multiple":32)", R"("pad_multiple":32.0)"),
      replaced(kYuNet, R"(,"pad_multiple":32)", ""),
      replaced(kYuNet, R"(["face"])", "[]"),
      replaced(kYuNet, R"(["face"])", R"(["face","plate"])"),
      replaced(kYuNet, R"("yunet")", R"("yunet2")"),
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

TEST(ModelInfo, ReadsAModelJsonFile) {
  const std::filesystem::path path = test::tmpDir() / "model_info_valid.json";
  std::ofstream(path) << kValid;
  const ModelInfo m = readModelInfo(path);
  EXPECT_EQ(m.name, "m");
  EXPECT_EQ(m.classes, (std::vector<std::string>{"face", "plate"}));
  EXPECT_EQ(m.inputWidth, 1920);
  EXPECT_EQ(m.inputHeight, 1088);
}

TEST(ModelInfo, NamesTheFileInItsErrors) {
  const std::filesystem::path missing = test::tmpDir() / "model_info_missing.json";
  std::filesystem::remove(missing);
  try {
    readModelInfo(missing);
    ADD_FAILURE() << "no throw for a missing file";
  } catch (const VcError& e) {
    EXPECT_EQ(std::string(e.what()), "cannot read " + missing.string());
  }

  const std::filesystem::path broken = test::tmpDir() / "model_info_broken.json";
  std::ofstream(broken) << replaced(kValid, R"("yolov8")", R"("yolov5")");
  try {
    readModelInfo(broken);
    ADD_FAILURE() << "no throw for an unsupported file";
  } catch (const VcError& e) {
    EXPECT_EQ(std::string(e.what()).rfind(broken.string() + ": ", 0), 0u) << e.what();
  }
}

}  // namespace
}  // namespace vc
