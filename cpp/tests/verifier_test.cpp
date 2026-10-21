#include "detect/verifier.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

fs::path constVerifier() { return fs::path(VC_TEST_DATA_DIR) / "const_verifier"; }

std::string readText(const fs::path& p) {
  std::ifstream in(p);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

TEST(VerifierInfo, ReadsTheConstModel) {
  const VerifierInfo info = readVerifierInfo(constVerifier() / "model.json");
  EXPECT_EQ(info.name, "const_verifier");
  EXPECT_EQ(info.classes, (std::vector<std::string>{"face", "plate"}));
  EXPECT_EQ(info.inputSize, 64);
  EXPECT_EQ(info.batch, 4);
  EXPECT_EQ(info.padValue, 114);
  EXPECT_DOUBLE_EQ(info.cropFactor, 2.5);
  EXPECT_DOUBLE_EQ(info.minCropPx, 32);
  EXPECT_EQ(info.passThreshold, (std::vector<float>{0.5f, 0.5f}));
  EXPECT_EQ(info.crop().size, 64);
}

TEST(VerifierInfo, RefusesAnotherKindOrMissingThresholds) {
  const std::string text = readText(constVerifier() / "model.json");
  const std::string kind = R"("kind": "verifier")";
  std::string wrongKind = text;
  wrongKind.replace(wrongKind.find(kind), kind.size(), R"("kind": "detector")");
  EXPECT_THROW(parseVerifierInfo(wrongKind, "m.json"), VcError);
  std::string noPlate = text;
  const std::string plate = R"(, "plate": 0.5)";
  noPlate.erase(noPlate.find(plate), plate.size());
  EXPECT_THROW(parseVerifierInfo(noPlate, "m.json"), VcError);
  EXPECT_THROW(parseVerifierInfo("[]", "m.json"), VcError);
  EXPECT_THROW(parseVerifierInfo("{", "m.json"), VcError);
}

TEST(Verifier, ScoresEachCropByItsClass) {
  Verifier v(constVerifier(), ExecutionProvider::Cpu, SessionSettings{});
  EXPECT_EQ(v.backend(), "CPU");
  std::vector<float> pixels(3 * 64 * 64, 0.5f);
  // 6 crops: more than one batch of 4, so the second batch is padded
  std::vector<VerifierInput> in;
  for (int i = 0; i < 6; ++i) in.push_back(VerifierInput{i % 2, pixels.data()});
  const std::vector<float> s = v.score(in);
  ASSERT_EQ(s.size(), 6u);
  const float face = 1.0f / (1.0f + std::exp(2.0f));
  const float plate = 1.0f / (1.0f + std::exp(-2.0f));
  for (int i = 0; i < 6; ++i) EXPECT_NEAR(s[i], i % 2 == 0 ? face : plate, 1e-5) << i;
  EXPECT_TRUE(v.score({}).empty());
  EXPECT_THROW(v.score({VerifierInput{2, pixels.data()}}), VcError);
}

TEST(Verifier, RefusesAMissingModelOrAShapeThatDisagreesWithModelJson) {
  EXPECT_THROW(Verifier(test::tmpDir() / "no_such_verifier", ExecutionProvider::Cpu, SessionSettings{}), VcError);
  const fs::path dir = test::tmpDir() / "verifier_wrong_batch";
  fs::create_directories(dir);
  fs::copy_file(constVerifier() / "model.onnx", dir / "model.onnx", fs::copy_options::overwrite_existing);
  std::string text = readText(constVerifier() / "model.json");
  const std::string batch = R"("batch": 4)";
  text.replace(text.find(batch), batch.size(), R"("batch": 8)");
  std::ofstream(dir / "model.json") << text;
  EXPECT_THROW(Verifier(dir, ExecutionProvider::Cpu, SessionSettings{}), VcError);
}

}  // namespace
}  // namespace vc
