#include "vc_redact/verify_dump.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;

TEST(VerifyDump, WritesTheCropsAndOneLineEachInOrder) {
  const fs::path dir = test::tmpDir() / "verify_dump";
  fs::remove_all(dir);
  fs::create_directories(dir);
  {
    VerifyDump dump(dir, 2);
    const std::vector<uint8_t> a(12, 7), b(12, 9);
    dump.add("v.mp4", 12, Box{1, 10.04f, 20.0f, 30.0f, 40.0f, 0.33333f}, "plate", "student", a.data());
    dump.add("v.mp4", 24, Box{0, 1.0f, 2.0f, 3.0f, 4.0f, 0.5f}, "face", "yunet", b.data());
    EXPECT_EQ(dump.count(), 2);
    dump.close();
  }
  std::ifstream bin(dir / "verify_crops.bin", std::ios::binary);
  const std::vector<char> bytes((std::istreambuf_iterator<char>(bin)), {});
  ASSERT_EQ(bytes.size(), 24u);
  EXPECT_EQ(bytes[0], 7);
  EXPECT_EQ(bytes[12], 9);
  std::ifstream jsonl(dir / "verify_crops.jsonl");
  std::string line;
  ASSERT_TRUE(std::getline(jsonl, line));
  const auto first = nlohmann::json::parse(line);
  EXPECT_EQ(first.at("index"), 0);
  EXPECT_EQ(first.at("frame"), 12);
  EXPECT_EQ(first.at("cls"), "plate");
  EXPECT_DOUBLE_EQ(first.at("x1").get<double>(), 10.0);
  EXPECT_DOUBLE_EQ(first.at("score").get<double>(), 0.3333);
  EXPECT_EQ(first.at("src"), "student");
  EXPECT_EQ(first.at("size"), 2);
  ASSERT_TRUE(std::getline(jsonl, line));
  EXPECT_EQ(nlohmann::json::parse(line).at("index"), 1);
  EXPECT_FALSE(std::getline(jsonl, line));
}

}  // namespace
}  // namespace vc
