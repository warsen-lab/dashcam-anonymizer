#include "detect/fake_detector.h"

#include <algorithm>
#include <fstream>
#include <utility>

#include <nlohmann/json.hpp>

#include "common/vc_error.h"

namespace vc {

FakeDetector::FakeDetector(const std::filesystem::path& jsonl, std::vector<std::string> classes)
    : classes_(std::move(classes)) {
  std::ifstream in(jsonl);
  if (!in) throw VcError("cannot read " + jsonl.string());
  std::string line;
  int lineNo = 0;
  while (std::getline(in, line)) {
    ++lineNo;
    if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
    const std::string where = jsonl.string() + " line " + std::to_string(lineNo);
    try {
      const nlohmann::json j = nlohmann::json::parse(line);
      const int64_t frame = j.at("frame").get<int64_t>();
      if (j.value("fail", false)) {
        failing_.insert(frame);
        continue;
      }
      const std::string cls = j.at("cls").get<std::string>();
      const auto it = std::find(classes_.begin(), classes_.end(), cls);
      if (it == classes_.end()) throw VcError(where + ": unknown class '" + cls + "'");
      Box b;
      b.cls = static_cast<int>(it - classes_.begin());
      b.x1 = j.at("x1").get<float>();
      b.y1 = j.at("y1").get<float>();
      b.x2 = j.at("x2").get<float>();
      b.y2 = j.at("y2").get<float>();
      b.score = j.value("score", 1.0f);
      byFrame_[frame].push_back(b);
    } catch (const nlohmann::json::exception& e) {
      throw VcError(where + ": " + e.what());
    }
  }
}

std::vector<Box> FakeDetector::detect(const VideoFrame& frame) {
  calls_.push_back(frame.index);
  if (failing_.count(frame.index) != 0) {
    throw VcError("fake detector failure at frame " + std::to_string(frame.index));
  }
  const auto it = byFrame_.find(frame.index);
  return it == byFrame_.end() ? std::vector<Box>{} : it->second;
}

}  // namespace vc
