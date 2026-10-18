#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "detect/detector.h"

namespace vc {

// Replays detections.jsonl records (spec §3.2) by frame number, for tests and for checking the
// pipeline without a model. A record {"frame": N, "fail": true} makes detect() throw for frame N;
// {"frame": N, "delay_ms": M} makes it sleep M milliseconds first (to simulate a slow detector).
class FakeDetector : public Detector {
 public:
  // Throws VcError naming the file and line for unreadable files, bad JSON or unknown classes.
  explicit FakeDetector(const std::filesystem::path& jsonl,
                        std::vector<std::string> classes = {"face", "plate"});

  std::vector<Box> detect(const VideoFrame& frame) override;
  const std::vector<std::string>& classes() const override { return classes_; }
  std::string source() const override { return "fake"; }

  // Frame numbers detect() was called for, in call order.
  const std::vector<int64_t>& calls() const { return calls_; }

 private:
  std::vector<std::string> classes_;
  std::map<int64_t, std::vector<Box>> byFrame_;
  std::set<int64_t> failing_;
  std::map<int64_t, int64_t> delayMs_;
  std::vector<int64_t> calls_;
};

}  // namespace vc
