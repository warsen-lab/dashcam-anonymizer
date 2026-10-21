#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "detect/box.h"

namespace vc {

// The verifier crops of a run, for training (spec 2026-10-21 §3.2; vc_redact --dump-verify-crops):
// verify_crops.bin holds the crops one after another, size * size * 3 bytes each (RGB, interleaved, the verifier's
// exact input before / 255), and verify_crops.jsonl one line per crop in the same order:
// {index, video, frame, cls, x1, y1, x2, y2, score, src, size}. Not thread-safe.
class VerifyDump {
 public:
  // Creates (truncates) both files in `dir`. Throws VcError when they cannot be created.
  VerifyDump(const std::filesystem::path& dir, int size);
  void add(const std::string& video, int64_t frame, const Box& box, const std::string& cls, const std::string& src,
           const uint8_t* rgb);
  int64_t count() const { return count_; }
  // Flushes and closes both files. Throws VcError when writing failed.
  void close();

 private:
  std::ofstream bin_;
  std::ofstream jsonl_;
  int size_;
  int64_t count_ = 0;
};

}  // namespace vc
