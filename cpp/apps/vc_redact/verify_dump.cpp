#include "vc_redact/verify_dump.h"

#include <cmath>

#include <nlohmann/json.hpp>

#include "common/vc_error.h"

namespace vc {
namespace {

// v rounded to `scale` steps per unit, +0.0 for -0.0 (which would read "-0.0").
double rounded(double v, double scale) {
  const double r = std::round(v * scale) / scale;
  return r == 0 ? 0.0 : r;
}

}  // namespace

VerifyDump::VerifyDump(const std::filesystem::path& dir, int size)
    : bin_(dir / "verify_crops.bin", std::ios::binary | std::ios::trunc),
      jsonl_(dir / "verify_crops.jsonl", std::ios::trunc),
      size_(size) {
  if (!bin_ || !jsonl_) throw VcError("cannot create verify_crops.bin / verify_crops.jsonl in " + dir.string());
}

void VerifyDump::add(const std::string& video, int64_t frame, const Box& box, const std::string& cls,
                     const std::string& src, const uint8_t* rgb) {
  bin_.write(reinterpret_cast<const char*>(rgb), static_cast<std::streamsize>(size_) * size_ * 3);
  nlohmann::ordered_json j;
  j["index"] = count_;
  j["video"] = video;
  j["frame"] = frame;
  j["cls"] = cls;
  j["x1"] = rounded(box.x1, 10);
  j["y1"] = rounded(box.y1, 10);
  j["x2"] = rounded(box.x2, 10);
  j["y2"] = rounded(box.y2, 10);
  j["score"] = rounded(box.score, 10000);
  j["src"] = src;
  j["size"] = size_;
  jsonl_ << j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << '\n';
  ++count_;
}

void VerifyDump::close() {
  bin_.close();
  jsonl_.close();
  if (!bin_ || !jsonl_) throw VcError("writing verify_crops.bin / verify_crops.jsonl failed");
}

}  // namespace vc
