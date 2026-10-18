#include "vc_redact/records.h"

#include <cmath>

#include <nlohmann/json.hpp>

#include "common/vc_error.h"

namespace vc {
namespace {

double round1(double v) { return std::round(v * 10.0) / 10.0; }
double round4(double v) { return std::round(v * 10000.0) / 10000.0; }

const std::string& className(const std::vector<std::string>& classes, int cls) {
  if (cls < 0 || static_cast<size_t>(cls) >= classes.size()) {
    throw VcError("class index " + std::to_string(cls) + " is outside the model's " +
                  std::to_string(classes.size()) + " classes");
  }
  return classes[static_cast<size_t>(cls)];
}

}  // namespace

std::string detectionRecord(const std::string& video, int64_t frame, double ptsMs, const Box& box,
                            const std::vector<std::string>& classes, const std::string& src) {
  nlohmann::ordered_json j;
  j["video"] = video;
  j["frame"] = frame;
  j["pts_ms"] = round1(ptsMs);
  j["cls"] = className(classes, box.cls);
  j["x1"] = round1(box.x1);
  j["y1"] = round1(box.y1);
  j["x2"] = round1(box.x2);
  j["y2"] = round1(box.y2);
  j["score"] = round4(box.score);
  j["src"] = src;
  return j.dump() + "\n";
}

std::string maskRecord(const std::string& video, int64_t frame, const std::vector<MaskBox>& boxes,
                       const std::vector<std::string>& classes) {
  nlohmann::ordered_json j;
  j["video"] = video;
  j["frame"] = frame;
  j["boxes"] = nlohmann::ordered_json::array();
  for (const MaskBox& m : boxes) {
    nlohmann::ordered_json b;
    b["cls"] = className(classes, m.cls);
    b["x1"] = round1(m.x1);
    b["y1"] = round1(m.y1);
    b["x2"] = round1(m.x2);
    b["y2"] = round1(m.y2);
    b["origin"] = toString(m.origin);
    j["boxes"].push_back(b);
  }
  return j.dump() + "\n";
}

}  // namespace vc
