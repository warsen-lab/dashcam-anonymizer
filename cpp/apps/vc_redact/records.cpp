#include "vc_redact/records.h"

#include <cmath>
#include <string>

#include <nlohmann/json.hpp>

#include "common/vc_error.h"

namespace vc {
namespace {

// v rounded to `scale` steps per unit, as a record writes it. Throws VcError when v is not finite
// or the scaling overflows to infinity (nlohmann would write null). Returns +0.0 for -0.0, which
// would otherwise read "-0.0".
double recordNumber(double v, double scale, const char* field) {
  const double rounded = std::round(v * scale) / scale;
  if (!std::isfinite(rounded)) throw VcError(std::string(field) + " is not finite");
  return rounded == 0 ? 0.0 : rounded;
}

double round1(double v, const char* field) { return recordNumber(v, 10.0, field); }
double round4(double v, const char* field) { return recordNumber(v, 10000.0, field); }

// One JSON line. Invalid UTF-8 (in a video or source name, say) is written as U+FFFD: it must not
// throw an exception that is not a VcError out of the pipeline.
std::string jsonLine(const nlohmann::ordered_json& j) {
  return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) + "\n";
}

const std::string& className(const std::vector<std::string>& classes, int cls) {
  if (cls < 0 || static_cast<size_t>(cls) >= classes.size()) {
    throw VcError("class index " + std::to_string(cls) + " is outside the model's " +
                  std::to_string(classes.size()) + " classes");
  }
  return classes[static_cast<size_t>(cls)];
}

}  // namespace

std::string detectionRecord(const std::string& video, int64_t frame, double ptsMs, const Box& box,
                            const std::vector<std::string>& classes, const std::string& src,
                            std::optional<bool> confirmed) {
  nlohmann::ordered_json j;
  j["video"] = video;
  j["frame"] = frame;
  j["pts_ms"] = round1(ptsMs, "pts_ms");
  j["cls"] = className(classes, box.cls);
  j["x1"] = round1(box.x1, "x1");
  j["y1"] = round1(box.y1, "y1");
  j["x2"] = round1(box.x2, "x2");
  j["y2"] = round1(box.y2, "y2");
  j["score"] = round4(box.score, "score");
  j["src"] = src;
  if (confirmed) j["confirmed"] = *confirmed;
  return jsonLine(j);
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
    b["x1"] = round1(m.x1, "x1");
    b["y1"] = round1(m.y1, "y1");
    b["x2"] = round1(m.x2, "x2");
    b["y2"] = round1(m.y2, "y2");
    b["track"] = m.track;
    b["origin"] = toString(m.origin);
    b["block"] = m.block;
    b["snap"] = m.snap;
    j["boxes"].push_back(b);
  }
  return jsonLine(j);
}

}  // namespace vc
