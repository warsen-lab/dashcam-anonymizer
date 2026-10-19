#include "detect/model_info.h"

#include <cmath>
#include <fstream>
#include <functional>
#include <sstream>

#include <nlohmann/json.hpp>

#include "common/vc_error.h"
#include "detect/letterbox.h"

namespace vc {
namespace {

// The only padding a YuNet input takes: its largest stride.
constexpr int kYuNetPadMultiple = 32;

using Fail = std::function<VcError(const std::string&)>;

// input.<field> as an int: get<int>() would silently truncate 1088.5 to 1088, so the JSON number
// must be an integer.
int integerField(const nlohmann::json& input, const char* field, const Fail& fail) {
  const nlohmann::json& v = input.at(field);
  if (!v.is_number_integer()) throw fail(std::string("input.") + field + " must be an integer");
  return v.get<int>();
}

void parseYolov8(const nlohmann::json& input, ModelInfo& m, const Fail& fail) {
  m.inputWidth = integerField(input, "width", fail);
  m.inputHeight = integerField(input, "height", fail);
  if (m.inputWidth <= 0 || m.inputHeight <= 0 || m.inputWidth % 32 != 0 || m.inputHeight % 32 != 0) {
    throw fail("input width and height must be positive multiples of 32");
  }
  if (input.at("color").get<std::string>() != "RGB") throw fail("input.color must be \"RGB\"");
  if (input.at("letterbox").get<std::string>() != "top_left") throw fail("input.letterbox must be \"top_left\"");
  if (integerField(input, "pad_value", fail) != kPadValue) throw fail("input.pad_value must be 114");
  if (std::abs(input.at("scale").get<double>() - 1.0 / 255.0) > 1e-9) throw fail("input.scale must be 1/255");
}

void parseYuNet(const nlohmann::json& input, ModelInfo& m, const Fail& fail) {
  if (m.classes.size() != 1) throw fail("a yunet model has exactly one class");
  if (input.at("color").get<std::string>() != "BGR") throw fail("input.color must be \"BGR\"");
  if (input.at("letterbox").get<std::string>() != "top_left") throw fail("input.letterbox must be \"top_left\"");
  if (integerField(input, "pad_value", fail) != 0) throw fail("input.pad_value must be 0");
  if (input.at("scale").get<double>() != 1.0) throw fail("input.scale must be 1");
  m.padMultiple = integerField(input, "pad_multiple", fail);
  if (m.padMultiple != kYuNetPadMultiple) throw fail("input.pad_multiple must be 32");
}

}  // namespace

ModelInfo parseModelInfo(const std::string& text, const std::string& origin) {
  const Fail fail = [&](const std::string& why) { return VcError(origin + ": " + why); };
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(text);
  } catch (const nlohmann::json::exception& e) {
    throw fail(e.what());
  }
  if (!j.is_object()) throw fail("the root must be a JSON object");
  try {
    ModelInfo m;
    m.name = j.value("name", std::string());
    const std::string format = j.at("output_format").get<std::string>();
    if (format == "yolov8") {
      m.outputFormat = OutputFormat::Yolov8;
    } else if (format == "yunet") {
      m.outputFormat = OutputFormat::YuNet;
    } else {
      throw fail("output_format must be \"yolov8\" or \"yunet\"");
    }
    m.classes = j.at("classes").get<std::vector<std::string>>();
    if (m.classes.empty()) throw fail("classes is empty");
    if (m.outputFormat == OutputFormat::Yolov8) {
      parseYolov8(j.at("input"), m, fail);
    } else {
      parseYuNet(j.at("input"), m, fail);
    }
    return m;
  } catch (const nlohmann::json::exception& e) {
    throw fail(e.what());
  }
}

ModelInfo readModelInfo(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw VcError("cannot read " + path.string());
  std::ostringstream text;
  text << in.rdbuf();
  return parseModelInfo(text.str(), path.string());
}

}  // namespace vc
