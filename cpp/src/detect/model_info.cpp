#include "detect/model_info.h"

#include <cmath>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "common/vc_error.h"
#include "detect/letterbox.h"

namespace vc {

ModelInfo parseModelInfo(const std::string& text, const std::string& origin) {
  const auto fail = [&](const std::string& why) { return VcError(origin + ": " + why); };
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
    if (j.at("output_format").get<std::string>() != "yolov8") throw fail("output_format must be \"yolov8\"");
    m.classes = j.at("classes").get<std::vector<std::string>>();
    if (m.classes.empty()) throw fail("classes is empty");
    const nlohmann::json& input = j.at("input");
    // get<int>() would silently truncate 1088.5 to 1088, so the JSON number must be an integer.
    const auto integerField = [&](const char* field) {
      const nlohmann::json& v = input.at(field);
      if (!v.is_number_integer()) throw fail(std::string("input.") + field + " must be an integer");
      return v.get<int>();
    };
    m.inputWidth = integerField("width");
    m.inputHeight = integerField("height");
    if (m.inputWidth <= 0 || m.inputHeight <= 0 || m.inputWidth % 32 != 0 || m.inputHeight % 32 != 0) {
      throw fail("input width and height must be positive multiples of 32");
    }
    if (input.at("color").get<std::string>() != "RGB") throw fail("input.color must be \"RGB\"");
    if (input.at("letterbox").get<std::string>() != "top_left") throw fail("input.letterbox must be \"top_left\"");
    if (integerField("pad_value") != kPadValue) throw fail("input.pad_value must be 114");
    if (std::abs(input.at("scale").get<double>() - 1.0 / 255.0) > 1e-9) throw fail("input.scale must be 1/255");
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
