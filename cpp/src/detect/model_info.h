#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace vc {

// The model.json written by `vc_train export` (proxy-validation spec §4.4).
struct ModelInfo {
  std::string name;
  std::vector<std::string> classes;
  int inputWidth = 0;
  int inputHeight = 0;
};

// Throws VcError("<origin>: …") for invalid JSON, missing fields or values this pipeline does not
// implement: color must be "RGB", letterbox "top_left", pad_value 114, scale 1/255,
// output_format "yolov8", and the input size positive multiples of 32.
ModelInfo parseModelInfo(const std::string& text, const std::string& origin);
ModelInfo readModelInfo(const std::filesystem::path& path);

}  // namespace vc
