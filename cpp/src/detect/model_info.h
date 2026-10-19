#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace vc {

enum class OutputFormat {
  Yolov8,  // the student: `vc_train export`, fixed letterboxed RGB input
  YuNet,   // the face model: `vc_train yunet`, BGR at the frame's own size padded to multiples of 32
};

// The model.json next to a model.onnx (proxy-validation spec §4.4 for the student).
struct ModelInfo {
  std::string name;
  std::vector<std::string> classes;
  OutputFormat outputFormat = OutputFormat::Yolov8;
  int inputWidth = 0;   // yolov8: the fixed model input; yunet: 0 (the input follows the frame)
  int inputHeight = 0;
  int padMultiple = 0;  // yunet: the frame is zero-padded on the right and bottom to multiples of this
};

// Throws VcError("<origin>: …") for invalid JSON, missing fields or values this pipeline does not
// implement. output_format must be "yolov8" or "yunet".
// - yolov8: color "RGB", letterbox "top_left", pad_value 114, scale 1/255, and the input size
//   positive multiples of 32.
// - yunet: classes exactly ["face"], color "BGR", letterbox "top_left", pad_value 0, scale 1 and
//   pad_multiple 32.
// Sizes, pad_value and pad_multiple must be JSON integers within the int range.
ModelInfo parseModelInfo(const std::string& text, const std::string& origin);
ModelInfo readModelInfo(const std::filesystem::path& path);

}  // namespace vc
