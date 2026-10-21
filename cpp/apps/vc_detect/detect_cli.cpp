#include "vc_detect/detect_cli.h"

#include <cmath>
#include <cstddef>
#include <exception>
#include <stdexcept>

#include "common/vc_error.h"

namespace vc {
namespace {

float parseScore(const std::string& text) {
  try {
    std::size_t used = 0;
    const float value = std::stof(text, &used);
    if (used == text.size() && std::isfinite(value) && value >= 0.0f && value <= 1.0f) return value;
  } catch (const std::exception&) {
  }
  throw UsageError("--score expects a number in [0, 1], got '" + text + "'");
}

bool looksLikeOption(const std::string& text) { return text.rfind("--", 0) == 0; }

}  // namespace

DetectImagesOptions parseDetectArgs(const std::vector<std::string>& args) {
  DetectImagesOptions o;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (arg != "--model" && arg != "--frames" && arg != "--out" && arg != "--ep" && arg != "--score") {
      throw UsageError(looksLikeOption(arg) ? "unknown option '" + arg + "'" : "unexpected argument '" + arg + "'");
    }
    // The value of the option at `arg`; an option name in its place is a missing value.
    if (i + 1 >= args.size()) throw UsageError(arg + " needs a value");
    const std::string& value = args[++i];
    if (looksLikeOption(value)) throw UsageError(arg + " needs a value, got option '" + value + "'");
    if (arg == "--model") {
      o.modelDir = value;
    } else if (arg == "--frames") {
      o.framesDir = value;
    } else if (arg == "--out") {
      o.out = value;
    } else if (arg == "--ep") {
      try {
        o.provider = parseExecutionProvider(value);
      } catch (const std::invalid_argument& e) {
        throw UsageError(e.what());
      }
    } else {
      o.decode.scoreThreshold = parseScore(value);
    }
  }
  if (o.modelDir.empty() || o.framesDir.empty() || o.out.empty()) {
    throw UsageError("--model, --frames and --out are required");
  }
  return o;
}

std::string detectUsage() {
  return R"(Usage: vc_detect --model <dir> --frames <dir> --out <jsonl> [--ep <provider>] [--score <x>]

Runs the ONNX student model on every frame listed in <frames>/frames.csv (a vc_extract output
directory) and writes detections.jsonl records, for checking C++ against Python inference.
The NMS IoU is the default 0.5 (there is no option for it).

Options:
  --model <dir>   model.onnx + model.json from vc_train export
  --frames <dir>  directory with frames.csv and the <frame:06d>.jpg files
  --out <jsonl>   detections file to write
  --ep <provider> cpu | coreml | auto | tensorrt | cuda (default cpu; see vc_redact --help)
  --score <x>     score threshold in [0, 1] (default 0.15)
  -h, --help      show this help

Exit codes: 0 ok, 1 processing error, 2 invalid arguments.
)";
}

}  // namespace vc
