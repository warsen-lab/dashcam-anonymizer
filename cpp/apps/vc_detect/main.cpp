#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "vc_detect/detect_images.h"

extern "C" {
#include <libavutil/log.h>
}

namespace {

constexpr int kExitOk = 0;
constexpr int kExitFailure = 1;
constexpr int kExitUsage = 2;

const char* const kUsage =
    R"(Usage: vc_detect --model <dir> --frames <dir> --out <jsonl> [--ep cpu|coreml] [--score <x>]

Runs the ONNX student model on every frame listed in <frames>/frames.csv (a vc_extract output
directory) and writes detections.jsonl records, for checking C++ against Python inference.

Options:
  --model <dir>   model.onnx + model.json from vc_train export
  --frames <dir>  directory with frames.csv and the <frame:06d>.jpg files
  --out <jsonl>   detections file to write
  --ep <provider> cpu | coreml (default cpu)
  --score <x>     score threshold in [0, 1] (default 0.15)
  -h, --help      show this help

Exit codes: 0 ok, 1 processing error, 2 invalid arguments.
)";

float parseScore(const std::string& text) {
  try {
    std::size_t used = 0;
    const float value = std::stof(text, &used);
    if (used == text.size() && std::isfinite(value) && value >= 0.0f && value <= 1.0f) return value;
  } catch (const std::exception&) {
  }
  throw vc::UsageError("--score expects a number in [0, 1], got '" + text + "'");
}

vc::DetectImagesOptions parse(const std::vector<std::string>& args) {
  vc::DetectImagesOptions o;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (arg != "--model" && arg != "--frames" && arg != "--out" && arg != "--ep" && arg != "--score") {
      throw vc::UsageError("unknown argument '" + arg + "'");
    }
    if (i + 1 >= args.size()) throw vc::UsageError(arg + " needs a value");
    const std::string& value = args[++i];
    if (arg == "--model") {
      o.modelDir = value;
    } else if (arg == "--frames") {
      o.framesDir = value;
    } else if (arg == "--out") {
      o.out = value;
    } else if (arg == "--ep") {
      try {
        o.provider = vc::parseExecutionProvider(value);
      } catch (const std::invalid_argument& e) {
        throw vc::UsageError(e.what());
      }
    } else {
      o.decode.scoreThreshold = parseScore(value);
    }
  }
  if (o.modelDir.empty() || o.framesDir.empty() || o.out.empty()) {
    throw vc::UsageError("--model, --frames and --out are required");
  }
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);
  for (const std::string& arg : args) {
    if (arg == "-h" || arg == "--help") {
      std::cout << kUsage;
      return kExitOk;
    }
  }

  vc::DetectImagesOptions options;
  try {
    options = parse(args);
  } catch (const vc::UsageError& e) {
    std::cerr << "vc_detect: " << e.what() << "\n\n" << kUsage;
    return kExitUsage;
  }

  av_log_set_level(AV_LOG_ERROR);
  try {
    const int64_t images = vc::runDetectImages(options);
    std::printf("vc_detect: %lld images -> %s\n", static_cast<long long>(images), options.out.string().c_str());
    return kExitOk;
  } catch (const std::exception& e) {
    std::cerr << "vc_detect: " << e.what() << '\n';
    return kExitFailure;
  }
}
