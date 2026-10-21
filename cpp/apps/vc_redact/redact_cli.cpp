#include "vc_redact/redact_cli.h"

#include <cstddef>
#include <optional>
#include <stdexcept>

#include "common/vc_error.h"

namespace vc {
namespace fs = std::filesystem;
namespace {

double parseNumber(const std::string& option, const std::string& text) {
  try {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used == text.size()) return value;
  } catch (const std::exception&) {
  }
  throw UsageError(option + " expects a number, got '" + text + "'");
}

bool looksLikeOption(const std::string& text) { return text.rfind("--", 0) == 0; }

}  // namespace

RedactOptions parseRedactArgs(const std::vector<std::string>& args) {
  RedactOptions o;
  std::optional<fs::path> config;
  std::optional<double> detectFps;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& arg = args[i];
    // The value of the option at `arg`; an option name in its place is a missing value.
    const auto value = [&]() -> const std::string& {
      if (i + 1 >= args.size()) throw UsageError(arg + " needs a value");
      const std::string& text = args[++i];
      if (looksLikeOption(text)) throw UsageError(arg + " needs a value, got option '" + text + "'");
      return text;
    };
    if (arg == "--input") {
      o.input = value();
    } else if (arg == "--out") {
      o.outRoot = value();
    } else if (arg == "--model") {
      o.modelDir = fs::path(value());
    } else if (arg == "--face-model") {
      o.faceModelDir = fs::path(value());
    } else if (arg == "--fake-detections") {
      o.fakeDetections = fs::path(value());
    } else if (arg == "--config") {
      config = fs::path(value());
    } else if (arg == "--detect-fps") {
      detectFps = parseNumber(arg, value());
    } else if (arg == "--detect-every-frame") {
      o.detectEveryFrame = true;
    } else if (arg == "--ep") {
      try {
        o.provider = parseExecutionProvider(value());
      } catch (const std::invalid_argument& e) {
        throw UsageError(e.what());
      }
    } else if (arg == "--hwaccel") {
      try {
        o.hwAccel = parseHwAccel(value());
      } catch (const std::invalid_argument& e) {
        throw UsageError(e.what());
      }
    } else if (arg == "--encoder") {
      const std::string& mode = value();
      if (mode == "auto") {
        o.preferHardwareEncoder = true;
      } else if (mode == "software") {
        o.preferHardwareEncoder = false;
      } else {
        throw UsageError("unknown --encoder '" + mode + "' (auto|software)");
      }
    } else if (arg == "--overwrite") {
      o.overwrite = true;
    } else {
      throw UsageError(looksLikeOption(arg) ? "unknown option '" + arg + "'" : "unexpected argument '" + arg + "'");
    }
  }
  if (o.input.empty()) throw UsageError("--input is required");
  if (o.outRoot.empty()) throw UsageError("--out is required");
  if (o.modelDir.has_value() == o.fakeDetections.has_value()) {
    throw UsageError("pass exactly one of --model or --fake-detections");
  }
  try {
    if (config) o.config = readRedactConfig(*config);
    if (detectFps) o.config.detectFps = *detectFps;
    validateRedactConfig(o.config, detectFps ? "--detect-fps" : config ? config->string() : "the default settings");
  } catch (const VcError& e) {
    throw UsageError(e.what());
  }
  return o;
}

std::string redactUsage() {
  return R"(Usage: vc_redact --input <video> --out <dir> (--model <dir> | --fake-detections <jsonl>) [options]

Masks faces and licence plates with a mosaic. Writes <dir>/<video stem>/:
<stem>.redacted.mp4, detections.jsonl, masks.jsonl (one line per frame) and stats.json.

Options:
  --model <dir>             model.onnx + model.json from vc_train export
  --fake-detections <file>  replay a detections.jsonl instead of running a model
  --face-model <dir>        also run the YuNet face model (model.onnx + model.json from vc_train yunet)
                            on every keyframe and mask what either detector finds
  --config <json>           parameters (see configs/redact.json; built-in defaults otherwise)
  --detect-fps <n>          detections per second of video, 5 to 60 (overrides the config): a
                            keyframe every round(video fps / n) frames, every frame once n reaches
                            the video frame rate
  --detect-every-frame      detect on every frame (reference run for temporal coverage)
  --ep <provider>           auto | tensorrt | cuda | coreml | cpu (default auto: TensorRT, CUDA, CPU on
                            Linux; CoreML, CPU on macOS); a provider that cannot be used falls back
                            (one line on stderr), an explicit one to cpu
  --hwaccel <mode>          decoder: none | auto | videotoolbox (default none)
  --encoder <mode>          auto (h264_nvenc on Linux / h264_videotoolbox on macOS, then libx264) |
                            software (libx264 only) (default auto)
  --overwrite               replace the output directory of a previous vc_redact run (it is removed
                            once the checks pass, before processing starts)
  -h, --help                show this help

Exit codes: 0 ok, 1 processing error, 2 invalid arguments, 3 output exists.
)";
}

}  // namespace vc
