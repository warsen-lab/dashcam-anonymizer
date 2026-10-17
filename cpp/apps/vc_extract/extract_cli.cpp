#include "vc_extract/extract_cli.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>

namespace vc {
namespace {

double parseDouble(const std::string& option, const std::string& text) {
  try {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used == text.size()) return value;
  } catch (const std::exception&) {
  }
  throw UsageError(option + " expects a number, got '" + text + "'");
}

int parseInt(const std::string& option, const std::string& text) {
  try {
    std::size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used == text.size()) return value;
  } catch (const std::exception&) {
  }
  throw UsageError(option + " expects an integer, got '" + text + "'");
}

constexpr std::array<std::string_view, 9> kValueOptions = {
    "--input", "--out",  "--fps",     "--max-fps", "--motion-threshold",
    "--dedup", "--clip", "--hwaccel", "--jpeg-q"};

bool takesValue(const std::string& option) {
  return std::find(kValueOptions.begin(), kValueOptions.end(), option) != kValueOptions.end();
}

bool looksLikeOption(const std::string& text) { return text.rfind("--", 0) == 0; }

// "<startSec>,<durationSec>"
ClipWindow parseClip(const std::string& text) {
  const std::size_t comma = text.find(',');
  if (comma == std::string::npos) {
    throw UsageError("--clip expects <startSec>,<durationSec>, got '" + text + "'");
  }
  const double start = parseDouble("--clip", text.substr(0, comma));
  const double duration = parseDouble("--clip", text.substr(comma + 1));
  return ClipWindow{start * 1000.0, duration * 1000.0};
}

}  // namespace

ExtractOptions parseExtractArgs(const std::vector<std::string>& args) {
  ExtractOptions options;
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string& option = args[i];
    if (option == "--overwrite") {
      options.overwrite = true;
      continue;
    }
    if (!takesValue(option)) {
      throw UsageError(looksLikeOption(option) ? "unknown option '" + option + "'"
                                               : "unexpected argument '" + option + "'");
    }
    if (i + 1 >= args.size()) throw UsageError(option + " needs a value");
    const std::string& value = args[++i];
    if (looksLikeOption(value)) {
      throw UsageError(option + " needs a value, got option '" + value + "'");
    }
    if (option == "--input") {
      options.input = value;
    } else if (option == "--out") {
      options.outRoot = value;
    } else if (option == "--fps") {
      options.sampler.baseFps = parseDouble(option, value);
    } else if (option == "--max-fps") {
      options.sampler.maxFps = parseDouble(option, value);
    } else if (option == "--motion-threshold") {
      options.sampler.motionThreshold = parseDouble(option, value);
    } else if (option == "--dedup") {
      options.sampler.dedupThreshold = parseDouble(option, value);
    } else if (option == "--clip") {
      options.sampler.clip = parseClip(value);
    } else if (option == "--hwaccel") {
      try {
        options.hwAccel = parseHwAccel(value);
      } catch (const std::invalid_argument& e) {
        throw UsageError(e.what());
      }
    } else if (option == "--jpeg-q") {
      options.jpegQscale = parseInt(option, value);
    } else {
      throw UsageError("unknown option '" + option + "'");
    }
  }

  if (options.input.empty()) throw UsageError("--input is required");
  if (options.outRoot.empty()) throw UsageError("--out is required");
  if (options.jpegQscale < kBestJpegQscale || options.jpegQscale > kWorstJpegQscale) {
    throw UsageError("--jpeg-q must be in [2, 31]");
  }
  try {
    FrameSampler validate(options.sampler);
  } catch (const std::invalid_argument& e) {
    throw UsageError(e.what());
  }
  return options;
}

std::string extractUsage() {
  return R"(Usage: vc_extract --input <video> --out <dir> [options]

Extracts JPEG frames into <dir>/<video stem>/ and writes frames.csv there.

Options:
  --fps <n>               base sampling rate, >= 5 (default 5)
  --max-fps <n>           rate while motion is high (default 15)
  --motion-threshold <x>  mean abs luma diff in [0,1] that counts as motion (default 0.05)
  --dedup <x>             skip frames whose diff to the last kept frame is below x
  --clip <start>,<dur>    keep every frame in [start, start+dur) seconds, nothing else
  --hwaccel <mode>        auto | none | videotoolbox (default auto)
  --jpeg-q <n>            MJPEG qscale 2 (best) .. 31 (default 2)
  --overwrite             replace an existing output directory
  -h, --help              show this help

Exit codes: 0 ok, 1 processing error, 2 invalid arguments, 3 output exists.
)";
}

}  // namespace vc
