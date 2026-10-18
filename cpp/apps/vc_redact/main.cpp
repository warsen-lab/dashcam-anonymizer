#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "vc_redact/redact_cli.h"
#include "vc_redact/redact_job.h"

extern "C" {
#include <libavutil/log.h>
}

namespace {

constexpr int kExitOk = 0;
constexpr int kExitFailure = 1;
constexpr int kExitUsage = 2;
constexpr int kExitOutputExists = 3;

std::string joined(const std::vector<std::string>& items) {
  if (items.empty()) return "none";
  std::string out;
  for (const std::string& item : items) out += (out.empty() ? "" : ", ") + item;
  return out;
}

void printStats(const vc::RedactStats& s) {
  const double fps = s.elapsedSec > 0 ? static_cast<double>(s.frames) / s.elapsedSec : 0;
  std::printf("output:        %s\n", s.video.string().c_str());
  std::printf("frames:        %lld (keyframes %lld, detections %lld, masked %lld frames / %lld boxes)\n",
              static_cast<long long>(s.frames), static_cast<long long>(s.keyframes),
              static_cast<long long>(s.detections), static_cast<long long>(s.maskedFrames),
              static_cast<long long>(s.maskBoxes));
  std::printf("detector:      %s\n", s.detector.c_str());
  std::printf("encoder:       %s\n", s.encoder.c_str());
  std::printf("audio:         %zu stream(s), %lld packets copied; dropped streams: %s\n", s.audioStreams,
              static_cast<long long>(s.audioPackets), joined(s.droppedStreams).c_str());
  std::printf("stages (fps):  decode %.1f, detect %.1f, track %.1f, mosaic %.1f, encode %.1f\n", s.decode.fps(),
              s.detect.fps(), s.track.fps(), s.mosaic.fps(), s.encode.fps());
  std::printf("decode errors: %lld\n", static_cast<long long>(s.decodeErrors));
  std::printf("elapsed:       %.2f s (%.1f fps)\n", s.elapsedSec, fps);
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);
  for (const std::string& arg : args) {
    if (arg == "-h" || arg == "--help") {
      std::cout << vc::redactUsage();
      return kExitOk;
    }
  }

  vc::RedactOptions options;
  try {
    options = vc::parseRedactArgs(args);
  } catch (const vc::UsageError& e) {
    std::cerr << "vc_redact: " << e.what() << "\n\n" << vc::redactUsage();
    return kExitUsage;
  }

  av_log_set_level(AV_LOG_ERROR);
  try {
    const vc::RedactStats stats = vc::runRedact(options);
    printStats(stats);
    for (const std::string& warning : vc::redactWarnings(stats)) std::cerr << "vc_redact: " << warning << '\n';
    return kExitOk;
  } catch (const vc::OutputExistsError& e) {
    std::cerr << "vc_redact: " << e.what() << '\n';
    return kExitOutputExists;
  } catch (const std::exception& e) {
    std::cerr << "vc_redact: " << e.what() << '\n';
    return kExitFailure;
  }
}
