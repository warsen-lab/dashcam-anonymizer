#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "vc_extract/extract_cli.h"
#include "vc_extract/extract_job.h"

extern "C" {
#include <libavutil/log.h>
}

namespace {

constexpr int kExitOk = 0;
constexpr int kExitFailure = 1;
constexpr int kExitUsage = 2;
constexpr int kExitOutputExists = 3;

void printStats(const vc::ExtractStats& s) {
  const double fps = s.elapsedSec > 0 ? static_cast<double>(s.decodedFrames) / s.elapsedSec : 0;
  std::printf("output:        %s\n", s.outputDir.string().c_str());
  std::printf("decoded:       %lld frames (%s decode)\n", static_cast<long long>(s.decodedFrames),
              s.usedHardware ? "hardware" : "software");
  std::printf("kept:          %lld (base %lld, motion %lld, clip %lld)\n",
              static_cast<long long>(s.keptFrames), static_cast<long long>(s.keptBase),
              static_cast<long long>(s.keptMotion), static_cast<long long>(s.keptClip));
  std::printf("decode errors: %lld\n", static_cast<long long>(s.decodeErrors));
  std::printf("elapsed:       %.2f s (%.1f fps)\n", s.elapsedSec, fps);
}

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);
  for (const std::string& arg : args) {
    if (arg == "-h" || arg == "--help") {
      std::cout << vc::extractUsage();
      return kExitOk;
    }
  }

  vc::ExtractOptions options;
  try {
    options = vc::parseExtractArgs(args);
  } catch (const vc::UsageError& e) {
    std::cerr << "vc_extract: " << e.what() << "\n\n" << vc::extractUsage();
    return kExitUsage;
  }

  av_log_set_level(AV_LOG_ERROR);
  try {
    printStats(vc::runExtract(options));
    return kExitOk;
  } catch (const vc::OutputExistsError& e) {
    std::cerr << "vc_extract: " << e.what() << '\n';
    return kExitOutputExists;
  } catch (const std::exception& e) {
    std::cerr << "vc_extract: " << e.what() << '\n';
    return kExitFailure;
  }
}
