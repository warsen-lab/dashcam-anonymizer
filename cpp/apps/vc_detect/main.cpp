#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "vc_detect/detect_cli.h"
#include "vc_detect/detect_images.h"

extern "C" {
#include <libavutil/log.h>
}

namespace {

constexpr int kExitOk = 0;
constexpr int kExitFailure = 1;
constexpr int kExitUsage = 2;

}  // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);
  for (const std::string& arg : args) {
    if (arg == "-h" || arg == "--help") {
      std::cout << vc::detectUsage();
      return kExitOk;
    }
  }

  vc::DetectImagesOptions options;
  try {
    options = vc::parseDetectArgs(args);
  } catch (const vc::UsageError& e) {
    std::cerr << "vc_detect: " << e.what() << "\n\n" << vc::detectUsage();
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
