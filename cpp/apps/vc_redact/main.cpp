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
    std::fputs(vc::redactSummary(stats).c_str(), stdout);
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
