#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "vc_extract/extract_options.h"

namespace vc {

// Invalid command line.
class UsageError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Parses the arguments after the program name. Throws UsageError.
ExtractOptions parseExtractArgs(const std::vector<std::string>& args);

std::string extractUsage();

}  // namespace vc
