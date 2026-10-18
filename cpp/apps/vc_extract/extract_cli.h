#pragma once

#include <string>
#include <vector>

#include "common/vc_error.h"
#include "vc_extract/extract_options.h"

namespace vc {

// Parses the arguments after the program name. Throws UsageError.
ExtractOptions parseExtractArgs(const std::vector<std::string>& args);

std::string extractUsage();

}  // namespace vc
