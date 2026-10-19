#pragma once

#include <string>
#include <vector>

#include "vc_redact/redact_job.h"

namespace vc {

// Parses the arguments after the program name. Throws UsageError, also for an unreadable or
// invalid --config and for a --detect-fps outside [5, 60].
RedactOptions parseRedactArgs(const std::vector<std::string>& args);

std::string redactUsage();

}  // namespace vc
