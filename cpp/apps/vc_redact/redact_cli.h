#pragma once

#include <string>
#include <vector>

#include "vc_redact/redact_job.h"

namespace vc {

// Parses the arguments after the program name. Throws UsageError, also for an unreadable or
// invalid --config and for a --detect-fps that makes the configuration invalid.
RedactOptions parseRedactArgs(const std::vector<std::string>& args);

std::string redactUsage();

}  // namespace vc
