#pragma once

#include <string>
#include <vector>

#include "vc_detect/detect_images.h"

namespace vc {

// Parses the arguments after the program name. Throws UsageError.
DetectImagesOptions parseDetectArgs(const std::vector<std::string>& args);

std::string detectUsage();

}  // namespace vc
