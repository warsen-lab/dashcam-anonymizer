#pragma once

#include <cstdint>
#include <string>

namespace vc {

// "<frame:06d>.jpg": the name vc_extract gives a kept frame's JPEG and vc_detect looks for. One
// definition, so the two tools cannot drift apart.
std::string frameFileName(int64_t frame);

}  // namespace vc
