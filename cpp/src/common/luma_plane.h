#pragma once

#include <cstdint>

namespace vc {

// Read-only view of an 8-bit luma plane.
struct LumaPlane {
  const uint8_t* data = nullptr;
  int width = 0;
  int height = 0;
  int stride = 0;  // bytes per row, >= width
};

}  // namespace vc
