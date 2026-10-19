#include "io/frame_file_name.h"

#include <cstdio>

namespace vc {

std::string frameFileName(int64_t frame) {
  char name[32];
  std::snprintf(name, sizeof(name), "%06lld.jpg", static_cast<long long>(frame));
  return name;
}

}  // namespace vc
