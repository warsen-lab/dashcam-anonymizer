#include "common/vc_error.h"

extern "C" {
#include <libavutil/error.h>
}

namespace vc {

std::string avErrorText(int errnum) {
  char buffer[AV_ERROR_MAX_STRING_SIZE] = {0};
  av_strerror(errnum, buffer, sizeof(buffer));
  return buffer;
}

int checkAv(int ret, const std::string& what) {
  if (ret < 0) {
    throw VcError(what + ": " + avErrorText(ret));
  }
  return ret;
}

}  // namespace vc
