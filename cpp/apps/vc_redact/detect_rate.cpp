#include "vc_redact/detect_rate.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/vc_error.h"

namespace vc {
namespace {

// Above this a "frame rate" is a broken header, and the stride would overflow int64_t.
constexpr double kMaxVideoFps = 1e6;

bool usable(AVRational rate) { return rate.num > 0 && rate.den > 0; }

void requireVideoFps(double videoFps) {
  if (!(videoFps > 0 && videoFps <= kMaxVideoFps)) {
    throw std::invalid_argument("the video frame rate must be in (0, 1e6] fps");
  }
}

}  // namespace

double streamFrameRate(AVRational averageRate, AVRational baseRate, const std::string& video) {
  if (usable(averageRate)) return av_q2d(averageRate);
  if (usable(baseRate)) return av_q2d(baseRate);
  throw VcError("cannot space the keyframes of " + video +
                ": its video stream has neither an average frame rate nor a base frame rate");
}

DetectSchedule detectSchedule(double videoFps, double detectFps) {
  requireVideoFps(videoFps);
  if (!(detectFps >= kMinDetectFps && detectFps <= kMaxDetectFps)) {
    throw std::invalid_argument("detect_fps must be in [5, 60]");
  }
  const auto stride = std::max<int64_t>(1, static_cast<int64_t>(std::floor(videoFps / detectFps + 0.5)));
  return DetectSchedule{stride, videoFps, videoFps / static_cast<double>(stride), detectFps > videoFps};
}

DetectSchedule everyFrameSchedule(double videoFps) {
  requireVideoFps(videoFps);
  return DetectSchedule{1, videoFps, videoFps, false};
}

bool onSchedule(const DetectSchedule& s, int64_t frameIndex) { return frameIndex % s.stride == 0; }

}  // namespace vc
