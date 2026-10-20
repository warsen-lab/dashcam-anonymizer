#pragma once

#include <cstdint>
#include <string>

extern "C" {
#include <libavutil/rational.h>
}

namespace vc {

// detect_fps bounds (spec 2026-10-19 §2).
inline constexpr double kMinDetectFps = 5.0;
inline constexpr double kMaxDetectFps = 60.0;

// The frame rate keyframes are spaced by: the video stream's average frame rate (avg_frame_rate),
// else its base rate (r_frame_rate). Throws VcError naming `video` when neither is a positive rate.
double streamFrameRate(AVRational averageRate, AVRational baseRate, const std::string& video);

// Where vc_redact detects: on every `stride`-th frame by frame index (index % stride == 0), the same
// over the whole video.
struct DetectSchedule {
  int64_t stride = 1;
  double videoFps = 0;
  double actualFps = 0;         // videoFps / stride: detections per second of video
  bool aboveVideoRate = false;  // the requested rate exceeded the video's, so every frame is a keyframe
};

// stride = max(1, floor(videoFps / detectFps + 0.5)). Throws std::invalid_argument unless videoFps is in
// (0, 1e6] and detectFps in [kMinDetectFps, kMaxDetectFps].
DetectSchedule detectSchedule(double videoFps, double detectFps);

// --detect-every-frame: stride 1. Throws std::invalid_argument unless videoFps is in (0, 1e6].
DetectSchedule everyFrameSchedule(double videoFps);

// True when frame `frameIndex` is a keyframe of the schedule.
bool onSchedule(const DetectSchedule& s, int64_t frameIndex);

}  // namespace vc
