#pragma once

#include <optional>

namespace vc {

enum class SampleReason { None, Base, Motion, Clip };

// "none" | "base" | "motion" | "clip" — the values written to frames.csv.
const char* toString(SampleReason reason);

struct ClipWindow {
  double startMs = 0;
  double durationMs = 0;
};

inline constexpr double kMinBaseFps = 5.0;

struct SamplerConfig {
  double baseFps = 5.0;           // must be >= kMinBaseFps
  double maxFps = 15.0;           // rate used while motionScore >= motionThreshold
  double motionThreshold = 0.05;  // mean abs luma diff to the previous frame, [0, 1]
  // A due frame whose diff to the last kept frame is below this is skipped.
  std::optional<double> dedupThreshold;
  // When set, every frame inside the window is kept and nothing else.
  std::optional<ClipWindow> clip;
};

// Decides which decoded frames to keep. Feed every decoded frame in display
// order; the caller computes the scores, so this class is pure logic.
class FrameSampler {
 public:
  // Throws std::invalid_argument for an invalid config.
  explicit FrameSampler(const SamplerConfig& config);

  // motionScore: diff to the previous decoded frame (0 for the first frame).
  // diffToLastKept: diff to the last kept frame (ignored before the first keep).
  SampleReason offer(double ptsMs, double motionScore, double diffToLastKept);

 private:
  SamplerConfig config_;
  std::optional<double> lastKeptMs_;
};

}  // namespace vc
