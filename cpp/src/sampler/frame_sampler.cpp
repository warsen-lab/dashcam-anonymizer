#include "sampler/frame_sampler.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace vc {
namespace {

// Container timestamps are rounded; accept a frame up to 1 ms early.
constexpr double kTimingToleranceMs = 1.0;

}  // namespace

const char* toString(SampleReason reason) {
  switch (reason) {
    case SampleReason::Base:
      return "base";
    case SampleReason::Motion:
      return "motion";
    case SampleReason::Clip:
      return "clip";
    case SampleReason::None:
      break;
  }
  return "none";
}

FrameSampler::FrameSampler(const SamplerConfig& config) : config_(config) {
  if (!std::isfinite(config_.baseFps) || config_.baseFps < kMinBaseFps) {
    char message[80];
    std::snprintf(message, sizeof(message), "fps must be >= %g, got %g", kMinBaseFps,
                  config_.baseFps);
    throw std::invalid_argument(message);
  }
  if (!std::isfinite(config_.maxFps) || config_.maxFps < config_.baseFps) {
    throw std::invalid_argument("max fps must be >= base fps");
  }
  if (!std::isfinite(config_.motionThreshold) || config_.motionThreshold < 0) {
    throw std::invalid_argument("motion threshold must be >= 0");
  }
  if (config_.dedupThreshold &&
      (!std::isfinite(*config_.dedupThreshold) || *config_.dedupThreshold < 0)) {
    throw std::invalid_argument("dedup threshold must be >= 0");
  }
  if (config_.clip &&
      (!std::isfinite(config_.clip->startMs) || !std::isfinite(config_.clip->durationMs) ||
       config_.clip->startMs < 0 || config_.clip->durationMs <= 0)) {
    throw std::invalid_argument("clip needs start >= 0 and duration > 0");
  }
}

SampleReason FrameSampler::offer(double ptsMs, double motionScore, double diffToLastKept) {
  if (config_.clip) {
    const double start = config_.clip->startMs;
    const bool inside = ptsMs >= start && ptsMs < start + config_.clip->durationMs;
    return inside ? SampleReason::Clip : SampleReason::None;
  }
  if (!lastKeptMs_) {
    lastKeptMs_ = ptsMs;
    return SampleReason::Base;
  }
  const double elapsed = ptsMs - *lastKeptMs_;
  const bool dueAtBase = elapsed >= 1000.0 / config_.baseFps - kTimingToleranceMs;
  const bool dueAtMax = elapsed >= 1000.0 / config_.maxFps - kTimingToleranceMs;
  const bool moving = motionScore >= config_.motionThreshold;
  if (!dueAtBase && !(moving && dueAtMax)) {
    return SampleReason::None;
  }
  if (config_.dedupThreshold && diffToLastKept < *config_.dedupThreshold) {
    return SampleReason::None;
  }
  lastKeptMs_ = ptsMs;
  return dueAtBase ? SampleReason::Base : SampleReason::Motion;
}

}  // namespace vc
