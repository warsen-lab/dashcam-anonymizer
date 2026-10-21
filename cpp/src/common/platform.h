#pragma once

// What this build runs on and what it was built with (spec 2026-10-20 §2-§3): the execution provider order of
// --ep auto, the decoder of --hwaccel auto and vc_redact's defaults depend on it.

namespace vc {

enum class Platform { Linux, MacOS, Other };

constexpr Platform buildPlatform() {
#if defined(__APPLE__)
  return Platform::MacOS;
#elif defined(__linux__)
  return Platform::Linux;
#else
  return Platform::Other;
#endif
}

// True when CMake configured the build with -DVC_WITH_CUDA=ON.
constexpr bool builtWithCuda() {
#if defined(VC_WITH_CUDA)
  return true;
#else
  return false;
#endif
}

}  // namespace vc
