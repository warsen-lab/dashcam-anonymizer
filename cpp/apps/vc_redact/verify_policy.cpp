#include "vc_redact/verify_policy.h"

namespace vc {

bool needsVerification(const SourcedBox& box, const std::string& className, const RedactConfig& config) {
  if (className == "plate") return box.box.score < config.verifyPlateBelow;
  if (className == "face") {
    const bool both = box.src.find('+') != std::string::npos;
    return !both || box.box.score < config.verifyFaceUnlessBothAbove;
  }
  return false;
}

float passThreshold(const VerifierInfo& info, int verifierClass, const RedactConfig& config) {
  const std::string& name = info.classes.at(static_cast<size_t>(verifierClass));
  if (name == "face" && config.verifyThresholdFace) return *config.verifyThresholdFace;
  if (name == "plate" && config.verifyThresholdPlate) return *config.verifyThresholdPlate;
  return info.passThreshold.at(static_cast<size_t>(verifierClass));
}

}  // namespace vc
