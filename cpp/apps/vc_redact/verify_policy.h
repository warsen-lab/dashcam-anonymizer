#pragma once

#include <string>

#include "detect/detector.h"
#include "detect/verifier.h"
#include "vc_redact/redact_config.h"

namespace vc {

// A box the verifier scored: its pass probability, and whether it reached the pass threshold.
struct VerifyResult {
  float score = 0;
  bool passed = true;
};

// Whether `box` (of class `className`) goes through the verifier (spec 2026-10-21 §3.1): a plate scoring below
// verify_plate_below; a face that only one detector found (no '+' in its src) or that scores below
// verify_face_unless_both_above. Other classes never do.
bool needsVerification(const SourcedBox& box, const std::string& className, const RedactConfig& config);

// The pass threshold of the verifier's class `verifierClass`: verify_threshold_face / verify_threshold_plate when
// set and the class is "face" / "plate", otherwise the model's pass_threshold.
float passThreshold(const VerifierInfo& info, int verifierClass, const RedactConfig& config);

}  // namespace vc
