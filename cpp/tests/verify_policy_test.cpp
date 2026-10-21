#include "vc_redact/verify_policy.h"

#include <gtest/gtest.h>

#include <string>

namespace vc {
namespace {

SourcedBox box(float score, const std::string& src) { return SourcedBox{Box{0, 0, 0, 10, 10, score}, src}; }

TEST(VerifyPolicy, PlatesBelowTheBandAreChecked) {
  const RedactConfig c;
  EXPECT_TRUE(needsVerification(box(0.49f, "student"), "plate", c));
  EXPECT_FALSE(needsVerification(box(0.5f, "student"), "plate", c));
}

TEST(VerifyPolicy, FacesAreCheckedUnlessBothDetectorsFoundThemConfidently) {
  const RedactConfig c;
  EXPECT_TRUE(needsVerification(box(0.95f, "yunet"), "face", c));
  EXPECT_TRUE(needsVerification(box(0.95f, "student"), "face", c));
  EXPECT_TRUE(needsVerification(box(0.95f, "fake"), "face", c));
  EXPECT_TRUE(needsVerification(box(0.49f, "student+yunet"), "face", c));
  EXPECT_FALSE(needsVerification(box(0.5f, "student+yunet"), "face", c));
}

TEST(VerifyPolicy, OtherClassesAreNeverChecked) {
  EXPECT_FALSE(needsVerification(box(0.01f, "student"), "vehicle", RedactConfig{}));
}

TEST(VerifyPolicy, TheConfigOverridesTheModelsThreshold) {
  VerifierInfo info;
  info.classes = {"face", "plate"};
  info.passThreshold = {0.2f, 0.3f};
  RedactConfig c;
  EXPECT_FLOAT_EQ(passThreshold(info, 0, c), 0.2f);
  EXPECT_FLOAT_EQ(passThreshold(info, 1, c), 0.3f);
  c.verifyThresholdFace = 0.6f;
  c.verifyThresholdPlate = 0.7f;
  EXPECT_FLOAT_EQ(passThreshold(info, 0, c), 0.6f);
  EXPECT_FLOAT_EQ(passThreshold(info, 1, c), 0.7f);
}

}  // namespace
}  // namespace vc
