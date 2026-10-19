#include "vc_redact/records.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "common/vc_error.h"

namespace vc {
namespace {

const std::vector<std::string> kClasses{"face", "plate"};

TEST(Records, DetectionLineFollowsTheContract) {
  const Box b{0, 812.06f, 433.5f, 846.2f, 475.0f, 0.87f};
  EXPECT_EQ(detectionRecord("trip.mp4", 1234, 20566.74, b, kClasses, "student"),
            R"({"video":"trip.mp4","frame":1234,"pts_ms":20566.7,"cls":"face","x1":812.1,"y1":433.5,)"
            R"("x2":846.2,"y2":475.0,"score":0.87,"src":"student"})"
            "\n");
}

TEST(Records, MaskLineListsBoxesWithTheirOrigin) {
  const std::vector<MaskBox> boxes{MaskBox{1, 1.0f, 2.0f, 3.5f, 4.0f, MaskOrigin::Interpolated}};
  EXPECT_EQ(maskRecord("a.mp4", 3, boxes, kClasses),
            R"({"video":"a.mp4","frame":3,"boxes":[{"cls":"plate","x1":1.0,"y1":2.0,"x2":3.5,"y2":4.0,"origin":"interpolated"}]})"
            "\n");
  EXPECT_EQ(maskRecord("a.mp4", 4, {}, kClasses), R"({"video":"a.mp4","frame":4,"boxes":[]})" "\n");
}

TEST(Records, RejectsClassIndicesOutsideTheModel) {
  EXPECT_THROW(detectionRecord("a.mp4", 0, 0, Box{2, 0, 0, 1, 1, 1}, kClasses, "x"), VcError);
}

TEST(Records, RefusesNonFiniteDetectionValues) {
  const float nanF = std::numeric_limits<float>::quiet_NaN();
  const float infF = std::numeric_limits<float>::infinity();
  const double nanD = std::numeric_limits<double>::quiet_NaN();
  const double infD = std::numeric_limits<double>::infinity();
  const Box good{0, 0, 0, 1, 1, 0.5f};
  EXPECT_THROW(detectionRecord("a.mp4", 0, 0.0, Box{0, nanF, 0, 1, 1, 0.5f}, kClasses, "x"), VcError);
  EXPECT_THROW(detectionRecord("a.mp4", 0, 0.0, Box{0, 0, nanF, 1, 1, 0.5f}, kClasses, "x"), VcError);
  EXPECT_THROW(detectionRecord("a.mp4", 0, 0.0, Box{0, 0, 0, infF, 1, 0.5f}, kClasses, "x"), VcError);
  EXPECT_THROW(detectionRecord("a.mp4", 0, 0.0, Box{0, 0, 0, 1, -infF, 0.5f}, kClasses, "x"), VcError);
  EXPECT_THROW(detectionRecord("a.mp4", 0, 0.0, Box{0, 0, 0, 1, 1, nanF}, kClasses, "x"), VcError);
  EXPECT_THROW(detectionRecord("a.mp4", 0, 0.0, Box{0, 0, 0, 1, 1, infF}, kClasses, "x"), VcError);
  EXPECT_THROW(detectionRecord("a.mp4", 0, nanD, good, kClasses, "x"), VcError);
  EXPECT_THROW(detectionRecord("a.mp4", 0, infD, good, kClasses, "x"), VcError);
  // Finite, but x10 overflows to infinity before rounding: nlohmann would write null.
  EXPECT_THROW(detectionRecord("a.mp4", 0, std::numeric_limits<double>::max(), good, kClasses, "x"), VcError);
}

TEST(Records, RefusesNonFiniteMaskCoordinates) {
  const float nanF = std::numeric_limits<float>::quiet_NaN();
  const float infF = std::numeric_limits<float>::infinity();
  const auto line = [&](float x1, float y1, float x2, float y2) {
    return maskRecord("a.mp4", 0, {MaskBox{0, x1, y1, x2, y2, MaskOrigin::Detected}}, kClasses);
  };
  EXPECT_THROW(line(nanF, 0, 1, 1), VcError);
  EXPECT_THROW(line(0, nanF, 1, 1), VcError);
  EXPECT_THROW(line(0, 0, infF, 1), VcError);
  EXPECT_THROW(line(0, 0, 1, -infF), VcError);
}

TEST(Records, WritesNegativeZeroAsZero) {
  // Rounding a small negative value gives -0.0, which nlohmann would print as "-0.0".
  const Box b{0, -0.0f, -0.04f, 1.0f, 1.0f, -0.00001f};
  EXPECT_EQ(detectionRecord("a.mp4", 0, -0.0, b, kClasses, "x"),
            R"({"video":"a.mp4","frame":0,"pts_ms":0.0,"cls":"face","x1":0.0,"y1":0.0,"x2":1.0,"y2":1.0,)"
            R"("score":0.0,"src":"x"})"
            "\n");
  const std::vector<MaskBox> boxes{MaskBox{1, -0.0f, -0.04f, 1.0f, 1.0f, MaskOrigin::Extended}};
  EXPECT_EQ(maskRecord("a.mp4", 0, boxes, kClasses),
            R"({"video":"a.mp4","frame":0,"boxes":[{"cls":"plate","x1":0.0,"y1":0.0,"x2":1.0,"y2":1.0,"origin":"extended"}]})"
            "\n");
}

TEST(Records, InvalidUtf8IsReplacedInsteadOfThrowing) {
  const std::string bad = "\xffz\xc3.mp4";  // a stray continuation-less lead byte and a truncated sequence
  const Box b{0, 0, 0, 1, 1, 0.5f};
  std::string line;
  EXPECT_NO_THROW(line = detectionRecord(bad, 0, 0.0, b, kClasses, bad));
  EXPECT_NE(line.find("\xEF\xBF\xBD"), std::string::npos) << line;  // U+FFFD
  EXPECT_EQ(line.back(), '\n');
  std::string mask;
  EXPECT_NO_THROW(mask = maskRecord(bad, 0, {}, kClasses));
  EXPECT_NE(mask.find("\xEF\xBF\xBD"), std::string::npos) << mask;
}

}  // namespace
}  // namespace vc
