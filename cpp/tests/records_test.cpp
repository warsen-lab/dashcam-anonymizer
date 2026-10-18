#include "vc_redact/records.h"

#include <gtest/gtest.h>

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

}  // namespace
}  // namespace vc
