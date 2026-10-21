#include <gtest/gtest.h>

#include "common/host_device.h"
#include "common/platform.h"

namespace vc {
namespace {

VC_HOST_DEVICE constexpr int twice(int value) { return 2 * value; }

// Without nvcc the macro is empty: a VC_HOST_DEVICE function is an ordinary (here constexpr) host function.
static_assert(twice(21) == 42);

TEST(BuildConfig, KnowsThePlatformItWasBuiltFor) {
#if defined(__APPLE__)
  EXPECT_EQ(buildPlatform(), Platform::MacOS);
#elif defined(__linux__)
  EXPECT_EQ(buildPlatform(), Platform::Linux);
#else
  EXPECT_EQ(buildPlatform(), Platform::Other);
#endif
}

// VC_TEST_EXPECT_CUDA is 1 exactly when CMake was configured with -DVC_WITH_CUDA=ON.
TEST(BuildConfig, KnowsWhetherCudaIsBuiltIn) { EXPECT_EQ(builtWithCuda(), VC_TEST_EXPECT_CUDA == 1); }

}  // namespace
}  // namespace vc
