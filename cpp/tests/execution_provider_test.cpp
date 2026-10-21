#include "detect/execution_provider.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/vc_error.h"
#include "detect/ort_session.h"
#include "test_media.h"

namespace vc {
namespace {
namespace fs = std::filesystem;
using P = ExecutionProvider;
using Order = std::vector<ExecutionProvider>;

// Sets (or, with nullptr, unsets) an environment variable for the lifetime of the object.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* old = std::getenv(name)) old_ = old;
    if (value != nullptr) {
      setenv(name, value, 1);
    } else {
      unsetenv(name);
    }
  }
  ~ScopedEnv() {
    if (old_) {
      setenv(name_.c_str(), old_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  std::string name_;
  std::optional<std::string> old_;
};

// A fake provider set: the providers in `working` succeed, the others fail, unattached unless in `attached`.
struct FakeProviders {
  Order working;
  Order attached{};
  Order calls{};
  std::vector<std::string> notes{};

  ExecutionProvider run(const Order& order) {
    return tryProviders(
        order,
        [this](ExecutionProvider p) {
          calls.push_back(p);
          const auto in = [p](const Order& list) { return std::find(list.begin(), list.end(), p) != list.end(); };
          return ProviderAttempt{in(working), in(working) || in(attached), "no " + providerLabel(p)};
        },
        [this](const std::string& line) { notes.push_back(line); });
  }
};

TEST(ExecutionProvider, ParsesEveryName) {
  EXPECT_EQ(parseExecutionProvider("auto"), P::Auto);
  EXPECT_EQ(parseExecutionProvider("tensorrt"), P::TensorRT);
  EXPECT_EQ(parseExecutionProvider("cuda"), P::Cuda);
  EXPECT_EQ(parseExecutionProvider("coreml"), P::CoreML);
  EXPECT_EQ(parseExecutionProvider("cpu"), P::Cpu);
  for (const char* bad : {"", "CPU", "trt", "rocm", "gpu"}) {
    try {
      parseExecutionProvider(bad);
      ADD_FAILURE() << "accepted '" << bad << "'";
    } catch (const std::invalid_argument& e) {
      EXPECT_EQ(std::string(e.what()), "unknown execution provider '" + std::string(bad) +
                                           "' (auto|tensorrt|cuda|coreml|cpu)");
    }
  }
}

TEST(ExecutionProvider, LabelsAreTheStatsNames) {
  EXPECT_EQ(providerLabel(P::TensorRT), "TensorRT");
  EXPECT_EQ(providerLabel(P::Cuda), "CUDA");
  EXPECT_EQ(providerLabel(P::CoreML), "CoreML");
  EXPECT_EQ(providerLabel(P::Cpu), "CPU");
  EXPECT_EQ(providerLabel(P::Auto), "auto");
}

// spec 2026-10-20 §3: auto is TensorRT, CUDA, CPU on Linux and CoreML, CPU on macOS; an explicit provider falls
// back to the CPU.
TEST(ExecutionProvider, OrderPerPlatform) {
  EXPECT_EQ(providerOrder(P::Auto, Platform::Linux), (Order{P::TensorRT, P::Cuda, P::Cpu}));
  EXPECT_EQ(providerOrder(P::Auto, Platform::MacOS), (Order{P::CoreML, P::Cpu}));
  EXPECT_EQ(providerOrder(P::Auto, Platform::Other), (Order{P::Cpu}));
  for (const Platform platform : {Platform::Linux, Platform::MacOS, Platform::Other}) {
    EXPECT_EQ(providerOrder(P::TensorRT, platform), (Order{P::TensorRT, P::Cpu}));
    EXPECT_EQ(providerOrder(P::Cuda, platform), (Order{P::Cuda, P::Cpu}));
    EXPECT_EQ(providerOrder(P::CoreML, platform), (Order{P::CoreML, P::Cpu}));
    EXPECT_EQ(providerOrder(P::Cpu, platform), (Order{P::Cpu}));
  }
}

TEST(ExecutionProvider, TheFirstWorkingProviderWinsWithoutANote) {
  FakeProviders fake{{P::TensorRT, P::Cuda, P::Cpu}};
  EXPECT_EQ(fake.run({P::TensorRT, P::Cuda, P::Cpu}), P::TensorRT);
  EXPECT_EQ(fake.calls, (Order{P::TensorRT}));
  EXPECT_TRUE(fake.notes.empty());
}

TEST(ExecutionProvider, LinuxAutoFallsFromTensorRtToCuda) {
  FakeProviders fake{{P::Cuda, P::Cpu}};
  EXPECT_EQ(fake.run(providerOrder(P::Auto, Platform::Linux)), P::Cuda);
  EXPECT_EQ(fake.calls, (Order{P::TensorRT, P::Cuda}));
  EXPECT_EQ(fake.notes, (std::vector<std::string>{"vc: TensorRT execution provider unavailable (no TensorRT); trying CUDA"}));
}

TEST(ExecutionProvider, LinuxAutoEndsOnTheCpu) {
  FakeProviders fake{{P::Cpu}, {P::Cuda}};
  EXPECT_EQ(fake.run(providerOrder(P::Auto, Platform::Linux)), P::Cpu);
  EXPECT_EQ(fake.calls, (Order{P::TensorRT, P::Cuda, P::Cpu}));
  EXPECT_EQ(fake.notes, (std::vector<std::string>{"vc: TensorRT execution provider unavailable (no TensorRT); trying CUDA",
                                                  "vc: CUDA session failed (no CUDA); using CPU"}));
}

// The macOS lines are the ones vc_redact printed before --ep auto existed.
TEST(ExecutionProvider, MacAutoFallsFromCoreMlToTheCpu) {
  FakeProviders unavailable{{P::Cpu}};
  EXPECT_EQ(unavailable.run(providerOrder(P::Auto, Platform::MacOS)), P::Cpu);
  EXPECT_EQ(unavailable.notes, (std::vector<std::string>{"vc: CoreML execution provider unavailable (no CoreML); using CPU"}));
  FakeProviders failing{{P::Cpu}, {P::CoreML}};
  EXPECT_EQ(failing.run(providerOrder(P::Auto, Platform::MacOS)), P::Cpu);
  EXPECT_EQ(failing.notes, (std::vector<std::string>{"vc: CoreML session failed (no CoreML); using CPU"}));
}

TEST(ExecutionProvider, AnExplicitProviderFallsBackToTheCpu) {
  FakeProviders fake{{P::Cpu}};
  EXPECT_EQ(fake.run(providerOrder(P::TensorRT, Platform::MacOS)), P::Cpu);
  EXPECT_EQ(fake.calls, (Order{P::TensorRT, P::Cpu}));
  EXPECT_EQ(fake.notes, (std::vector<std::string>{"vc: TensorRT execution provider unavailable (no TensorRT); using CPU"}));
}

TEST(ExecutionProvider, ACpuFailureIsAnErrorAndTheOrderMustEndOnTheCpu) {
  FakeProviders none{{}};
  EXPECT_THROW(none.run({P::CoreML, P::Cpu}), VcError);
  FakeProviders fake{{P::Cuda}};
  EXPECT_THROW(fake.run({P::Cuda}), std::invalid_argument);
  EXPECT_THROW(fake.run({}), std::invalid_argument);
  EXPECT_TRUE(fake.calls.empty());
}

// TensorRT and CUDA cannot run a session in this build; the reason says which piece is missing.
TEST(ExecutionProvider, SaysWhyAGpuProviderIsUnavailable) {
  const std::vector<std::string> mac = {"CoreMLExecutionProvider", "CPUExecutionProvider"};
  const std::vector<std::string> gpu = {"TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"};
  EXPECT_EQ(gpuProviderUnavailableReason(P::TensorRT, mac, false), "TensorrtExecutionProvider is not in this onnxruntime build");
  EXPECT_EQ(gpuProviderUnavailableReason(P::Cuda, mac, true), "CUDAExecutionProvider is not in this onnxruntime build");
  EXPECT_EQ(gpuProviderUnavailableReason(P::Cuda, gpu, false), "this build was made without VC_WITH_CUDA");
  EXPECT_EQ(gpuProviderUnavailableReason(P::TensorRT, gpu, true),
            "the TensorRT / CUDA session set-up comes with stage 1c of the Linux plan");
}

// The TensorRT / CUDA branch asks onnxruntime for its providers before any session exists; the telemetry (which
// uploads past proxy settings) must already be off by then. The environment is set once per process, so the first
// call runs in a fresh process (a "threadsafe" death test re-executes the binary).
TEST(ExecutionProvider, SwitchesTheTelemetryOffBeforeAskingForGpuProviders) {
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";  // GTEST_FLAG_SET needs googletest 1.12; Ubuntu 22.04 has 1.11
  EXPECT_EXIT(
      {
        unsetenv("ORT_DISABLE_TELEMETRY");
        const std::vector<std::string> providers = availableOrtProviders();
        const char* value = std::getenv("ORT_DISABLE_TELEMETRY");
        std::exit(!providers.empty() && value != nullptr && std::string(value) == "1" ? 0 : 1);
      },
      testing::ExitedWithCode(0), "");
}

// The whole path: a CUDA request on a build that has no CUDA falls back to the CPU with the telemetry off.
TEST(ExecutionProvider, ACudaRequestWithoutCudaRunsOnTheCpuWithTheTelemetryOff) {
  const fs::path onnx = fs::path(VC_TEST_DATA_DIR) / "const_yolo" / "model.onnx";
  const CreatedSession created = createOrtSession(onnx, P::Cuda, SessionCaches{}, SessionSettings{});
  EXPECT_EQ(created.backend, "CPU");  // this build has no CUDA session set-up
  const char* value = std::getenv("ORT_DISABLE_TELEMETRY");
  ASSERT_NE(value, nullptr);
  EXPECT_STREQ(value, "1");
}

// spec 2026-10-20 §3: $VC_CACHE/trt-cache/<sha256 of model.onnx>/<width>x<height>/, like the CoreML cache.
TEST(ExecutionProvider, KeysTheTensorRtCacheByModelAndInputSize) {
  const fs::path onnx = fs::path(VC_TEST_DATA_DIR) / "const_yolo" / "model.onnx";
  // `shasum -a 256 tests/data/const_yolo/model.onnx`
  const std::string sha = "cf8f5bf0e958cf75b422ac5b6991c6c4c6f50c389e389eb5ec69262076b35d51";
  const fs::path root = test::tmpDir() / "trt_cache_root";
  {
    const ScopedEnv cache("VC_CACHE", root.c_str());
    EXPECT_EQ(tensorRtCacheDirectory(onnx, 1920, 1088), root / "trt-cache" / sha / "1920x1088");
    EXPECT_EQ(tensorRtCacheDirectory(onnx, 3840, 2176), root / "trt-cache" / sha / "3840x2176");
    EXPECT_FALSE(fs::exists(root));  // nothing is created
    EXPECT_THROW(tensorRtCacheDirectory(test::tmpDir() / "no_such_model.onnx", 64, 64), VcError);
  }
  const ScopedEnv home("HOME", "/nonexistent-home-for-test");
  const ScopedEnv noCache("VC_CACHE", nullptr);
  EXPECT_EQ(tensorRtCacheDirectory(onnx, 64, 64),
            fs::path("/nonexistent-home-for-test") / ".cache" / "video-compliance" / "trt-cache" / sha / "64x64");
  const ScopedEnv noHome("HOME", nullptr);
  EXPECT_FALSE(tensorRtCacheDirectory(onnx, 64, 64).has_value());
}

}  // namespace
}  // namespace vc
