#pragma once

// The onnxruntime set-up the ONNX detectors share. Internal to src/detect: only their .cpp files
// include it (it pulls in the onnxruntime C++ API).

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "detect/execution_provider.h"

namespace vc {

// The process-wide environment, created with onnxruntime's usage telemetry switched off.
Ort::Env& ortEnv();

// The execution providers this onnxruntime build has (Ort::GetAvailableProviders()), after ortEnv() has switched
// the telemetry off: call this, not onnxruntime's function, so that no call comes before that.
std::vector<std::string> availableOrtProviders();

// Model-specific session options (free dimension overrides, say), applied to every attempt.
using SessionConfigurer = std::function<void(Ort::SessionOptions&)>;

// What createOrtSession returns. (Not named OrtSession: onnxruntime's C API has a type of that name.)
struct CreatedSession {
  Ort::Session session{nullptr};
  std::string backend = "CPU";  // providerLabel of the provider the session runs with, e.g. "CoreML"
};

// The outcome of trying one execution provider.
struct ProviderAttempt {
  bool ok = false;        // a session exists on it
  bool attached = false;  // the provider could be registered (its session may still have failed)
  std::string error{};    // why not, otherwise
};
using ProviderTry = std::function<ProviderAttempt(ExecutionProvider)>;
using NoteSink = std::function<void(const std::string&)>;

// Tries `order` front to back with `attempt` until one is ok and returns it. For each provider passed over, `note`
// gets one line naming the next: "vc: <P> execution provider unavailable (<error>); trying <next>" when it could not
// be attached, "vc: <P> session failed (<error>); ..." when its session failed, "using <next>" instead of "trying
// <next>" when the next is the last. `order` must end with Cpu (std::invalid_argument otherwise); a CPU attempt
// that throws propagates, one that reports failure throws VcError.
ExecutionProvider tryProviders(const std::vector<ExecutionProvider>& order, const ProviderTry& attempt,
                               const NoteSink& note);

// Why the TensorRT or CUDA provider cannot run a session in this build: not in this onnxruntime (its
// GetAvailableProviders() is `available`), a build made without VC_WITH_CUDA, or (with it) the GPU session
// set-up of stage 1c not being in this build yet.
std::string gpuProviderUnavailableReason(ExecutionProvider provider, const std::vector<std::string>& available,
                                         bool withCuda);

// The caches of the providers that compile the model, computed only when that provider is tried (hashing the
// model takes a moment, and an unreadable file only matters then).
using CacheDirectory = std::function<std::optional<std::filesystem::path>()>;
struct SessionCaches {
  CacheDirectory coreMl{};    // CoreML's compile cache (coreMlCacheDirectory, or per input size for YuNet)
  CacheDirectory tensorRt{};  // TensorRT's engine cache (tensorRtCacheDirectory), used from stage 1c on
};

// A session for `onnx` with all graph optimisations and `settings`, on the first provider of
// providerOrder(provider, buildPlatform()) that works (tryProviders, notes on stderr). CoreML compiles into
// caches.coreMl() (created; compiling into a temporary directory when there is none or it cannot be created, with
// a note on stderr for the latter); when that session fails with the cache it is created once more without it,
// leaving the shared directory alone (another process may be writing it), with one line on stderr, before the next
// provider is tried. TensorRT and CUDA are unavailable in this build (gpuProviderUnavailableReason). Throws
// Ort::Exception when the CPU session fails.
CreatedSession createOrtSession(const std::filesystem::path& onnx, ExecutionProvider provider,
                                const SessionCaches& caches, const SessionSettings& settings,
                                const SessionConfigurer& configure = {});

struct TensorSpec {
  ONNXTensorElementDataType type;
  std::vector<int64_t> shape;  // -1 for a symbolic dimension
};

// `what` names the tensor in messages ("input", "output cls_8"). GetTensorData<float> does no type
// check, so a float16 export would be read as garbage boxes: throws VcError unless the value is a
// float32 tensor.
TensorSpec readFloatTensorSpec(const Ort::TypeInfo& info, const std::string& what,
                               const std::filesystem::path& onnx);

// "[1, 3, 640, 640]"
std::string shapeText(const std::vector<int64_t>& shape);

}  // namespace vc
