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

// Model-specific session options (free dimension overrides, say), applied to every attempt.
using SessionConfigurer = std::function<void(Ort::SessionOptions&)>;

struct OrtSession {
  Ort::Session session{nullptr};
  std::string backend = "CPU";  // "CoreML" when the session runs with the CoreML provider registered
};

// A session for `onnx` with all graph optimisations. CoreML first when asked for, compiling into
// `coreMlCache` (created; compiling into a temporary directory when it is nullopt or cannot be
// created, with a note on stderr for the latter). When that session fails with the cache it is
// created once more without it, leaving the shared directory alone (another process may be writing
// it), with one line on stderr; only when that fails too, or when the provider cannot be attached, does
// the CPU provider take over (also with a line on stderr). Throws Ort::Exception when the CPU session
// fails.
OrtSession createOrtSession(const std::filesystem::path& onnx, ExecutionProvider provider,
                            const std::optional<std::filesystem::path>& coreMlCache,
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
