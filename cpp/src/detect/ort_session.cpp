#include "detect/ort_session.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <utility>

#include "common/vc_error.h"

extern "C" {
#include <libavutil/mem.h>
#include <libavutil/sha.h>
}

namespace vc {
namespace fs = std::filesystem;
namespace {

const char* elementTypeName(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "bfloat16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "float64";
    default: return "a non-float32 type";
  }
}

constexpr int kSha256Bits = 256;
constexpr size_t kHashChunkBytes = 1 << 20;

// Lower-case hex SHA-256 of the file's bytes.
std::string sha256Hex(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw VcError("cannot read " + path.string());
  const std::unique_ptr<AVSHA, decltype(&av_free)> sha(av_sha_alloc(), &av_free);
  if (!sha || av_sha_init(sha.get(), kSha256Bits) < 0) throw VcError("cannot set up SHA-256");
  std::vector<char> chunk(kHashChunkBytes);
  while (in) {
    in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    const std::streamsize got = in.gcount();
    if (got > 0) av_sha_update(sha.get(), reinterpret_cast<const uint8_t*>(chunk.data()), static_cast<size_t>(got));
  }
  if (in.bad()) throw VcError("cannot read " + path.string());
  std::array<uint8_t, kSha256Bits / 8> digest{};
  av_sha_final(sha.get(), digest.data());
  static constexpr char kHex[] = "0123456789abcdef";
  std::string hex;
  for (const uint8_t byte : digest) {
    hex += kHex[byte >> 4];
    hex += kHex[byte & 0xF];
  }
  return hex;
}

// A set, non-empty environment variable.
std::optional<std::string> envValue(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return std::nullopt;
  return std::string(value);
}

// `dir`, created; nullopt (CoreML then compiles into a temporary directory each time) when there is
// none or it cannot be created, with a note on stderr for the latter.
std::optional<fs::path> usableCoreMlCache(const std::optional<fs::path>& dir) {
  if (!dir) return std::nullopt;
  std::error_code error;
  fs::create_directories(*dir, error);
  if (error || !fs::is_directory(*dir)) {
    std::fprintf(stderr, "vc: cannot create the CoreML cache %s (%s); compiling without it\n", dir->c_str(),
                 error ? error.message().c_str() : "not a directory");
    return std::nullopt;
  }
  return dir;
}

Ort::SessionOptions makeOptions(const SessionConfigurer& configure) {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (configure) configure(options);
  return options;
}

// One try at a session on the CoreML provider, compiling into `cache` (a temporary directory when
// there is none).
struct CoreMlAttempt {
  Ort::Session session{nullptr};
  bool attached = false;  // the provider was registered (the session may still have failed)
  bool ok = false;        // `session` is usable
  std::string error;      // why not, otherwise
};

CoreMlAttempt attemptCoreMl(const fs::path& onnx, const std::optional<fs::path>& cache,
                            const SessionConfigurer& configure) {
  CoreMlAttempt attempt;
  Ort::SessionOptions options = makeOptions(configure);
  try {
    std::unordered_map<std::string, std::string> coreMl{
        {"ModelFormat", "MLProgram"}, {"MLComputeUnits", "ALL"}, {"RequireStaticInputShapes", "1"}};
    if (cache) coreMl["ModelCacheDirectory"] = cache->string();
    options.AppendExecutionProvider("CoreML", coreMl);
  } catch (const Ort::Exception& e) {
    attempt.error = e.what();
    return attempt;
  }
  attempt.attached = true;
  try {
    // CoreML compiles the model while the session is created.
    attempt.session = Ort::Session(ortEnv(), onnx.c_str(), options);
    attempt.ok = true;
  } catch (const Ort::Exception& e) {
    attempt.error = e.what();
  }
  return attempt;
}

}  // namespace

// onnxruntime's macOS / Linux build ships Microsoft 1DS telemetry: it uploads usage events with a
// device id over the system network stack (proxy environment variables do not stop it) and queues
// them in ~/Library/Application Support/Microsoft/DeveloperTools/.onnxruntime. The link is metered
// and the data is the user's, so it is switched off before the environment exists (the variable is
// read then) and once more through the API.
Ort::Env& ortEnv() {
  static Ort::Env env = [] {
    ::setenv("ORT_DISABLE_TELEMETRY", "1", 1);
    Ort::Env created(ORT_LOGGING_LEVEL_WARNING, "vc");
    created.DisableTelemetryEvents();
    return created;
  }();
  return env;
}

std::string shapeText(const std::vector<int64_t>& shape) {
  std::string text = "[";
  for (size_t i = 0; i < shape.size(); ++i) text += (i ? ", " : "") + std::to_string(shape[i]);
  return text + "]";
}

TensorSpec readFloatTensorSpec(const Ort::TypeInfo& info, const std::string& what, const fs::path& onnx) {
  if (info.GetONNXType() != ONNX_TYPE_TENSOR) {
    throw VcError(onnx.string() + ": the " + what + " is not a tensor");
  }
  const auto tensor = info.GetTensorTypeAndShapeInfo();
  TensorSpec spec{tensor.GetElementType(), tensor.GetShape()};
  if (spec.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw VcError(onnx.string() + ": the " + what + " tensor is " + elementTypeName(spec.type) +
                  ", but the model must be float32 (e.g. not a float16 export)");
  }
  return spec;
}

std::optional<fs::path> coreMlCacheDirectory(const fs::path& onnx) {
  std::optional<fs::path> root;
  if (const auto cache = envValue("VC_CACHE")) {
    root = fs::path(*cache);
  } else if (const auto home = envValue("HOME")) {
    root = fs::path(*home) / ".cache" / "video-compliance";
  }
  if (!root) return std::nullopt;
  return *root / "coreml-cache" / sha256Hex(onnx);
}

ExecutionProvider parseExecutionProvider(const std::string& text) {
  if (text == "coreml") return ExecutionProvider::CoreML;
  if (text == "cpu") return ExecutionProvider::Cpu;
  throw std::invalid_argument("unknown execution provider '" + text + "' (coreml|cpu)");
}

CreatedSession createOrtSession(const fs::path& onnx, ExecutionProvider provider,
                                const std::optional<fs::path>& coreMlCache, const SessionConfigurer& configure) {
  CreatedSession result;
  if (provider == ExecutionProvider::CoreML) {
    const std::optional<fs::path> cache = usableCoreMlCache(coreMlCache);
    CoreMlAttempt coreMl = attemptCoreMl(onnx, cache, configure);
    if (coreMl.attached && !coreMl.ok && cache) {
      // The cache is shared: processes that compile at the same time race on it ("Error copying
      // compiled model to cache path"), and a compile that died half-way leaves a model that cannot
      // be loaded. Another process may be writing the directory, so it is left alone; compile
      // without it once before giving the CoreML provider up.
      std::fprintf(stderr,
                   "vc: CoreML session failed with the compile cache %s (%s); retrying without the cache "
                   "(delete %s to rebuild the cache)\n",
                   cache->c_str(), coreMl.error.c_str(), cache->c_str());
      coreMl = attemptCoreMl(onnx, std::nullopt, configure);
    }
    if (coreMl.ok) {
      result.session = std::move(coreMl.session);
      result.backend = "CoreML";
      return result;
    }
    if (coreMl.attached) {
      std::fprintf(stderr, "vc: CoreML session failed (%s); using CPU\n", coreMl.error.c_str());
    } else {
      std::fprintf(stderr, "vc: CoreML execution provider unavailable (%s); using CPU\n", coreMl.error.c_str());
    }
  }
  const Ort::SessionOptions cpuOptions = makeOptions(configure);
  result.session = Ort::Session(ortEnv(), onnx.c_str(), cpuOptions);
  result.backend = "CPU";
  return result;
}

}  // namespace vc
