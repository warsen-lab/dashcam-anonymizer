#include "detect/onnx_yolo_detector.h"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/vc_error.h"
#include "detect/letterbox.h"
#include "detect/model_info.h"

extern "C" {
#include <libavutil/mem.h>
#include <libavutil/sha.h>
}

namespace vc {
namespace fs = std::filesystem;
namespace {

Ort::Env& ortEnv() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "vc");
  return env;
}

std::string shapeText(const std::vector<int64_t>& shape) {
  std::string text = "[";
  for (size_t i = 0; i < shape.size(); ++i) text += (i ? ", " : "") + std::to_string(shape[i]);
  return text + "]";
}

const char* elementTypeName(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "bfloat16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "float64";
    default: return "a non-float32 type";
  }
}

struct TensorSpec {
  ONNXTensorElementDataType type;
  std::vector<int64_t> shape;
};

// `what` is "input" or "output", for messages. GetTensorData<float> does no type check, so a
// float16 export would be read as garbage boxes; refuse anything but float32 tensors up front.
TensorSpec readFloatTensorSpec(const Ort::TypeInfo& info, const char* what, const fs::path& onnx) {
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

// coreMlCacheDirectory(onnx), created; nullopt (CoreML then compiles into a temporary directory each
// time) when there is none or it cannot be created, with a note on stderr for the latter.
std::optional<fs::path> usableCoreMlCache(const fs::path& onnx) {
  const std::optional<fs::path> dir = coreMlCacheDirectory(onnx);
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

Ort::SessionOptions makeOptions() {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
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

CoreMlAttempt attemptCoreMl(const fs::path& onnx, const std::optional<fs::path>& cache) {
  CoreMlAttempt attempt;
  Ort::SessionOptions options = makeOptions();
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

struct OnnxYoloDetector::Impl {
  ModelInfo info;
  DecodeParams params;
  Letterbox lb;
  std::unique_ptr<InputTensorFiller> filler;
  Ort::Session session{nullptr};
  std::string inputName;
  std::string outputName;
  std::string backend = "CPU";
  int64_t anchors = 0;
  std::vector<float> input;
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  void checkShapes(const fs::path& onnx) {
    if (session.GetInputCount() != 1 || session.GetOutputCount() < 1) {
      throw VcError(onnx.string() + ": expected one input and at least one output");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    inputName = session.GetInputNameAllocated(0, allocator).get();
    outputName = session.GetOutputNameAllocated(0, allocator).get();
    const std::vector<int64_t> inShape =
        readFloatTensorSpec(session.GetInputTypeInfo(0), "input", onnx).shape;
    const std::vector<int64_t> wanted{1, 3, info.inputHeight, info.inputWidth};
    if (inShape != wanted) {
      throw VcError(onnx.string() + ": input shape " + shapeText(inShape) +
                    " does not match model.json input " + shapeText(wanted));
    }
    const std::vector<int64_t> outShape =
        readFloatTensorSpec(session.GetOutputTypeInfo(0), "output", onnx).shape;
    const int64_t rows = 4 + static_cast<int64_t>(info.classes.size());
    if (outShape.size() != 3 || outShape[0] != 1 || outShape[1] != rows || outShape[2] <= 0) {
      throw VcError(onnx.string() + ": output shape " + shapeText(outShape) + " does not match the " +
                    std::to_string(info.classes.size()) + " classes in model.json (expected [1, " +
                    std::to_string(rows) + ", N])");
    }
    anchors = outShape[2];
  }
};

OnnxYoloDetector::OnnxYoloDetector(const fs::path& modelDir, ExecutionProvider provider,
                                   const DecodeParams& params, int frameWidth, int frameHeight)
    : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.info = readModelInfo(modelDir / "model.json");
  s.params = params;
  s.lb = makeLetterbox(frameWidth, frameHeight, s.info.inputWidth, s.info.inputHeight);
  s.filler = std::make_unique<InputTensorFiller>(s.lb);
  s.input.resize(3 * static_cast<size_t>(s.info.inputWidth) * s.info.inputHeight);
  const fs::path onnx = modelDir / "model.onnx";
  if (!fs::is_regular_file(onnx)) throw VcError("missing " + onnx.string());
  try {
    if (provider == ExecutionProvider::CoreML) {
      const std::optional<fs::path> cache = usableCoreMlCache(onnx);
      CoreMlAttempt coreMl = attemptCoreMl(onnx, cache);
      if (coreMl.attached && !coreMl.ok && cache) {
        // The cache is shared: processes that compile at the same time race on it ("Error copying
        // compiled model to cache path"), and a compile that died half-way leaves a model that cannot
        // be loaded. Another process may be writing the directory, so it is left alone; compile
        // without it once before giving the CoreML provider up.
        std::fprintf(stderr,
                     "vc: CoreML session failed with the compile cache %s (%s); retrying without the cache "
                     "(delete %s to rebuild the cache)\n",
                     cache->c_str(), coreMl.error.c_str(), cache->c_str());
        coreMl = attemptCoreMl(onnx, std::nullopt);
      }
      if (coreMl.ok) {
        s.session = std::move(coreMl.session);
        s.backend = "CoreML";
      } else if (coreMl.attached) {
        std::fprintf(stderr, "vc: CoreML session failed (%s); using CPU\n", coreMl.error.c_str());
      } else {
        std::fprintf(stderr, "vc: CoreML execution provider unavailable (%s); using CPU\n", coreMl.error.c_str());
      }
    }
    if (s.backend != "CoreML") {
      const Ort::SessionOptions cpuOptions = makeOptions();
      s.session = Ort::Session(ortEnv(), onnx.c_str(), cpuOptions);
    }
    s.checkShapes(onnx);  // its VcError is a model problem, never retried
  } catch (const Ort::Exception& e) {
    throw VcError("onnxruntime: " + onnx.string() + ": " + e.what());
  }
}

OnnxYoloDetector::~OnnxYoloDetector() = default;

std::vector<Box> OnnxYoloDetector::detect(const VideoFrame& frame) {
  Impl& s = *impl_;
  if (frame.av == nullptr) throw VcError("detector: frame has no pixels");
  s.filler->fill(*frame.av, s.input.data());
  try {
    const std::array<int64_t, 4> shape{1, 3, s.info.inputHeight, s.info.inputWidth};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(s.memory, s.input.data(), s.input.size(),
                                                        shape.data(), shape.size());
    const char* inputs[] = {s.inputName.c_str()};
    const char* outputs[] = {s.outputName.c_str()};
    std::vector<Ort::Value> result =
        s.session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 1);
    return decodeYolov8(result[0].GetTensorData<float>(), static_cast<int>(s.info.classes.size()),
                        s.anchors, s.lb, s.params);
  } catch (const Ort::Exception& e) {
    throw VcError(std::string("onnxruntime: inference failed: ") + e.what());
  }
}

const std::vector<std::string>& OnnxYoloDetector::classes() const { return impl_->info.classes; }

std::string OnnxYoloDetector::backend() const { return impl_->backend; }

}  // namespace vc
