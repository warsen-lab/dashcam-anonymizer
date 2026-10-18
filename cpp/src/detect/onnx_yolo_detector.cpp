#include "detect/onnx_yolo_detector.h"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "common/vc_error.h"
#include "detect/letterbox.h"
#include "detect/model_info.h"

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

}  // namespace

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

  void checkShapes(const fs::path& onnx) {
    if (session.GetInputCount() != 1 || session.GetOutputCount() < 1) {
      throw VcError(onnx.string() + ": expected one input and at least one output");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    inputName = session.GetInputNameAllocated(0, allocator).get();
    outputName = session.GetOutputNameAllocated(0, allocator).get();
    const std::vector<int64_t> inShape =
        session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const std::vector<int64_t> wanted{1, 3, info.inputHeight, info.inputWidth};
    if (inShape != wanted) {
      throw VcError(onnx.string() + ": input shape " + shapeText(inShape) +
                    " does not match model.json input " + shapeText(wanted));
    }
    const std::vector<int64_t> outShape =
        session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
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
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (provider == ExecutionProvider::CoreML) {
      try {
        options.AppendExecutionProvider(
            "CoreML", {{"ModelFormat", "MLProgram"}, {"MLComputeUnits", "ALL"},
                       {"RequireStaticInputShapes", "1"}});
        s.backend = "CoreML";
      } catch (const Ort::Exception& e) {
        std::fprintf(stderr, "vc: CoreML execution provider unavailable (%s); using CPU\n", e.what());
      }
    }
    s.session = Ort::Session(ortEnv(), onnx.c_str(), options);
    s.checkShapes(onnx);
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
    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> shape{1, 3, s.info.inputHeight, s.info.inputWidth};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(memory, s.input.data(), s.input.size(),
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
