#include "detect/yunet_detector.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/vc_error.h"
#include "detect/model_info.h"
#include "detect/ort_session.h"
#include "io/av_ptr.h"

namespace vc {
namespace fs = std::filesystem;
namespace {

// The outputs decodeYuNet reads, in YuNetLevel order: cls, obj, bbox of strides 8, 16, 32.
constexpr std::array<const char*, 9> kOutputs = {"cls_8", "cls_16", "cls_32", "obj_8",  "obj_16",
                                                 "obj_32", "bbox_8", "bbox_16", "bbox_32"};
constexpr size_t kLevels = kYuNetStrides.size();

// Values per anchor of kOutputs[i].
int64_t valuesPerAnchor(size_t i) { return i < 2 * kLevels ? 1 : 4; }
int strideOf(size_t i) { return kYuNetStrides[i % kLevels]; }

// `size` rounded up to a multiple of `multiple` (both positive), as FaceDetectorYN pads.
int padded(int size, int multiple) { return ((size - 1) / multiple + 1) * multiple; }

std::string sizeText(int64_t width, int64_t height) { return std::to_string(width) + "x" + std::to_string(height); }

}  // namespace

std::optional<fs::path> yunetCoreMlCacheDirectory(const fs::path& onnx, int padWidth, int padHeight) {
  const std::optional<fs::path> base = coreMlCacheDirectory(onnx);
  if (!base) return std::nullopt;
  return *base / sizeText(padWidth, padHeight);
}

struct YuNetDetector::Impl {
  ModelInfo info;
  fs::path onnx;
  ExecutionProvider provider = ExecutionProvider::Cpu;
  SessionSettings settings{};
  YuNetParams params;
  Ort::Session session{nullptr};
  std::string backend = "CPU";
  std::string inputName;
  int frameWidth = 0;
  int frameHeight = 0;
  int padWidth = 0;
  int padHeight = 0;
  std::vector<float> input;  // [1, 3, padHeight, padWidth]: planes B, G, R
  av::SwsPtr sws = av::allocSws();
  av::FramePtr bgr = av::allocFrame();
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // (Re)builds the session for frames of width x height. Everything is built in locals and committed
  // at the end by moves and swaps that cannot throw, so a failure anywhere leaves the detector as it
  // was: the next frame of the old size still runs on the old session and buffer, and a frame of the
  // size that failed is refused again instead of running on half a state.
  void open(int width, int height) {
    if (width <= 0 || height <= 0) throw VcError("face detector: frame size " + sizeText(width, height));
    const int pw = padded(width, info.padMultiple);
    const int ph = padded(height, info.padMultiple);
    // Fixes the symbolic input size, so that the graph has static shapes (CoreML needs them).
    const SessionConfigurer fixSize = [&](Ort::SessionOptions& options) {
      options.AddFreeDimensionOverrideByName("H", ph);
      options.AddFreeDimensionOverrideByName("W", pw);
    };
    CreatedSession created;
    std::string newInputName;
    try {
      const SessionCaches caches{[&] { return yunetCoreMlCacheDirectory(onnx, pw, ph); },
                                 [&] { return tensorRtCacheDirectory(onnx, pw, ph); }};
      created = createOrtSession(onnx, provider, caches, settings, fixSize);
      newInputName = checkModel(created.session, width, height, pw, ph);  // its VcError is a model problem, never retried
    } catch (const Ort::Exception& e) {
      throw VcError("onnxruntime: " + onnx.string() + ": " + e.what());
    }
    // The padding stays 0: fill() only ever writes the frame's own pixels.
    std::vector<float> buffer;
    try {
      buffer.assign(3 * static_cast<size_t>(pw) * static_cast<size_t>(ph), 0.0f);
    } catch (const std::bad_alloc&) {
      throw VcError("face detector: no memory for the input of a " + sizeText(width, height) + " frame");
    } catch (const std::length_error&) {
      throw VcError("face detector: the input of a " + sizeText(width, height) + " frame is too large");
    }
    static_assert(std::is_nothrow_move_assignable_v<Ort::Session> && std::is_nothrow_move_assignable_v<std::string>);
    input.swap(buffer);  // before the sizes: fill() trusts them to describe `input`
    session = std::move(created.session);
    backend = std::move(created.backend);
    inputName = std::move(newInputName);
    frameWidth = width;
    frameHeight = height;
    padWidth = pw;
    padHeight = ph;
  }

  // Checks the input and the outputs (found by the names in kOutputs) of `checked` for a frame of
  // width x height padded to pw x ph; returns the input's name.
  std::string checkModel(const Ort::Session& checked, int width, int height, int pw, int ph) const {
    if (checked.GetInputCount() != 1) throw VcError(onnx.string() + ": expected one input");
    Ort::AllocatorWithDefaultOptions allocator;
    std::string name = checked.GetInputNameAllocated(0, allocator).get();
    const std::vector<int64_t> in = readFloatTensorSpec(checked.GetInputTypeInfo(0), "input", onnx).shape;
    if (in.size() != 4 || (in[0] != 1 && in[0] != -1) || in[1] != 3) {
      throw VcError(onnx.string() + ": input shape " + shapeText(in) + " is not [1, 3, H, W]");
    }
    if ((in[2] != -1 && in[2] != ph) || (in[3] != -1 && in[3] != pw)) {
      throw VcError(onnx.string() + ": the input is fixed at " + sizeText(in[3], in[2]) + ", but a " +
                    sizeText(width, height) + " frame needs " + sizeText(pw, ph) +
                    "; write the model with `python -m vc_train yunet`, which frees the input size");
    }
    for (size_t i = 0; i < kOutputs.size(); ++i) {
      bool found = false;
      for (size_t k = 0; k < checked.GetOutputCount() && !found; ++k) {
        if (std::string(checked.GetOutputNameAllocated(k, allocator).get()) != kOutputs[i]) continue;
        found = true;
        const std::vector<int64_t> shape =
            readFloatTensorSpec(checked.GetOutputTypeInfo(k), std::string("output ") + kOutputs[i], onnx).shape;
        if (shape.size() != 3 || shape[2] != valuesPerAnchor(i)) {
          throw VcError(onnx.string() + ": output " + kOutputs[i] + " has shape " + shapeText(shape) +
                        ", expected [1, N, " + std::to_string(valuesPerAnchor(i)) + "]");
        }
      }
      if (!found) throw VcError(onnx.string() + ": no output " + kOutputs[i] + " (not a YuNet model?)");
    }
    return name;
  }

  // Frame -> BGR floats 0-255 in the top-left corner of the input.
  void fill(const AVFrame& frame) {
    av_frame_unref(bgr.get());
    bgr->format = AV_PIX_FMT_BGR24;
    bgr->width = frame.width;
    bgr->height = frame.height;
    checkAv(sws_scale_frame(sws.get(), bgr.get(), &frame), "convert frame to BGR");
    const size_t plane = static_cast<size_t>(padWidth) * static_cast<size_t>(padHeight);
    for (int y = 0; y < frameHeight; ++y) {
      const uint8_t* row = bgr->data[0] + static_cast<ptrdiff_t>(y) * bgr->linesize[0];
      float* b = input.data() + static_cast<size_t>(y) * static_cast<size_t>(padWidth);
      float* g = b + plane;
      float* r = g + plane;
      for (int x = 0; x < frameWidth; ++x) {
        b[x] = row[3 * x];
        g[x] = row[3 * x + 1];
        r[x] = row[3 * x + 2];
      }
    }
  }

  std::vector<Box> run() {
    const std::array<int64_t, 4> shape{1, 3, padHeight, padWidth};
    Ort::Value tensor =
        Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
    const char* inputs[] = {inputName.c_str()};
    std::vector<Ort::Value> result =
        session.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, kOutputs.data(), kOutputs.size());
    std::array<const float*, kOutputs.size()> data{};
    for (size_t i = 0; i < kOutputs.size(); ++i) {
      const int s = strideOf(i);
      const std::vector<int64_t> wanted{1, static_cast<int64_t>(padWidth / s) * (padHeight / s), valuesPerAnchor(i)};
      const std::vector<int64_t> got = result[i].GetTensorTypeAndShapeInfo().GetShape();
      if (got != wanted) {
        throw VcError(std::string("face model output ") + kOutputs[i] + " has shape " + shapeText(got) +
                      ", expected " + shapeText(wanted));
      }
      data[i] = result[i].GetTensorData<float>();
    }
    std::array<YuNetLevel, kLevels> levels{};
    for (size_t k = 0; k < kLevels; ++k) {
      levels[k] = YuNetLevel{data[k], data[kLevels + k], data[2 * kLevels + k]};
    }
    return decodeYuNet(levels, padWidth, padHeight, frameWidth, frameHeight, params);
  }
};

YuNetDetector::YuNetDetector(const fs::path& modelDir, ExecutionProvider provider, const YuNetParams& params,
                             int frameWidth, int frameHeight, const SessionSettings& settings)
    : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  const fs::path json = modelDir / "model.json";
  s.info = readModelInfo(json);
  if (s.info.outputFormat != OutputFormat::YuNet) {
    throw VcError(json.string() + ": output_format must be \"yunet\" for a face model (write it with "
                  "`python -m vc_train yunet`; a student model goes to --model)");
  }
  s.onnx = modelDir / "model.onnx";
  if (!fs::is_regular_file(s.onnx)) throw VcError("missing " + s.onnx.string());
  s.provider = provider;
  s.settings = settings;
  s.params = params;
  s.sws->flags = SWS_BILINEAR;
  s.open(frameWidth, frameHeight);
}

YuNetDetector::~YuNetDetector() = default;

std::vector<Box> YuNetDetector::detect(const VideoFrame& frame) {
  Impl& s = *impl_;
  if (frame.av == nullptr) throw VcError("face detector: frame has no pixels");
  if (frame.av->width != s.frameWidth || frame.av->height != s.frameHeight) s.open(frame.av->width, frame.av->height);
  s.fill(*frame.av);
  try {
    return s.run();
  } catch (const Ort::Exception& e) {
    throw VcError(std::string("onnxruntime: face model inference failed: ") + e.what());
  }
}

const std::vector<std::string>& YuNetDetector::classes() const { return impl_->info.classes; }

std::string YuNetDetector::backend() const { return impl_->backend; }

}  // namespace vc
