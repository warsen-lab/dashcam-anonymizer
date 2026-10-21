#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "detect/detector.h"
#include "detect/execution_provider.h"
#include "detect/yolo_decode.h"

namespace vc {

// YOLOv8 student model (model.onnx + model.json from `vc_train export`) on onnxruntime.
// backend() names the execution provider the session was created with (providerLabel: "CoreML", "CPU", ...);
// onnxruntime may still run nodes the provider does not support on the CPU. The model must be float32.
// detect() reuses internal buffers and is not thread-safe: use one detector per thread.
class OnnxYoloDetector : public Detector {
 public:
  // Frames passed to detect() must be frameWidth x frameHeight. The providers of providerOrder(provider) are tried
  // in turn (createOrtSession), with `settings` for every session. CoreML keeps its compiled model in
  // coreMlCacheDirectory(model.onnx) (created; without it CoreML compiles on every construction).
  // When the session cannot be created with that cache (processes that compile at the same time race
  // on it, and a compile that died half-way leaves an unloadable model) it is created once more
  // without the cache, leaving the shared directory as it is (another process may be writing it),
  // with one line on stderr; only when that fails too, or when the provider cannot be attached, does
  // the detector fall back to the CPU provider (also with a message on stderr). Throws VcError when
  // the files are missing or invalid (model.json of another output_format, a YuNet face model say,
  // included), when the ONNX input/output is not a float32 tensor, when model.json disagrees with the
  // ONNX input shape or class count, or when onnxruntime fails (also on the CPU retry).
  OnnxYoloDetector(const std::filesystem::path& modelDir, ExecutionProvider provider,
                   const DecodeParams& params, int frameWidth, int frameHeight, const SessionSettings& settings = {});
  ~OnnxYoloDetector() override;
  OnnxYoloDetector(const OnnxYoloDetector&) = delete;
  OnnxYoloDetector& operator=(const OnnxYoloDetector&) = delete;

  std::vector<Box> detect(const VideoFrame& frame) override;
  const std::vector<std::string>& classes() const override;
  std::string source() const override { return "student"; }
  std::string backend() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vc
