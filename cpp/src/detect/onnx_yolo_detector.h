#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "detect/detector.h"
#include "detect/yolo_decode.h"

namespace vc {

enum class ExecutionProvider { CoreML, Cpu };

// "coreml" | "cpu"; throws std::invalid_argument.
ExecutionProvider parseExecutionProvider(const std::string& text);

// YOLOv8 student model (model.onnx + model.json from `vc_train export`) on onnxruntime.
class OnnxYoloDetector : public Detector {
 public:
  // Frames passed to detect() must be frameWidth x frameHeight. CoreML falls back to the CPU
  // provider (with a message on stderr) when it cannot be attached. Throws VcError when the files
  // are missing or invalid, when model.json disagrees with the ONNX input shape or class count, or
  // when onnxruntime fails.
  OnnxYoloDetector(const std::filesystem::path& modelDir, ExecutionProvider provider,
                   const DecodeParams& params, int frameWidth, int frameHeight);
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
