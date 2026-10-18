#pragma once

#include <cstdint>
#include <filesystem>

#include "detect/onnx_yolo_detector.h"

namespace vc {

struct DetectImagesOptions {
  std::filesystem::path modelDir;   // model.onnx + model.json
  std::filesystem::path framesDir;  // vc_extract output: frames.csv + <frame:06d>.jpg
  std::filesystem::path out;        // detections.jsonl to write
  ExecutionProvider provider = ExecutionProvider::Cpu;
  DecodeParams decode;
};

// Runs the ONNX student model on every frame listed in framesDir/frames.csv and writes their
// detections (src "student") to `out`, for comparing the C++ and Python inference paths. `out` is
// written as <out>.partial and renamed on success, so a failed run leaves no output. Returns the
// number of images. Throws VcError, also when frames.csv lists no frames at all.
int64_t runDetectImages(const DetectImagesOptions& options);

}  // namespace vc
