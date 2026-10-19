#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "detect/detector.h"
#include "detect/execution_provider.h"
#include "detect/yunet_decode.h"

namespace vc {

// Where CoreML keeps the face model compiled for a padWidth x padHeight input:
// coreMlCacheDirectory(onnx) / "<padWidth>x<padHeight>". The compiled model is fixed to one input size,
// so each size gets its own directory. nullopt / VcError as coreMlCacheDirectory.
std::optional<std::filesystem::path> yunetCoreMlCacheDirectory(const std::filesystem::path& onnx, int padWidth,
                                                              int padHeight);

// The YuNet face detector (OpenCV Zoo, MIT; model.onnx + model.json from `vc_train yunet`) on
// onnxruntime. It sees the frame at its own resolution: converted to BGR, as floats 0-255 (no scale,
// no mean), zero-padded on the right and bottom to multiples of 32 (a 3840x2160 frame is a 3840x2176
// input), and decodes with decodeYuNet. The session is set up like OnnxYoloDetector's (shared
// onnxruntime environment with telemetry off, CoreML with the compile cache, one retry without it,
// then the CPU); the input height and width are fixed for it (AddFreeDimensionOverrideByName "H" /
// "W"), so CoreML sees static shapes, and the session is rebuilt when a frame of another size comes.
// A model whose input is fixed (the OpenCV Zoo file itself, 640x640) is accepted only for frames that
// pad to exactly that size. classes() is model.json's single class ("face"); source() "yunet".
// detect() reuses internal buffers and is not thread-safe.
class YuNetDetector : public Detector {
 public:
  // Sets the session up for frameWidth x frameHeight (CoreML compiles here). Throws VcError when the
  // files are missing or invalid (model.json must say output_format "yunet"), when the ONNX lacks
  // YuNet's float32 input or the outputs cls_8 ... bbox_32, when a fixed input does not fit, or when
  // onnxruntime fails (also on the CPU retry).
  YuNetDetector(const std::filesystem::path& modelDir, ExecutionProvider provider, const YuNetParams& params,
                int frameWidth, int frameHeight);
  ~YuNetDetector() override;
  YuNetDetector(const YuNetDetector&) = delete;
  YuNetDetector& operator=(const YuNetDetector&) = delete;

  // Any pixel format FFmpeg converts to BGR24. Throws VcError when the frame has no pixels, when an
  // output does not have the expected anchor count, for non-finite values (decodeYuNet), and when
  // onnxruntime fails.
  std::vector<Box> detect(const VideoFrame& frame) override;
  const std::vector<std::string>& classes() const override;
  std::string source() const override { return "yunet"; }
  // Of the current session ("CoreML" or "CPU").
  std::string backend() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vc
