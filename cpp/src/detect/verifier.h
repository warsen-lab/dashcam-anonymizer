#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "detect/execution_provider.h"
#include "detect/verifier_crop.h"

namespace vc {

// The model.json of a verifier (`vc_train verifier`, spec 2026-10-21 §4.1).
struct VerifierInfo {
  std::string name;
  std::vector<std::string> classes;  // the order of the model's logits
  int inputSize = 0;                 // crops are inputSize x inputSize RGB
  int batch = 0;                     // the model's fixed batch
  int padValue = 114;
  double cropFactor = 2.5;
  double minCropPx = 32;
  std::vector<float> passThreshold;  // per class, in `classes` order, in [0, 1]

  CropParams crop() const { return CropParams{inputSize, cropFactor, minCropPx, padValue}; }
};

// Throws VcError("<origin>: …") for invalid JSON, a kind other than "verifier", a color other than "RGB", a scale
// other than 1/255, sizes that are not positive JSON integers, a pad_value outside [0, 255], a crop factor not > 0,
// a min_px < 0, or a class without a pass_threshold in [0, 1].
VerifierInfo parseVerifierInfo(const std::string& text, const std::string& origin);
VerifierInfo readVerifierInfo(const std::filesystem::path& path);

struct VerifierInput {
  int cls = 0;                    // index into VerifierInfo::classes
  const float* pixels = nullptr;  // 3 * inputSize * inputSize floats (toPlanarFloat)
};

// The verifier's onnxruntime session: model.onnx with input [batch, 3, size, size] and output [batch, n classes],
// both float32. score() is not thread-safe.
class Verifier {
 public:
  // Throws VcError when the files are missing or invalid, when the ONNX shapes disagree with model.json, or when
  // onnxruntime fails (also on the CPU fallback).
  Verifier(const std::filesystem::path& modelDir, ExecutionProvider provider, const SessionSettings& settings);
  ~Verifier();
  Verifier(const Verifier&) = delete;
  Verifier& operator=(const Verifier&) = delete;

  const VerifierInfo& info() const;
  std::string backend() const;  // providerLabel of the session's provider
  // The pass probability of each input: the sigmoid of its class's logit, in input order. Runs ceil(n / batch)
  // batches, the last one padded with pad-value crops. Throws VcError when a class index is out of range or
  // inference fails.
  std::vector<float> score(const std::vector<VerifierInput>& inputs);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vc
