#include "detect/verifier.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>

#include <nlohmann/json.hpp>

#include "common/vc_error.h"
#include "detect/ort_session.h"

namespace vc {
namespace fs = std::filesystem;
namespace {

const nlohmann::json& member(const nlohmann::json& j, const char* key, const std::string& origin) {
  if (!j.is_object()) throw VcError(origin + ": expected an object holding " + key);
  const auto it = j.find(key);
  if (it == j.end()) throw VcError(origin + ": missing " + key);
  return *it;
}

int positiveInt(const nlohmann::json& j, const char* key, const std::string& origin) {
  const nlohmann::json& v = member(j, key, origin);
  if (!v.is_number_integer() || v.get<int64_t>() < 1 || v.get<int64_t>() > std::numeric_limits<int>::max()) {
    throw VcError(origin + ": " + key + " must be a positive integer");
  }
  return static_cast<int>(v.get<int64_t>());
}

double number(const nlohmann::json& j, const char* key, const std::string& origin) {
  const nlohmann::json& v = member(j, key, origin);
  if (!v.is_number() || !std::isfinite(v.get<double>())) throw VcError(origin + ": " + key + " must be a number");
  return v.get<double>();
}

std::string text(const nlohmann::json& j, const char* key, const std::string& origin) {
  const nlohmann::json& v = member(j, key, origin);
  if (!v.is_string()) throw VcError(origin + ": " + key + " must be a string");
  return v.get<std::string>();
}

}  // namespace

VerifierInfo parseVerifierInfo(const std::string& json, const std::string& origin) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(json);
  } catch (const nlohmann::json::exception& e) {
    throw VcError(origin + ": " + e.what());
  }
  if (!j.is_object()) throw VcError(origin + ": the root must be a JSON object");
  if (text(j, "kind", origin) != "verifier") throw VcError(origin + ": kind must be \"verifier\"");
  VerifierInfo info;
  info.name = text(j, "name", origin);
  const nlohmann::json& classes = member(j, "classes", origin);
  if (!classes.is_array() || classes.empty()) throw VcError(origin + ": classes must be a non-empty array");
  for (const auto& c : classes) {
    if (!c.is_string()) throw VcError(origin + ": classes must be strings");
    info.classes.push_back(c.get<std::string>());
  }
  const nlohmann::json& input = member(j, "input", origin);
  info.inputSize = positiveInt(input, "size", origin);
  info.batch = positiveInt(input, "batch", origin);
  if (text(input, "color", origin) != "RGB") throw VcError(origin + ": input color must be \"RGB\"");
  if (std::abs(number(input, "scale", origin) - 1.0 / 255.0) > 1e-9) {
    throw VcError(origin + ": input scale must be 1/255");
  }
  const nlohmann::json& pad = member(input, "pad_value", origin);
  if (!pad.is_number_integer() || pad.get<int64_t>() < 0 || pad.get<int64_t>() > 255) {
    throw VcError(origin + ": pad_value must be an integer in [0, 255]");
  }
  info.padValue = static_cast<int>(pad.get<int64_t>());
  const nlohmann::json& crop = member(j, "crop", origin);
  info.cropFactor = number(crop, "factor", origin);
  info.minCropPx = number(crop, "min_px", origin);
  if (!(info.cropFactor > 0) || info.minCropPx < 0) throw VcError(origin + ": crop factor must be > 0 and min_px >= 0");
  const nlohmann::json& thresholds = member(j, "pass_threshold", origin);
  for (const std::string& c : info.classes) {
    const double t = number(thresholds, c.c_str(), origin);
    if (t < 0 || t > 1) throw VcError(origin + ": pass_threshold of " + c + " must be in [0, 1]");
    info.passThreshold.push_back(static_cast<float>(t));
  }
  return info;
}

VerifierInfo readVerifierInfo(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw VcError("cannot read " + path.string());
  std::ostringstream content;
  content << in.rdbuf();
  return parseVerifierInfo(content.str(), path.string());
}

struct Verifier::Impl {
  VerifierInfo info;
  Ort::Session session{nullptr};
  std::string backend = "CPU";
  std::string inputName;
  std::string outputName;
  std::vector<float> input;
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  void checkShapes(const fs::path& onnx) {
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
      throw VcError(onnx.string() + ": expected one input and one output");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    inputName = session.GetInputNameAllocated(0, allocator).get();
    outputName = session.GetOutputNameAllocated(0, allocator).get();
    const std::vector<int64_t> in = readFloatTensorSpec(session.GetInputTypeInfo(0), "input", onnx).shape;
    const std::vector<int64_t> wantIn{info.batch, 3, info.inputSize, info.inputSize};
    if (in != wantIn) {
      throw VcError(onnx.string() + ": input shape " + shapeText(in) + " does not match model.json " +
                    shapeText(wantIn));
    }
    const std::vector<int64_t> out = readFloatTensorSpec(session.GetOutputTypeInfo(0), "output", onnx).shape;
    const std::vector<int64_t> wantOut{info.batch, static_cast<int64_t>(info.classes.size())};
    if (out != wantOut) {
      throw VcError(onnx.string() + ": output shape " + shapeText(out) + " does not match model.json " +
                    shapeText(wantOut));
    }
  }
};

Verifier::Verifier(const fs::path& modelDir, ExecutionProvider provider, const SessionSettings& settings)
    : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  s.info = readVerifierInfo(modelDir / "model.json");
  const fs::path onnx = modelDir / "model.onnx";
  if (!fs::is_regular_file(onnx)) throw VcError("missing " + onnx.string());
  const int size = s.info.inputSize;
  try {
    const SessionCaches caches{[&] { return coreMlCacheDirectory(onnx); },
                               [&] { return tensorRtCacheDirectory(onnx, size, size); }};
    CreatedSession created = createOrtSession(onnx, provider, caches, settings);
    s.session = std::move(created.session);
    s.backend = created.backend;
    s.checkShapes(onnx);  // its VcError is a model problem, never retried
  } catch (const Ort::Exception& e) {
    throw VcError("onnxruntime: " + onnx.string() + ": " + e.what());
  }
  s.input.resize(static_cast<size_t>(s.info.batch) * 3 * size * size);
}

Verifier::~Verifier() = default;

const VerifierInfo& Verifier::info() const { return impl_->info; }

std::string Verifier::backend() const { return impl_->backend; }

std::vector<float> Verifier::score(const std::vector<VerifierInput>& inputs) {
  Impl& s = *impl_;
  const size_t per = 3 * static_cast<size_t>(s.info.inputSize) * s.info.inputSize;
  const size_t classes = s.info.classes.size();
  const auto batch = static_cast<size_t>(s.info.batch);
  for (const VerifierInput& in : inputs) {
    if (in.cls < 0 || static_cast<size_t>(in.cls) >= classes) throw VcError("verifier: class index out of range");
  }
  std::vector<float> scores;
  scores.reserve(inputs.size());
  for (size_t start = 0; start < inputs.size(); start += batch) {
    const size_t count = std::min(batch, inputs.size() - start);
    std::fill(s.input.begin(), s.input.end(), static_cast<float>(s.info.padValue) / 255.0f);
    for (size_t i = 0; i < count; ++i) std::copy_n(inputs[start + i].pixels, per, s.input.data() + i * per);
    try {
      const std::array<int64_t, 4> shape{s.info.batch, 3, s.info.inputSize, s.info.inputSize};
      Ort::Value tensor =
          Ort::Value::CreateTensor<float>(s.memory, s.input.data(), s.input.size(), shape.data(), shape.size());
      const char* inNames[] = {s.inputName.c_str()};
      const char* outNames[] = {s.outputName.c_str()};
      std::vector<Ort::Value> out = s.session.Run(Ort::RunOptions{nullptr}, inNames, &tensor, 1, outNames, 1);
      const float* logits = out[0].GetTensorData<float>();
      for (size_t i = 0; i < count; ++i) {
        const float logit = logits[i * classes + static_cast<size_t>(inputs[start + i].cls)];
        scores.push_back(1.0f / (1.0f + std::exp(-logit)));
      }
    } catch (const Ort::Exception& e) {
      throw VcError(std::string("onnxruntime: verifier inference failed: ") + e.what());
    }
  }
  return scores;
}

}  // namespace vc
