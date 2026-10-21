#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/platform.h"

namespace vc {

// --ep (spec 2026-10-20 §3). Auto tries TensorRT, CUDA, then the CPU on Linux and CoreML, then the CPU on macOS;
// an explicit provider that cannot be used falls back to the CPU.
enum class ExecutionProvider { Auto, TensorRT, Cuda, CoreML, Cpu };

// "auto" | "tensorrt" | "cuda" | "coreml" | "cpu"; throws std::invalid_argument.
ExecutionProvider parseExecutionProvider(const std::string& text);

// The name a detector's backend() and stats.json's detector use: "TensorRT", "CUDA", "CoreML", "CPU" ("auto" for
// Auto, which never runs a session).
std::string providerLabel(ExecutionProvider provider);

// The providers createOrtSession tries for `requested`, in order, always ending with the CPU: Auto is TensorRT,
// CUDA, CPU on Linux, CoreML, CPU on macOS and the CPU alone elsewhere; any other provider is tried before the
// CPU (the CPU alone for Cpu).
std::vector<ExecutionProvider> providerOrder(ExecutionProvider requested, Platform platform);

// Settings every onnxruntime session of a run shares (configs/redact.json).
struct SessionSettings {
  int intraOpThreads = 0;    // ort_intra_threads: 0 keeps onnxruntime's default, more sets the intra-op threads
  bool tensorRtFp16 = true;  // trt_fp16, for the TensorRT provider (stage 1c of the Linux plan)
};

// Where CoreML keeps the model it compiles from `onnx`: $VC_CACHE/coreml-cache/<sha256 of the file>,
// VC_CACHE defaulting to ~/.cache/video-compliance (shared by all checkouts). Keyed by the content,
// so a changed model is compiled afresh. nullopt when neither VC_CACHE nor HOME is set. Throws
// VcError when the file cannot be read.
std::optional<std::filesystem::path> coreMlCacheDirectory(const std::filesystem::path& onnx);

// Where TensorRT keeps the engines it builds from `onnx` for a width x height input (spec 2026-10-20 §3):
// $VC_CACHE/trt-cache/<sha256 of the file>/<width>x<height>, VC_CACHE defaulting as for coreMlCacheDirectory. An
// engine is fixed to its model and input size, so each gets its own directory. nullopt / VcError as
// coreMlCacheDirectory. Creates nothing.
std::optional<std::filesystem::path> tensorRtCacheDirectory(const std::filesystem::path& onnx, int width, int height);

}  // namespace vc
