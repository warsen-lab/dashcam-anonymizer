#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace vc {

enum class ExecutionProvider { CoreML, Cpu };

// "coreml" | "cpu"; throws std::invalid_argument.
ExecutionProvider parseExecutionProvider(const std::string& text);

// Where CoreML keeps the model it compiles from `onnx`: $VC_CACHE/coreml-cache/<sha256 of the file>,
// VC_CACHE defaulting to ~/.cache/video-compliance (shared by all checkouts). Keyed by the content,
// so a changed model is compiled afresh. nullopt when neither VC_CACHE nor HOME is set. Throws
// VcError when the file cannot be read.
std::optional<std::filesystem::path> coreMlCacheDirectory(const std::filesystem::path& onnx);

}  // namespace vc
