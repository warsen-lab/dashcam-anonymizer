#pragma once

#include <cstdint>
#include <filesystem>

#include "common/vc_error.h"
#include "vc_extract/extract_options.h"

namespace vc {

// The output directory already exists and overwrite was not requested.
class OutputExistsError : public VcError {
 public:
  using VcError::VcError;
};

struct ExtractStats {
  std::filesystem::path outputDir;
  int64_t decodedFrames = 0;
  int64_t keptFrames = 0;
  int64_t keptBase = 0;
  int64_t keptMotion = 0;
  int64_t keptClip = 0;
  int64_t decodeErrors = 0;
  bool usedHardware = false;
  double elapsedSec = 0;
};

// Decodes options.input and writes <outRoot>/<stem>/<frame:06d>.jpg plus
// frames.csv. The input is opened before the output directory is touched, so a
// bad input never deletes existing output. Throws OutputExistsError or VcError.
ExtractStats runExtract(const ExtractOptions& options);

}  // namespace vc
