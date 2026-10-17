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
// frames.csv, where <stem> is the input's file stem.
//
// The input is opened and every check below runs before anything is deleted or
// created, so a bad input or a refused output never destroys existing files:
//  - an input whose stem is empty, "." or ".." has no usable directory name
//    -> VcError;
//  - an existing <outRoot>/<stem> -> OutputExistsError unless options.overwrite;
//    with overwrite it is replaced only if it is a directory that is empty or
//    holds a frames.csv (a previous vc_extract output), otherwise
//    OutputExistsError ("not a vc_extract output directory");
//  - an output directory that is, or would contain, the input file -> VcError.
//
// Throws OutputExistsError or VcError for the cases above and for I/O and
// decode failures; std::invalid_argument for an invalid sampler or JPEG
// setting; std::filesystem::filesystem_error when the filesystem cannot be
// inspected or modified.
ExtractStats runExtract(const ExtractOptions& options);

}  // namespace vc
